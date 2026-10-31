"""Explicit PC FP32 nodule detection with a supplied predicted lung mask.

No CT file reader, lung U-Net, annotations or FPGA commands are accepted here.
Torch is imported only when detection is requested, preserving desktop browsing.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT/'data/lung_nodule_five_bundle_20261021_v1/model.pt'
RANKER = ROOT/'data/lung_nodule_continuity_20261019_boundary_v2/ranker.json'
MODEL_SHA = 'c1fefe55a74d1cdc81ede2f2e249190aef7d0b71a8a81378c950427ddcd8dc06'
RANKER_SHA = 'b7afba327d6375539b745b40173e7d3903d7e0899665a4c2e903aa93fc747e67'
THRESHOLD = 0.6418673229668569


def array_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


class NoduleService:
    def __init__(self, model_path=None):
        self.path = Path(model_path or DEFAULT_MODEL).resolve()
        self.net = self.config = self.device = None

    def availability(self):
        reason = ('缺少结节模型，请安装已核验的445例训练权重' if not self.path.is_file()
                  else '缺少冻结跨层评分文件' if not RANKER.is_file()
                  else '当前Python环境未安装PyTorch' if importlib.util.find_spec('torch') is None else None)
        return dict(available=reason is None, reason=reason, backend='PC_CNN_FP32',
                    model_sha256=MODEL_SHA, threshold=THRESHOLD)

    def predict(self, hu, lung_mask, geometry, progress, cancel):
        available = self.availability()
        if not available['available']:
            raise ValueError(available['reason'])
        if (hashlib.sha256(self.path.read_bytes()).hexdigest() != MODEL_SHA or
                hashlib.sha256(RANKER.read_bytes()).hexdigest() != RANKER_SHA):
            raise ValueError('结节模型或跨层评分文件身份不匹配，未执行检测')
        hu, lung_mask = np.asarray(hu), np.asarray(lung_mask)
        spacing = np.asarray(geometry['spacing_zyx_mm'], float)
        if (hu.ndim != 3 or len(hu) < 2 or hu.shape != lung_mask.shape or
                lung_mask.dtype != np.bool_ or not np.isfinite(hu).all() or
                spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any()):
            raise ValueError('需要完整连续CT、同尺寸预测肺掩膜和毫米几何信息')
        import torch
        from lung_nodule_cnn import model, config_from_dict, DetectorConfig, search_mask, tiles, decode, consolidate
        from lung_nodule_whole_background import FloatPlaneCache
        from lung_nodule_sampling_cache import sample_patch_reuse
        from lung_nodule_continuity_boundary import evidence_for_packet
        from lung_nodule_continuity import rerank_scores
        def check():
            if cancel.is_set():
                raise InterruptedError('结节检测已在安全边界停止')
        check(); start = perf_counter(); load_start = start
        if self.net is None:
            cp = torch.load(self.path, map_location='cpu', weights_only=True)
            cfg = config_from_dict(cp['config'])
            if cfg != DetectorConfig().checked():
                raise ValueError('结节网络配置与已验证五层模型不同')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            net = model(cfg).to(device).float().eval(); net.load_state_dict(cp['state'])
            self.net, self.config, self.device = net, cfg, device
        load_ms = (perf_counter()-load_start)*1000
        old_tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.inference_mode():
                progress('生成肺内搜索范围', 0, 1); tick = perf_counter()
                search = search_mask(lung_mask, spacing, self.config)
                jobs, schedule = tiles(search, spacing, self.config)
                search_ms = (perf_counter()-tick)*1000; check()
                cached = FloatPlaneCache(hu); accumulated = [[], [], []]
                progress('CNN检测', 0, len(jobs)); tick = perf_counter()
                for begin in range(0, len(jobs), 48):
                    check(); batch = jobs[begin:begin+48]
                    image = np.stack([sample_patch_reuse(cached, spacing, j, self.config) for j in batch])
                    tensor = torch.from_numpy(image).to(self.device).float().clamp(-1000,400).add_(1000).div_(1400)
                    prediction = self.net(tensor).cpu().numpy()
                    for dest, values in zip(accumulated, decode(prediction, batch, geometry, self.config, search)):
                        dest.append(values)
                    progress('CNN检测', min(begin+48,len(jobs)), len(jobs))
                forward_decode_ms = (perf_counter()-tick)*1000; check()
                progress('合并与跨层评分', len(jobs), len(jobs)); tick = perf_counter()
                candidates = consolidate(*(np.concatenate(v) for v in accumulated), geometry, self.config) if jobs else []
                for c in candidates:
                    box = np.asarray(c['bbox_zyx']).reshape(3,2)
                    box[:,0] = np.maximum(box[:,0],0); box[:,1] = np.minimum(box[:,1],hu.shape)
                    c['bbox_zyx'] = box.ravel().tolist()
                features, _ = evidence_for_packet(hu, spacing, candidates); check()
                scores = np.array([c['score'] for c in candidates])
                cross = rerank_scores(scores, features, json.loads(RANKER.read_text(encoding='utf-8')))
                selected = []
                for i in np.flatnonzero(cross >= THRESHOLD):
                    c = dict(candidates[i]); c['raw_score'] = c['score']; c['score'] = float(cross[i])
                    c['score_kind'] = 'frozen_cross_layer_score_not_malignancy_probability'
                    selected.append(c)
                selected.sort(key=lambda c:(-c['score'], c['candidate_id']))
                post_ms = (perf_counter()-tick)*1000
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old_tf32
        check()
        return dict(schema='ct_nodule_prediction_v1', backend='PC_CNN_FP32', device=self.device,
            model_sha256=MODEL_SHA, ranker_sha256=RANKER_SHA, threshold=THRESHOLD,
            candidates=selected, candidate_count=len(selected), raw_candidate_count=len(candidates), schedule=schedule,
            input_hu_sha256=array_sha(hu), lung_mask_sha256=array_sha(lung_mask),
            segmentation_reused=True, lung_inference_rerun=False, annotations_used=False,
            timing=dict(model_load_ms=load_ms, search_ms=search_ms,
                        sampling_forward_decode_ms=forward_decode_ms, merge_evidence_scoring_ms=post_ms,
                        inference_ms=(perf_counter()-start)*1000,
                        scope='电脑FP32；复用当前CT和人体约束后肺掩膜；含模型首次加载，不含肺分割、读CT、落盘、通信和显示'),
            note='候选中心及大小，不是精确结节轮廓；分数不是恶性概率。当前肺掩膜来源可能不同于开发评价，不直接套用历史检出率。')
