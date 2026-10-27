"""Lung-guided full-case canvas PL detector plus frozen cross-layer scoring."""
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
import json
from pathlib import Path
import time
import numpy as np
from ct_dual_io_client import CachedCanvasClient as CanvasClient
from nodule_canvas_contract import digest, shapes
from nodule_exact_host import quantize_exact as quantize_canvas, search_mask_exact as search_mask, ExactSamplerCache
from dual_io_contract import INTERFACE
from shared_patch_contract import validate_assets
from nodule_vendor.nodule_canvas import lower_canvas, extract_tile_heads, valid_dependency_proof
from nodule_vendor.nodule_fast_input import SharedPlaneSampler
from nodule_vendor.lung_nodule_cnn import config_from_dict, tiles, decode, consolidate
from nodule_frozen_scoring import load_scoring, score_candidates

PIPELINE = 'int8_canvas_cached_exact_frozen_cross_layer_v3'


class Cancelled(Exception): pass


def plane_groups(jobs, config):
    """Keep every original job and ordering; split rectangles only for bounds."""
    def split(group):
        y0 = min(j[1] for j in group); y1 = max(j[1] for j in group)+config.patch
        x0 = min(j[2] for j in group); x1 = max(j[2] for j in group)+config.patch
        if max(y1-y0, x1-x0) <= 1024:
            shapes(y1-y0, x1-x0)
            yield group
        else:
            # Contiguous subsequences preserve original tile decode/merge order.
            if len(group) < 2: raise ValueError('Unsupported canvas geometry')
            half = len(group)//2
            yield from split(group[:half]); yield from split(group[half:])
    for _, values in groupby(jobs, key=lambda j: j[0]):
        yield from split(list(values))


class NoduleCaseProvider:
    def __init__(self, package, legacy_client):
        self.package = Path(package).resolve()
        self.bundle = json.loads((self.package/'shared_host_platform.json').read_text())
        self.model = validate_assets(self.package/'ct_studio', self.bundle['shared_nodule'])
        self.config = config_from_dict(self.model['detector_config'])
        self.client = CanvasClient(legacy_client.url, legacy_client.token, self.bundle)
        self.scoring, self.ranker = load_scoring(self.package)
        self.proof = valid_dependency_proof(self.model)
        self.default_scoring_mode = 'frozen_cross_layer'
        self.code_hashes = {str(p.relative_to(Path(__file__).parent)):digest(p.read_bytes()) for p in
            [Path(__file__),Path(__file__).with_name('nodule_frozen_scoring.py'),Path(__file__).with_name('nodule_exact_host.py')]+list(
                Path(__file__).parent.joinpath('nodule_vendor').glob('nodule_canvas.py'))}

    def run(self, hu, lung, geometry, context, folder, progress, cancelled):
        self.client.health()
        started = time.perf_counter()
        folder = Path(folder)
        def check():
            if cancelled(): raise Cancelled('已在安全整层边界停止；未发布部分候选')
        check(); progress('search', 0, 0)
        tick = time.perf_counter()
        search = search_mask(lung, geometry['spacing_zyx_mm'], self.config)
        jobs, schedule = tiles(search, geometry['spacing_zyx_mm'], self.config)
        groups = list(plane_groups(jobs, self.config))
        timing = dict(search_ms=(time.perf_counter()-tick)*1000, sampling_ms=0., sampling_wait_ms=0.,
            canvas_roundtrip_ms=0., network_ms=0., pl_active_ms=0., canvas_setup_ms=0., decode_ms=0., merge_ms=0.)
        points = [[], [], []]
        sampler_cache=ExactSamplerCache(hu,geometry['spacing_zyx_mm'])
        def prepare(index):
            check(); tick = time.perf_counter()
            group = groups[index]
            sampler = sampler_cache.sample(group,self.config)
            h, w = sampler.context.shape[1:]
            codes = quantize_canvas(sampler.context)
            return group, sampler.y0, sampler.x0, h, w, codes, (time.perf_counter()-tick)*1000
        done = 0
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='canvas-sample') as producer, \
                (folder/'canvas_receipts.jsonl').open('x', encoding='utf8') as receipts:
            pending = producer.submit(prepare, 0) if groups else None
            for index in range(len(groups)):
                check(); progress('inference', done, len(jobs)); tick = time.perf_counter()
                group, y0, x0, h, w, codes, sample_ms = pending.result()
                timing['sampling_wait_ms'] += (time.perf_counter()-tick)*1000
                timing['sampling_ms'] += sample_ms
                pending = producer.submit(prepare, index+1) if index+1 < len(groups) else None
                head, receipt = self.client.predict(codes, h, w)
                receipts.write(json.dumps(dict(index=index, jobs=group, y0=y0, x0=x0, receipt=receipt))+'\n')
                receipts.flush()
                for dst, key in (('canvas_roundtrip_ms','roundtrip_ms'),): timing[dst] += receipt[key]
                timing['network_ms'] += receipt['stats']['host_total_ms']
                timing['pl_active_ms'] += receipt['stats']['pl_command_cycles']/100000.
                timing['canvas_setup_ms'] += receipt['stats']['canvas_setup_ms']
                tick = time.perf_counter()
                pred = (head.astype(np.float32)-128)*np.asarray(self.model['head_scales'],np.float32)[:,None,None]
                recovered = extract_tile_heads(pred, group, y0, x0, self.config)
                for dest, values in zip(points, decode(recovered, group, geometry, self.config, search)):
                    dest.append(values)
                timing['decode_ms'] += (time.perf_counter()-tick)*1000
                done += len(group)
                progress('inference', done, len(jobs))
                (folder/'timing_progress.json').write_text(json.dumps(dict(completed=done,total=len(jobs),
                    canvases_completed=index+1,canvases_total=len(groups),timing=timing,elapsed_ms=(time.perf_counter()-started)*1000)))
        check(); progress('merge', len(jobs), len(jobs)); tick = time.perf_counter()
        candidates = consolidate(*(np.concatenate(p) for p in points), geometry, self.config) if jobs else []
        timing['merge_ms'] = (time.perf_counter()-tick)*1000
        if len(candidates) > 10000: raise ValueError('Raw candidate safety bound exceeded; no silent truncation')
        progress('cross_layer_scoring', len(jobs), len(jobs))
        try:
            selected, score_record = score_candidates(hu, geometry['spacing_zyx_mm'], candidates,
                self.scoring, self.ranker, folder, cancelled)
        except InterruptedError as exc: raise Cancelled(str(exc))
        check()
        timing['scoring_ms'] = score_record['scoring_ms']
        timing['whole_case_ms'] = (time.perf_counter()-started)*1000
        provenance = dict(backend='FPGA_NODULE_SHARED',pipeline=PIPELINE,canvas_interface=INTERFACE,
            scoring_mode='frozen_cross_layer',detection_quality_accepted=False,
            input_conditioning='original' if context['analysis_source_id']=='original' else 'enhanced_unvalidated',
            source_hu_sha256=context['source_hu_sha256'],lung_mask_sha256=digest(np.ascontiguousarray(lung).tobytes()),
            lung_mask_stage='body',geometry_sha256=context['geometry_sha256'],segmentation_signature=context['segmentation_signature'],
            service_bundle_sha256=self.bundle['service_bundle_sha256'],shared_nodule=self.bundle['shared_nodule'],
            detector_config=self.model['detector_config'],sampling_decode_code_sha256=self.code_hashes,
            raw_decode_floor=self.config.decode_floor,additional_score_filter=self.scoring['threshold'],
            ranker_applied=True,score_filter=score_record,schedule=schedule,timing=timing,
            transport=dict(canvases=len(groups),original_tiles=len(jobs),prefetch_canvases=1,
                sampling_overlaps_transfer=True,input='zlib-five-plane-uint8-native-CMA-binary-persistent',no_patch_fallback=True),
            equivalence=dict(scope='Only decoder-required [15:49] neighbourhood, not unused patch borders',
                dependency_proof=self.proof['status'],tile_order_preserved=True),
            note='实际FPGA整层复用推理＋冻结跨层评分；分数不是恶性概率；新版整条链路尚待实板质量验收。')
        return selected, provenance
