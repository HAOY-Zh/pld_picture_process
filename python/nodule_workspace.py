"""Context-bound candidates and explicit whole-case FPGA raw-CNN experiments."""
import copy
import base64
import hashlib
import json
import math
import re
import uuid
import threading
import time

import numpy as np

SCHEMA = 'ct_nodule_prediction_v1'
REGION = 'predicted_center_and_size_not_voxel_segmentation'
BUSY = ('loading', 'connecting', 'computing', 'reconstructing', 'processing')
IDENTIFIER = re.compile(r'[A-Za-z0-9_.:-]{1,128}\Z')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def identifier(value, field):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError('无效候选结果标识：' + field)
    return value


def numbers(value, length, field, integer=False):
    try:
        valid = (isinstance(value, list) and len(value) == length and
            all(type(v) in ((int,) if integer else (int, float)) and math.isfinite(v) for v in value))
    except (OverflowError, TypeError): valid = False
    if not valid:
        raise ValueError('无效候选数值：' + field)
    return list(value)


def display_bounds(candidate, shape, spacing, padding_mm):
    if type(padding_mm) not in (int, float) or not math.isfinite(padding_mm) or not 0 <= padding_mm <= 50:
        raise ValueError('ROI 观察边界须在 0 至 50 mm 内')
    box = candidate['bbox_zyx']
    return [max(0, min(shape[a], (box[2*a+j] + (1 if j else -1)*math.ceil(padding_mm/spacing[a]))))
            for a in range(3) for j in range(2)]


def validate_packet(packet, context, *, allow_enhanced=False):
    if not isinstance(packet, dict) or packet.get('schema') != SCHEMA:
        raise ValueError('需要 ct_nodule_prediction_v1 候选结果')
    for key in ('case_id', 'analysis_source_id', 'context_id', 'source_hu_sha256', 'geometry_sha256', 'segmentation_signature'):
        if packet.get(key) != context.get(key):
            raise ValueError('候选的病例、来源、几何或分割版本已失效：' + key)
    identifier(packet.get('result_id'), 'result_id')
    if type(packet.get('is_test_data')) is not bool:
        raise ValueError('必须明确声明 is_test_data；测试候选不能作为检出结果')
    if not packet['is_test_data'] and ((context['analysis_source_id'] != 'original' and not allow_enhanced) or not context['whole_case_ready']):
        raise ValueError('真实整病例候选需要完整原始 CT 和完整肺分割')
    candidates = packet.get('candidates')
    if (not isinstance(candidates, list) or len(candidates) > 10000 or
            type(packet.get('candidate_count')) is not int or packet['candidate_count'] != len(candidates)):
        raise ValueError('候选数量不匹配或超过 10000')
    result = copy.deepcopy(packet)
    seen = set()
    geom, shape = context['geometry'], context['shape']
    for c in result['candidates']:
        if not isinstance(c, dict): raise ValueError('候选须为对象')
        cid = identifier(c.get('candidate_id'), 'candidate_id')
        if cid in seen: raise ValueError('候选编号重复')
        seen.add(cid)
        center = numbers(c.get('center_index_zyx'), 3, 'center_index_zyx')
        box = numbers(c.get('bbox_zyx'), 6, 'bbox_zyx', integer=True)
        world = numbers(c.get('center_lps_xyz_mm'), 3, 'center_lps_xyz_mm')
        for a, size in enumerate(shape):
            # Match the detector's nearest-native-voxel gate, including fractional
            # centers at the half-voxel boundary. Never move the model's center.
            if not 0 <= int(np.rint(center[a])) < size or not box[2*a] <= center[a] < box[2*a+1]:
                raise ValueError('候选中心越界或不在原始包围框内')
            if max(0,box[2*a]) >= min(size,box[2*a+1]):
                raise ValueError('候选框与原生 CT 没有可显示的交集')
            if max(abs(box[2*a]), abs(box[2*a+1])) > 2**31-1:
                raise ValueError('候选框坐标过大')
        expected = np.asarray(geom['origin_lps_mm']) + np.asarray(geom['basis_lps_zyx']) @ center
        if not np.allclose(expected, world, atol=.05, rtol=0):
            raise ValueError('候选 LPS 中心与当前 CT 几何不一致')
        diameter = numbers([c.get('estimated_scale_diameter_mm')], 1, 'estimated_scale_diameter_mm')[0]
        if diameter <= 0: raise ValueError('估计直径须为正数')
        numbers([c.get('score')], 1, 'score')
        if 'raw_score' in c: numbers([c['raw_score']], 1, 'raw_score')
        if not isinstance(c.get('score_kind'), str) or not 0 < len(c['score_kind']) <= 256:
            raise ValueError('需要 score_kind 说明最终评分含义')
        if c.get('region_kind') != REGION: raise ValueError('当前只支持中心和大小估计候选区域')
    # Keep service order, IDs, original out-of-bounds boxes and scores unchanged.
    result['ingestion_kind'] = 'test_fixture' if packet['is_test_data'] else 'imported_result'
    return result


def test_packet(context, empty=False):
    """Explicit analytic UI fixture, independent of HU/annotations/model output."""
    result = {k: copy.deepcopy(context[k]) for k in ('case_id', 'analysis_source_id', 'context_id',
        'source_hu_sha256', 'geometry_sha256', 'segmentation_signature')}
    result.update(schema=SCHEMA, result_id='test_'+uuid.uuid4().hex, is_test_data=True,
        candidates=[], candidate_count=0, detector_provenance=dict(backend='SYNTHETIC_UI_TEST',
        note='由体数据尺寸生成的联动测试候选；未读取影像内容，未执行检测，不代表医学效果'))
    shape, geom = context['shape'], context['geometry']
    if not empty:
        for i, fractions in enumerate(((.35,.43,.61), (.58,.61,.34), (.76,.35,.69)), 1):
            center = [float(round((n-1)*f)) for n, f in zip(shape, fractions)]
            world = (np.asarray(geom['origin_lps_mm']) + np.asarray(geom['basis_lps_zyx']) @ center).tolist()
            box = [int(center[a]+sign*max(1, math.ceil(4/geom['spacing_zyx_mm'][a])))+(1 if sign==1 else 0)
                   for a in range(3) for sign in (-1,1)]
            result['candidates'].append(dict(candidate_id=f'test_{i:05d}', center_index_zyx=center,
                bbox_zyx=box, center_lps_xyz_mm=world, estimated_scale_diameter_mm=8., score=round(.9-i*.12,2),
                score_kind='synthetic_ui_test_score_not_detection', region_kind=REGION))
    result['candidate_count'] = len(result['candidates'])
    return result


class NoduleWorkspaceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodule_result = None
        self.nodule_context_id = None
        self.nodule_generation = 0
        self.nodule_packet_hashes = {}
        self.nodule_provider = None
        self.nodule_job = None
        self.nodule_cancel_event = threading.Event()

    def configure_nodule(self, package):
        if self.client is None:raise ValueError('真实结节检测需要匹配的统一 FPGA 服务')
        from nodule_detector import NoduleCaseProvider
        self.nodule_provider=NoduleCaseProvider(package,self.client)

    def nodule_context(self):
        with self.lock:
            if self.hu is None: return None
            segmentation = digest(dict(receipts=self.receipts, known=np.flatnonzero(self.known).tolist(),
                whole=bool(self.status.get('full_volume_ready')), generation=self.nodule_generation))
            context = dict(case_id=self.status['case_id'], analysis_source_id=self.analysis_source_id,
                source_hu_sha256=self.details['hu_sha256'], geometry_sha256=digest(self.geom),
                segmentation_signature=segmentation, shape=list(self.hu.shape), geometry=copy.deepcopy(self.geom),
                source_slice_indices=list(self.details['source_slice_indices']),
                whole_case_ready=bool(self.status.get('full_volume_ready') and not self.details.get('partial_input')),
                input_scope='partial_input' if self.details.get('partial_input') else 'loaded_input')
            context['context_id'] = digest(context)
            if context['context_id'] != self.nodule_context_id:
                self.nodule_result = None
                self.nodule_context_id = context['context_id']
            return context

    def snapshot(self):
        with self.lock:
            value = super().snapshot()
            context = self.nodule_context()
            packet = self.nodule_result
            value['nodule'] = dict(service_available=self.nodule_provider is not None,
                service_status=('原始 CNN 分数实验版：使用当前所选 HU；实际 FPGA 推理，检测质量尚未验收。'
                    if self.nodule_provider is not None else '等待整病例检测服务与冻结评分参数；可导入匹配结果进行联动'),
                job=copy.deepcopy(self.nodule_job),context=context,
                result={k:packet[k] for k in ('case_id','analysis_source_id','context_id','result_id',
                    'candidate_count','is_test_data','ingestion_kind')} if packet else None)
            return value

    def submit(self, action, values):
        result = super().submit(action, values)
        if action in ('load', 'run'):
            with self.lock:
                self.nodule_generation += 1
                self.nodule_result = None
                self._update()
                return self.snapshot()
        return result

    def nodule_action(self, action, values):
        with self.lock:
            if action=='cancel':
                if not self.nodule_job or self.nodule_job['status']!='RUNNING' or values.get('job_id')!=self.nodule_job['job_id']:
                    raise ValueError('结节任务不存在或已经结束')
                self.nodule_cancel_event.set();return self.snapshot()
            if self.status['phase'] in BUSY: raise BlockingIOError('请等待当前任务完成')
            context = self.nodule_context()
            if context is None or any(values.get(k) != context[k] for k in ('case_id','analysis_source_id','context_id')):
                raise ValueError('病例或候选上下文已变化，请重新获取当前上下文')
            if action == 'clear': self.nodule_result = None
            else:
                if action == 'fixture':
                    if type(values.get('empty', False)) is not bool: raise ValueError('empty 须为布尔值')
                    packet = test_packet(context, values.get('empty', False))
                else: packet = values.get('packet')
                record = validate_packet(packet, context)
                key = (record['context_id'],record['result_id'])
                fingerprint = digest(record)
                if key in self.nodule_packet_hashes and self.nodule_packet_hashes[key] != fingerprint:
                    raise ValueError('同一 result_id 的结果不可改写；重新检测必须使用新 result_id')
                folder = self.case_folder/'nodule'; folder.mkdir(exist_ok=True)
                with (folder/(uuid.uuid4().hex+'.json')).open('x', encoding='utf-8') as stream:
                    json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
                self.nodule_result = record
                self.nodule_packet_hashes[key] = fingerprint
            self._update()
            return self.snapshot()

    def start_nodule(self, values):
        with self.lock:
            if self.status['phase'] in BUSY:raise BlockingIOError('请等待当前任务完成')
            context=self.nodule_context()
            if context is None or any(values.get(k)!=context[k] for k in ('case_id','analysis_source_id','context_id')):
                raise ValueError('病例或当前 HU 来源已变化')
            if self.nodule_provider is None:raise ValueError('尚未配置真实检测服务')
            if not self.status.get('board_connected'):raise ValueError('先连接统一 FPGA 服务')
            if not context['whole_case_ready']:raise ValueError('先完成当前所选 CT 的整例肺分割')
            if values.get('scoring_mode')!='raw_cnn':raise ValueError('当前只提供明确的原始 CNN 实验模式')
            job_id='nodule_'+uuid.uuid4().hex
            folder=self.case_folder/'nodule'/job_id;folder.mkdir(parents=True,exist_ok=False)
            self.nodule_job=dict(job_id=job_id,status='RUNNING',stage='search',completed=0,total=0,error=None,
                analysis_source_id=self.analysis_source_id,case_id=context['case_id'],scoring_mode='raw_cnn')
            self.nodule_cancel_event.clear();self._update(phase='processing',error=None,completed=0,total=0)
            threading.Thread(target=self._nodule_work,args=(copy.deepcopy(context),folder,self.hu,self.body),daemon=True).start()
            return self.snapshot()

    def _nodule_work(self, context, folder, hu, lung):
        from nodule_detector import Cancelled
        began=time.perf_counter()
        def progress(stage,completed,total):
            waiting=time.perf_counter()
            with self.lock:
                waited=(time.perf_counter()-waiting)*1000
                self.nodule_job['progress_lock_wait_ms']=self.nodule_job.get('progress_lock_wait_ms',0.)+waited
                self.nodule_job.update(stage=stage,completed=completed,total=total,elapsed_ms=(time.perf_counter()-began)*1000)
                self._update(completed=completed,total=total)
        try:
            candidates,provenance=self.nodule_provider.run(hu,lung,context['geometry'],context,folder,progress,self.nodule_cancel_event.is_set)
            if self.nodule_cancel_event.is_set():raise Cancelled('已停止；未发布部分候选')
            packet={k:copy.deepcopy(context[k]) for k in ('case_id','analysis_source_id','context_id','source_hu_sha256','geometry_sha256','segmentation_signature')}
            packet.update(schema=SCHEMA,result_id=self.nodule_job['job_id'],is_test_data=False,
                candidate_count=len(candidates),candidates=candidates,detector_provenance=provenance)
            with self.lock:
                if context['context_id']!=self.nodule_context()['context_id']:raise ValueError('检测期间输入上下文已变化，未发布结果')
                record=validate_packet(packet,context,allow_enhanced=True);record['ingestion_kind']='computed_live_fpga'
                with (folder/'prediction.json').open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,allow_nan=False,indent=2)
                self.nodule_result=record;self.nodule_packet_hashes[(record['context_id'],record['result_id'])]=digest(record)
                self.nodule_job.update(status='COMPLETE',stage='complete',candidate_count=len(candidates),elapsed_ms=(time.perf_counter()-began)*1000)
                self._update(phase='done',error=None)
        except Cancelled as exc:
            with self.lock:
                self.nodule_job.update(status='CANCELLED',stage='cancelled',error=str(exc));self._update(phase='cancelled',error=None)
        except Exception as exc:
            with self.lock:
                self.nodule_job.update(status='FAILED',stage='failed',error=type(exc).__name__+': '+str(exc))
                self._update(phase='error',error=self.nodule_job['error'],board_connected=False)
        finally:
            with (folder/'job.json').open('x',encoding='utf8') as f:json.dump(self.nodule_job,f,ensure_ascii=False,indent=2)

    def nodule_payload(self, case_id, source_id, context_id, result_id):
        with self.lock:
            context = self.nodule_context()
            if (context is None or self.nodule_result is None or
                    (case_id, source_id, context_id, result_id) != (context['case_id'],context['analysis_source_id'],
                    context['context_id'],self.nodule_result['result_id'])):
                raise ValueError('候选结果已失效')
            return copy.deepcopy(self.nodule_result)

    def nodule_mip(self, case_id, source_id, context_id, result_id, candidate_id, padding_mm):
        """Display-only native-Z projection, bound to the current candidate context."""
        with self.lock:
            packet = self.nodule_payload(case_id, source_id, context_id, result_id)
            candidate = next((c for c in packet['candidates'] if c['candidate_id'] == candidate_id), None)
            if candidate is None: raise ValueError('候选 ROI 编号不存在')
            box = display_bounds(candidate, self.hu.shape, self.geom['spacing_zyx_mm'], padding_mm)
            z0,z1,y0,y1,x0,x1 = box
            if z0 >= z1 or y0 >= y1 or x0 >= x1: raise ValueError('候选没有可显示的 ROI')
            values = np.max(self.hu[z0:z1,y0:y1,x0:x1], axis=0).astype('<f4')
            return dict(case_id=case_id, analysis_source_id=source_id, context_id=context_id,
                result_id=result_id, candidate_id=candidate_id, padding_mm=padding_mm,
                bbox_zyx=box, shape_yx=list(values.shape), spacing_yx_mm=self.geom['spacing_zyx_mm'][1:],
                hu=base64.b64encode(values.tobytes()).decode('ascii'), backend='CPU_DISPLAY_ONLY',
                is_test_data=packet['is_test_data'], projection_axis='native_z',
                note='当前影像来源 HU 的局部最大投影；仅用于显示，不是病灶分割或新增检测')

    def nodule_reference(self, reference, roi_yx, z_bounds):
        if reference is None: return None
        if not isinstance(reference, dict): raise ValueError('无效候选 ROI 来源')
        context = self.nodule_context()
        packet = self.nodule_result
        if (context is None or packet is None or reference.get('context_id') != context['context_id']
                or reference.get('result_id') != packet['result_id']):
            raise ValueError('候选 ROI 来源已失效，请重新选择候选')
        candidate = next((c for c in packet['candidates'] if c['candidate_id']==reference.get('candidate_id')), None)
        if candidate is None: raise ValueError('候选 ROI 编号不存在')
        pad = reference.get('padding_mm', 0)
        box = display_bounds(candidate, context['shape'], context['geometry']['spacing_zyx_mm'], pad)
        return dict(context_id=context['context_id'], result_id=packet['result_id'], candidate_id=candidate['candidate_id'],
            is_test_data=packet['is_test_data'], original_bbox_zyx=list(candidate['bbox_zyx']), padding_mm=pad,
            display_roi_yx=list(roi_yx), display_z_bounds=list(z_bounds),
            user_adjusted=list(roi_yx)!=box[2:] or list(z_bounds)!=box[:2])
