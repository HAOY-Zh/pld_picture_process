"""PC ROI jobs on the current immutable HU source; no hardware ownership."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from roi_vendor.ct_roi_reference import (CpuRoiProcessor, MODULE_VERSION,
    ALGORITHM_ID, PARAMS_VERSION, validate_bounds, validate_slice)

ACTIVE = ('loading', 'connecting', 'computing', 'reconstructing', 'processing')
VENDOR = Path(__file__).parent / 'roi_vendor'


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def reference_identity():
    source = json.loads((VENDOR / 'provenance.json').read_text(encoding='utf-8'))
    if hashlib.sha256((VENDOR / source['reference_file']).read_bytes()).hexdigest() != source['reference_sha256']:
        raise ValueError('ROI reference differs from member C frozen delivery')
    return {key: source[key] for key in ('module_version', 'algorithm_id', 'reference_sha256')}


def physical_spacing(geometry):
    try:
        spacing = np.asarray(geometry['spacing_zyx_mm'], dtype=float)
        basis = np.asarray(geometry['basis_lps_zyx'], dtype=float)
        if spacing.shape != (3,) or basis.shape != (3, 3): return None
        if not np.isfinite(spacing).all() or not np.isfinite(basis).all() or (spacing <= 0).any(): return None
        directions = basis / spacing[None, :]
        if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5, rtol=0): return None
        return spacing
    except (KeyError, TypeError, ValueError):
        return None


def summary(record):
    return {key: value for key, value in record.items()
            if key not in ('slice_results', 'geometry', 'lung_mask', 'reference_identity')}


class ROIWorkspaceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi_identity = reference_identity()
        self.roi_result = None
        self.roi_records = {}
        self.roi_job = None
        self.roi_sessions = {}
        self.roi_cancel_event = threading.Event()

    def snapshot(self):
        with self.lock:
            value = super().snapshot()
            value['roi'] = dict(available=True, backend='CPU_REFERENCE', fpga_available=False,
                fpga_status='待接板验证', identity=self.roi_identity,
                result=summary(self.roi_result) if self.roi_result else None,
                history=[summary(row) for row in list(self.roi_records.values())[-30:]],
                job=copy.deepcopy(self.roi_job),
                mask_known_indices=np.flatnonzero(self.known).tolist() if self.known is not None else [],
                cleaned_mask_available=self.cleaned is not None)
            return value

    def _work(self, action, values):
        previous = self.status.get('case_id')
        super()._work(action, values)
        if action == 'load' and self.status.get('case_id') != previous:
            with self.lock:
                self.roi_result = self.roi_job = None
                self.roi_records = {}; self.roi_sessions = {}

    def select_analysis_source(self, source_id):
        with self.lock:
            if source_id not in self.source_choices:
                raise ValueError('影像来源尚未完整生成')
            if source_id != self.analysis_source_id:
                self.roi_sessions[self.analysis_source_id] = (self.roi_result, self.roi_records, self.roi_job)
                super().select_analysis_source(source_id)
                self.roi_result, self.roi_records, self.roi_job = self.roi_sessions.get(source_id, (None, {}, None))
            return self.snapshot()

    def submit(self, action, values):
        if action != 'roi': return super().submit(action, values)
        with self.lock:
            if self.status['phase'] in ACTIVE: raise BlockingIOError('请等待当前任务完成')
            if (self.hu is None or values.get('case_id') != self.status['case_id']
                    or values.get('analysis_source_id') != self.analysis_source_id):
                raise ValueError('病例或原图/增强图来源已变化，请刷新后分析')
            if values.get('backend', 'CPU_REFERENCE') != 'CPU_REFERENCE':
                raise ValueError('ROI FPGA 尚未接入验证；当前仅提供明确的电脑参考')
            mode = values.get('mode', 'slice'); index = values.get('index')
            if mode not in ('slice', 'range') or type(index) is not int or not 0 <= index < len(self.hu):
                raise ValueError('ROI 切片或处理模式无效')
            z0, z1 = ((index, index + 1) if mode == 'slice' else
                      validate_bounds(values.get('z_bounds'), (len(self.hu),), 'z_bounds'))
            if z0 == z1: raise ValueError('请选择至少一层；结束层不能早于起始层')
            use_lung = values.get('use_lung_mask', False)
            if type(use_lung) is not bool: raise ValueError('肺区域限制须为布尔值')
            mask_stage = values.get('mask_stage', 'body')
            if mask_stage not in ('body', 'cleaned'): raise ValueError('请选择有效的预测肺掩膜')
            hu, roi, thresholds, use_hu, _ = validate_slice(self.hu[z0],
                values.get('roi_yx'), values.get('hu_range'), values.get('use_hu', False))
            lung = None
            if use_lung:
                if not self.known[z0:z1].all():
                    raise ValueError('所选层还没有完整的当前来源肺分割；请先分割，或关闭肺区域限制')
                lung = self.body if mask_stage == 'body' else self.cleaned
                if lung is None or lung.shape != self.hu.shape:
                    raise ValueError('当前来源尚无所选预测肺掩膜，不使用专家标注或其他来源结果替代')
            job_id = uuid.uuid4().hex
            output = self.case_folder / 'roi' / job_id
            output.mkdir(parents=True, exist_ok=False)
            record = dict(schema='ct_workspace_roi_v1', result_id=job_id,
                case_id=self.status['case_id'], analysis_source_id=self.analysis_source_id,
                analysis_source=copy.deepcopy(self.source_choices[self.analysis_source_id]),
                backend='CPU_REFERENCE', module_version=MODULE_VERSION, algorithm_id=ALGORITHM_ID,
                params_version=PARAMS_VERSION, reference_identity=copy.deepcopy(self.roi_identity),
                shape=list(self.hu.shape), geometry=copy.deepcopy(self.geom),
                source_hu_sha256=self.details['hu_sha256'], mode=mode, index=index,
                roi_yx=list(roi), z_bounds=[z0, z1], use_hu=use_hu,
                hu_range=list(thresholds) if use_hu else None, use_lung_mask=use_lung, lung_mask_stage=mask_stage,
                lung_mask=dict(stage=mask_stage, analysis_source_id=self.analysis_source_id,
                    segmentation_receipts={str(z):self.receipts.get(z) for z in range(z0,z1)}) if use_lung else None,
                input_scope='partial_input' if self.details.get('partial_input') else 'loaded_input',
                requested_slices=z1-z0, processed_slices=0, complete=False, status='RUNNING',
                valid_count=0, slice_results=[], processed_indices=[],
                warning='按矩形/HU/预测肺区域筛选的区域，不是自动结节检测或精确病灶体积。')
            # Freeze identity/options now. Existing task exclusion keeps arrays immutable.
            self.roi_cancel_event.clear()
            self.roi_job = dict(result_id=job_id, running=True, completed=0, total=z1-z0)
            self._update(phase='processing', error=None, completed=0, total=z1-z0)
            threading.Thread(target=self._roi_work, args=(record, output, self.hu, lung,
                list(self.details['source_slice_indices'])), daemon=True).start()
            return self.snapshot()

    def cancel_roi(self, values):
        with self.lock:
            if (not self.roi_job or not self.roi_job['running']
                    or values.get('case_id') != self.status['case_id']
                    or values.get('analysis_source_id') != self.analysis_source_id
                    or values.get('result_id') != self.roi_job['result_id']):
                raise ValueError('ROI 任务已结束或来源已变化')
            self.roi_cancel_event.set()
            return self.snapshot()

    def _roi_work(self, record, output, volume, lung, source_indices):
        began = time.perf_counter(); failure = None
        total_sum = 0; low = high = None; cpu_ms = 0.
        spacing = physical_spacing(record['geometry'])
        try:
            processor = CpuRoiProcessor()
            for z in range(*record['z_bounds']):
                if self.roi_cancel_event.is_set(): break
                lung_slice = None if lung is None else lung[z]
                result = processor.process_slice(volume[z], roi_yx=record['roi_yx'],
                    hu_range=record['hu_range'], use_hu=record['use_hu'], lung_mask=lung_slice)
                mask = result['filtered_mask']; stats = result['hu_statistics']
                if result['backend'] != 'CPU_REFERENCE' or int(mask.sum()) != result['valid_count']:
                    raise ValueError('ROI 参考返回身份或计数不一致')
                rounded = np.rint(volume[z]).astype(np.int16)
                row = dict(index=z, source_slice_index=source_indices[z], valid_count=result['valid_count'],
                    area_mm2=result['valid_count']*float(spacing[1]*spacing[2]) if spacing is not None else None,
                    hu_statistics=stats, input_hu_i16_sha256=digest(rounded), mask_sha256=digest(mask),
                    lung_mask_sha256=digest(lung_slice.astype(np.uint8)) if lung_slice is not None else None,
                    timing=result['timing'])
                with (output / f'slice_{z:06d}.npz').open('xb') as stream:
                    np.savez_compressed(stream, filtered_mask=mask)
                record['slice_results'].append(row); record['processed_indices'].append(z)
                record['valid_count'] += result['valid_count']; total_sum += stats['sum']
                if result['valid_count']:
                    low = stats['min'] if low is None else min(low, stats['min'])
                    high = stats['max'] if high is None else max(high, stats['max'])
                cpu_ms += result['timing']['total_seconds']*1000
                with self.lock:
                    self.roi_job['completed'] = len(record['slice_results'])
                    self._update(completed=len(record['slice_results']), active_slice=z)
        except Exception as exc:
            failure = type(exc).__name__ + ': ' + str(exc)
        record['processed_slices'] = len(record['slice_results'])
        record['complete'] = failure is None and record['processed_slices'] == record['requested_slices']
        record['status'] = 'FAILED' if failure else 'SUCCESS' if record['complete'] else 'CANCELLED'
        record['error'] = failure
        count = record['valid_count']
        record['hu_statistics'] = dict(count=count, sum=total_sum, min=low, max=high,
            mean=total_sum/count if count else None, computed_on='PC_FROM_RETURNED_MASKS')
        # A single plane reports area; a slab reports only actually computed voxels.
        volume_mm3 = (count*float(np.prod(spacing)) if spacing is not None
                      and record['mode'] == 'range' and record['processed_slices'] >= 2 else None)
        record.update(region_volume_mm3=volume_mm3,
            region_volume_ml=None if volume_mm3 is None else volume_mm3/1000,
            physical_geometry_verified=spacing is not None,
            timing=dict(cpu_reference_ms=cpu_ms, host_result_ready_ms=(time.perf_counter()-began)*1000,
                        scope='电脑参考及电脑落盘；非FPGA耗时，不含浏览器显示'))
        try:
            with (output / 'result.json').open('x', encoding='utf-8') as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            with self.lock:
                self.roi_records[record['result_id']] = record
                self.roi_result = record
                self.roi_job.update(running=False, status=record['status'], error=failure)
                self._update(phase='error' if failure else 'done' if record['complete'] else 'cancelled', error=failure)
        except Exception as exc:
            with self.lock:
                self.roi_job.update(running=False, status='FAILED', error=str(exc))
                self._update(phase='error', error='ROI 输出保存失败：' + str(exc))

    def roi_export(self, case_id, source_id, result_id):
        with self.lock:
            if (case_id != self.status['case_id'] or source_id != self.analysis_source_id
                    or result_id not in self.roi_records):
                raise ValueError('ROI 结果属于其他病例或影像来源，请切回对应来源查看')
            return copy.deepcopy(self.roi_records[result_id])

    def roi_payload(self, case_id, source_id, result_id, index):
        with self.lock:
            record = self.roi_export(case_id, source_id, result_id)
            if not 0 <= index < record['shape'][0]: raise ValueError('切片越界')
            row = next((row for row in record['slice_results'] if row['index'] == index), None)
            packed = None
            if row is not None:
                path = self.case_folder / 'roi' / result_id / f'slice_{index:06d}.npz'
                with np.load(path, allow_pickle=False) as data: mask = data['filtered_mask']
                if digest(mask) != row['mask_sha256']: raise ValueError('已保存 ROI 掩膜哈希不一致')
                packed = dict(shape=list(mask.shape), dtype='uint8', pixels=base64.b64encode(mask.tobytes()).decode())
            return dict(case_id=case_id, analysis_source_id=source_id, result_id=result_id,
                        index=index, known=row is not None, slice_result=row, mask=packed,
                        metadata=summary(record))
