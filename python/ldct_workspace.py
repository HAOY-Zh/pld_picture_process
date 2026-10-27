"""LDCT jobs and immutable, independently cached analysis sources on the PC."""
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from ct_studio_client import StudioClient
from ldct_contract import METHODS, quantize_slice, validate_release

SESSION_FIELDS = ('hu','raw','body','cleaned','known','geom','details','case_folder',
    'expert','expert_info','scores','surfaces','receipts','basic_result','basic_image',
    'threshold_result','threshold_image','history')


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pack_hu(array):
    value = np.ascontiguousarray(array,dtype='<i2')
    return dict(shape=list(value.shape),dtype='int16',pixels=base64.b64encode(value.tobytes()).decode())


class LDCTWorkspaceMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ldct_release = None; self.ldct_identity = None; self._ldct_arrays = None
        self.series_id = None; self.original_hu = None; self.original_details = None
        self.analysis_source_id = 'original'; self.analysis_sessions = {}; self.source_choices = {}
        self.ldct_groups = {}; self.ldct_group = None; self.ldct_running = False

    def configure_ldct(self, folder):
        manifest, descriptor = validate_release(folder)
        self.ldct_release = Path(folder).resolve(); self.ldct_identity = descriptor
        self._ldct_spec = manifest['model']

    def snapshot(self):
        with self.lock:
            value = super().snapshot()
            group = self.ldct_groups.get(self.ldct_group,{})
            value.update(series_id=self.series_id, analysis_source_id=self.analysis_source_id,
                analysis_source=self.source_choices.get(self.analysis_source_id,dict(label='原始 CT')),
                analysis_sources=list(self.source_choices.values()),
                ldct=dict(available=self.ldct_identity is not None,identity=self.ldct_identity,
                    fpga_configured=isinstance(self.client,StudioClient),running=self.ldct_running,
                    group_id=self.ldct_group,completed_slices=len(group),total_slices=len(self.original_hu) if self.original_hu is not None else 0,
                    results=[group[z]['metadata'] for z in sorted(group)]))
            return value

    def _work(self, action, values):
        previous = self.status.get('case_id')
        super()._work(action,values)
        if action == 'load' and self.status.get('case_id') != previous:
            with self.lock:
                self.series_id = self.status['case_id']; self.original_hu = self.hu
                self.original_details = copy.deepcopy(self.details)
                self.analysis_source_id='original'; self.analysis_sessions={}
                self.source_choices={'original':dict(source_id='original',label='原始 CT',backend='ORIGINAL',
                    hu_sha256=self.details['hu_sha256'],case_id=self.status['case_id'])}
                self.ldct_groups={}; self.ldct_group=None
                self.details['analysis_source']=copy.deepcopy(self.source_choices['original'])

    def submit(self, action, values):
        if action not in ('ldct','analysis_source'):
            return super().submit(action,values)
        with self.lock:
            if self.status['phase'] in ('loading','connecting','computing','reconstructing','processing'):
                raise BlockingIOError('请等待当前任务完成')
            if self.hu is None or values.get('case_id') != self.status['case_id']:
                raise ValueError('病例或分析来源已变化，请刷新后重试')
            if action=='analysis_source':
                return self.select_analysis_source(values.get('source_id'))
            if self.ldct_identity is None:
                raise ValueError('未配置经过校验的 LDCT 模型')
            backend=values.get('backend'); method=values.get('method','cnn')
            if backend not in ('CPU_REFERENCE','FPGA_LDCT') or not isinstance(method,str) or method not in METHODS:
                raise ValueError('请选择有效的执行位置和增强方法')
            if backend=='FPGA_LDCT':
                if not isinstance(self.client,StudioClient) or not self.status['board_connected']:
                    raise ValueError('统一 LDCT FPGA 服务尚未连接；可明确选择电脑参考')
                if self.client.manifest['ldct'] != self.ldct_identity:
                    raise ValueError('电脑参考与板端 LDCT 版本不匹配')
            mode=values.get('mode','slice'); z=values.get('index')
            if mode not in ('slice','volume') or type(z) is not int or not 0<=z<len(self.original_hu):
                raise ValueError('增强范围或切片索引无效')
            indices=list(range(len(self.original_hu))) if mode=='volume' else [z]
            group_id=hashlib.sha256(json.dumps([backend,method,self.ldct_identity['model_id']],sort_keys=True).encode()).hexdigest()
            group=self.ldct_groups.setdefault(group_id,{})
            if any(i in group for i in indices):
                # Every execution gets a new generation; older groups/analysis sources remain available.
                group_id=uuid.uuid4().hex; group={}; self.ldct_groups[group_id]=group
            self.ldct_group=group_id; self.ldct_running=True; self.cancel.clear()
            self._update(phase='processing',error=None,completed=0,total=len(indices))
            threading.Thread(target=self._enhance,args=(indices,backend,method,group_id),daemon=True).start()
            return self.snapshot()

    def cancel_ldct(self):
        with self.lock:
            if not self.ldct_running: raise ValueError('当前没有增强任务')
            self.cancel.set()
            return self.snapshot()

    def _enhance(self,indices,backend,method,group_id):
        try:
            root=self.output/self.series_id/'ldct'; root.mkdir(exist_ok=True)
            if backend=='CPU_REFERENCE' and self._ldct_arrays is None:
                from ctdenoise.cnn_fixed import load_quantized
                self._ldct_arrays,self._ldct_spec=load_quantized(self.ldct_release/'model')
            def save_slice(done,z,original,output,elapsed,receipt):
                if output.dtype!=np.int16 or output.shape!=original.shape:
                    raise ValueError('LDCT 输出形状/类型与原始 CT 不一致')
                result_id=uuid.uuid4().hex
                metadata=dict(result_id=result_id,series_id=self.series_id,group_id=group_id,index=z,
                    source_slice_index=self.original_details['source_slice_indices'][z],backend=backend,
                    method=method,model_id=self.ldct_identity['model_id'],input_sha256=digest(original),
                    output_sha256=digest(output),processing_ms=elapsed,hardware_receipt=receipt,
                    timing_scope=receipt['stats']['timing_scope'] if receipt else 'PC integer reference only',
                    quality_reference_available=False)
                with (root/(result_id+'.npz')).open('xb') as f:
                    np.savez_compressed(f,hu=output)
                with (root/(result_id+'.json')).open('x',encoding='utf-8') as f:
                    json.dump(metadata,f,ensure_ascii=False,indent=2,allow_nan=False)
                with self.lock:
                    self.ldct_groups[group_id][z]=dict(hu=output.copy(),metadata=metadata)
                    self._update(completed=done)
            with ThreadPoolExecutor(max_workers=1,thread_name_prefix='ldct-save') as writer:
                pending=None
                for done,z in enumerate(indices,1):
                    if self.cancel.is_set():
                        self._update(phase='cancelled'); return
                    original=quantize_slice(self.original_hu[z])
                    start=time.perf_counter(); receipt=None
                    if backend=='FPGA_LDCT':
                        output,receipt=self.client.denoise(original,method)
                        elapsed=receipt['stats']['total_ms']
                    else:
                        if method=='cnn':
                            from ctdenoise.cnn_fixed import infer_fixed
                            output=infer_fixed(original,self._ldct_arrays,self._ldct_spec)[0]
                        else:
                            from ctdenoise.pl_reference import fixed_filter
                            output=fixed_filter(original,METHODS[method])
                        elapsed=(time.perf_counter()-start)*1000
                    if pending is not None:pending.result()
                    pending=writer.submit(save_slice,done,z,original,output.copy(),elapsed,receipt)
                if pending is not None:pending.result()
            with self.lock:
                self._register_enhanced_source(group_id,backend,method)
                self._update(phase='done')
        except Exception as exc:
            self._update(phase='error',error=type(exc).__name__+': '+str(exc),
                **({'board_connected':False} if backend=='FPGA_LDCT' else {}))
        finally:
            with self.lock:
                self.ldct_running=False; self._update()

    def _register_enhanced_source(self,group_id,backend,method):
        group=self.ldct_groups[group_id]
        if len(group)!=len(self.original_hu): return
        hu=np.stack([group[z]['hu'] for z in range(len(self.original_hu))]).astype(np.float32)
        source_id='enhanced:'+group_id
        case_id=uuid.uuid4().hex; folder=self.output/case_id; folder.mkdir()
        descriptor=dict(source_id=source_id,label='LDCT 增强 CT · '+method.upper()+' · '+('电脑参考' if backend=='CPU_REFERENCE' else 'FPGA'),
            backend=backend,method=method,model_id=self.ldct_identity['model_id'],
            hu_sha256=digest(hu),case_id=case_id,parent_series_id=self.series_id,group_id=group_id)
        details=copy.deepcopy(self.original_details)
        details.update(hu_sha256=digest(hu),analysis_source=descriptor)
        with (folder/'input.json').open('x',encoding='utf-8') as f:
            json.dump(details,f,ensure_ascii=False,indent=2,allow_nan=False)
        self.source_choices[source_id]=descriptor
        self.analysis_sessions[source_id]=dict(hu=hu,raw=np.zeros(hu.shape,bool),body=np.zeros(hu.shape,bool),
            cleaned=None,known=np.zeros(len(hu),bool),geom=self.geom,details=details,case_folder=folder,
            expert=self.expert,expert_info=self.expert_info,scores={n:{} for n in ('raw','body','cleaned')},
            surfaces={},receipts={},basic_result=None,basic_image=None,threshold_result=None,
            threshold_image=None,history=[],status=dict(self.status,case_id=case_id,phase='idle',error=None,
                latest=None,full_volume_ready=False,volume_ready=False,volume_scope=None,reconstruction=None,
                region_volume_ml=None,elapsed_ms=None,completed=0,total=0,active_slice=None))

    def select_analysis_source(self,source_id):
        if source_id not in self.source_choices:
            raise ValueError('增强范围尚未全部完成，不能混入未增强切片进行分析')
        if source_id==self.analysis_source_id: return self.snapshot()
        self.analysis_sessions[self.analysis_source_id]={name:getattr(self,name) for name in SESSION_FIELDS}
        self.analysis_sessions[self.analysis_source_id]['status']=copy.deepcopy(self.status)
        target=self.analysis_sessions[source_id]
        revision=self.status['revision']; connected=self.status['board_connected']
        for name in SESSION_FIELDS: setattr(self,name,target[name])
        self.status=copy.deepcopy(target['status'])
        self.status.update(revision=revision,board_connected=connected)
        self.analysis_source_id=source_id
        self.basic_request_id=None
        self._update(phase='idle',error=None)
        return self.snapshot()

    def ldct_payload(self,series_id,group_id,index):
        with self.lock:
            if (series_id!=self.series_id or group_id not in (self.ldct_group,'') or type(index) is not int
                    or not 0<=index<len(self.original_hu)):
                raise ValueError('增强结果已过期，请刷新当前病例')
            row=self.ldct_groups.get(group_id,{}).get(index)
            return dict(metadata=row['metadata'] if row else None,
                original=pack_hu(quantize_slice(self.original_hu[index])),enhanced=pack_hu(row['hu']) if row else None)
