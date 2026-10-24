"""Whole-case raw-CNN experiment using the accepted sole-owner FPGA endpoint."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,time
from pathlib import Path
import numpy as np
from ct_studio_shared_client import SharedPatchClient
from shared_patch_contract import validate_assets
from shared_patch_contract import BATCH_INTERFACE,MAX_BATCH
from nodule_vendor.lung_nodule_cnn import config_from_dict,search_mask,tiles,decode,consolidate
from nodule_vendor.lung_nodule_sampling_cache import sample_patch_reuse
from nodule_vendor.nodule_plan import quantize_hu,lower_stem,dequantize_head

PIPELINE='raw_cnn_experimental_v1'
ROOT=Path(__file__).resolve().parents[1]
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Cancelled(Exception):pass


class FloatPlaneCache:
    def __init__(self,hu):self.hu=hu;self.planes=OrderedDict()
    def __len__(self):return len(self.hu)
    def __getitem__(self,z):
        z=int(z)
        if z not in self.planes:self.planes[z]=np.asarray(self.hu[z],np.float32)
        self.planes.move_to_end(z)
        if len(self.planes)>16:self.planes.popitem(last=False)
        return self.planes[z]


class NoduleCaseProvider:
    def __init__(self,package,legacy_client,batch_size=8):
        package=Path(package).resolve();self.package=package
        self.bundle=json.loads((package/'shared_host_platform.json').read_text())
        self.model=validate_assets(package/'ct_studio',self.bundle['shared_nodule'])
        self.client=SharedPatchClient(legacy_client.url,legacy_client.token,self.bundle)
        self.config=config_from_dict(self.model['detector_config'])
        if type(batch_size)is not int or not 1<=batch_size<=MAX_BATCH:raise ValueError('Invalid bounded batch size')
        self.batch_size=batch_size
        files=('lung_nodule_cnn.py','lung_nodule_sampling_cache.py','nodule_plan.py')
        self.code_sha256={n:hashlib.sha256((Path(__file__).parent/'nodule_vendor'/n).read_bytes()).hexdigest() for n in files}

    def run(self,hu,lung,geometry,context,folder,progress,cancelled):
        health=self.client.health();started=time.perf_counter()
        if self.batch_size>1 and health.get('shared_batch')!=dict(interface=BATCH_INTERFACE,max_patches=MAX_BATCH):
            raise ValueError('Batch service is not accepted; no single-patch fallback')
        timing=dict(search_ms=0.,sample_quantize_ms=0.,patch_module_ms=0.,patch_roundtrip_ms=0.,decode_ms=0.,merge_ms=0.,pl_active_ms=0.,
            progress_callback_ms=0.,patch_client_outer_ms=0.,receipt_io_ms=0.,input_hash_ms=0.,prepare_wait_ms=0.)
        def check():
            if cancelled():raise Cancelled('已在安全 patch 边界停止；未发布部分候选')
        def report(stage,done,total):
            tick=time.perf_counter();progress(stage,done,total)
            timing['progress_callback_ms']+=(time.perf_counter()-tick)*1000
        check();report('search',0,0)
        tick=time.perf_counter();search=search_mask(lung,geometry['spacing_zyx_mm'],self.config)
        jobs,schedule=tiles(search,geometry['spacing_zyx_mm'],self.config);timing['search_ms']=(time.perf_counter()-tick)*1000
        cache=FloatPlaneCache(hu);points=[[],[],[]];tick=time.perf_counter();mask_sha=sha(lung)
        timing['input_hash_ms']=(time.perf_counter()-tick)*1000
        def prepare(start):
            # Exactly one producer owns both interpolation caches; order is frozen.
            began=time.perf_counter();values=[]
            for job in jobs[start:start+self.batch_size]:
                check();patch=sample_patch_reuse(cache,geometry['spacing_zyx_mm'],job,self.config)
                values.append(lower_stem(quantize_hu(patch)))
            return values,(time.perf_counter()-began)*1000
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='nodule-sample') as producer, \
                (folder/'patch_receipts.jsonl').open('x',encoding='utf8') as receipts:
            pending=producer.submit(prepare,0) if jobs else None
            for start in range(0,len(jobs),self.batch_size):
                check();report('inference',start,len(jobs));tick=time.perf_counter()
                images,work_ms=pending.result();timing['prepare_wait_ms']+=(time.perf_counter()-tick)*1000
                timing['sample_quantize_ms']+=work_ms;check()
                stop=start+len(images)
                pending=producer.submit(prepare,stop) if stop<len(jobs) else None
                tick=time.perf_counter()
                outputs=self.client.nodule_batch(images) if self.batch_size>1 else [self.client.nodule_patch(images[0])]
                timing['patch_client_outer_ms']+=(time.perf_counter()-tick)*1000
                if len(outputs)!=len(images):raise ValueError('Incomplete batch result; no partial publication')
                for i,(head,receipt) in enumerate(outputs,start):
                    job=jobs[i]
                    timing['patch_module_ms']+=receipt['stats']['host_total_ms'];timing['patch_roundtrip_ms']+=receipt['roundtrip_ms']
                    timing['pl_active_ms']+=receipt['stats']['pl_command_cycles']/100000.
                    tick=time.perf_counter();receipts.write(json.dumps(dict(index=i,job=job,receipt=receipt),allow_nan=False)+'\n')
                    timing['receipt_io_ms']+=(time.perf_counter()-tick)*1000
                    tick=time.perf_counter();prediction=dequantize_head(head,self.model)
                    for dest,value in zip(points,decode(prediction[None],[job],geometry,self.config,search)):dest.append(value)
                    timing['decode_ms']+=(time.perf_counter()-tick)*1000
                tick=time.perf_counter();receipts.flush();timing['receipt_io_ms']+=(time.perf_counter()-tick)*1000
                if stop%128==0 or stop==len(jobs):
                    tick=time.perf_counter()
                    (folder/'timing_progress.json').write_text(json.dumps(dict(completed=stop,total=len(jobs),timing=timing,
                        elapsed_ms=(time.perf_counter()-started)*1000)),encoding='utf8')
                    timing['receipt_io_ms']+=(time.perf_counter()-tick)*1000
            report('merge',len(jobs),len(jobs));check();tick=time.perf_counter()
            candidates=consolidate(*(np.concatenate(v) for v in points),geometry,self.config) if jobs else []
            timing['merge_ms']=(time.perf_counter()-tick)*1000
        if len(candidates)>10000:raise ValueError('原始候选超过 10000；未截断或擅自提高评分阈值，请检查实验配置')
        for c in candidates:c['raw_score']=c['score']
        timing['whole_case_ms']=(time.perf_counter()-started)*1000
        timing['unaccounted_ms']=timing['whole_case_ms']-sum(timing[k] for k in ('search_ms','prepare_wait_ms',
            'patch_client_outer_ms','decode_ms','merge_ms','progress_callback_ms','receipt_io_ms','input_hash_ms'))
        source=context['analysis_source_id']
        provenance=dict(backend='FPGA_NODULE_SHARED',pipeline=PIPELINE,scoring_mode='raw_cnn',
            detection_quality_accepted=False,input_conditioning='original' if source=='original' else 'enhanced_unvalidated',
            source_hu_sha256=context['source_hu_sha256'],lung_mask_sha256=mask_sha,lung_mask_stage='body',
            geometry_sha256=context['geometry_sha256'],segmentation_signature=context['segmentation_signature'],
            service_bundle_sha256=self.bundle['service_bundle_sha256'],shared_nodule=self.bundle['shared_nodule'],
            detector_config=self.model['detector_config'],sampling_decode_code_sha256=self.code_sha256,
            raw_decode_floor=self.config.decode_floor,additional_score_filter=None,ranker_applied=False,
            candidate_order='descending raw CNN score, then native Z/Y/X',schedule=schedule,timing=timing,
            transport=dict(batch_size=self.batch_size,requests=(len(jobs)+self.batch_size-1)//self.batch_size,
                prefetch_batches=1,sampling_work_overlaps_transfer=True),
            note='真实 FPGA 原始 CNN 实验候选；分数未经校准，非恶性概率。增强 HU 的检测效果未验收。')
        return candidates,provenance
