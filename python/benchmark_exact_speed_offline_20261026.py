"""Real saved-input equivalence and PC-only timing; no network/FPGA access.

LDCT timing is explicitly recorded-latency replay, not a physical speed result.
Nodule timing covers host search/sampling only, never whole-case FPGA inference.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_ExactSpeed_20261026_03'
OLD=ROOT/'data/CT_Studio_DualExact_20261026_02'
OUT=ROOT/'data/exact_speed_iteration_20261026/offline'
sys.path[:0]=[str(APP/'python')]


def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    OUT.mkdir(exist_ok=False)
    from lung_live_data import read_ct
    from nodule_exact_host import ExactSamplerCache,quantize_exact,search_mask_exact
    from nodule_canvas_contract import quantize_canvas
    from nodule_vendor.nodule_fast_input import SharedPlaneSampler
    from nodule_vendor.lung_nodule_cnn import search_mask,tiles,config_from_dict
    from ldct_workspace import LDCTWorkspaceMixin as NewMixin
    spec=importlib.util.spec_from_file_location('old_ldct_for_replay',OLD/'python/ldct_workspace.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    state=json.loads((ROOT/'data/current_effect_audit_20261026/state.json').read_text(encoding='utf8'))
    results=ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results'
    case=results/state['case_id'];base=results/state['series_id']
    info=json.loads((base/'input.json').read_text(encoding='utf8'))
    original,geom,details=read_ct(info['source'],confirm_hu=True,series_uid=info.get('series_uid',''),
        allow_inconsistent_frame_uids=info.get('frame_uid_exception_applied') is True)
    assert digest(original)==info['hu_sha256']
    rows=sorted(state['ldct']['results'],key=lambda r:r['index'])
    hu=np.empty_like(original)
    for row in rows:
        with np.load(base/'ldct'/(row['result_id']+'.npz'),allow_pickle=False)as a:hu[row['index']]=a['hu']
    assert digest(hu)==state['analysis_source']['hu_sha256']
    with np.load(case/(state['volume_job_id']+'_masks.npz'),allow_pickle=False)as a:lung=a['body']
    oldjob=case/'nodule'/state['nodule']['job']['job_id']
    packet=json.loads((oldjob/'prediction.json').read_text(encoding='utf8'))
    assert digest(lung)==packet['detector_provenance']['lung_mask_sha256']
    records=[json.loads(line)for line in(oldjob/'canvas_receipts.jsonl').open()]
    model=json.loads((APP/'data/ct_studio_integration_20261025/ct_studio/shared_cnn/models/nodule/model.json').read_text())
    cfg=config_from_dict(model['detector_config']);timing={};baseline_search=None
    for name,fn in [('baseline',search_mask),('optimized',search_mask_exact),('optimized',search_mask_exact),('baseline',search_mask)]:
        t=time.perf_counter();value=fn(lung,geom['spacing_zyx_mm'],cfg);ms=(time.perf_counter()-t)*1000
        timing.setdefault(name,[]).append(ms)
        if baseline_search is None:baseline_search=value
        else:np.testing.assert_array_equal(value,baseline_search)
        print('SEARCH_FULL_VOLUME_EXACT',name,round(ms,2),flush=True)
    jobs,_=tiles(baseline_search,geom['spacing_zyx_mm'],cfg)
    assert [list(j)for j in jobs]==[j for r in records for j in r['jobs']]
    write('search.json',dict(scope='PC-only full search, ABBA order; complete mask/jobs exact.',times_ms=timing,
        slices=len(hu),speedup=float(np.median(timing['baseline'])/np.median(timing['optimized']))))
    prep={};counts={}
    for name in ['baseline','optimized']:
        cache=ExactSamplerCache(hu,geom['spacing_zyx_mm']);began=time.perf_counter()
        for i,r in enumerate(records):
            sampler=(SharedPlaneSampler(hu,geom['spacing_zyx_mm'],r['jobs'],cfg)if name=='baseline'
                else cache.sample(r['jobs'],cfg))
            q=quantize_canvas(sampler.context)if name=='baseline'else quantize_exact(sampler.context)
            assert digest(q)==r['receipt']['input_sha256']
            if i%40==0 or i==len(records)-1:print('ALL_REAL_CANVAS_INPUTS_EXACT',name,i+1,'/',len(records),flush=True)
        prep[name]=(time.perf_counter()-began)*1000
        counts[name]=dict(hits=cache.hits,misses=cache.misses,peak_cache_bytes=cache.peak_bytes)
    write('nodule_inputs.json',dict(status='PASS_ALL_ACTUAL_SAVED_CANVAS_INPUT_HASHES',canvases=len(records),
        original_tiles=len(jobs),times_ms=prep,cache=counts,speedup=prep['baseline']/prep['optimized'],
        scope='Exclusive PC preparation time only, not end-to-end; original board head hashes remain the reference, no new FPGA forward.'))

    # Delays replay recorded real requests; no mock is installed in any app.
    selected=list(range(35,351,21))
    class ReplayClient:
        def __init__(self):self.index=0
        def denoise(self,image,method):
            z=selected[self.index];row=rows[z];assert digest(image)==row['input_sha256']
            time.sleep(row['hardware_receipt']['roundtrip_ms']/1000)
            self.index+=1
            return hu[z].astype(np.int16),copy.deepcopy(row['hardware_receipt'])
    def replay(mixin,name):
        class Harness(mixin):
            def __init__(self):
                self.output=OUT/('ldct_'+name);self.output.mkdir();self.series_id='recorded_latency_replay'
                (self.output/self.series_id).mkdir();self.original_hu=original[selected]
                self.original_details=dict(source_slice_indices=selected);self.ldct_identity=state['ldct']['identity']
                self.ldct_groups={'group':{}};self.lock=threading.RLock();self.cancel=threading.Event()
                self.client=ReplayClient();self.ldct_running=True;self.status={};self.registered=False
            def _update(self,**fields):self.status.update(fields)
            def _register_enhanced_source(self,*args):self.registered=True
        h=Harness();t=time.perf_counter();h._enhance(list(range(len(selected))),'FPGA_LDCT','cnn','group')
        elapsed=(time.perf_counter()-t)*1000
        assert h.registered and h.status.get('phase')=='done',h.status
        for i,z in enumerate(selected):assert h.ldct_groups['group'][i]['metadata']['output_sha256']==rows[z]['output_sha256']
        return elapsed
    rt={}
    for name,mixin in [('baseline',legacy.LDCTWorkspaceMixin),('optimized',NewMixin)]:
        rt[name]=replay(mixin,name);print('LDCT_RECORDED_LATENCY_REPLAY_EXACT',name,round(rt[name],2),flush=True)
    write('ldct_replay.json',dict(status='PASS_OUTPUTS_IN_RECORDED_LATENCY_REPLAY',times_ms=rt,
        latency_reduction_percent=100*(1-rt['optimized']/rt['baseline']),slices=len(selected),
        physical_board_measured=False,scope='Scheduling/disk replay using fixed recorded HTTP delays; not a new board speed measurement or guaranteed live gain.'))
    write('complete.json',dict(status='OFFLINE_CHECKS_COMPLETE_BOARD_ACCEPTANCE_PENDING',
        board_accessed=False,all_canvas_inputs_exact=len(records),ldct_replay_outputs_exact=len(selected),
        speed_target_accepted=False))


if __name__=='__main__':main()
