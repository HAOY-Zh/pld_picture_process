"""One actual web full-case run, compared to saved raw/scored old predictions."""
import gzip
import json
from pathlib import Path
import re
import time
import urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/nodule_full_web_upgrade_20261025'
CASE=ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results/d2ab9db1e644493388c93b0fbe3b2f40'
OLD=CASE/'nodule/nodule_737071f016ad4a0b91d0c0f9fce84103'


def main():
    dest=OUT/'full_case_live_20261026';dest.mkdir(exist_ok=False)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));url='http://127.0.0.1:8099'
    html=opener.open(url,timeout=5).read().decode()
    token=re.search(r'<meta name="ui-token" content="([^"]+)"',html)[1]
    def call(path,data=None):
        req=urllib.request.Request(url+path,data=None if data is None else json.dumps(data).encode(),
            headers={'X-UI-Token':token,**({'Content-Type':'application/json'}if data is not None else{})})
        return json.load(opener.open(req,timeout=10))
    before=call('/api/state')
    assert before['phase']=='done'and before['case_id']==CASE.name
    context=call('/api/nodule/context')['context']
    started=time.perf_counter()
    call('/api/nodule/run',dict(case_id=context['case_id'],analysis_source_id=context['analysis_source_id'],
        context_id=context['context_id'],scoring_mode='frozen_cross_layer'))
    previous=-100
    while True:
        s=call('/api/state');job=s['nodule']['job']
        if job['status']in ('FAILED','CANCELLED'):raise RuntimeError(job)
        (dest/'progress.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf8')
        if job['completed']-previous>=256 or job['status']=='COMPLETE':
            print('LIVE_NEW_NODULE',job['stage'],job['completed'],'/',job['total'],
                'elapsed_s',round(job.get('elapsed_ms',0)/1000,2),flush=True);previous=job['completed']
        if job['status']=='COMPLETE':break
        if time.perf_counter()-started>1800:raise TimeoutError('Long job retained, inspect progress before continuing')
        time.sleep(1)
    new=CASE/'nodule'/job['job_id']
    packet=json.loads((new/'prediction.json').read_text(encoding='utf8'))
    expected=json.loads((OUT/'saved_case_scoring/scored_prediction.json').read_text(encoding='utf8'))
    oldraw=json.loads((OLD/'prediction.json').read_text(encoding='utf8'))
    with gzip.open(new/'all_ranked_candidates.json.gz','rt',encoding='utf8')as f:allnew=json.load(f)
    with gzip.open(OUT/'saved_case_scoring/all_ranked_candidates.json.gz','rt',encoding='utf8')as f:allold=json.load(f)
    assert packet['source_hu_sha256']==oldraw['source_hu_sha256']
    assert packet['detector_provenance']['lung_mask_sha256']==oldraw['detector_provenance']['lung_mask_sha256']
    assert len(allnew)==len(allold)
    for a,b in zip(allnew,allold):
        for key in ('candidate_id','center_index_zyx','center_lps_xyz_mm','bbox_zyx',
                    'estimated_scale_diameter_mm','raw_score','score'):
            if a[key]!=b[key]:raise AssertionError('Candidate changed: '+key)
    assert packet['candidates']==expected['candidates']
    old_job=json.loads((OLD/'job.json').read_text())
    p=packet['detector_provenance'];old_t=oldraw['detector_provenance']['timing'];new_t=p['timing']
    result=dict(status='LIVE_WEB_FULL_CASE_ALL_RAW_AND_FINAL_CANDIDATES_EXACT_PASS',case_id=CASE.name,
        job_id=job['job_id'],raw_candidates=len(allnew),selected_candidates=len(packet['candidates']),
        all_raw_geometry_scores_exact=True,final_selected_exact=True,model_and_input_unchanged=True,
        old_raw_pipeline_job_ms=old_job['elapsed_ms'],new_complete_scored_job_ms=job['elapsed_ms'],
        before_after_speedup=old_job['elapsed_ms']/job['elapsed_ms'],old_timing=old_t,new_timing=new_t,
        new_transport=p['transport'],scope='Same saved native CT and lung mask; new run includes frozen cross-layer scoring, old run produced unfiltered raw candidates. No expert labels; timings from separate runs.')
    (dest/'result.json').write_text(json.dumps(result,indent=2))
    (dest/'web_state.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items()if k not in ('old_timing','new_timing')},indent=2),flush=True)


if __name__=='__main__':main()
