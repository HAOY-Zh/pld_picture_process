"""Actual full-case compact transport vs saved complete canvas pipeline."""
import gzip
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/dual_pipeline_optimization_20261026'
CASE=ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results/08ea2953f1664d608f11b863f07fe68e'
OLD=CASE/'nodule/nodule_cd90efb94313430d8b0f649176db60dc'


def main():
    dest=OUT/'nodule_full_case';dest.mkdir(exist_ok=True)
    if any(dest.iterdir()):raise FileExistsError('Existing benchmark output retained')
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}));url='http://127.0.0.1:8099'
    html=op.open(url,timeout=5).read().decode();token=re.search(r'<meta name="ui-token" content="([^"]+)"',html)[1]
    def call(path,data=None):
        req=urllib.request.Request(url+path,data=None if data is None else json.dumps(data).encode(),
            headers={'X-UI-Token':token,**({'Content-Type':'application/json'}if data is not None else{})})
        return json.load(op.open(req,timeout=10))
    s=call('/api/state');assert s['phase']in('done','idle')and s['case_id']==CASE.name and s['known_slices']==351
    c=call('/api/nodule/context')['context'];begin=time.perf_counter()
    call('/api/nodule/run',dict(case_id=c['case_id'],analysis_source_id=c['analysis_source_id'],context_id=c['context_id'],scoring_mode='frozen_cross_layer'))
    printed=-256
    while True:
        s=call('/api/state');job=s['nodule']['job'];(dest/'progress.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf8')
        if job['status']in('FAILED','CANCELLED'):raise RuntimeError(job)
        if job['completed']-printed>=384 or job['status']=='COMPLETE':
            print('COMPACT_WEB',job['stage'],job['completed'],'/',job['total'],'seconds',round(job.get('elapsed_ms',0)/1000,2),flush=True);printed=job['completed']
        if job['status']=='COMPLETE':break
        if time.perf_counter()-begin>1800:raise TimeoutError('Job retained for inspection')
        time.sleep(1)
    new=CASE/'nodule'/job['job_id']
    packet=json.loads((new/'prediction.json').read_text(encoding='utf8'));expected=json.loads((OLD/'prediction.json').read_text(encoding='utf8'))
    with gzip.open(new/'all_ranked_candidates.json.gz','rt',encoding='utf8')as f:allnew=json.load(f)
    with gzip.open(OLD/'all_ranked_candidates.json.gz','rt',encoding='utf8')as f:allold=json.load(f)
    assert allnew==allold,'Raw or ranked candidates changed'
    assert packet['candidates']==expected['candidates'],'Final candidates changed'
    oldrecords=[json.loads(l)for l in(OLD/'canvas_receipts.jsonl').open()]
    newrecords=[json.loads(l)for l in(new/'canvas_receipts.jsonl').open()]
    assert len(oldrecords)==len(newrecords)
    for a,b in zip(oldrecords,newrecords):
        assert a['jobs']==b['jobs']
        assert a['receipt']['output_sha256']==b['receipt']['output_sha256'],'Full head hash changed'
        assert a['receipt']['output_shape']==b['receipt']['output_shape']
    oldjob=json.loads((OLD/'job.json').read_text());a=oldjob['elapsed_ms'];b=job['elapsed_ms']
    def summarize(records):
        times={};raw=wire=received=transmitted=reused=0
        for r in records:
            row=r['receipt'];transmitted+=row.get('transmitted_raw_input_bytes',row.get('raw_input_bytes',0));reused+=row.get('reused_planes',0);raw+=row.get('raw_input_bytes',0);wire+=row.get('compressed_input_bytes',0);received+=row.get('response_bytes',0)
            for k,v in row['stats']['timing_ms'].items():times[k]=times.get(k,0)+v
        return dict(timing_ms=times,raw_input_bytes=raw,compressed_input_bytes=wire,binary_response_bytes=received,transmitted_raw_input_bytes=transmitted,reused_planes=reused)
    result=dict(status='DUAL_LIVE_FULL_CASE_ALL_HEADS_AND_CANDIDATES_EXACT_PASS',case_id=CASE.name,job_id=job['job_id'],
        full_head_hashes_exact=len(newrecords),raw_candidates=len(allnew),selected_candidates=len(packet['candidates']),
        old_complete_job_ms=a,new_complete_job_ms=b,speedup=a/b,latency_reduction_percent=100*(1-b/a),
        old_timing=expected['detector_provenance']['timing'],new_timing=packet['detector_provenance']['timing'],
        old_transfer=summarize(oldrecords),new_transfer=summarize(newrecords),
        scope='Same CT, lung mask, full model, thresholds and complete scoring. All full canvas head hashes and raw/final candidates exact; separate runs, no expert labels.')
    (dest/'result.json').write_text(json.dumps(result,indent=2));(dest/'web_state.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items()if k not in('old_timing','new_timing','old_transfer','new_transfer')},indent=2),flush=True)


if __name__=='__main__':main()
