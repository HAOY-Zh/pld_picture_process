"""Whole-case original/new segmentation, including actual native postprocessing."""
import argparse
import json
from pathlib import Path
import re
import time
import urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/dual_pipeline_optimization_20261026'
CASE=ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results/08ea2953f1664d608f11b863f07fe68e'


def api():
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}));url='http://127.0.0.1:8099'
    html=op.open(url,timeout=5).read().decode();token=re.search(r'<meta name="ui-token" content="([^"]+)"',html)[1]
    def call(path,data=None):
        req=urllib.request.Request(url+path,data=None if data is None else json.dumps(data).encode(),
            headers={'X-UI-Token':token,**({'Content-Type':'application/json'}if data is not None else{})})
        return json.load(op.open(req,timeout=10))
    return call


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('baseline','optimized'));a=p.parse_args()
    dest=OUT/('seg_'+a.mode);dest.mkdir(exist_ok=False)
    call=api();s=call('/api/state')
    if s['case_id']!=CASE.name or s['phase'] not in ('done','idle'):raise ValueError('Different or active case')
    if (s.get('nodule',{}).get('job')or{}).get('status')=='RUNNING':raise ValueError('Active nodule job')
    (dest/'before.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    previous=s.get('volume_job_id');call('/api/run',dict(mode='volume',index=0));began=time.perf_counter();printed=-50
    while True:
        s=call('/api/state');(dest/'progress.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
        if s['phase']in('error','cancelled'):raise RuntimeError(s.get('error')or s['phase'])
        if s['completed']-printed>=50 or s['phase']=='done':
            print(a.mode,s['phase'],s['completed'],'/',s['total'],round(s.get('elapsed_ms',0)/1000,1),'s',flush=True);printed=s['completed']
        if s['phase']=='done'and s.get('volume_job_id')!=previous:break
        if time.perf_counter()-began>1200:raise TimeoutError('Job retained for inspection')
        time.sleep(1)
    job=s['volume_job_id'];meta=json.loads((CASE/(job+'_volume.json')).read_text(encoding='utf8'))
    records=[json.loads((CASE/r/'receipt.json').read_text(encoding='utf8')) for r in meta['slice_receipts'].values()]
    mean=lambda key:float(np.mean([r['stats'].get(key,0)for r in records]))
    timings={k:float(np.mean([r['stats']['timing_ms'][k]for r in records]))for k in records[0]['stats']['timing_ms']}
    result=dict(status='BASELINE_COMPLETE',job_id=job,case_id=CASE.name,slices=len(records),
        total_elapsed_ms=meta['total_elapsed_ms'],network_mean_ms=mean('host_total_ms'),
        pl_active_mean_ms=mean('pl_command_cycles')/100000,roundtrip_mean_ms=float(np.mean([r['roundtrip_ms']for r in records])),
        timing_mean_ms=timings,volume_ml=meta['volume_ml'])
    if a.mode=='optimized':
        old=json.loads((OUT/'seg_baseline/result.json').read_text());oldmeta=json.loads((CASE/(old['job_id']+'_volume.json')).read_text(encoding='utf8'))
        for z,rid in meta['slice_receipts'].items():
            with np.load(CASE/rid/'actual_head.npz',allow_pickle=False)as new,np.load(CASE/oldmeta['slice_receipts'][z]/'actual_head.npz',allow_pickle=False)as original:
                for name in ('head_codes','input_codes'):np.testing.assert_array_equal(new[name],original[name])
        with np.load(CASE/(job+'_masks.npz'),allow_pickle=False)as new,np.load(CASE/(old['job_id']+'_masks.npz'),allow_pickle=False)as original:
            assert set(new.files)==set(original.files)
            for name in new.files:np.testing.assert_array_equal(new[name],original[name])
        assert meta['volume_ml']==oldmeta['volume_ml']
        result.update(status='OPTIMIZED_ALL_351_HEADS_INPUTS_AND_3D_MASKS_EXACT_PASS',
            speedup=old['total_elapsed_ms']/meta['total_elapsed_ms'],baseline_total_elapsed_ms=old['total_elapsed_ms'])
    (dest/'result.json').write_text(json.dumps(result,indent=2));(dest/'state.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
