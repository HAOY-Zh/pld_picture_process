"""Real compact/native-CMA heads versus unchanged real lowered canvas route."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_DualExact_20261026_02'
PACKAGE=APP/'data/ct_studio_integration_20261025'
OUT=ROOT/'data/dual_pipeline_optimization_20261026'
sys.path.insert(0,str(APP/'python'))


def main():
    from start_ct_studio_demo import configuration
    from ct_nodule_canvas_client import CanvasClient
    from ct_dual_io_client import CachedCanvasClient as CompactClient
    from nodule_canvas_contract import quantize_canvas,digest
    from lung_live_data import read_ct
    from nodule_vendor.nodule_canvas import lower_canvas
    from nodule_vendor.nodule_fast_input import SharedPlaneSampler
    from nodule_vendor.lung_nodule_cnn import config_from_dict
    result=OUT/'wire_pairs';result.mkdir(exist_ok=False)
    case=ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results/08ea2953f1664d608f11b863f07fe68e'
    info=json.loads((case/'input.json').read_text(encoding='utf8'))
    hu,g,details=read_ct(info['source'],confirm_hu=True,series_uid=info.get('series_uid',''),
        allow_inconsistent_frame_uids=info.get('frame_uid_exception_applied')is True)
    if details['hu_sha256']!=info['hu_sha256']:raise ValueError('Case changed')
    from ct_nodule_compact_client import CompactClient as PreviousClient
    client,_=configuration(PACKAGE/'host_config.json');client.health()
    bundle=json.loads((PACKAGE/'shared_host_platform.json').read_text())
    old=PreviousClient(client.url,client.token,bundle);new=CompactClient(client.url,client.token,bundle)
    old.health();new.health()
    model=json.loads((PACKAGE/'ct_studio/shared_cnn/models/nodule/model.json').read_text());cfg=config_from_dict(model['detector_config'])
    plans=[json.loads(line)for line in(case/'nodule/nodule_cd90efb94313430d8b0f649176db60dc/canvas_receipts.jsonl').open()]
    rows=[]
    for index in range(89,102):
        sampler=SharedPlaneSampler(hu,g['spacing_zyx_mm'],plans[index]['jobs'],cfg)
        q=quantize_canvas(sampler.context);h,w=q.shape[1:];results={};receipts={}
        for label,transport in ([('old',old),('new',new)]if index%2 else [('new',new),('old',old)]):
            results[label],receipts[label]=transport.predict(q,h,w)
        np.testing.assert_array_equal(results['old'],results['new'])
        assert digest(results['new'].tobytes())==plans[index]['receipt']['output_sha256']
        if index>89:
            rows.append(dict(index=index,old=receipts['old'],new=receipts['new']))
            print('PAIRED_WIRE',index,'old',round(receipts['old']['roundtrip_ms'],2),
                'new',round(receipts['new']['roundtrip_ms'],2),'reused',receipts['new']['reused_planes'],flush=True)
    means={label:float(np.mean([r[label]['roundtrip_ms']for r in rows]))for label in ('old','new')}
    value=dict(status='PAIRED_ALL_FULL_HEADS_EXACT_PASS',planes=len(rows),mean_roundtrip_ms=means,
        speedup=means['old']/means['new'],rows=rows,scope='Same process, same board runtime, same original planes, alternating request order; first plane warmup excluded')
    (result/'result.json').write_text(json.dumps(value,indent=2));print(json.dumps(means),flush=True)


if __name__=='__main__':main()
