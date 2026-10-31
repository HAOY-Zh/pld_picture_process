"""Real compact/native-CMA heads versus unchanged real lowered canvas route."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_NoduleCompact_20261026_01'
PACKAGE=APP/'data/ct_studio_integration_20261025'
OUT=ROOT/'data/nodule_transport_upgrade_20261026'
sys.path.insert(0,str(APP/'python'))


def main():
    from start_ct_studio_demo import configuration
    from ct_nodule_canvas_client import CanvasClient
    from ct_nodule_compact_client import CompactClient
    from nodule_canvas_contract import quantize_canvas,digest
    from lung_live_data import read_ct
    from nodule_vendor.nodule_canvas import lower_canvas
    from nodule_vendor.nodule_fast_input import SharedPlaneSampler
    from nodule_vendor.lung_nodule_cnn import config_from_dict
    result=OUT/'physical_planes';result.mkdir(exist_ok=False)
    case=ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results/d2ab9db1e644493388c93b0fbe3b2f40'
    info=json.loads((case/'input.json').read_text(encoding='utf8'))
    hu,g,details=read_ct(info['source'],confirm_hu=True,series_uid=info.get('series_uid',''),
        allow_inconsistent_frame_uids=info.get('frame_uid_exception_applied')is True)
    if details['hu_sha256']!=info['hu_sha256']:raise ValueError('Case changed')
    client,_=configuration(PACKAGE/'host_config.json');client.health()
    bundle=json.loads((PACKAGE/'shared_host_platform.json').read_text())
    old=CanvasClient(client.url,client.token,bundle);new=CompactClient(client.url,client.token,bundle);new.health()
    model=json.loads((PACKAGE/'ct_studio/shared_cnn/models/nodule/model.json').read_text());cfg=config_from_dict(model['detector_config'])
    previous=case/'nodule/nodule_bd55846451b14dca99eaa12074c63042/canvas_receipts.jsonl'
    plans=[json.loads(line)for line in previous.open()]
    rows=[];arrays={}
    for index in (0,140,280):
        jobs=plans[index]['jobs'];sampler=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg)
        q=quantize_canvas(sampler.context);h,w=q.shape[1:]
        expected,baseline=old.predict(lower_canvas(q),h,w)
        for repeat in range(2):
            actual,record=new.predict(q,h,w)
            np.testing.assert_array_equal(actual,expected)
            arrays[f'new_{index}_{repeat}']=actual
            rows.append(dict(plane_index=index,repeat=repeat,full_head_exact=True,old_receipt=baseline,new_receipt=record))
            print('COMPACT_PHYSICAL_FULL_HEAD_EXACT',index,repeat,'network_ms',round(record['stats']['host_total_ms'],3),
                'roundtrip_ms',round(record['roundtrip_ms'],3),flush=True)
        arrays['old_'+str(index)]=expected
    np.savez_compressed(result/'actual_heads.npz',**arrays)
    (result/'result.json').write_text(json.dumps(dict(status='PHYSICAL_COMPACT_FULL_HEAD_EXACT_PASS',rows=rows,
        service_bundle_sha256=bundle['service_bundle_sha256'],arrays_sha256=digest((result/'actual_heads.npz').read_bytes())),indent=2))


if __name__=='__main__':main()
