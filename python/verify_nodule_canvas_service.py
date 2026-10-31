"""Real board canvas versus real board patch heads on saved CT plane inputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_NoduleFull_20261025_01'
PACKAGE=APP/'data/ct_studio_integration_20261025'
sys.path.insert(0,str(APP/'python'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-result',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'data/nodule_full_web_upgrade_20261025/board_planes')
    p.add_argument('--planes',type=int,default=3);a=p.parse_args()
    if not 1<=a.planes<=8:raise ValueError('Use 1..8 real plane comparisons')
    a.out.mkdir(exist_ok=False)
    from start_ct_studio_demo import configuration
    from ct_nodule_canvas_client import CanvasClient
    from ct_studio_shared_client import SharedPatchClient
    from nodule_canvas_contract import quantize_canvas,digest
    from lung_live_data import read_ct
    from nodule_vendor.nodule_canvas import lower_canvas,extract_tile_heads
    from nodule_vendor.nodule_fast_input import SharedPlaneSampler
    from nodule_vendor.nodule_plan import quantize_hu,lower_stem
    from nodule_vendor.lung_nodule_cnn import config_from_dict,decode
    case=a.raw_result.parents[2]
    info=json.loads((case/'input.json').read_text(encoding='utf8'))
    packet=json.loads(a.raw_result.read_text(encoding='utf8'))
    hu,g,source=read_ct(info['source'],confirm_hu=True,series_uid=info.get('series_uid',''),
        allow_inconsistent_frame_uids=info.get('frame_uid_exception_applied')is True)
    if source['hu_sha256']!=packet['source_hu_sha256']:raise ValueError('Original CT hash differs')
    client,_=configuration(PACKAGE/'host_config.json');client.health()
    bundle=json.loads((PACKAGE/'shared_host_platform.json').read_text())
    canv=CanvasClient(client.url,client.token,bundle);canv.health()
    patches=SharedPatchClient(client.url,client.token,bundle)
    model=json.loads((PACKAGE/'ct_studio/shared_cnn/models/nodule/model.json').read_text())
    cfg=config_from_dict(model['detector_config']);groups=defaultdict(list)
    with a.raw_result.with_name('patch_receipts.jsonl').open()as stream:
        for line in stream:
            row=json.loads(line);groups[row['job'][0]].append(row['job'])
    for volume_file in sorted(case.glob('*_volume.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        meta=json.loads(volume_file.read_text(encoding='utf8'))
        if meta.get('source',{}).get('hu_sha256')!=packet['source_hu_sha256']:continue
        with np.load(volume_file.with_name(meta['job_id']+'_masks.npz'))as f:lung=f['body']
        if lung.shape!=hu.shape:raise ValueError('Need saved full-volume body mask')
        from nodule_vendor.lung_nodule_cnn import search_mask
        search=search_mask(lung,g['spacing_zyx_mm'],cfg)
        break
    else:raise ValueError('Matching saved full-volume lung mask missing')
    chosen=np.linspace(0,len(groups)-1,min(a.planes,len(groups))).round().astype(int)
    report=[];arrays={}
    for index in chosen:
        jobs=list(groups.values())[index]
        sampler=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg);h,w=sampler.context.shape[1:]
        actual,receipt=canv.predict(lower_canvas(quantize_canvas(sampler.context)),h,w)
        recovered=extract_tile_heads(actual,jobs,sampler.y0,sampler.x0,cfg)
        old=[];old_ms=0.;arrays['canvas_'+str(index)]=actual
        for start in range(0,len(jobs),8):
            outputs=patches.nodule_batch([lower_stem(quantize_hu(sampler.patch(j)))for j in jobs[start:start+8]])
            for head,record in outputs:old.append(head);old_ms+=record['stats']['host_total_ms']
        old=np.stack(old)
        np.testing.assert_array_equal(old[:,:,15:49,15:49],recovered[:,:,15:49,15:49])
        scale=np.asarray(model['head_scales'],np.float32)[None,:,None,None]
        lhs=decode((old.astype(np.float32)-128)*scale,jobs,g,cfg,search)
        rhs=decode((recovered.astype(np.float32)-128)*scale,jobs,g,cfg,search)
        for x,y in zip(lhs,rhs):np.testing.assert_array_equal(x,y)
        arrays['patches_'+str(index)]=old
        report.append(dict(plane_index=int(index),tiles=len(jobs),canvas_hw=[h,w],
            exact_required_values=True,raw_decode_exact=True,canvas_receipt=receipt,
            old_patch_network_total_ms=old_ms))
        print('PHYSICAL_CANVAS_EXACT',index,'tiles',len(jobs),'new network',receipt['stats']['host_total_ms'],flush=True)
    np.savez_compressed(a.out/'actual_outputs.npz',**arrays)
    result=dict(status='PHYSICAL_CANVAS_REQUIRED_CELLS_AND_DECODE_EXACT',planes=len(report),rows=report,
        service_bundle_sha256=bundle['service_bundle_sha256'],arrays_sha256=digest((a.out/'actual_outputs.npz').read_bytes()),
        scope='Actual board inputs and outputs; required cells and full raw decode equality, not a labeled full-dataset quality test')
    (a.out/'result.json').write_text(json.dumps(result,indent=2))
    print('PHYSICAL_PLANE_ACCEPTANCE_PASS',flush=True)


if __name__=='__main__':main()
