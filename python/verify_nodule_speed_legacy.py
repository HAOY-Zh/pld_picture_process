"""Physical old-function regression after batch deployment; no full-case rerun."""
import argparse,json
from pathlib import Path
import numpy as np
from start_ct_studio_demo import configuration,read_web_info
from ct_studio_shared_client import SharedPatchClient
from ct_basic_processing import process
from ctdenoise.cnn_fixed import load_quantized,infer_fixed
from ldct_workspace import digest

def run(package,out):
    s=read_web_info('http://127.0.0.1:8099/')[2]
    assert s['phase']=='done' and s['nodule']['job']['status']=='COMPLETE','Wait for detection'
    c,_=configuration(package/'host_config.json');c.health()
    original=next(x for x in s['analysis_sources']if x['source_id']=='original')
    base=Path('data/ct_workspace_demo/results')/original['case_id'];found=None
    for path in (base/'ldct').glob('*.json'):
        meta=json.loads(path.read_text(encoding='utf8'))
        if meta['group_id']==s['analysis_source']['group_id'] and meta['index']==180:
            found=path;break
    assert found is not None
    with np.load(found.with_suffix('.npz'))as z:image=z['hu'].copy()
    assert digest(image)==meta['output_sha256']
    enhanced,ldct=c.denoise(image,'cnn')
    weights,spec=load_quantized(package/'ct_studio/ldct/model')
    np.testing.assert_array_equal(enhanced,infer_fixed(image,weights,spec)[0])
    params=dict(operation='window');actual,basic=c.basic(image,params)
    np.testing.assert_array_equal(actual,process(image,params))
    case=Path('data/ct_workspace_demo/results')/s['case_id']
    volume=json.loads((case/(s['volume_job_id']+'_volume.json')).read_text(encoding='utf8'))
    with np.load(case/volume['slice_receipts']['180']/'actual_head.npz')as z:
        codes=z['input_codes'].copy();expected=z['head_codes'].copy()
    head,seg=c.infer(codes);np.testing.assert_array_equal(head,expected)
    descriptor=json.loads((package/'shared_host_platform.json').read_text())
    shared=SharedPatchClient(c.url,c.token,descriptor)
    with np.load('data/ct_studio_shared_20261024_01/physical_board_01/actual_and_reference.npz')as z:
        rows=shared.nodule_batch([z['patch_0_codes']]);np.testing.assert_array_equal(rows[0][0],z['patch_0_reference_head'])
    after=read_web_info('http://127.0.0.1:8099/')[2]
    assert after['case_id']==s['case_id'] and after['analysis_source_id']==s['analysis_source_id']
    assert after['known_slices']==386 and after['volume_job_id']==s['volume_job_id']
    r=dict(status='PASS_PHYSICAL_LEGACY_FUNCTIONS_AND_SWITCHING',light_cnn_max_hu_error=0,
        original_lung_head_bit_exact=True,basic_output_bit_exact=True,nodule_after_legacy_bit_exact=True,
        web_case_and_386_masks_preserved=True,input_slice_sha256=digest(image),ldct_receipt=ldct,
        basic_receipt=basic,segmentation_receipt=seg,nodule_receipt=rows[0][1],shared_health=shared.health())
    with out.open('x',encoding='utf8')as f:json.dump(r,f,indent=2)
    print(r['status'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.package,a.out)
