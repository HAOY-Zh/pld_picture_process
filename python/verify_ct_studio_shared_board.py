"""Real-CT patch and legacy-route physical acceptance, no retries/fallback."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
from start_ct_studio_demo import configuration,read_web_info
from ct_studio_shared_client import SharedPatchClient
from lung_live_data import read_ct,input_codes
from roi_vendor.ct_roi_reference import CpuRoiProcessor
from ct_basic_processing import process
from ctdenoise.cnn_fixed import load_quantized,infer_fixed


def run(package,source,shared_source,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    legacy,manifest=configuration(package/'host_config.json')
    descriptor=json.loads((package/'shared_host_platform.json').read_text())
    client=SharedPatchClient(legacy.url,legacy.token,descriptor);initial=client.health();legacy.health()
    # Use the supplied independent FP64 convolution oracle; never a PC fallback
    # for the service. The hardware request is always executed separately.
    path=shared_source/'python/lung_nodule_shared_public_package.py'
    spec=importlib.util.spec_from_file_location('independent_shared_reference',path)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    folder=package/'ct_studio/shared_cnn'
    sys.path.insert(0,str(package/'ct_studio'))
    from shared_cnn.nodule_plan import quantize_hu,lower_stem
    from shared_cnn.lung_nodule_cnn import config_from_dict,sample_patch
    model=json.loads((folder/'models/nodule/model.json').read_text())
    with np.load(folder/'models/nodule/parameters.npz') as z:params={n:z[n] for n in z.files}
    hu,geometry,details=read_ct(source,slices='176-184',allow_inconsistent_frame_uids=True)
    cfg=config_from_dict(model['detector_config']);jobs=[(4.,160,100),(4.,160,260),(4.,250,170)]
    arrays={};checks=[];inputs=[]
    for index,job in enumerate(jobs):
        patch=sample_patch(hu,geometry['spacing_zyx_mm'],job,cfg)
        codes=lower_stem(quantize_hu(patch));expected=oracle.dense_reference(model,params,codes)
        actual,receipt=client.nodule_patch(codes);np.testing.assert_array_equal(actual,expected['head'])
        arrays[f'patch_{index}_hu']=patch;arrays[f'patch_{index}_codes']=codes
        arrays[f'patch_{index}_actual_head']=actual;arrays[f'patch_{index}_reference_head']=expected['head']
        inputs.append(codes)
        checks.append(dict(kind='real_CT_patch',index=index,local_sampling_job=job,receipt=receipt,max_code_error=0))
    # Check all pre-existing paths through the original client contract.
    image=np.rint(hu[4]).clip(-32768,32767).astype(np.int16)
    enhanced,ldct=legacy.denoise(image,'cnn');weights,cnn_spec=load_quantized(package/'ct_studio/ldct/model')
    np.testing.assert_array_equal(enhanced,infer_fixed(image,weights,cnn_spec)[0])
    after,receipt=client.nodule_patch(inputs[0]);np.testing.assert_array_equal(after,arrays['patch_0_reference_head'])
    checks.append(dict(kind='after_ldct',receipt=receipt))
    roi=legacy.roi(image,(150,360,100,400),(-1000,-300),True)
    ref=CpuRoiProcessor().process_slice(image,(150,360,100,400),(-1000,-300),True)
    np.testing.assert_array_equal(roi['filtered_mask'],ref['filtered_mask'])
    for key in ('count','sum','min','max','mean'):assert roi['hu_statistics'][key]==ref['hu_statistics'][key]
    after,receipt=client.nodule_patch(inputs[0]);np.testing.assert_array_equal(after,arrays['patch_0_reference_head'])
    checks.append(dict(kind='after_roi',receipt=receipt))
    basic,basic_receipt=legacy.basic(image,dict(operation='window'))
    np.testing.assert_array_equal(basic,process(image,dict(operation='window')))
    after,receipt=client.nodule_patch(inputs[0]);np.testing.assert_array_equal(after,arrays['patch_0_reference_head'])
    checks.append(dict(kind='after_basic',receipt=receipt))
    old=read_web_info('http://127.0.0.1:8099')[2]
    case=Path('data/ct_workspace_demo/results')/old['case_id']
    previous=json.loads((case/(old['volume_job_id']+'_volume.json')).read_text())
    with np.load(case/previous['slice_receipts']['180']/'actual_head.npz') as z:
        prior_head=z['head_codes'].copy();prior_codes=z['input_codes'].copy()
    segmentation_model=json.loads((package/'model.json').read_text())
    codes=input_codes(hu[4],segmentation_model);np.testing.assert_array_equal(codes,prior_codes)
    head,segmentation=legacy.infer(codes);np.testing.assert_array_equal(head,prior_head)
    after,receipt=client.nodule_patch(inputs[0]);np.testing.assert_array_equal(after,arrays['patch_0_reference_head'])
    checks.append(dict(kind='after_segmentation',receipt=receipt))
    for repeat in range(12):
        index=repeat%3;actual,receipt=client.nodule_patch(inputs[index])
        np.testing.assert_array_equal(actual,arrays[f'patch_{index}_reference_head'])
        checks.append(dict(kind='repeated_real_CT_patch',index=index,receipt=receipt,max_code_error=0))
    client.health();legacy.health();current=read_web_info('http://127.0.0.1:8099')[2]
    assert current['case_id']==old['case_id'] and current['full_volume_ready'] and current['known_slices']==386
    assert current['roi']['result']['result_id']==old['roi']['result']['result_id']
    np.savez_compressed(out/'actual_and_reference.npz',**arrays)
    record=dict(status='PASS_PHYSICAL_SHARED_CORE_AND_LEGACY_ROUTES',service_bundle_sha256=descriptor['service_bundle_sha256'],
        source=dict(path=str(source),source_slice_indices=details['source_slice_indices'],hu_sha256=details['hu_sha256']),
        checks=checks,ldct_receipt=ldct,roi_receipt=roi['hardware_receipt'],basic_receipt=basic_receipt,
        segmentation_receipt=segmentation,initial_shared_health=initial,final_shared_health=client.health(),
        actual_nodule_calls=len(checks),unique_real_CT_patches=3,max_head_code_error=0,
        previous_lung_head_exact=True,legacy_cnn_max_hu_error=0,web_case_and_volume_preserved=True,
        whole_case_detection=False,detection_quality_accepted=False)
    (out/'physical_acceptance.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(dict(status=record['status'],nodule_calls=len(checks),real_patches=3,max_code_error=0,
        nodule_median_ms=float(np.median([c['receipt']['stats']['host_total_ms'] for c in checks])),
        nodule_median_PL_active_ms=float(np.median([c['receipt']['stats']['pl_command_cycles']/100000 for c in checks])),
        legacy_cnn_max_hu_error=0,previous_lung_head_exact=True,web_case_preserved=True),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--shared-source',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    a=p.parse_args();run(a.package.resolve(),a.source,a.shared_source.resolve(),a.outdir)
