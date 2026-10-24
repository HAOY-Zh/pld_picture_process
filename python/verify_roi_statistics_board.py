"""Explicit physical acceptance through the single owner. No FPGA retry/fallback."""
import argparse,json,time
from pathlib import Path
import numpy as np
from start_ct_studio_demo import configuration
from lung_live_data import read_ct,input_codes
from roi_vendor.ct_roi_reference import CpuRoiProcessor
from roi_statistics_contract import STATISTICS_ABI,validate_statistics
from ctdenoise.cnn_fixed import load_quantized,infer_fixed
from lung_unet import torch_module


def verify(config,source,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch_module().set_num_threads(2)
    client,manifest=configuration(config);health=client.health()
    assert manifest['roi']['statistics_abi']==STATISTICS_ABI
    hu,geom,details=read_ct(source,slices='180-182',allow_inconsistent_frame_uids=True)
    images=np.rint(hu).clip(-32768,32767).astype(np.int16)
    arrays={'original_hu':images};checks=[];ldct=[]
    oracle=CpuRoiProcessor()
    def roi(image,name,box=None,limits=None,use_hu=False,lung=None):
        actual=client.roi(image,box,limits,use_hu,lung)
        expected=oracle.process_slice(image,box,limits,use_hu,lung)
        np.testing.assert_array_equal(actual['filtered_mask'],expected['filtered_mask'])
        stats=validate_statistics(actual['hu_statistics'],expected['valid_count'])
        values=image[expected['filtered_mask']!=0].astype(np.int64)
        for key in ('count','sum','min','max','mean'):assert stats[key]==expected['hu_statistics'][key],key
        assert stats['sumsq']==int((values*values).sum())
        if values.size:np.testing.assert_allclose(stats['std_population'],values.std(),rtol=1e-12,atol=1e-12)
        arrays[name+'_mask']=actual['filtered_mask']
        checks.append(dict(name=name,statistics=stats,timing=actual['timing'],receipt=actual['hardware_receipt']))
        return actual
    for z,image in enumerate(images):
        roi(image,'original_full_'+str(z))
        roi(image,'original_box_hu_'+str(z),(150,360,100,400),(-1000,-300),True)
        test_mask=(np.indices(image.shape).sum(0)%3!=0).astype(np.uint8)
        roi(image,'original_box_hu_test_mask_'+str(z),(150,360,100,400),(-1000,-300),True,test_mask)
    model_dir=config.parent/'ct_studio/ldct/model'
    weights,spec=load_quantized(model_dir)
    for z,image in enumerate(images):
        enhanced,receipt=client.denoise(image,'cnn')
        reference=infer_fixed(image,weights,spec)[0]
        np.testing.assert_array_equal(enhanced,reference)
        arrays['enhanced_'+str(z)]=enhanced
        ldct.append(dict(index=180+z,receipt=receipt,max_hu_error=0))
        roi(enhanced,'enhanced_full_'+str(z))
        roi(enhanced,'enhanced_box_'+str(z),(150,360,100,400),(-1000,-300),True)
    basic,receipt=client.basic(images[0],dict(operation='window'))
    assert basic.shape==images[0].shape
    roi(images[0],'after_window')
    model=json.loads((config.parent/'ct_studio/segmentation/model.json').read_text())
    head,segmentation=client.infer(input_codes(images[0],model))
    arrays['segmentation_head']=head
    roi(images[0],'after_segmentation')
    for repeat in range(20):roi(images[0],'repeat_'+str(repeat),(140,380,80,420),(-1000,300),True)
    final=client.health()
    np.savez_compressed(out/'physical_outputs.npz',**arrays)
    evidence=dict(status='PASS_PHYSICAL_BOARD',platform_manifest_sha256=manifest['platform_manifest_sha256'],
        statistics_abi=STATISTICS_ABI,roi=manifest['roi'],ldct_identity=manifest['ldct'],
        source=dict(path=str(source),indices=details['source_slice_indices'],hu_sha256=details['hu_sha256']),
        checks=checks,ldct=ldct,basic_receipt=receipt,segmentation_receipt=segmentation,
        initial_health=health,final_health=final,roi_checks=len(checks),cnn_slices=3,
        note='ROI moments checked against independent masked int64 oracle; CNN pixel-exact integer reference; no clinical scoring.')
    (out/'physical_acceptance.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
    print(json.dumps(dict(status=evidence['status'],roi_checks=len(checks),cnn_slices=3,
        cnn_max_hu_error=0,median_roi_dma_ms=float(np.median([r['timing']['dma_compute_seconds']*1000 for r in checks])),
        median_roi_total_ms=float(np.median([r['timing']['total_seconds']*1000 for r in checks])),
        median_cnn_ms=float(np.median([r['receipt']['stats']['total_ms'] for r in ldct]))),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();verify(a.config.resolve(),a.source,a.outdir)
