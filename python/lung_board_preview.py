"""Render ACTUAL accepted PL head codes against native expert labels.

No model forward, retraining or CPU replacement of hardware head predictions.
Sigmoid, native-size bilinear interpolation and body-envelope gate run on CPU.
One slice cannot perform the volume's 3D connected-component cleanup.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from lung_body_gate import gate_slice,PARAMETERS
from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256,load_cached_source,mapped_npy,read_manifest,torch_module
from lungseg.metrics import overlap_metrics
from verify_lung_axil import check_hashes

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_continuous_test import load_package


def native_mask(codes,scale,zero,shape):
    if codes.dtype!=np.uint8 or codes.ndim!=3 or codes.shape[0]!=1:
        raise ValueError('Need actual one-channel CHW UINT8 hardware head')
    if not np.isfinite(scale) or scale<=0 or type(zero) is not int or not 0<=zero<=255:
        raise ValueError('Invalid frozen head quantization grid')
    if len(shape)!=2 or min(shape)<1: raise ValueError('Invalid native geometry')
    torch=torch_module()
    with torch.inference_mode():
        logits=(torch.from_numpy(np.array(codes,copy=True)).to(torch.float32)[None]-zero)*scale
        # Match the accepted order: sigmoid BEFORE interpolation and threshold.
        probability=torch.nn.functional.interpolate(torch.sigmoid(logits),size=tuple(shape),
                                                     mode='bilinear',align_corners=False)
    return probability[0,0].numpy()>=.5


def draw(hu,expert,prediction,path,case,index,dice):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,4.7))
    titles=['Original CT','Expert lung mask (cyan)',
            'Actual FPGA core + CPU native resize','Red: extra / blue: missed / gray: correct']
    for ax,title in zip(axes,titles):
        ax.imshow(hu,cmap='gray',vmin=-1000,vmax=400); ax.set_title(title,fontsize=10); ax.axis('off')
    for ax,mask,color in [(axes[1],expert,(0,1,1)),(axes[2],prediction,(1,.60,0))]:
        overlay=np.zeros((*mask.shape,4),np.float32); overlay[mask,:3]=color; overlay[mask,3]=.30
        ax.imshow(overlay)
        if mask.any() and not mask.all(): ax.contour(mask,levels=[.5],colors=[color],linewidths=.65)
    error=np.zeros((*prediction.shape,3),np.float32)
    error[expert&prediction]=.6
    error[prediction&~expert]=(1,0,0); error[expert&~prediction]=(0,.4,1)
    axes[3].imshow(error)
    fig.suptitle('{} | axial {} | native {}x{} | slice Dice {:.4f}\n'
                 'Actual board output; no 3D cleanup; engineering education only'.format(case,index,*hu.shape,dice),fontsize=11)
    fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)


def run(args):
    workspace=Path(__file__).resolve().parents[1]; root=Path(args.root).resolve()
    manifest,model,_=load_package(root/'lung_continuous')
    acceptance_path=root/'board_report_verification.json'
    accepted=json.loads(acceptance_path.read_text(encoding='utf-8'))
    report_path=root/'lung_continuous_result.json'; head_path=root/'lung_continuous_result_head.npz'
    if (accepted['status']!='verified_returned_real_board_continuous_output'
            or accepted['actual_head_independently_verified'] is not True
            or accepted['report_sha256']!=file_sha256(report_path)
            or accepted['head_codes_sha256']!=file_sha256(head_path)
            or accepted['package_sha256']!=file_sha256(root/'lung_continuous/continuous_manifest.json')):
        raise ValueError('Actual returned hardware output must be independently accepted first')
    check_hashes(accepted['source_sha256'])
    dataset_path=workspace/'data/lung_learning_20261009/dataset.json'
    tracked={str(acceptance_path):file_sha256(acceptance_path),str(head_path):file_sha256(head_path),
             str(report_path):file_sha256(report_path),str(dataset_path):manifest['source_sha256'][str(dataset_path)],
             str(Path(__file__).resolve()):file_sha256(__file__)}
    cache_root,dataset=read_manifest(dataset_path)
    record=next(r for r in dataset['cases'] if r['case']==manifest['case'])
    if record['split']!='validation': raise ValueError('This demonstration uses only the fixed validation fixture')
    index=manifest['slice_index']; reference_path=cache_root/record['cache_folder']/'reference_native.npy'
    frozen_path=workspace/'data/lung_fixed_network_20261014/screen/evaluation.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    prior=next(r for r in frozen['cases'] if r['case']==record['case'])
    prediction_path=frozen_path.parent/record['case']/'predictions_native.npz'
    tracked[str(frozen_path)]=file_sha256(frozen_path)
    tracked[str(prediction_path)]=prior['prediction_sha256']
    for file in [reference_path]+[Path(p) for p in record['source_ct_files_sorted']]:
        tracked[str(file)]=frozen['protocol']['input_sha256'][str(file)]
    check_hashes(tracked)
    with np.load(head_path,allow_pickle=False) as archive: codes=archive['head_codes']
    raw=native_mask(codes,model['output_scale'],model['output_zero'],record['native_shape'][1:])
    with np.load(prediction_path,allow_pickle=False) as archive:
        expected=archive['fixed_raw'][index]
    if not np.array_equal(raw,expected): raise ValueError('Native FPGA-derived mask differs from frozen fixed-network slice')
    # Reuse identity/order/spacing-checked source loading; never treat a display PNG as HU.
    hu=load_cached_source(record)[index]
    with mapped_npy(reference_path) as volume: expert=np.array(volume[index]!=0,copy=True)
    body,plan=gate_slice(hu,raw,record['spacing_zyx_mm'][1:],**PARAMETERS)
    scores={stage:overlap_metrics(mask,expert) for stage,mask in [('raw',raw),('body',body)]}
    output=fresh_output(args.outdir)
    np.savez_compressed(output/'native_masks.npz',fpga_raw=raw,fpga_body=body)
    draw(hu,expert,raw,output/'comparison.png',record['case'],index,scores['raw']['dice'])
    check_hashes(tracked)
    result=dict(status='complete_actual_board_single_slice_native_preview',case=record['case'],slice_index=index,
                model_input_shape=model['input_shape'],native_shape=list(raw.shape),
                source='Actual returned FPGA head codes; CPU sigmoid/native bilinear threshold; no model rerun',
                native_mask_matches_frozen_fixed_slice=True,native_different_pixels=0,scores=scores,body_plan=plan,
                three_dimensional_cleanup_applied=False,clinical_use=False,
                warning='One fixed validation slice only; not full-volume or held-out accuracy, sustained throughput or medical diagnosis',
                source_sha256=tracked,files_sha256={p.name:file_sha256(p) for p in output.iterdir()})
    write_json(output/'metrics.json',result)
    print('Actual FPGA native preview ready:',output/'comparison.png',flush=True)
    print('Slice scores:',json.dumps(scores,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_continuous_20261015')
    parser.add_argument('--outdir',default='data/lung_continuous_20261015/preview')
    run(parser.parse_args())
