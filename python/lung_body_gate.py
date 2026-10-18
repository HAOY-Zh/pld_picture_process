"""CT-only 2D body-envelope OUTPUT constraint for the frozen original model.

No input cropping/rescaling change, no new model, no labels in gate decisions.
Uses a conservative body bounding envelope, not an exact skin segmentation.
Optional 3D island cleanup runs AFTER this 2D gate, before surface reconstruction.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from lung_body_crop import body_plan
from lung_case_browser import checked_inputs
from lung_island_cleanup import PRIMARY, clean_prediction, score, evaluation_tables, summarize
from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256, load_cached_source, mapped_npy, resize_stack, torch_module
from lungseg.metrics import overlap_metrics

KIND='body_envelope_output_gate_v1'
PARAMETERS=dict(margin_mm=20.,closing_mm=3.)


def gate_slice(hu, prediction, spacing_yx, margin_mm=20., closing_mm=3.):
    """Estimate a CT-only envelope and intersect native prediction; no GT input."""
    hu,prediction=np.asarray(hu),np.asarray(prediction)
    if hu.shape!=prediction.shape or prediction.ndim!=2 or not np.isin(prediction,[0,1,255]).all():
        raise ValueError('Need matching 2D CT and binary native prediction')
    plan=body_plan(hu,spacing_yx,margin_mm=margin_mm,closing_mm=closing_mm)
    y0,y1,x0,x1=plan['box_y0_y1_x0_x1']
    result=np.zeros(prediction.shape,bool)
    result[y0:y1,x0:x1]=prediction[y0:y1,x0:x1]!=0
    return result,plan


def gate_volume(hu, prediction, spacing_yx):
    if np.shape(hu)!=np.shape(prediction) or np.ndim(hu)!=3 or not len(hu):
        raise ValueError('Need matching nonempty native CT/prediction stacks')
    start=perf_counter();result=np.empty(np.shape(prediction),bool);plans=[]
    for z in range(len(hu)):
        result[z],plan=gate_slice(hu[z],prediction[z],spacing_yx,**PARAMETERS)
        plans.append(plan)
    return result,plans,(perf_counter()-start)*1000


def predict_slice_with_body(model, hu, spacing_yx, device, image_size=256):
    """Unknown single-slice API; identical original normalization and .5 threshold.

    Caller supplies the frozen original eval-mode 1-channel model. No reference,
    patient identity, layer class, neighbor slice or 3D data required by this gate.
    """
    if getattr(model,'context_slices',1)!=1 or model.training or image_size<32 or image_size%8:
        raise ValueError('Original eval-mode single-slice model and valid image size required')
    hu=np.asarray(hu)
    if hu.ndim!=2 or not np.isfinite(hu).all():
        raise ValueError('Need finite native HU slice')
    normalized=((resize_stack(np.clip(hu,-1000,400)[None],image_size,1)+1000)/1400).astype(np.float16).astype(np.float32)
    torch=torch_module()
    with torch.inference_mode():
        probability=torch.sigmoid(model(torch.from_numpy(normalized[:,None]).to(device)))
        probability=torch.nn.functional.interpolate(probability,size=hu.shape,mode='bilinear',align_corners=False)
        raw=(probability[0,0]>=.5).cpu().numpy()
    constrained,plan=gate_slice(hu,raw,spacing_yx,**PARAMETERS)
    return dict(original=raw,body=constrained,body_plan=plan)


def evaluate_native(prediction,reference,original):
    """Scoring only, separated from CT-only inference/filtering functions."""
    prediction,reference,original=[np.asarray(a)!=0 for a in [prediction,reference,original]]
    if prediction.shape!=original.shape or (prediction&~original).any():
        raise ValueError('Output constraint must be a subset of original prediction')
    # One label per foreground suffices: score() independently derives slice metrics.
    tables=evaluation_tables(prediction.astype(np.uint8),reference)
    result=score(tables,[])
    removed=original&~prediction
    result['removed_true_lung_voxels']=int((removed&reference).sum())
    result['removed_false_positive_voxels']=int((removed&~reference).sum())
    result['removed_components']=0 # Not a component count for 2D envelope clipping.
    result['lost_positive_slices']=int(((reference.sum((1,2))>0)&(original.sum((1,2))>0)&(prediction.sum((1,2))==0)).sum())
    return result


def save_preview(hu,expert,original,body,combined,plan,path,case,index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,mask,title in zip(axes,[expert,original,body,combined],['Expert','Original','2D body envelope','Body then 3D cleanup']):
        ax.imshow(hu,cmap='gray',vmin=-1000,vmax=400)
        if mask.any():ax.contour(mask,levels=[.5],colors=['cyan' if title=='Expert' else 'orange'],linewidths=.7)
        ax.set_title(title+' | pixels '+str(int(mask.sum())));ax.axis('off')
    y0,y1,x0,x1=plan['box_y0_y1_x0_x1']
    for ax in axes[2:]:ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='#75e98d',linewidth=1))
    fig.suptitle(f'{case} | axial {index} | green box=conservative body envelope | education only')
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def run(args):
    root,manifest,old,records,all_tracked=checked_inputs(args.dataset,args.screen,args.checkpoint)
    # Keep all medical/model hashes, but presentation source can evolve separately.
    tracked={p:h for p,h in all_tracked.items() if Path(p).suffix not in ['.py','.html']}
    for p in [Path(__file__).resolve(),Path(__file__).with_name('lung_body_crop.py'),Path(__file__).with_name('lung_island_cleanup.py')]:
        tracked[str(p)]=file_sha256(p)
    output=fresh_output(args.outdir)
    protocol=dict(kind=KIND,parameters=PARAMETERS,body_threshold_hu=-500,
        body_detection='Largest closed HU>-500 tissue component, holes filled only in temporary body detector; 20 mm expanded square bounding envelope; uncertain/tiny/uniform slices full-frame fallback.',
        order='Full-field 2D model -> native .5 prediction -> 2D body-envelope intersection -> unchanged optional 3D island cleanup -> metrics/surface display.',
        expert_used_for_gate=False,input_crop=False,retraining=False,inference_rerun=False,
        validation_cases=[r['case'] for r in records],train_cases=[r['case'] for r in manifest['cases'] if r['split']=='train'],
        model_epoch=old['protocol']['model_epoch'],model_sha256=old['protocol']['model_sha256'],
        dataset_fingerprint=manifest['fingerprint'],input_sha256=tracked,
        island_parameters=PRIMARY,acceptance='No expert voxels outside estimated envelope, no raw TP lost, no case Dice decrease, and body gate removes some FP.',
        warning='Development validation only; approximate envelope, not skin/organ diagnosis, not PL. No Test/COVID accessed.')
    write_json(output/'protocol.json',protocol)
    report=dict(status='running',protocol=protocol,cases=[],original_model_retained=True)
    write_json(output/'evaluation.json',report)
    try:
        for r in records:
            print('2D output body constraint: '+r['case'],flush=True)
            old_row=next(row for row in old['cases'] if row['case']==r['case'])
            hu=load_cached_source(r)
            with np.load(Path(args.screen).parent/r['case']/'predictions_native.npz') as archive:
                original,old_cleaned=archive['original'],archive['primary']
            body,plans,gate_ms=gate_volume(hu,original,r['spacing_zyx_mm'][1:])
            combined,geometry=clean_prediction(body,r['spacing_zyx_mm'],**PRIMARY)
            # All gate/component decisions are made before expert mask is opened.
            folder=output/r['case'];folder.mkdir()
            write_json(folder/'body_geometry.json',dict(kind=KIND,parameters=PARAMETERS,plans=plans,expert_used=False))
            write_json(folder/'combined_geometry.json',geometry)
            with mapped_npy(root/r['cache_folder']/'reference_native.npy') as ref:
                scores={name:evaluate_native(mask,ref,original) for name,mask in
                    [('original',original),('old_cleaned',old_cleaned),('body',body),('body_cleaned',combined)]}
                for metric in ['dice','true_positive','false_positive','false_negative']:
                    if scores['original'][metric]!=old_row['scores']['original'][metric]:raise ValueError('Original screen no longer matches native reference')
                uncovered=0
                for z,plan in enumerate(plans):
                    y0,y1,x0,x1=plan['box_y0_y1_x0_x1'];uncovered+=int(np.count_nonzero(ref[z])-np.count_nonzero(ref[z,y0:y1,x0:x1]))
                np.savez_compressed(folder/'predictions_native.npz',original=original,old_cleaned=old_cleaned,body=body,body_cleaned=combined)
                index=186 if r['case']=='LCTSC-Train-S3-002' and len(hu)>186 else int((original&~body).sum((1,2)).argmax())
                save_preview(hu[index],ref[index]!=0,original[index],body[index],combined[index],plans[index],folder/'preview.png',r['case'],index)
                inspected=dict(index=index,expert_pixels=int(np.count_nonzero(ref[index])),original_pixels=int(original[index].sum()),body_pixels=int(body[index].sum()),body_cleaned_pixels=int(combined[index].sum()))
            row=dict(case=r['case'],split='validation',shape=r['native_shape'],spacing_zyx_mm=r['spacing_zyx_mm'],scores=scores,
                expert_outside_envelope_voxels=uncovered,fallback_slices=sum(p['fallback_full_frame'] for p in plans),
                body_gate_cpu_ms=gate_ms,combined_island_cpu_ms=geometry['cpu_analysis_ms'],inference_ms=old_row['inference_ms'],
                inspected_slice=inspected,reference_sha256=old_row['reference_sha256'],
                prediction_sha256=file_sha256(folder/'predictions_native.npz'),body_geometry_sha256=file_sha256(folder/'body_geometry.json'))
            report['cases'].append(row);write_json(output/'evaluation.json',report)
            print(f'  Original {scores["original"]["dice"]:.6f} -> body {scores["body"]["dice"]:.6f} -> combined {scores["body_cleaned"]["dice"]:.6f}; body removed TP {scores["body"]["removed_true_lung_voxels"]}, uncovered expert {uncovered}',flush=True)
            del hu,original,old_cleaned,body,combined
        summaries={name:summarize([r['scores'][name] for r in report['cases']]) for name in ['original','old_cleaned','body','body_cleaned']}
        accept=bool(summaries['body']['removed_false_positive_voxels']>0 and
            all(r['expert_outside_envelope_voxels']==0 and r['scores']['body']['removed_true_lung_voxels']==0 and
                r['scores']['body_cleaned']['removed_true_lung_voxels']==0 and
                r['scores']['body']['dice']>=r['scores']['original']['dice'] and
                r['scores']['body_cleaned']['dice']>=r['scores']['original']['dice'] for r in report['cases']))
        for p,h in tracked.items():
            if file_sha256(p)!=h:raise ValueError('Protected input changed: '+p)
        report.update(status='complete_body_gate_development_screen',summaries=summaries,protected_inputs_unchanged=True,
            accepted_for_optional_view=accept,decision='offer_body_gate_and_combined_optional' if accept else 'do_not_adopt_body_gate')
        write_json(output/'evaluation.json',report)
        print(json.dumps(dict(decision=report['decision'],summaries=summaries),indent=2),flush=True)
    except BaseException as e:
        report.update(status='aborted_incomplete',error=str(e));write_json(output/'evaluation.json',report);raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='data/lung_learning_20261009/dataset.json')
    p.add_argument('--screen',default='data/lung_island_cleanup_20261014/screen/evaluation.json')
    p.add_argument('--checkpoint',default='data/lung_training_20261011/early_stop_run/best.pt')
    p.add_argument('--outdir',required=True)
    return p


if __name__=='__main__':run(parser().parse_args())
