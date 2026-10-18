"""Fixed difficult-slice R231 comparison and native-HU boundary diagnostics.

No training, mask repair of our model, new labels, or FPGA deployment. Four
already-seen slices are descriptive diagnostics, NOT an unbiased benchmark.
R231 network output and official volume postprocessing are reported separately.
"""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

from lung_error_audit import binary, error_maps
from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256, load_cached_source, mapped_npy, select_device, torch_module
from lungseg.metrics import overlap_metrics


TARGETS = (('external','coronacases_003',80), ('external','coronacases_009',114),
           ('validation','LCTSC-Train-S2-006',54), ('validation','LCTSC-Train-S3-008',79))
DISTANCES_MM = np.arange(-3.,4.)
OFFICIAL_URL = 'https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth'


def _signed_distance(mask, spacing):
    padded = np.pad(mask,1)
    return (ndi.distance_transform_edt(padded,sampling=spacing)-
            ndi.distance_transform_edt(~padded,sampling=spacing))[1:-1,1:-1]


def boundary_profiles(hu, reference, prediction, spacing, max_points=512):
    """Sample inward -> outward HU profiles, using fixed physical distances.

    Expert group: outer contours only, not legitimate enclosed expert holes.
    Wrong group: predicted foreground edge facing an FN, >=2mm inside expert;
    every sampled +/-3mm location must remain entirely inside the expert mask.
    Filling the expert mask here ONLY identifies its exterior for diagnostics;
    the original expert target and predictions are NEVER filled/changed.
    """
    hu = np.asarray(hu,dtype=np.float32)
    reference,prediction = binary(reference),binary(prediction)
    spacing = np.asarray(spacing,dtype=float)
    if (hu.ndim != 2 or hu.shape != reference.shape or hu.shape != prediction.shape or
            min(hu.shape)<3 or not np.isfinite(hu).all() or spacing.shape!=(2,) or
            not np.isfinite(spacing).all() or (spacing<=0).any() or max_points<1):
        raise ValueError('Need aligned finite HU/binary masks and positive millimeter spacing')
    cross = ndi.generate_binary_structure(2,1)
    filled = ndi.binary_fill_holes(reference,structure=cross)
    expert_edge = reference & ndi.binary_dilation(~filled,structure=cross)
    pred_edge = prediction & ~ndi.binary_erosion(prediction,structure=cross,border_value=0)
    depth = ndi.distance_transform_edt(np.pad(reference,1),sampling=spacing)[1:-1,1:-1]
    fn = reference & ~prediction
    wrong_edge = pred_edge & reference & (depth>=2) & ndi.binary_dilation(fn,structure=cross)
    groups = {}
    for name,edge,normal_mask in [('expert_outer',expert_edge,filled),
                                  ('wrong_internal',wrong_edge,prediction)]:
        # Local signed-distance normals, not HU gradients: avoid picking only strong edges.
        signed = ndi.gaussian_filter(_signed_distance(normal_mask,spacing),sigma=1)
        gy,gx = np.gradient(signed,*spacing)
        magnitude = np.hypot(gy,gx)
        coordinates = np.argwhere(edge & (magnitude>1e-6))
        if len(coordinates):
            normals = -np.stack([gy[edge & (magnitude>1e-6)],gx[edge & (magnitude>1e-6)]],axis=1)
            normals /= np.linalg.norm(normals,axis=1)[:,None]
            samples = coordinates[:,None,:]+DISTANCES_MM[None,:,None]*normals[:,None,:]/spacing
            valid = ((samples[:,:,0]>=0)&(samples[:,:,0]<=hu.shape[0]-1)&
                     (samples[:,:,1]>=0)&(samples[:,:,1]<=hu.shape[1]-1)).all(axis=1)
            if name=='wrong_internal':
                inside=ndi.map_coordinates(reference.astype(np.float32),
                    [samples[:,:,0],samples[:,:,1]],order=1,mode='nearest')
                valid &= (inside>=1-1e-6).all(axis=1)
            coordinates,normals,samples = coordinates[valid],normals[valid],samples[valid]
        else:
            normals=np.empty((0,2));samples=np.empty((0,len(DISTANCES_MM),2))
        eligible = len(coordinates)
        if eligible>max_points:
            keep = np.linspace(0,eligible-1,max_points,dtype=int)
            coordinates,normals,samples = coordinates[keep],normals[keep],samples[keep]
        values = ndi.map_coordinates(hu,[samples[:,:,0],samples[:,:,1]],order=1,mode='nearest')
        inner,outer = values[:,:3].mean(axis=1),values[:,4:].mean(axis=1)
        jump = outer-inner
        summary = {'eligible_points':eligible,'sampled_points':len(values),
            'median_profile_hu':np.median(values,axis=0).tolist() if len(values) else None,
            'median_inside_hu':float(np.median(inner)) if len(values) else None,
            'median_outside_hu':float(np.median(outer)) if len(values) else None,
            'median_signed_jump_hu':float(np.median(jump)) if len(values) else None,
            'absolute_jump_quartiles_hu':np.quantile(np.abs(jump),[.25,.5,.75]).tolist() if len(values) else None}
        groups[name] = {'summary':summary,'coordinates_yx':coordinates,'normals_yx':normals,'values_hu':values}
    return groups


def reference_outputs(inferer, hu):
    """One forward pass, original pinned lungmask preprocessing/postprocessing.

    Mirrors LMInferer._inference in lungmask 0.2.20, exposing both outputs rather
    than silently merging hole filling with network quality. Tests compare this
    adapter with the installed vendor API; no vendor source file is modified.
    """
    from lungmask import utils
    torch = torch_module()
    hu=np.asarray(hu,dtype=np.float32)
    if hu.ndim!=3 or min(hu.shape)<1 or not np.isfinite(hu).all():
        raise ValueError('Need a finite LPS-oriented native HU volume')
    started=perf_counter()
    prepared,boxes=utils.preprocess(hu,resolution=[256,256])
    prepared[prepared>600]=600
    prepared=(prepared+1024)/1624
    labels=[]
    with torch.inference_mode():
        for offset in range(0,len(hu),inferer.batch_size):
            inputs=torch.as_tensor(prepared[offset:offset+inferer.batch_size,None],dtype=torch.float32,device=inferer.device)
            labels.append(inferer.model(inputs).argmax(1).cpu().numpy().astype(np.uint8))
            if offset and offset%64==0:
                print(f'  R231 network slices {offset}/{len(hu)}',flush=True)
    grid=np.concatenate(labels)
    if not np.isin(grid,[0,1,2]).all():
        raise ValueError('R231 must emit background/right/left labels, not lobe labels')
    def restore(array):
        return np.stack([utils.reshape_mask(array[i],boxes[i],hu.shape[1:]) for i in range(len(hu))]).astype(np.uint8)
    raw=restore(grid)
    network_seconds=perf_counter()-started
    print('  R231 raw network output ready; official volume postprocessing follows',flush=True)
    started=perf_counter()
    processed=restore(utils.postprocessing(grid.copy(),disable_tqdm=True))
    return raw,processed,{'network_preprocess_restore_seconds':network_seconds,
        'official_postprocessing_restore_seconds':perf_counter()-started}


def _errors(prediction,reference):
    colors=np.zeros((*reference.shape,3),np.float32)
    colors[prediction & reference]=.6
    colors[prediction & ~reference]=[1,0,0]
    colors[~prediction & reference]=[0,.4,1]
    return colors


def save_figures(hu,reference,original,raw,processed,groups,folder,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,3,figsize=(15,10))
    panels=[(reference,'cyan','Expert (cyan)'),(original,'yellow','Our original (yellow)'),
            (raw,'lime','R231 NETWORK, no mask postprocess'),
            (processed,'orange','R231 + OFFICIAL volume postprocess')]
    for axis,(mask,color,name) in zip([axes[0,0],axes[0,1],axes[0,2],axes[1,0]],panels):
        axis.imshow(hu,cmap='gray',vmin=-1000,vmax=400)
        if mask.any() and not mask.all():axis.contour(mask,levels=[.5],colors=[color],linewidths=.7)
        axis.set_title(name+f'\nSlice Dice={overlap_metrics(mask,reference)["dice"]:.4f}')
    for axis,mask,name in [(axes[1,1],original,'Our errors'),(axes[1,2],raw,'R231 NETWORK errors')]:
        axis.imshow(_errors(mask,reference));axis.set_title(name+'\nred extra / blue missed / gray correct')
    for axis in axes.flat:axis.axis('off')
    figure.suptitle(title+' | selected difficult slice; education only')
    figure.tight_layout();figure.savefig(folder/'comparison.png',dpi=140);plt.close(figure)
    figure,axes=plt.subplots(1,3,figsize=(16,4.5))
    jump_max=max([1.]+[float(np.max(np.abs(g['values_hu'][:,4:].mean(1)-g['values_hu'][:,:3].mean(1))))
                      for g in groups.values() if len(g['values_hu'])])
    for name,color,label in [('expert_outer','cyan','Expert outer'),('wrong_internal','magenta','Our wrong internal')]:
        group=groups[name];values=group['values_hu'];coords=group['coordinates_yx']
        if not len(values):continue
        q=np.quantile(values,[.25,.5,.75],axis=0)
        axes[0].plot(DISTANCES_MM,q[1],color=color,label=f'{label} (n={len(values)})')
        axes[0].fill_between(DISTANCES_MM,q[0],q[2],color=color,alpha=.15)
        axes[1].hist(np.abs(values[:,4:].mean(1)-values[:,:3].mean(1)),bins=np.linspace(0,jump_max,25),
                     density=True,alpha=.45,color=color,label=label)
        axes[2].scatter(coords[:,1],coords[:,0],s=2,color=color,label=label)
    axes[0].axvline(0,color='gray',linestyle='--');axes[0].set_xlabel('mm: inside (-) -> outside (+)')
    axes[0].set_ylabel('Native HU, median / interquartile band');axes[0].legend()
    axes[1].set_xlabel('Absolute mean HU jump, +/-1..3mm');axes[1].set_ylabel('Density');axes[1].legend()
    axes[2].imshow(hu,cmap='gray',vmin=-1000,vmax=400);axes[2].axis('off');axes[2].legend()
    figure.suptitle(title+' | descriptive profiles; NOT a learned HU threshold')
    figure.tight_layout();figure.savefig(folder/'hu_profiles.png',dpi=140);plt.close(figure)
    with (folder/'hu_profiles.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['group','y','x','outward_normal_y','outward_normal_x']+[f'HU_{d:+.0f}mm' for d in DISTANCES_MM])
        for name,group in groups.items():
            for coord,normal,values in zip(group['coordinates_yx'],group['normals_yx'],group['values_hu']):
                writer.writerow([name,*coord,*normal,*values])


def run(args):
    if args.reuse_outputs:
        return refresh_profiles(args)
    torch=torch_module()
    if args.batch_size<1 or args.threads<1:raise ValueError('Positive batch/threads required')
    if importlib.metadata.version('lungmask')!='0.2.20':raise ValueError('Adapter is pinned to lungmask 0.2.20')
    from lungmask.mask import LMInferer
    weights=Path(args.weights)
    digest=file_sha256(weights)
    if not digest.startswith('d5d2fc3d'):raise ValueError('Weight SHA256 does not match official R231 release prefix')
    development=Path(args.development)
    external=Path(args.external)
    baseline=Path(args.baseline)
    previous=Path(args.external_predictions)
    original_checkpoint=Path(args.original_checkpoint)
    original_digest=file_sha256(original_checkpoint)
    dev=json.loads(development.read_text(encoding='utf-8'))
    ext=json.loads((external/'external_dataset.json').read_text(encoding='utf-8'))
    base=json.loads((baseline/'baseline.json').read_text(encoding='utf-8'))
    external_scores=json.loads((previous/'evaluation.json').read_text(encoding='utf-8'))
    base_protocol=json.loads((baseline/'protocol.json').read_text(encoding='utf-8'))
    ext_protocol=json.loads((previous/'protocol.json').read_text(encoding='utf-8'))
    if (base_protocol['input_sha256'][str(original_checkpoint.resolve())]!=original_digest or
            ext_protocol['source_sha256']['checkpoint']!=original_digest):
        raise ValueError('Saved original predictions must belong to the retained model')
    tracked={str(p.resolve()):file_sha256(p) for p in [weights,development,external/'external_dataset.json',
        baseline/'baseline.json',baseline/'protocol.json',previous/'evaluation.json',previous/'protocol.json',
        original_checkpoint,Path(__file__)]}
    target_records=[]
    for cohort,name,index in TARGETS:
        record=next(r for r in (ext if cohort=='external' else dev)['cases'] if r['case']==name)
        if record['split']!=('external_test' if cohort=='external' else 'validation'):
            raise ValueError('Fixed target cohort differs')
        if not 0<=index<record['slices']:raise ValueError('Fixed slice index outside source case')
        root=external if cohort=='external' else development.parent
        folder=root/record['cache_folder']
        prediction=(previous/name/'float_prediction_native.npy' if cohort=='external' else
                    baseline/'baseline'/name/'prediction_native.npy')
        paths=[folder/'reference_native.npy',prediction]
        if cohort=='external':paths += [folder/'hu_native.npy',folder/'case.json']
        else:paths += [Path(p) for p in record['source_ct_files_sorted']]
        for p in paths:tracked[str(p.resolve())]=file_sha256(p)
        target_records.append((cohort,name,index,record,folder,prediction))
    output=fresh_output(args.outdir)
    device=select_device(args.device)
    old_tf32=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    protocol={'targets':list(TARGETS),'scope':'Four already-seen difficult slices, NOT a cohort/clinical test',
        'package':'lungmask==0.2.20','model':'R231 (NOT R231CovidWeb)','weight_url':OFFICIAL_URL,
        'input_sha256':tracked,'device':str(device),'batch_size':args.batch_size,
        'preprocessing':'Official body crop, 256x256, HU [-1024,600] normalization, native nearest label restore',
        'outputs':'Network argmax left/right union; separate official complete-volume postprocess then union',
        'no_training':True,'no_target_edits':True,'no_original_prediction_edits':True,
        'hu_profile':'Native HU; +/-1..3mm signed-distance normal; max512 systematic points per group; '
            'wrong-internal profile samples entirely within original expert mask',
        'limitations':['Targets were chosen because of visible errors, not random or independent.',
            'R231 training-case overlap with evaluation cohorts has not been audited.',
            'R231 may use different dense-area/effusion semantics from these expert references.',
            'HU samples within slices are correlated; no optimal threshold or statistical generalization claim.',
            'Normals at digitized foreground edge pixels are approximate; profiles are not subpixel surface measurements.',
            'Computer reference only; no evidence that R231 fits PYNQ-Z2.']}
    write_json(output/'protocol.json',protocol)
    report={'status':'running_fixed_comparison','protocol':protocol,'cases':[]}
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    try:
        inferer=LMInferer(modelname='R231',modelpath=str(weights),force_cpu=device.type=='cpu',
            batch_size=args.batch_size,volume_postprocessing=False,tqdm_disable=True)
        if inferer.device!=device:raise ValueError('Reference device differs from requested device')
        report['r231_parameters']=sum(p.numel() for p in inferer.model.parameters())
        for cohort,name,index,record,folder,prediction_path in target_records:
            print(f'Comparing {name}, fixed slice {index}; full volume for official postprocessing',flush=True)
            if cohort=='external':
                for filename,key in [('reference_native.npy','reference_sha256'),('hu_native.npy','hu_cache_sha256')]:
                    if file_sha256(folder/filename)!=record[key]:raise ValueError('External native cache changed')
                with mapped_npy(folder/'hu_native.npy') as array:hu=np.array(array,dtype=np.float32)
                spacing=record['ct_geometry']['spacing_zyx']
            else:
                hu=load_cached_source(record)
                spacing=record['spacing_zyx_mm']
            with mapped_npy(folder/'reference_native.npy') as array:reference=binary(array)
            with mapped_npy(prediction_path) as array:original=binary(array)
            if hu.shape!=reference.shape or original.shape!=reference.shape:raise ValueError('HU/GT/prediction grid differs')
            prior=next(r for r in (external_scores if cohort=='external' else base)['cases'] if r['case']==name)
            if overlap_metrics(original,reference)!=(prior['fp32'] if cohort=='external' else prior['metrics']):
                raise ValueError('Saved original prediction and prior expert metrics differ')
            raw,processed,timing=reference_outputs(inferer,hu)
            destination=output/name/f'slice_{index:03d}'
            destination.mkdir(parents=True)
            np.savez_compressed(destination/'slice_masks.npz',hu=hu[index],expert=reference[index],
                original=original[index],r231_raw=raw[index],r231_official=processed[index])
            np.save(output/name/'r231_raw_labels_native.npy',raw)
            np.save(output/name/'r231_official_labels_native.npy',processed)
            groups=boundary_profiles(hu[index],reference[index],original[index],spacing[1:])
            row={'case':name,'cohort':cohort,'slice':index,'spacing_zyx_mm':spacing,'timing':timing,
                 'original':overlap_metrics(original[index],reference[index]),
                 'r231_raw':overlap_metrics(raw[index]!=0,reference[index]),
                 'r231_official':overlap_metrics(processed[index]!=0,reference[index]),
                 'hu_profiles':{key:value['summary'] for key,value in groups.items()},
                 'official_vs_raw_different_pixels':int(((raw[index]!=0)!=(processed[index]!=0)).sum())}
            for key,mask in [('original',original[index]),('r231_raw',raw[index]!=0),('r231_official',processed[index]!=0)]:
                row[key+'_geometry']=error_maps(mask,reference[index],spacing[1:])[0]
            save_figures(hu[index],reference[index],original[index],raw[index]!=0,processed[index]!=0,
                groups,destination,f'{name} | slice {index}')
            write_json(destination/'metrics.json',row)
            report['cases'].append(row)
            write_json(output/'evaluation.json',report)
            print(f'  Slice Dice original {row["original"]["dice"]:.4f}, R231 raw '
                  f'{row["r231_raw"]["dice"]:.4f}, official {row["r231_official"]["dice"]:.4f}',flush=True)
            del hu,reference,original,raw,processed
        for path,expected in tracked.items():
            if file_sha256(path)!=expected:raise ValueError(f'Protected source changed: {path}')
        report.update(status='complete_fixed_difficult_slice_comparison',protected_inputs_unchanged=True,
            original_model_retained=True,warning='No training, no new independent test accuracy, no FPGA deployment')
        write_json(output/'evaluation.json',report)
    except Exception as error:
        report.update(status='aborted_incomplete_comparison',execution_failure=str(error))
        write_json(output/'evaluation.json',report)
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old_tf32


def refresh_profiles(args):
    """Improve sampling safeguards using saved masks, without rerunning R231.

    New output directory; never rewrites an earlier comparison or its protocol.
    Historic inference-code hash stays recorded separately from current code.
    """
    source=Path(args.reuse_outputs)
    source_report=source/'evaluation.json'
    previous=json.loads(source_report.read_text(encoding='utf-8'))
    if (previous['status']!='complete_fixed_difficult_slice_comparison' or
            previous['protocol']['targets']!=[list(t) for t in TARGETS]):
        raise ValueError('Reuse requires a completed comparison of exactly the fixed four targets')
    current_code=str(Path(__file__).resolve())
    tracked={path:digest for path,digest in previous['protocol']['input_sha256'].items() if path!=current_code}
    tracked[current_code]=file_sha256(__file__)
    tracked[str(source_report.resolve())]=file_sha256(source_report)
    for row in previous['cases']:
        path=source/row['case']/f'slice_{row["slice"]:03d}'/'slice_masks.npz'
        tracked[str(path.resolve())]=file_sha256(path)
    for path,digest in tracked.items():
        if file_sha256(path)!=digest:raise ValueError(f'Protected input changed before replot: {path}')
    output=fresh_output(args.outdir)
    protocol=dict(previous['protocol'],input_sha256=tracked,source_evaluation=str(source_report.resolve()),
        historical_inference_code_sha256=previous['protocol']['input_sha256'][current_code],
        no_new_inference=True,
        hu_profile='Native HU; +/-1..3mm signed-distance normal; max512 systematic points; '
            'each wrong-internal sampled position entirely inside unchanged expert mask',
        change='Sampling guard only: reject internal profiles crossing the expert boundary; '
            'reuse original masks/metrics, no model execution, tuning or label edits')
    write_json(output/'protocol.json',protocol)
    report=dict(previous,protocol=protocol,cases=[],status='refreshing_profiles')
    try:
        for old in previous['cases']:
            row=dict(old)
            folder=source/row['case']/f'slice_{row["slice"]:03d}'
            destination=output/row['case']/f'slice_{row["slice"]:03d}'
            destination.mkdir(parents=True)
            with np.load(folder/'slice_masks.npz') as arrays:
                hu,expert,original=arrays['hu'],arrays['expert'],arrays['original']
                raw,processed=arrays['r231_raw']!=0,arrays['r231_official']!=0
                for key,mask in [('original',original),('r231_raw',raw),('r231_official',processed)]:
                    if overlap_metrics(mask,expert)!=row[key]:raise ValueError('Saved slice masks/metrics differ')
                groups=boundary_profiles(hu,expert,original,row['spacing_zyx_mm'][1:])
                row['hu_profiles']={key:value['summary'] for key,value in groups.items()}
                save_figures(hu,expert,original,raw,processed,groups,destination,
                    f'{row["case"]} | slice {row["slice"]}')
            write_json(destination/'metrics.json',row)
            report['cases'].append(row)
        for path,digest in tracked.items():
            if file_sha256(path)!=digest:raise ValueError(f'Protected input changed during replot: {path}')
        report.update(status='complete_fixed_difficult_slice_comparison',protected_inputs_unchanged=True)
        write_json(output/'evaluation.json',report)
        print('Completed guarded HU profiles from saved predictions; no new inference',flush=True)
    except Exception as error:
        report.update(status='aborted_incomplete_profile_refresh',execution_failure=str(error))
        write_json(output/'evaluation.json',report)
        raise


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--outdir',required=True)
    result.add_argument('--reuse-outputs',help='Completed fixed comparison: redraw guarded HU profiles without model execution')
    result.add_argument('--weights',default='data/reference_models/unet_r231-d5d2fc3d.pth')
    result.add_argument('--development',default='data/lung_learning_20261009/dataset.json')
    result.add_argument('--external',default='data/lung_external_20261012/prepared')
    result.add_argument('--external-predictions',default='data/lung_external_20261012/evaluation')
    result.add_argument('--baseline',default='data/lung_refine_20261012/local_brightness')
    result.add_argument('--original-checkpoint',default='data/lung_training_20261011/early_stop_run/best.pt')
    result.add_argument('--batch-size',type=int,default=4)
    result.add_argument('--threads',type=int,default=4)
    result.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    return result


if __name__=='__main__':
    run(parser().parse_args())
