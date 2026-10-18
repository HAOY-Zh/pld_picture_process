"""Paired outer-contour supervision, same TRAIN patients and small network.

Closed holes are ignored ONLY for outer-distance diagnostics/weight geometry.
Human targets, actual predictions and full-mask Dice are never filled/changed.
No official Test/external inference or selection, no automatic deployment.
"""
import argparse
from bisect import bisect_right
import copy
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from lung_body_crop import check_hashes
from lung_distill import GuidedDataset, checked_teacher, guidance_loss, source_hashes, train_records
from lung_error_audit import binary
from lung_segmentation import fresh_output, write_json
from lung_unet import (_case_prediction, aggregate_slice_quality, load_cached_source,
    make_model, mapped_npy, model_summary, read_manifest, save_checkpoint,
    save_learned_preview, segmentation_loss, select_device, slice_quality_metrics, torch_module)
from lungseg.metrics import overlap_metrics

KIND = 'train_outer_distance_v1'
CROSS = ndi.generate_binary_structure(2,1)


def spacing_yx(spacing):
    a=np.asarray(spacing,dtype=float)
    if a.shape!=(2,) or not np.isfinite(a).all() or (a<=0).any():raise ValueError('Positive physical YX spacing required')
    return a


def outer_edge(mask):
    mask=binary(mask)
    if mask.ndim!=2:raise ValueError('Outer contour is explicitly 2D')
    diagnostic=ndi.binary_fill_holes(mask,structure=CROSS)
    return diagnostic & ~ndi.binary_erosion(diagnostic,structure=CROSS,border_value=0),diagnostic


def distance_weight(label,spacing,clip_mm=10.):
    """Normalized distance to EXTERIOR reference edge; expert holes weight=0.

    Empty targets receive zero distance loss (original BCE/Dice/KD still active).
    This is a class-balanced, nonnegative distance-error variant inspired by
    boundary loss, not a verbatim reimplementation of the original paper.
    """
    label=binary(label);spacing=spacing_yx(spacing)
    if not np.isfinite(clip_mm) or clip_mm<=0:raise ValueError('Positive distance clipping required')
    edge,filled=outer_edge(label)
    if not label.any():return np.zeros(label.shape,np.float32)
    result=(np.minimum(ndi.distance_transform_edt(~edge,sampling=spacing),clip_mm)/clip_mm).astype(np.float32)
    result[filled & ~label]=0  # Never train legitimate expert holes as lung.
    return result


def boundary_loss(logits,label,weight):
    torch=torch_module()
    if logits.shape!=label.shape or label.shape!=weight.shape:raise ValueError('Distance grids differ')
    if not torch.isfinite(weight).all() or (weight<0).any() or (weight>1).any():raise ValueError('Invalid distance weights')
    p=logits.float().sigmoid();penalty=(p-label).abs()*weight
    dims=tuple(range(1,logits.ndim));active=weight.sum(dims)>0
    terms=[]
    for group in [label,1-label]:
        counts=group.sum(dims);valid=(counts>0)&active
        if valid.any():terms.append(((penalty*group).sum(dims)/counts.clamp_min(1))[valid].mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0


def identity(r):
    return r['case'],r['split'],r['patient_key'],r['frame_uid'],r['slices'],r['native_shape']


def prepare(args):
    root,manifest=read_manifest(args.dataset);records=train_records(manifest)
    if not np.isfinite(args.clip_mm) or args.clip_mm<=0:raise ValueError('Invalid physical distance limit')
    tracked=source_hashes([args.dataset,__file__])
    for r in records:
        tracked.update(source_hashes([root/r['cache_folder']/n for n in ['images.npy','labels.npy','reference_native.npy']]))
    output=fresh_output(args.outdir)
    report=dict(status='running',kind=KIND,dataset_sha256=file_hash(args.dataset),
        dataset_fingerprint=manifest['fingerprint'],image_size=manifest['image_size'],clip_mm=args.clip_mm,
        cases=[],input_sha256=tracked,description='TRAIN human masks only; existing nearest-resized labels unchanged; '
            'physical spacing scaled to student grid; diagnostic outer distance clipped/normalized, expert holes zero weight')
    write_json(output/'distance_dataset.json',report)
    try:
        for r in records:
            print(f'TRAIN distance map: {r["case"]}',flush=True)
            folder=output/r['case'];folder.mkdir()
            with mapped_npy(root/r['cache_folder']/'labels.npy') as labels:
                shape=(r['slices'],manifest['image_size'],manifest['image_size'])
                if labels.shape!=shape or not np.isin(labels,[0,1]).all():raise ValueError('Original student target differs')
                spacing=spacing_yx(r['spacing_zyx_mm'][1:])*np.array(r['native_shape'][1:])/np.array(shape[1:])
                weights=np.empty(shape,np.float16)
                for i,label in enumerate(labels):weights[i]=distance_weight(label,spacing,args.clip_mm)
            path=folder/'weight.npy';np.save(path,weights)
            report['cases'].append(dict(r,weight_file=str(path.relative_to(output)),weight_sha256=file_hash(path),
                student_spacing_yx_mm=spacing.tolist()))
            write_json(output/'distance_dataset.json',report)
        check_hashes(tracked);report.update(status='complete',protected_inputs_unchanged=True)
        write_json(output/'distance_dataset.json',report)
    except BaseException as error:
        report.update(status='aborted_incomplete',error=str(error));write_json(output/'distance_dataset.json',report);raise


def file_hash(path):
    from lung_unet import file_sha256
    return file_sha256(path)


def checked_distance(path,manifest,dataset_path):
    path=Path(path);root=path.parent;cache=json.loads(path.read_text(encoding='utf-8'))
    rows=train_records(manifest)
    if (cache['status']!='complete' or cache['kind']!=KIND or cache['dataset_sha256']!=file_hash(dataset_path) or
            cache['dataset_fingerprint']!=manifest['fingerprint'] or cache['image_size']!=manifest['image_size'] or
            list(map(identity,rows))!=list(map(identity,cache['cases']))):raise ValueError('Distance TRAIN provenance/grid differs')
    check_hashes(cache['input_sha256'])
    for r in cache['cases']:
        target=(root/r['weight_file']).resolve()
        if not target.is_relative_to(root.resolve()) or file_hash(target)!=r['weight_sha256']:raise ValueError('Distance cache hash/path differs')
        with mapped_npy(target) as a:
            if (a.shape!=(r['slices'],manifest['image_size'],manifest['image_size']) or
                    not np.isfinite(a).all() or (a<0).any() or (a>1).any()):raise ValueError('Invalid distance cache')
    return root,cache


class ContourDataset(GuidedDataset):
    def __init__(self,root,records,teacher_root,teacher_records,distance_root,distance_records,augment=True):
        super().__init__(root,records,teacher_root,teacher_records,augment=False)
        if list(map(identity,records))!=list(map(identity,distance_records)):raise ValueError('TRAIN distance case order differs')
        self.distance_root,self.distance_records=Path(distance_root),distance_records
        self.distance_arrays={};self.contour_augment=augment

    def __getitem__(self,index):
        image,label,teacher=super().__getitem__(index)
        k=bisect_right(self.ends,index);j=index-(self.ends[k-1] if k else 0)
        if k not in self.distance_arrays:
            a=self._mappings.enter_context(mapped_npy(self.distance_root/self.distance_records[k]['weight_file']))
            if a.shape!=self.arrays[k][1].shape:raise ValueError('Distance grid differs')
            self.distance_arrays[k]=a
        weight=torch_module().from_numpy(np.array(self.distance_arrays[k][j,None],dtype=np.float32))
        if self.contour_augment and random.random()<.5:
            image,label,teacher,weight=[t.flip(-1) for t in [image,label,teacher,weight]]
        return image,label,teacher,weight

    def close(self):
        self.distance_arrays.clear();super().close()


def reference_contour(mask,spacing):
    edge,_=outer_edge(mask);points=np.argwhere(edge)*spacing_yx(spacing)
    return dict(points=points,tree=cKDTree(points) if len(points) else None)


def contour_metrics(prediction,reference,spacing):
    """Bidirectional 2D outer-boundary pixel-center distances, physical mm.

    Not DeepMind's area-weighted 3D surfel implementation. Closed holes ignored
    diagnostically; open notches and ALL foreground components remain visible.
    Missing foreground is undefined distance, explicitly counted (never zero).
    """
    edge,_=outer_edge(prediction);points=np.argwhere(edge)*spacing_yx(spacing)
    ref=reference['points']
    if not len(ref):return dict(status='empty_reference',prediction_present=bool(len(points)),hd95_mm=None)
    if not len(points):return dict(status='missing_foreground',bad5_fraction=1.,agreement2_fraction=0.,
        hd95_mm=None,mean_distance_mm=None,reference_points=len(ref),prediction_points=0)
    to_ref=reference['tree'].query(points)[0];to_pred=cKDTree(points).query(ref)[0]
    both=np.concatenate([to_ref,to_pred])
    return dict(status='ok',bad5_fraction=float(np.mean(both>5.)),agreement2_fraction=float(np.mean(both<=2.)),
        hd95_mm=float(max(np.quantile(to_ref,.95),np.quantile(to_pred,.95))),mean_distance_mm=float(both.mean()),
        reference_points=len(ref),prediction_points=len(points))


def contour_summary(rows):
    case_rows=[];main=[];small=[]
    for row in rows:
        selected=[s for s in row['outer_slices'] if s['reference_pixels']>1000]
        small.extend(s for s in row['outer_slices'] if 0<s['reference_pixels']<=1000)
        main.extend(selected)
        if selected:
            case_rows.append(dict(case=row['case'],main_slices=len(selected),
                bad5_fraction=float(np.mean([s['bad5_fraction'] for s in selected])),
                agreement2_fraction=float(np.mean([s['agreement2_fraction'] for s in selected])),
                missing_foreground=sum(s['status']=='missing_foreground' for s in selected),
                mean_hd95_mm=float(np.mean([s['hd95_mm'] for s in selected if s['hd95_mm'] is not None]))
                    if any(s['hd95_mm'] is not None for s in selected) else None))
    if not main:raise ValueError('No main-reference (>1000 native pixels) slices for outer-contour selection')
    rates=sorted([s['bad5_fraction'] for s in main],reverse=True)
    tail=max(1,int(np.ceil(.05*len(rates))))
    return dict(definition='2D physical outer boundary pixel centers; main: native expert area>1000 pixels; '
        'patient-equal mean rates, worst 5% pooled main-slice bad5; no diagnosis or clinical tolerances',
        main_slices=len(main),small_slices=len(small),cases=case_rows,
        mean_case_bad5_fraction=float(np.mean([r['bad5_fraction'] for r in case_rows])),
        mean_case_agreement2_fraction=float(np.mean([r['agreement2_fraction'] for r in case_rows])),
        worst5_percent_bad5_mean=float(np.mean(rates[:tail])),
        main_missing_foreground=sum(s['status']=='missing_foreground' for s in main),
        small_missing_foreground=sum(s['status']=='missing_foreground' for s in small),
        mean_case_hd95_mm=float(np.mean([r['mean_hd95_mm'] for r in case_rows if r['mean_hd95_mm'] is not None]))
            if any(r['mean_hd95_mm'] is not None for r in case_rows) else None)


class OuterValidator:
    def __init__(self,root,records):
        self.root,self.records=Path(root),records;self.contours={}
        if any(r['split']!='validation' for r in records):raise ValueError('Only unchanged validation selects models')

    def evaluate(self,model,device,batch_size,outdir=None):
        model.eval();rows=[]
        for r in self.records:
            prediction,elapsed=_case_prediction(model,self.root,r,device,batch_size,.5)
            with mapped_npy(self.root/r['cache_folder']/'reference_native.npy') as reference:
                if r['case'] not in self.contours:
                    self.contours[r['case']]=[reference_contour(mask,r['spacing_zyx_mm'][1:]) for mask in reference]
                slices=[]
                for i,(pred,expert,info) in enumerate(zip(prediction,reference,self.contours[r['case']])):
                    metrics=contour_metrics(pred,info,r['spacing_zyx_mm'][1:])
                    slices.append(dict(slice=i,reference_pixels=int(np.count_nonzero(expert)),**metrics))
                rows.append(dict(case=r['case'],metrics=overlap_metrics(prediction,reference),
                    slice_quality=slice_quality_metrics(prediction,reference),outer_slices=slices,inference_ms=elapsed))
            if outdir:
                folder=Path(outdir)/r['case'];folder.mkdir(parents=True,exist_ok=True)
                np.save(folder/'prediction_native.npy',prediction.astype(np.uint8)*255)
        summary=dict(mean_case_dice=float(np.mean([r['metrics']['dice'] for r in rows])),
            false_negative=sum(r['metrics']['false_negative'] for r in rows),
            false_positive=sum(r['metrics']['false_positive'] for r in rows),
            slice_quality=aggregate_slice_quality([r['slice_quality'] for r in rows]),outer=contour_summary(rows))
        return rows,summary


def contour_gate(candidate,baseline,max_dice_drop=.005):
    c,b=candidate['outer'],baseline['outer'];cq,bq=candidate['slice_quality'],baseline['slice_quality']
    if not np.isfinite(max_dice_drop) or not 0<=max_dice_drop<=.005:raise ValueError('Dice relative drop at most 0.5%')
    cs,bs=cq['groups']['small']['mean_slice_dice'],bq['groups']['small']['mean_slice_dice']
    previous={r['case']:r for r in b['cases']}
    checks=dict(dice_relative_drop_at_most_0_5_percent=candidate['mean_case_dice']>=baseline['mean_case_dice']*(1-max_dice_drop),
        main_bad5_reduced_at_least_10_percent=c['mean_case_bad5_fraction']<=b['mean_case_bad5_fraction']*.9+1e-12,
        worst5_bad5_reduced_at_least_5_percent=c['worst5_percent_bad5_mean']<=b['worst5_percent_bad5_mean']*.95+1e-12,
        no_more_main_missing_slices=c['main_missing_foreground']<=b['main_missing_foreground'],
        no_case_main_bad5_increase_over_2_percentage_points=all(r['bad5_fraction']<=previous[r['case']]['bad5_fraction']+.02 for r in c['cases']),
        overall_fp_increase_at_most_5_percent=candidate['false_positive']<=baseline['false_positive']*1.05,
        empty_fp_increase_at_most_10_percent=cq['empty_false_positive_pixels']<=bq['empty_false_positive_pixels']*1.1,
        empty_positive_count_not_increased=cq['empty_slices_with_prediction']<=bq['empty_slices_with_prediction'],
        small_dice_drop_at_most_0_01=bs is None or (cs is not None and cs>=bs-.01))
    return dict(passed=all(checks.values()),checks=checks,dice_floor=baseline['mean_case_dice']*(1-max_dice_drop))


def rank(summary,baseline,drop):
    if summary['mean_case_dice']<baseline['mean_case_dice']*(1-drop):return (1,-summary['mean_case_dice'],0,0)
    o=summary['outer']
    return (0,o['main_missing_foreground'],o['mean_case_bad5_fraction'],o['worst5_percent_bad5_mean'])


def previews(root,manifest,output,report):
    pairs=[('LCTSC-Train-S2-006',54),('LCTSC-Train-S3-008',79)]
    # Also show a largest ORIGINAL main-contour deviation, not just familiar examples.
    all_main=[(s['bad5_fraction'],r['case'],s['slice']) for r in report['baseline_cases']
        for s in r['outer_slices'] if s['reference_pixels']>1000]
    if all_main:
        _,case,index=max(all_main);pairs.append((case,index))
    saved=[]
    for case,index in dict.fromkeys(pairs):
        records=[r for r in manifest['cases'] if r['case']==case]
        if not records:continue
        r=records[0];hu=load_cached_source(r)[index]
        with mapped_npy(root/r['cache_folder']/'reference_native.npy') as a:expert=a[index]!=0
        with mapped_npy(output/'distance'/'selected'/case/'prediction_native.npy') as a:pred=a[index]!=0
        for name,folder in [('original','baseline'),('paired_control','teacher_control/selected')]:
            with mapped_npy(output/folder/case/'prediction_native.npy') as a:previous=a[index]!=0
            path=output/f'comparison_{case}_slice_{index:03d}_vs_{name}.png'
            save_learned_preview(hu,expert,pred,previous,path,case,index,
                baseline_title=f'{name} (yellow)',learned_title='Outer-distance SMALL U-Net (green)')
            saved.append(str(path))
    return saved


def paired(args):
    if (args.epochs<1 or args.batch_size<1 or args.threads<1 or args.ramp_epochs<1 or
            not 0<args.lr<1 or not 0<=args.boundary_weight<=2 or not 0<=args.max_dice_drop<=.005):
        raise ValueError('Invalid bounded training settings')
    root,manifest=read_manifest(args.dataset);records=train_records(manifest)
    teacher_root,teacher=checked_teacher(args.teacher_dataset,manifest)
    distance_root,distance=checked_distance(args.distance_dataset,manifest,args.dataset)
    if teacher['dataset_sha256']!=file_hash(args.dataset):raise ValueError('Teacher dataset bytes differ')
    torch=torch_module();original=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if (original['dataset_fingerprint']!=manifest['fingerprint'] or original.get('context_slices',1)!=1 or
            original['image_size']!=manifest['image_size'] or original['hu_clip']!=[-1000,400] or original['threshold']!=.5):
        raise ValueError('Retained original full-field 2D initialization required')
    tracked=source_hashes([args.dataset,args.teacher_dataset,args.distance_dataset,args.checkpoint,__file__]+[
        Path(__file__).with_name(n) for n in ['lung_unet.py','lung_body_crop.py','lung_distill.py','lung_error_audit.py']])
    tracked.update(teacher['input_sha256']);tracked.update(distance['input_sha256'])
    for r in manifest['cases']:
        tracked.update(source_hashes([root/r['cache_folder']/n for n in ['images.npy','labels.npy','reference_native.npy']]))
    for r in teacher['cases']:tracked.update(source_hashes([teacher_root/r['probability_file']]))
    for r in distance['cases']:tracked.update(source_hashes([distance_root/r['weight_file']]))
    device=select_device(args.device);torch.set_num_threads(args.threads)
    model=make_model(original['base_channels'],1);model.load_state_dict(original['model'])
    architecture=model_summary(model,manifest['image_size']);output=fresh_output(args.outdir)
    val=[r for r in manifest['cases'] if r['split']=='validation'];validator=OuterValidator(root,val)
    protocol=dict({k:v for k,v in vars(args).items() if k!='func'},architecture=architecture,input_sha256=tracked,
        train_cases=[r['case'] for r in records],validation_cases=[r['case'] for r in val],
        held_out_test_cases=manifest.get('held_out_test_cases',[]),teacher_weight=.25,teacher_confidence=.8,
        sampling='Original uniform shuffle; original number of slices, same paired seed/order/flips',
        distance_loss='Class-balanced soft probability error * clipped/normalized distance to human OUTER contour; '
            'expert holes zero EXTRA weight, original targets/region/KD supervision retained; empty distance maps zero',
        validation='Full native masks retained; closed holes excluded ONLY from 2D outer diagnostics. '
            'Physical pixel-center HD95=max(two directional 95th percentiles), not surfel-area weighted/3D',
        selection='Among baseline-gate passing epochs choose smallest main outer bad5; otherwise exploratory '
            'smallest main bad5 among Dice-eligible epochs; original never auto-replaced',
        warning='Same existing 18/6 patients; no official Test/COVID training, evaluation or parameter selection. '
            '2mm/5mm and improvement gates are engineering choices, not clinical standards; teacher overlap unknown.')
    write_json(output/'protocol.json',protocol)
    report=dict(status='running',protocol=protocol,arms={},original_model_retained=True)
    write_json(output/'evaluation.json',report)
    old=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False;model.to(device)
        rows,baseline=validator.evaluate(model,device,args.batch_size,output/'baseline')
        report.update(baseline=baseline,baseline_cases=rows)
        write_json(output/'baseline.json',dict(cases=rows,summary=baseline));write_json(output/'evaluation.json',report)
        for arm in ['teacher_control','distance']:
            model.load_state_dict(original['model']);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
            scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');folder=output/arm;folder.mkdir()
            history=[];best=accepted=None
            with ContourDataset(root,records,teacher_root,teacher['cases'],distance_root,distance['cases']) as dataset:
                for epoch in range(1,args.epochs+1):
                    torch.manual_seed(args.seed+epoch);random.seed(args.seed+epoch);np.random.seed(args.seed+epoch)
                    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,
                        pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(args.seed+epoch))
                    coefficient=args.boundary_weight*min(epoch/args.ramp_epochs,1.) if arm=='distance' else 0.
                    model.train();total=kd_total=bd_total=count=0;start=perf_counter()
                    for step,batch in enumerate(loader,1):
                        images,labels,probabilities,weights=[x.to(device) for x in batch];optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                            logits=model(images).float();kd=guidance_loss(logits,labels,probabilities,.8)
                            bd=boundary_loss(logits,labels,weights) if coefficient else logits.sum()*0
                            loss=segmentation_loss(logits,labels)+.25*kd+coefficient*bd
                        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                        scaler.scale(loss).backward();scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update()
                        total+=float(loss.detach())*len(images);kd_total+=float(kd.detach())*len(images)
                        bd_total+=float(bd.detach())*len(images);count+=len(images)
                        if step%100==0:print(f'{arm} epoch {epoch} batch {step}/{len(loader)}',flush=True)
                    rows,summary=validator.evaluate(model,device,args.batch_size)
                    gate=contour_gate(summary,baseline,args.max_dice_drop);key=rank(summary,baseline,args.max_dice_drop)
                    history.append(dict(epoch=epoch,loss=total/count,guidance_kl=kd_total/count,
                        distance_loss=bd_total/count,distance_coefficient=coefficient,summary=summary,cases=rows,gate=gate,
                        seconds_including_validation=perf_counter()-start))
                    payload={k:copy.deepcopy(original[k]) for k in ['base_channels','image_size','hu_clip','threshold']}
                    payload.update(model=model.state_dict(),epoch=epoch,context_slices=1,seed=args.seed,history=history,
                        dataset_fingerprint=manifest['fingerprint'],preprocessing={'kind':'full_frame'},
                        outer_experiment=dict(arm=arm,initialization_sha256=file_hash(args.checkpoint),
                            teacher_weight=.25,teacher_confidence=.8,boundary_weight=args.boundary_weight,
                            clip_mm=distance['clip_mm'],ramp_epochs=args.ramp_epochs),
                        warning='Only training objective differs; unchanged inference graph; not quantized/deployed')
                    save_checkpoint(folder/'last.pt',payload)
                    if best is None or key<best:best=key;save_checkpoint(folder/'exploratory_best.pt',payload)
                    if gate['passed'] and (accepted is None or key<accepted):
                        accepted=key;save_checkpoint(folder/'accepted_candidate.pt',payload)
                    write_json(folder/'history.json',history)
                    print(f'{arm} epoch {epoch}: Dice={summary["mean_case_dice"]:.6f}, '
                        f'main bad5={summary["outer"]["mean_case_bad5_fraction"]:.6f}, '
                        f'tail={summary["outer"]["worst5_percent_bad5_mean"]:.6f}, gate={gate["passed"]}',flush=True)
            chosen=folder/('accepted_candidate.pt' if accepted is not None else 'exploratory_best.pt')
            selected=torch.load(chosen,map_location='cpu',weights_only=True);model.load_state_dict(selected['model'])
            rows,summary=validator.evaluate(model,device,args.batch_size,folder/'selected')
            report['arms'][arm]=dict(selected_epoch=selected['epoch'],selected_file=chosen.name,cases=rows,
                summary=summary,gate=contour_gate(summary,baseline,args.max_dice_drop),history=history)
            write_json(output/'evaluation.json',report)
        c=report['arms']['distance']['summary'];b=report['arms']['teacher_control']['summary']
        report['beats_paired_control']=dict(
            main_bad5_reduced_at_least_10_percent=c['outer']['mean_case_bad5_fraction']<=b['outer']['mean_case_bad5_fraction']*.9+1e-12,
            worst5_bad5_reduced_at_least_5_percent=c['outer']['worst5_percent_bad5_mean']<=b['outer']['worst5_percent_bad5_mean']*.95+1e-12,
            no_more_main_missing_slices=c['outer']['main_missing_foreground']<=b['outer']['main_missing_foreground'])
        report['previews']=previews(root,manifest,output,report);check_hashes(tracked)
        report.update(status='complete_paired_outer_contour_screen',protected_inputs_unchanged=True,
            decision='candidate_needs_further_validation' if report['arms']['distance']['gate']['passed'] and
                all(report['beats_paired_control'].values()) else 'do_not_replace_original')
        write_json(output/'evaluation.json',report);print(f'Completed outer screen: {report["decision"]}',flush=True)
    except BaseException as error:
        report.update(status='aborted_incomplete',error=str(error));write_json(output/'evaluation.json',report);raise
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old


def parser():
    result=argparse.ArgumentParser(description=__doc__);sub=result.add_subparsers(dest='command',required=True)
    for name,func in [('prepare',prepare),('paired',paired)]:
        p=sub.add_parser(name);p.add_argument('--dataset',default='data/lung_learning_20261009/dataset.json')
        p.add_argument('--outdir',required=True);p.set_defaults(func=func)
        if name=='prepare':p.add_argument('--clip-mm',type=float,default=10.)
        else:
            p.add_argument('--teacher-dataset',default='data/lung_distillation_20261013/teacher_cache/teacher_dataset.json')
            p.add_argument('--distance-dataset',required=True)
            p.add_argument('--checkpoint',default='data/lung_training_20261011/early_stop_run/best.pt')
            p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=8)
            p.add_argument('--seed',type=int,default=20261013);p.add_argument('--threads',type=int,default=4)
            p.add_argument('--lr',type=float,default=.0001);p.add_argument('--boundary-weight',type=float,default=.5)
            p.add_argument('--ramp-epochs',type=int,default=4);p.add_argument('--max-dice-drop',type=float,default=.005)
            p.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    return result


if __name__=='__main__':
    args=parser().parse_args();args.func(args)
