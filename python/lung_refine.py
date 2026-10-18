"""One controlled bright-lung fine-tuning experiment, CPU/GPU only.

Never repairs predictions or changes expert labels. Architecture and inference
threshold are unchanged. Selection uses development validation, never test cases.
"""
import argparse
import copy
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from lung_segmentation import fresh_output, write_json
from lung_unet import (SliceDataset, _case_prediction, aggregate_slice_quality,
    file_sha256, make_model, mapped_npy, model_summary, read_manifest, resize_stack,
    save_checkpoint, save_learned_preview, select_device, slice_quality_metrics, torch_module)
from lungseg.metrics import overlap_metrics


def emphasis_weights(images, target, weight=2., threshold_hu=-400.):
    torch = torch_module()
    if (not np.isfinite(weight) or weight < 1 or not np.isfinite(threshold_hu) or
            not -1000 <= threshold_hu <= 400 or images.shape != target.shape or
            images.ndim != 4 or images.shape[1] != 1):
        raise ValueError('Need matching single-channel batches and valid emphasis settings')
    if (not torch.isfinite(images).all() or not torch.isfinite(target).all() or
            not ((target == 0) | (target == 1)).all()):
        raise ValueError('Finite inputs and unchanged binary expert labels are required')
    selected = (target == 1) & (images.detach() >= (threshold_hu + 1000)/1400)
    return torch.where(selected, torch.full_like(target, weight), torch.ones_like(target))


def refinement_loss(logits, target, images, weight=2., threshold_hu=-400.):
    torch = torch_module()
    if logits.shape != target.shape:
        raise ValueError('Logits/expert target shapes differ')
    weights = emphasis_weights(images, target, weight, threshold_hu)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    weighted = (bce * weights).sum()/weights.sum()
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    dice = (2*(probability*target).sum(dims)+1)/(probability.sum(dims)+target.sum(dims)+1)
    return weighted + 1 - dice.mean()


def local_brightness(images, target, generator, probability=.5, delta_hu=(150.,450.), radius=(12.,32.)):
    """Training-only synthetic bright patch; never change labels or background.

    A private CPU RNG avoids changing sample order or paired flip randomness.
    This is intensity robustness augmentation, NOT a simulated diagnosed lesion.
    """
    torch = torch_module()
    if (images.device.type != 'cpu' or images.shape != target.shape or images.ndim != 4 or
            images.shape[1] != 1 or not images.is_floating_point() or
            not np.isfinite([probability,*delta_hu,*radius]).all() or not 0 <= probability <= 1 or
            not 0 < delta_hu[0] <= delta_hu[1] <= 1400 or not 0 < radius[0] <= radius[1] or
            not torch.isfinite(images).all() or not ((images >= 0) & (images <= 1)).all() or
            not ((target == 0) | (target == 1)).all()):
        raise ValueError('Need CPU normalized single-channel inputs, binary labels and bounded augmentation')
    output = images.clone()
    mask = target.float()
    interior = torch.nn.functional.avg_pool2d(mask,9,stride=1,padding=4) >= 1-1e-6
    # Fade near the protected interior edge; original expert boundary stays untouched.
    taper = torch.nn.functional.avg_pool2d(interior.float(),9,stride=1,padding=4)*interior
    yy,xx = torch.meshgrid(torch.arange(images.shape[-2]),torch.arange(images.shape[-1]),indexing='ij')
    for index in range(len(images)):
        if torch.rand((),generator=generator).item() >= probability:
            continue
        locations = torch.nonzero(interior[index,0],as_tuple=False)
        if not len(locations):
            continue
        center = locations[torch.randint(len(locations),(),generator=generator)]
        patch_radius = radius[0]+torch.rand((),generator=generator).item()*(radius[1]-radius[0])
        delta = delta_hu[0]+torch.rand((),generator=generator).item()*(delta_hu[1]-delta_hu[0])
        squared = ((yy-center[0]).float()**2+(xx-center[1]).float()**2)/(patch_radius**2)
        edge = np.exp(-2.)
        profile = ((torch.exp(-2*squared)-edge)/(1-edge)).clamp(min=0)*taper[index,0]
        output[index,0] = (images[index,0]+profile*(delta/1400)).clamp(0,1)
    changed = output != images
    return output, {'augmented_slices':int(changed.flatten(1).any(1).sum()),
                    'augmented_pixels':int(changed.sum())}


def validation_summary(rows):
    if not rows or any(row['metrics']['dice'] is None for row in rows):
        raise ValueError('Need nonempty annotated validation cases')
    return {'mean_case_dice': float(np.mean([row['metrics']['dice'] for row in rows])),
        'false_positive': sum(row['metrics']['false_positive'] for row in rows),
        'false_negative': sum(row['metrics']['false_negative'] for row in rows),
        'bright_reference_pixels': sum(row['bright_reference_pixels'] for row in rows),
        'bright_false_negative': sum(row['bright_false_negative'] for row in rows),
        'slice_quality': aggregate_slice_quality([row['slice_quality'] for row in rows])}


def acceptance(candidate, baseline):
    """Fixed exploratory engineering gates, not clinical performance thresholds."""
    a = candidate['slice_quality']
    b = baseline['slice_quality']
    small_a, small_b = a['groups']['small']['mean_slice_dice'], b['groups']['small']['mean_slice_dice']
    checks = {
        'dice_drop_at_most_0.001': candidate['mean_case_dice'] >= baseline['mean_case_dice']-.001,
        'bright_fn_reduced_at_least_10_percent': (baseline['bright_false_negative'] > 0 and
            candidate['bright_false_negative'] <= baseline['bright_false_negative']*.9),
        'overall_fn_not_increased': candidate['false_negative'] <= baseline['false_negative'],
        'overall_fp_increase_at_most_5_percent': candidate['false_positive'] <= baseline['false_positive']*1.05,
        'empty_fp_increase_at_most_10_percent': a['empty_false_positive_pixels'] <= b['empty_false_positive_pixels']*1.1,
        'empty_positive_slice_count_not_increased': a['empty_slices_with_prediction'] <= b['empty_slices_with_prediction'],
        'small_slice_dice_drop_at_most_0.01': small_b is None or (small_a is not None and small_a >= small_b-.01),
    }
    return {'passed': all(checks.values()), 'checks': checks}


def validate(model, root, records, device, batch_size, threshold_hu, outdir=None):
    model.eval()
    rows = []
    for record in records:
        if record['split'] != 'validation':
            raise ValueError('Only development validation may select this experiment')
        prediction, elapsed = _case_prediction(model, root, record, device, batch_size, .5)
        folder = root/record['cache_folder']
        with mapped_npy(folder/'reference_native.npy') as cached, mapped_npy(folder/'images.npy') as images:
            reference = cached != 0
            # Same input-brightness condition as training, nearest mapped to the native grid.
            # This is NOT a new expert label or an infection mask, nor raw native HU.
            bright = resize_stack(images.astype(np.float32) >= (threshold_hu+1000)/1400, reference.shape[1], 0) >= .5
            bright &= reference
            rows.append(dict(case=record['case'], metrics=overlap_metrics(prediction, reference),
                bright_reference_pixels=int(bright.sum()), bright_false_negative=int((bright & ~prediction).sum()),
                slice_quality=slice_quality_metrics(prediction, reference), inference_ms=elapsed))
        if outdir is not None:
            target = Path(outdir)/record['case']
            target.mkdir(parents=True, exist_ok=True)
            np.save(target/'prediction_native.npy', prediction.astype(np.uint8)*255)
    return rows, validation_summary(rows)


def save_previews(root, records, output, threshold_hu):
    """Two fixed validation cases only; original CT pixels, not a test-set sweep."""
    import pydicom
    images = []
    for record in records:
        if record['case'] not in {'LCTSC-Train-S2-006','LCTSC-Train-S3-008'}:
            continue
        name, folder = record['case'],root/record['cache_folder']
        with mapped_npy(folder/'reference_native.npy') as cached, \
                mapped_npy(folder/'images.npy') as small, \
                mapped_npy(output/'baseline'/name/'prediction_native.npy') as old, \
                mapped_npy(output/'candidate'/name/'prediction_native.npy') as new:
            reference=cached!=0
            bright=resize_stack(small.astype(np.float32)>=(threshold_hu+1000)/1400,reference.shape[1],0)>=.5
            index=int(np.argmax((bright & reference & (old==0)).sum(axis=(1,2))))
            ds=pydicom.dcmread(record['source_ct_files_sorted'][index])
            if str(ds.FrameOfReferenceUID)!=record['frame_uid']:
                raise ValueError('Preview CT frame differs from cached expert target')
            hu=ds.pixel_array.astype(np.float32)*float(ds.RescaleSlope)+float(ds.RescaleIntercept)
            path=output/f'comparison_{name}_slice_{index:03d}.png'
            save_learned_preview(hu,reference[index],new[index]!=0,old[index]!=0,path,name,index,
                learned_title='Refined candidate (green), computer',baseline_title='Original model (yellow)')
            images.append(str(path))
    return images


def run(args):
    torch = torch_module()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < 1 or args.threads < 1:
        raise ValueError('Invalid bounded training settings')
    if not np.isfinite(args.weight) or args.weight < 1 or not -1000 <= args.threshold_hu <= 400:
        raise ValueError('Invalid train-only brightness emphasis')
    if (not 0 <= args.augmentation_probability <= 1 or
            not 0 < args.brightness_min_hu <= args.brightness_max_hu <= 1400 or
            not 0 < args.patch_radius_min <= args.patch_radius_max or
            not np.isfinite([args.augmentation_probability,args.brightness_min_hu,
                             args.brightness_max_hu,args.patch_radius_min,args.patch_radius_max]).all()):
        raise ValueError('Invalid local brightness augmentation settings')
    if args.augmentation != 'none' and args.weight != 1:
        raise ValueError('Brightness experiment must use original loss (weight 1) to isolate augmentation')
    root, manifest = read_manifest(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if (checkpoint['dataset_fingerprint'] != manifest['fingerprint'] or
            checkpoint['image_size'] != manifest['image_size'] or
            checkpoint.get('context_slices', 1) != 1 or checkpoint['hu_clip'] != [-1000,400] or
            checkpoint['threshold'] != .5):
        raise ValueError('Initialization model and unchanged 2D dataset/preprocessing differ')
    train = [r for r in manifest['cases'] if r['split'] == 'train']
    val = [r for r in manifest['cases'] if r['split'] == 'validation']
    if not train or not val or any(not r['case'].startswith('LCTSC-Train-') for r in train+val):
        raise ValueError('Only original official training development split is accepted')
    device = select_device(args.device)
    torch.set_num_threads(args.threads)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    output = fresh_output(args.outdir)
    tracked = {str(Path(p).resolve()): file_sha256(p) for p in [args.dataset, args.checkpoint, __file__]}
    for record in train+val:
        for name in (['images.npy','labels.npy'] if record['split']=='train' else ['images.npy','reference_native.npy']):
            path = root/record['cache_folder']/name
            tracked[str(path)] = file_sha256(path)
    model = make_model(checkpoint['base_channels'], 1)
    model.load_state_dict(checkpoint['model'])
    architecture = model_summary(model,manifest['image_size'])
    model.to(device)
    settings = {k:v for k,v in vars(args).items()}
    settings.update(source_epoch=checkpoint['epoch'], dataset_fingerprint=manifest['fingerprint'],
        base_channels=checkpoint['base_channels'], image_size=manifest['image_size'], context_slices=1,
        train_cases=[r['case'] for r in train], validation_cases=[r['case'] for r in val],
        threshold=.5, optimizer='New Adam; NOT a resume of old optimizer/early stopping',
        target='Unchanged expert lung union',
        augmentation_scope='TRAIN only; one smooth bright patch in protected expert-positive interior; '
            'no validation/inference augmentation, no label edit, no claim of realistic lesion synthesis',
        weighting='Only expert-positive input HU >= threshold_hu gets weight; all other pixels remain weight 1',
        brightness_metric='Nearest-mapped input brightness on native expert lung; NOT native-HU/infection labels',
        selection='Highest patient Dice among candidates passing all predeclared gates; otherwise keep original',
        architecture=architecture, input_sha256=tracked,
        limitation='Development fine-tune, not an independent test; a paired control requires '
            'a separate identical weight-1 run without augmentation; CUDA bitwise determinism not enforced',
        device_resolved=str(device), warning='Computer-only development; no PL or clinical inference')
    write_json(output/'protocol.json', settings)
    print(f'Fine-tune {args.epochs} epochs | {len(train)} TRAIN / {len(val)} VALIDATION | '
          f'{settings["architecture"]["parameters"]} parameters | device={device}', flush=True)
    history, accepted, best_exploratory = [], None, -1.
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        base_rows, baseline = validate(model, root, val, device, args.batch_size, args.threshold_hu, output/'baseline')
        write_json(output/'baseline.json', {'summary': baseline, 'cases': base_rows})
        print(f'Original Dice={baseline["mean_case_dice"]:.6f}, bright FN={baseline["bright_false_negative"]}, '
              f'FP={baseline["false_positive"]}', flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
        with SliceDataset(root, train, augment=True, context_slices=1) as dataset:
            for epoch in range(1,args.epochs+1):
                torch.manual_seed(args.seed+epoch)
                random.seed(args.seed+epoch)
                np.random.seed(args.seed+epoch)
                generator = torch.Generator().manual_seed(args.seed+epoch)
                augmentation_generator = torch.Generator().manual_seed(args.seed+epoch+100000)
                loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, pin_memory=device.type=='cuda', generator=generator)
                model.train()
                total, count, start = 0.,0,perf_counter()
                augmentation_counts = {'augmented_slices':0,'augmented_pixels':0}
                for step,(images,labels) in enumerate(loader,1):
                    if args.augmentation == 'local-brightness':
                        images, counts = local_brightness(images,labels,augmentation_generator,
                            args.augmentation_probability,(args.brightness_min_hu,args.brightness_max_hu),
                            (args.patch_radius_min,args.patch_radius_max))
                        for key,value in counts.items():
                            augmentation_counts[key] += value
                    images,labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                        loss = refinement_loss(model(images).float(),labels,images,args.weight,args.threshold_hu)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite refinement loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    scaler.step(optimizer)
                    scaler.update()
                    total += float(loss.detach())*len(images)
                    count += len(images)
                    if step%100==0:
                        print(f'epoch {epoch}/{args.epochs}, batch {step}/{len(loader)}, loss {total/count:.5f}',flush=True)
                rows, summary = validate(model,root,val,device,args.batch_size,args.threshold_hu)
                gate = acceptance(summary,baseline)
                row = dict(epoch=epoch,loss=total/count,summary=summary,gate=gate,
                           seconds=perf_counter()-start,cases=rows,augmentation_counts=augmentation_counts)
                history.append(row)
                payload = {key:copy.deepcopy(checkpoint[key]) for key in
                    ['base_channels','seed','image_size','hu_clip','dataset_fingerprint','threshold']}
                payload.update(model=model.state_dict(),epoch=epoch,context_slices=1,
                    best_dice=summary['mean_case_dice'],history=history,
                    initialization_checkpoint_sha256=tracked[str(Path(args.checkpoint).resolve())],
                    refinement=dict(weight=args.weight,threshold_hu=args.threshold_hu,lr=args.lr,
                                    augmentation=args.augmentation,
                                    augmentation_probability=args.augmentation_probability,
                                    brightness_delta_hu=[args.brightness_min_hu,args.brightness_max_hu],
                                    patch_radius=[args.patch_radius_min,args.patch_radius_max],
                                    source_epoch=checkpoint['epoch'],gate=gate),
                    warning='Exploratory fine-tune; epoch counts are local, not old training epochs')
                save_checkpoint(output/'last.pt',payload)
                if summary['mean_case_dice']>best_exploratory:
                    best_exploratory=summary['mean_case_dice']
                    save_checkpoint(output/'exploratory_best.pt',payload)
                if gate['passed'] and (accepted is None or summary['mean_case_dice']>accepted['summary']['mean_case_dice']):
                    accepted=row
                    save_checkpoint(output/'accepted_candidate.pt',payload)
                write_json(output/'history.json',history)
                print(f'epoch {epoch}: Dice {summary["mean_case_dice"]:.6f}; bright FN '
                      f'{summary["bright_false_negative"]}; FP {summary["false_positive"]}; '
                      f'accepted={gate["passed"]}; {row["seconds"]:.1f}s',flush=True)
        selected_path = output/('accepted_candidate.pt' if accepted else 'exploratory_best.pt')
        selected = torch.load(selected_path,map_location='cpu',weights_only=True)
        model.load_state_dict(selected['model'])
        final_rows, final = validate(model,root,val,device,args.batch_size,args.threshold_hu,output/'candidate')
        previews=save_previews(root,val,output,args.threshold_hu)
        for path,digest in tracked.items():
            if file_sha256(path)!=digest:
                raise ValueError(f'Original input/code changed during experiment: {path}')
        report=dict(status='complete_exploratory_validation',baseline=baseline,candidate=final,
                    selected_epoch=selected['epoch'],selected_file=selected_path.name,
                    gate=acceptance(final,baseline),original_model_retained=True,
                    decision='candidate_for_further_validation' if accepted else 'do_not_replace_original',
                    cases=final_rows,history=history,input_sha256_unchanged=tracked,previews=previews,
                    warning='No test data used; no INT8 reconversion or FPGA inference')
        write_json(output/'evaluation.json',report)
        print(json.dumps({k:report[k] for k in ['baseline','candidate','gate','decision']},indent=2),flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=previous_tf32


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset',default='data/lung_learning_20261009/dataset.json')
    result.add_argument('--checkpoint',default='data/lung_training_20261011/early_stop_run/best.pt')
    result.add_argument('--outdir',required=True)
    result.add_argument('--epochs',type=int,default=4)
    result.add_argument('--batch-size',type=int,default=8)
    result.add_argument('--weight',type=float,default=2.)
    result.add_argument('--augmentation',choices=['none','local-brightness'],default='none')
    result.add_argument('--augmentation-probability',type=float,default=.5)
    result.add_argument('--brightness-min-hu',type=float,default=150.)
    result.add_argument('--brightness-max-hu',type=float,default=450.)
    result.add_argument('--patch-radius-min',type=float,default=12.)
    result.add_argument('--patch-radius-max',type=float,default=32.)
    result.add_argument('--threshold-hu',type=float,default=-400.)
    result.add_argument('--lr',type=float,default=.0001)
    result.add_argument('--threads',type=int,default=4)
    result.add_argument('--seed',type=int,default=20261012)
    result.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    return result


if __name__=='__main__':
    run(parser().parse_args())
