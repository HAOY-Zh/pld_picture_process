"""CT-only body cropping and a paired small-U-Net fine-tune, computer only.

Never changes native expert masks, the original checkpoint, or FPGA files.
Crop geometry is independent of labels. Validation is development, not blind test.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import shutil
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

from lung_segmentation import fresh_output, write_json
from lung_unet import (SliceDataset, _case_prediction, aggregate_slice_quality,
    context_batch, file_sha256, load_cached_source, make_model, mapped_npy,
    model_summary, read_manifest, resize_stack, save_checkpoint,
    save_learned_preview, segmentation_loss, select_device, slice_quality_metrics,
    torch_module)
from lungseg.metrics import overlap_metrics

KIND = 'body_crop_v1'


def checked_plan(plan, shape):
    if list(shape) != plan['native_yx'] or len(shape) != 2:
        raise ValueError('Crop plan and native image grid differ')
    box = plan['box_y0_y1_x0_x1']
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError('Need integer crop bounds')
    y0,y1,x0,x1 = box
    if not (0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1]):
        raise ValueError('Crop box outside original grid')
    pad = plan['pad_top_bottom_left_right']
    if len(pad) != 4 or any(type(v) is not int or v < 0 for v in pad):
        raise ValueError('Invalid square padding')
    side = max(y1-y0,x1-x0)
    if (y1-y0+pad[0]+pad[1] != side or x1-x0+pad[2]+pad[3] != side):
        raise ValueError('Padding is inconsistent with square crop')
    return box,pad,side


def _plan(shape, box, fallback):
    y0,y1,x0,x1 = map(int,box)
    height,width = y1-y0,x1-x0
    side = max(height,width)
    dy,dx = side-height,side-width
    return dict(native_yx=list(map(int,shape)), box_y0_y1_x0_x1=[y0,y1,x0,x1],
        pad_top_bottom_left_right=[dy//2,dy-dy//2,dx//2,dx-dx//2],
        fallback_full_frame=bool(fallback))


def body_plan(hu, spacing_yx, margin_mm=20., closing_mm=3.):
    """Largest HU>-500 tissue component after closing; fill body, NOT lung labels.

    Expand to square where the native field allows; otherwise letterbox with air.
    Uniform/tiny/no-tissue slices conservatively use the full field of view.
    No expert masks, network output or learned case thresholds enter this function.
    """
    hu = np.asarray(hu)
    spacing = np.asarray(spacing_yx,dtype=float)
    if (hu.ndim != 2 or min(hu.shape) < 16 or not np.isfinite(hu).all() or
            spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any() or
            not np.isfinite([margin_mm,closing_mm]).all() or margin_mm < 0 or closing_mm <= 0):
        raise ValueError('Need finite 2D HU, physical spacing and valid margins')
    h,w = hu.shape
    full = _plan(hu.shape,[0,h,0,w],True)
    if hu.max() == hu.min():
        return full
    radius = np.ceil(closing_mm/spacing).astype(int)
    if (radius > max(h,w)).any():
        raise ValueError('Closing radius too large for this image grid')
    yy,xx = np.meshgrid(np.arange(-radius[0],radius[0]+1)*spacing[0],
        np.arange(-radius[1],radius[1]+1)*spacing[1],indexing='ij')
    kernel = yy**2+xx**2 <= closing_mm**2+1e-8
    tissue = hu > -500
    padded = np.pad(tissue,[(radius[0],radius[0]),(radius[1],radius[1])])
    closed = ndi.binary_closing(padded,structure=kernel)
    closed = closed[radius[0]:radius[0]+h,radius[1]:radius[1]+w] | tissue
    labels,count = ndi.label(closed)
    if not count:
        return full
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    largest = int(sizes.argmax())
    if sizes[largest] < .01*h*w:
        return full
    # Fill holes only in the temporary body detector, never in the expert target.
    body = ndi.binary_fill_holes(labels == largest)
    locations = np.argwhere(body)
    lo = locations.min(0)-np.ceil(margin_mm/spacing).astype(int)
    hi = locations.max(0)+1+np.ceil(margin_mm/spacing).astype(int)
    lo = np.maximum(lo,0); hi = np.minimum(hi,[h,w])
    side = int(max(hi-lo))
    for axis,limit in enumerate([h,w]):
        extent = min(side,limit)
        center = (int(lo[axis])+int(hi[axis]))/2
        begin = min(max(int(np.floor(center-extent/2)),0),limit-extent)
        lo[axis],hi[axis] = begin,begin+extent
    return _plan(hu.shape,[lo[0],hi[0],lo[1],hi[1]],False)


def crop_resize(image, plan, size, order, padding_value=0.):
    image = np.asarray(image)
    box,pad,_ = checked_plan(plan,image.shape)
    y0,y1,x0,x1 = box
    crop = np.pad(image[y0:y1,x0:x1],[(pad[0],pad[1]),(pad[2],pad[3])],
        constant_values=padding_value)
    return resize_stack(crop[None],size,order)[0]


def restore_probability(probability, plan):
    """Pixel-center bilinear resize, remove letterbox, paste; zero outside crop."""
    probability = np.asarray(probability,dtype=np.float32)
    if (probability.ndim != 2 or probability.shape[0] != probability.shape[1] or
            probability.shape[0] < 16 or not np.isfinite(probability).all() or
            ((probability < 0) | (probability > 1)).any()):
        raise ValueError('Need square finite probabilities in [0,1]')
    box,pad,side = checked_plan(plan,plan['native_yx'])
    resized = resize_stack(probability[None],side,1)[0]
    y0,y1,x0,x1 = box
    result = np.zeros(plan['native_yx'],dtype=np.float32)
    result[y0:y1,x0:x1] = resized[pad[0]:pad[0]+y1-y0,pad[2]:pad[2]+x1-x0]
    return result


def check_hashes(tracked):
    for path,digest in tracked.items():
        if file_sha256(path) != digest:
            raise ValueError(f'Protected input changed: {path}')


def prepare(args):
    root,original = read_manifest(args.dataset)
    if original['hu_clip'] != [-1000,400]:
        raise ValueError('Keep the original HU normalization unchanged')
    size = original['image_size']
    if size < 32 or size%8:
        raise ValueError('Input size must be >=32 and divisible by 8')
    tracked = {str(Path(args.dataset).resolve()):file_sha256(args.dataset)}
    for row in original['cases']:
        for p in [root/row['cache_folder']/'reference_native.npy',*map(Path,row['source_ct_files_sorted'])]:
            tracked[str(p.resolve())] = file_sha256(p)
    output = fresh_output(args.outdir)
    result = copy.deepcopy(original)
    result.update(status='preparing',cases=[],failures=[],preprocessing=dict(kind=KIND,
        body_threshold_hu=-500,margin_mm=args.margin_mm,closing_mm=args.closing_mm,
        detection='CT-only largest closed tissue component; full-frame fallback for uniform/tiny/no tissue',
        resize='pixel-center linear HU, nearest TRAIN labels; square pixel-aspect crop/air letterbox',
        restoration='bilinear probability resize, unpad, native paste, threshold 0.5; background outside crop',
        label_policy='Native expert labels byte-identical, holes retained; labels audit coverage but never choose boxes'),
        source_dataset=str(Path(args.dataset).resolve()),source_fingerprint=original['fingerprint'],
        input_sha256=tracked)
    write_json(output/'dataset.json',result)
    try:
        for k,row in enumerate(original['cases'],1):
            print(f'[{k}/{len(original["cases"])}] CT-only body crop {row["case"]} ({row["split"]})',flush=True)
            hu = load_cached_source(row)
            folder = output/'cases'/row['case']; folder.mkdir(parents=True)
            images = np.empty((len(hu),size,size),np.float16)
            labels = np.empty((len(hu),size,size),np.uint8)
            plans,missed,scale,fallback = [],0,[],0
            source_reference = root/row['cache_folder']/'reference_native.npy'
            with mapped_npy(source_reference) as reference:
                if reference.shape != hu.shape or not np.isin(reference,[0,1]).all():
                    raise ValueError('Native reference grid/values differ')
                for i,image in enumerate(hu):
                    plan = body_plan(image,row['spacing_zyx_mm'][1:],args.margin_mm,args.closing_mm)
                    plans.append(plan)
                    y0,y1,x0,x1 = plan['box_y0_y1_x0_x1']
                    missed += int(reference[i].sum()-reference[i,y0:y1,x0:x1].sum())
                    images[i] = (crop_resize(np.clip(image,-1000,400),plan,size,1,-1000)+1000)/1400
                    labels[i] = crop_resize(reference[i],plan,size,0,0) >= .5
                    scale.append(max(image.shape)/max(y1-y0,x1-x0))
                    fallback += int(plan['fallback_full_frame'])
                index = int(np.argmax(reference.sum((1,2))))
                save_crop_preview(hu[index],reference[index],images[index],plans[index],folder/'crop_preview.png',row['case'],index)
            # Do not fix failing crop boxes using ground truth, silently drop slices or edit expert labels.
            if missed:
                raise ValueError(f'CT-only crop excluded {missed} expert-positive pixels; experiment is unsafe')
            np.save(folder/'images.npy',images); np.save(folder/'labels.npy',labels)
            shutil.copyfile(source_reference,folder/'reference_native.npy')
            if file_sha256(source_reference) != file_sha256(folder/'reference_native.npy'):
                raise ValueError('Native expert reference copy changed')
            write_json(folder/'crop_geometry.json',plans)
            new = dict(row,cache_folder=str(folder.relative_to(output)),crop_geometry_file='crop_geometry.json',
                native_reference_sha256=file_sha256(source_reference),crop_audit=dict(
                    excluded_expert_pixels=missed,fallback_slices=fallback,
                    linear_scale_median=float(np.median(scale)),linear_scale_min=float(min(scale)),
                    linear_scale_max=float(max(scale))))
            result['cases'].append(new)
            write_json(folder/'case.json',new)
            write_json(output/'dataset.json',result)
            print(f'  Lung pixels outside crop: {missed}; median enlargement {np.median(scale):.3f}x',flush=True)
        check_hashes(tracked)
        import hashlib
        fingerprint = dict(source=original['fingerprint'],preprocessing=result['preprocessing'],
            geometries=[file_sha256(output/r['cache_folder']/r['crop_geometry_file']) for r in result['cases']])
        result.update(status='complete',protected_inputs_unchanged=True,
            fingerprint=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest())
        write_json(output/'dataset.json',result)
    except BaseException as error:
        result.update(status='incomplete',failures=[dict(error=str(error))])
        write_json(output/'dataset.json',result)
        raise


def save_crop_preview(hu,reference,small,plan,path,case,index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figure,axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(hu,cmap='gray',vmin=-1000,vmax=400)
    y0,y1,x0,x1 = plan['box_y0_y1_x0_x1']
    axes[0].add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='orange'))
    if reference.any() and not reference.all():
        axes[0].contour(reference,levels=[.5],colors=['cyan'],linewidths=.6)
    axes[0].set_title('Native CT; orange=CT-only crop\ncyan=unchanged expert (audit only)')
    full = (resize_stack(np.clip(hu[None],-1000,400),small.shape[0],1)[0]+1000)/1400
    axes[1].imshow(full,cmap='gray',vmin=0,vmax=1); axes[1].set_title('Original full-field network input')
    axes[2].imshow(small,cmap='gray',vmin=0,vmax=1); axes[2].set_title('Body-cropped network input')
    for axis in axes:axis.axis('off')
    figure.suptitle(f'{case} | slice {index} | educational development only')
    figure.tight_layout();figure.savefig(path,dpi=140);plt.close(figure)


def crop_prediction(model,root,record,device,batch_size):
    torch = torch_module()
    folder = Path(root)/record['cache_folder']
    plans = json.loads((folder/record['crop_geometry_file']).read_text(encoding='utf-8'))
    native = tuple(record['native_shape'])
    if len(plans) != native[0]:raise ValueError('Crop geometry slice count differs')
    prediction = np.empty(native,dtype=bool)
    start = perf_counter()
    with mapped_npy(folder/'images.npy') as images,torch.inference_mode():
        if images.ndim != 3 or len(images) != native[0]:raise ValueError('Input slice count differs')
        for begin in range(0,len(images),batch_size):
            centers = np.arange(begin,min(begin+batch_size,len(images)))
            image = torch.from_numpy(context_batch(images,centers,1)).to(device)
            probabilities = torch.sigmoid(model(image))[:,0].cpu().numpy()
            for i,probability in zip(centers,probabilities):
                checked_plan(plans[int(i)],native[1:])
                prediction[i] = restore_probability(probability,plans[int(i)]) >= .5
    return prediction,(perf_counter()-start)*1000


def validate(model,root,records,device,batch_size,cropped=False,outdir=None):
    model.eval();rows=[]
    for record in records:
        if record['split'] != 'validation':raise ValueError('Only original validation may select this experiment')
        prediction,elapsed = (crop_prediction(model,root,record,device,batch_size) if cropped else
            _case_prediction(model,root,record,device,batch_size,.5))
        with mapped_npy(root/record['cache_folder']/'reference_native.npy') as ref:
            rows.append(dict(case=record['case'],metrics=overlap_metrics(prediction,ref),
                slice_quality=slice_quality_metrics(prediction,ref),inference_ms=elapsed))
        if outdir:
            folder=Path(outdir)/record['case'];folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/'prediction_native.npy',prediction.astype(np.uint8)*255)
    summary=dict(mean_case_dice=float(np.mean([r['metrics']['dice'] for r in rows])),
        false_negative=sum(r['metrics']['false_negative'] for r in rows),
        false_positive=sum(r['metrics']['false_positive'] for r in rows),
        slice_quality=aggregate_slice_quality([r['slice_quality'] for r in rows]))
    return rows,summary


def acceptance(candidate,baseline):
    a,b = candidate['slice_quality'],baseline['slice_quality']
    sa,sb = a['groups']['small']['mean_slice_dice'],b['groups']['small']['mean_slice_dice']
    checks=dict(dice_drop_at_most_0_001=candidate['mean_case_dice']>=baseline['mean_case_dice']-.001,
        overall_fn_reduced_at_least_5_percent=candidate['false_negative']<=baseline['false_negative']*.95,
        fp_increase_at_most_5_percent=candidate['false_positive']<=baseline['false_positive']*1.05,
        empty_fp_increase_at_most_10_percent=a['empty_false_positive_pixels']<=b['empty_false_positive_pixels']*1.1,
        empty_positive_slice_count_not_increased=a['empty_slices_with_prediction']<=b['empty_slices_with_prediction'],
        small_slice_dice_drop_at_most_0_01=sb is None or (sa is not None and sa>=sb-.01))
    return dict(passed=all(checks.values()),checks=checks)


def paired(args):
    torch = torch_module()
    if (args.epochs<1 or args.batch_size<1 or args.threads<1 or not np.isfinite(args.lr) or not 0<args.lr<1):
        raise ValueError('Invalid bounded training settings')
    full_root,full = read_manifest(args.dataset)
    crop_root,crop = read_manifest(args.cropped_dataset,allow_body_crop=True)
    if (crop.get('preprocessing',{}).get('kind')!=KIND or crop.get('source_fingerprint')!=full['fingerprint'] or
            crop['image_size']!=full['image_size'] or crop['hu_clip']!=full['hu_clip']):
        raise ValueError('Cropped and full-field provenance/preprocessing differ')
    original = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if (original['dataset_fingerprint']!=full['fingerprint'] or original.get('context_slices',1)!=1 or
            original['image_size']!=full['image_size'] or original['hu_clip']!=[-1000,400] or original['threshold']!=.5):
        raise ValueError('Use retained original 2D initialization and unchanged normalization')
    if full.get('case_split')!=crop.get('case_split'):
        raise ValueError('Case assignment differs')
    identity=lambda r:(r['case'],r['split'],r['patient_key'],r['frame_uid'],r['native_shape'],r['slices'])
    if list(map(identity,full['cases']))!=list(map(identity,crop['cases'])):
        raise ValueError('Case identity/grid/order differs')
    tracked={str(Path(p).resolve()):file_sha256(p) for p in [args.dataset,args.cropped_dataset,args.checkpoint,__file__]}
    # Cache construction protected CT bytes too; keep that provenance throughout training/previews.
    check_hashes(crop.get('input_sha256',{}))
    tracked.update(crop.get('input_sha256',{}))
    shared_code=Path(__file__).with_name('lung_unet.py')
    tracked[str(shared_code.resolve())]=file_sha256(shared_code)
    for r,s in zip(full['cases'],crop['cases']):
        if not r['case'].startswith('LCTSC-Train-'):raise ValueError('No official test or external data allowed')
        for root,row in [(full_root,r),(crop_root,s)]:
            for name in ['images.npy','labels.npy','reference_native.npy']:
                path=root/row['cache_folder']/name;tracked[str(path.resolve())]=file_sha256(path)
        gp=crop_root/s['cache_folder']/s['crop_geometry_file'];tracked[str(gp.resolve())]=file_sha256(gp)
        if file_sha256(full_root/r['cache_folder']/'reference_native.npy')!=s['native_reference_sha256']:
            raise ValueError('Expert label source differs')
        if tracked[str((crop_root/s['cache_folder']/'reference_native.npy').resolve())]!=s['native_reference_sha256']:
            raise ValueError('Native expert target changed')
        if s['crop_audit']['excluded_expert_pixels']!=0:raise ValueError('Unsafe crop coverage')
    device=select_device(args.device);torch.set_num_threads(args.threads)
    previous_tf32=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    output=fresh_output(args.outdir)
    model=make_model(original['base_channels'],1);model.load_state_dict(original['model'])
    architecture=model_summary(model,full['image_size'])
    protocol=dict({k:v for k,v in vars(args).items() if k!='func'},architecture=architecture,input_sha256=tracked,initialization_epoch=original['epoch'],
        input_change_only=True,threshold=.5,loss='Original BCE + soft Dice',
        optimization='New Adam; paired epochs reset same RNG/order/horizontal flips; no synthetic augmentation',
        preprocessing=crop['preprocessing'],train_cases=[r['case'] for r in full['cases'] if r['split']=='train'],
        validation_cases=[r['case'] for r in full['cases'] if r['split']=='validation'],
        selection='Each arm highest Dice among epochs passing fixed FN/FP/empty/small gates; '
            'otherwise highest-Dice exploratory epoch. Original is never automatically replaced.',
        target_check='After selection, require >=5% reduction of native false negatives >=2mm inside expert '
            'on ALL validation slices; enclosed-FN counts are descriptive, never anatomical diagnoses',
        warning='4-epoch screen by default, not exhaustive optimization or independent test; no teacher/INT8/PL changes',
        device_resolved=str(device))
    write_json(output/'protocol.json',protocol)
    report=dict(status='running',original_model_retained=True,arms={},protocol=protocol)
    write_json(output/'evaluation.json',report)
    try:
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        val=[r for r in full['cases'] if r['split']=='validation']
        model.to(device)
        base_rows,baseline=validate(model,full_root,val,device,args.batch_size,outdir=output/'baseline')
        write_json(output/'baseline.json',dict(cases=base_rows,summary=baseline))
        report['baseline']=baseline
        for arm,root,manifest,is_crop in [('full_control',full_root,full,False),('body_crop',crop_root,crop,True)]:
            print(f'Paired arm {arm}: {args.epochs} epochs; same original initialization',flush=True)
            model.load_state_dict(original['model'])
            optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
            scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
            folder=output/arm;folder.mkdir()
            rows,initial=validate(model,root,[r for r in manifest['cases'] if r['split']=='validation'],
                device,args.batch_size,is_crop)
            write_json(folder/'initial.json',dict(summary=initial,cases=rows))
            history=[];best=-1.;accepted=-1.
            with SliceDataset(root,[r for r in manifest['cases'] if r['split']=='train'],augment=True) as dataset:
                for epoch in range(1,args.epochs+1):
                    torch.manual_seed(args.seed+epoch);random.seed(args.seed+epoch);np.random.seed(args.seed+epoch)
                    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,
                        pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(args.seed+epoch))
                    model.train();total,count,start=0.,0,perf_counter()
                    for step,(images,labels) in enumerate(loader,1):
                        images,labels=images.to(device),labels.to(device);optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                            loss=segmentation_loss(model(images).float(),labels)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                        scaler.scale(loss).backward();scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                        scaler.step(optimizer);scaler.update()
                        total+=float(loss.detach())*len(images);count+=len(images)
                        if step%100==0:print(f'  {arm} epoch {epoch}: batch {step}/{len(loader)}, loss={total/count:.5f}',flush=True)
                    rows,summary=validate(model,root,[r for r in manifest['cases'] if r['split']=='validation'],
                        device,args.batch_size,is_crop)
                    gate=acceptance(summary,baseline)
                    history.append(dict(epoch=epoch,loss=total/count,summary=summary,gate=gate,cases=rows,
                        seconds_including_validation=perf_counter()-start))
                    payload={key:copy.deepcopy(original[key]) for key in ['base_channels','image_size','hu_clip','threshold']}
                    payload.update(model=model.state_dict(),epoch=epoch,seed=args.seed,context_slices=1,history=history,
                        dataset_fingerprint=manifest['fingerprint'],body_crop_experiment=dict(arm=arm,
                        initialization_sha256=tracked[str(Path(args.checkpoint).resolve())],original_epoch=original['epoch']),
                        preprocessing=crop['preprocessing'] if is_crop else {'kind':'full_frame'},
                        warning='Bounded exploratory continuation, not original training resume or hardware model')
                    save_checkpoint(folder/'last.pt',payload)
                    if summary['mean_case_dice']>best:
                        best=summary['mean_case_dice'];save_checkpoint(folder/'exploratory_best.pt',payload)
                    if gate['passed'] and summary['mean_case_dice']>accepted:
                        accepted=summary['mean_case_dice'];save_checkpoint(folder/'accepted_candidate.pt',payload)
                    write_json(folder/'history.json',history)
                    print(f'{arm} epoch {epoch}: Dice={summary["mean_case_dice"]:.6f}, '
                        f'FN={summary["false_negative"]}, FP={summary["false_positive"]}, gate={gate["passed"]}',flush=True)
            chosen=folder/('accepted_candidate.pt' if accepted>=0 else 'exploratory_best.pt')
            selected=torch.load(chosen,map_location='cpu',weights_only=True);model.load_state_dict(selected['model'])
            rows,summary=validate(model,root,[r for r in manifest['cases'] if r['split']=='validation'],
                device,args.batch_size,is_crop,folder/'selected')
            report['arms'][arm]=dict(initial=initial,selected_epoch=selected['epoch'],selected_file=chosen.name,
                summary=summary,cases=rows,gate=acceptance(summary,baseline),history=history)
            write_json(output/'evaluation.json',report)
        report['selected_geometry']=selected_geometry(full_root,full,output)
        original_deep=report['selected_geometry']['totals']['baseline']['deep_fn_pixels']
        crop_deep=report['selected_geometry']['totals']['body_crop']['deep_fn_pixels']
        report['internal_target_passed']=original_deep>0 and crop_deep<=original_deep*.95
        check_hashes(tracked)
        report.update(status='complete_paired_development_screen',protected_inputs_unchanged=True,
            decision='candidate_needs_further_validation' if (report['arms']['body_crop']['gate']['passed']
                and report['internal_target_passed']) else 'do_not_replace_original')
        report['previews']=paired_previews(full_root,full,output)
        write_json(output/'evaluation.json',report)
        print(json.dumps({k:report[k] for k in ['status','baseline','decision']},indent=2),flush=True)
    except BaseException as error:
        report.update(status='aborted_incomplete',error=str(error));write_json(output/'evaluation.json',report);raise
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=previous_tf32


def selected_geometry(root,manifest,output,arms=None):
    """Native internal-FN audit of all selected validation outputs; no inference/repair."""
    if arms is None:
        arms={'baseline':'baseline','full_control':'full_control/selected','body_crop':'body_crop/selected'}
    totals={arm:dict(fn_pixels=0,deep_fn_pixels=0,closed_fn_pixels=0) for arm in arms}
    rows=[]
    for record in manifest['cases']:
        if record['split']!='validation':continue
        case=record['case']
        print(f'Native internal-FN audit: {case}, all {record["slices"]} slices',flush=True)
        with mapped_npy(root/record['cache_folder']/'reference_native.npy') as source:
            reference=source!=0
        predictions={}
        for arm,folder in arms.items():
            with mapped_npy(output/folder/case/'prediction_native.npy') as a:
                if a.shape!=reference.shape or not np.isin(a,[0,1,255]).all():
                    raise ValueError('Selected prediction grid/values differ')
                predictions[arm]=a!=0
        counts={arm:dict(fn_pixels=0,deep_fn_pixels=0,closed_fn_pixels=0) for arm in arms}
        for i,expert in enumerate(reference):
            if not expert.any():continue
            depth=ndi.distance_transform_edt(np.pad(expert,1),sampling=record['spacing_zyx_mm'][1:])[1:-1,1:-1]
            for arm,pred in predictions.items():
                fn=expert & ~pred[i]
                if not fn.any():continue
                enclosed=ndi.binary_fill_holes(pred[i]) & ~pred[i]
                counts[arm]['fn_pixels']+=int(fn.sum())
                counts[arm]['deep_fn_pixels']+=int((fn & (depth>=2.)).sum())
                counts[arm]['closed_fn_pixels']+=int((fn & enclosed).sum())
        for arm in arms:
            for key in totals[arm]:totals[arm][key]+=counts[arm][key]
        rows.append(dict(case=case,counts=counts))
    return dict(totals=totals,cases=rows,
        definition='deep FN: >=2mm from unchanged native expert background; closed FN: enclosed 2D '
            'prediction background intersected with expert lung; not pathology/clinical labels')


def paired_previews(root,manifest,output):
    from lung_error_audit import error_maps
    paths=[]
    for case,index in [('LCTSC-Train-S2-006',54),('LCTSC-Train-S3-008',79)]:
        records=[r for r in manifest['cases'] if r['case']==case]
        if not records:continue  # Small synthetic tests need not contain these real fixed targets.
        record=records[0]
        hu=load_cached_source(record)[index]
        with mapped_npy(root/record['cache_folder']/'reference_native.npy') as ref:
            expert=np.array(ref[index],dtype=bool)
        def mask(arm):
            with mapped_npy(output/arm/case/'prediction_native.npy') as a:return a[index]!=0
        baseline=mask('baseline');control=mask('full_control/selected');candidate=mask('body_crop/selected')
        for label,previous in [('original',baseline),('paired_control',control)]:
            path=output/f'comparison_{case}_slice_{index:03d}_vs_{label}.png'
            save_learned_preview(hu,expert,candidate,previous,path,case,index,
                baseline_title=f'{label} (yellow)',learned_title='Body-cropped small U-Net (green)')
            paths.append(str(path))
        stats={label:error_maps(prediction,expert,record['spacing_zyx_mm'][1:])[0]
            for label,prediction in [('original',baseline),('full_control',control),('body_crop',candidate)]}
        write_json(output/f'geometry_{case}_slice_{index:03d}.json',stats)
    return paths


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    sub=result.add_subparsers(dest='command',required=True)
    common='data/lung_learning_20261009/dataset.json'
    prep=sub.add_parser('prepare');prep.add_argument('--dataset',default=common)
    prep.add_argument('--outdir',required=True);prep.add_argument('--margin-mm',type=float,default=20.)
    prep.add_argument('--closing-mm',type=float,default=3.);prep.set_defaults(func=prepare)
    run=sub.add_parser('paired');run.add_argument('--dataset',default=common)
    run.add_argument('--cropped-dataset',required=True);run.add_argument('--outdir',required=True)
    run.add_argument('--checkpoint',default='data/lung_training_20261011/early_stop_run/best.pt')
    run.add_argument('--epochs',type=int,default=4);run.add_argument('--lr',type=float,default=.0001)
    run.add_argument('--seed',type=int,default=20261013);run.add_argument('--batch-size',type=int,default=8)
    run.add_argument('--threads',type=int,default=4);run.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    run.set_defaults(func=paired)
    return result


if __name__=='__main__':
    args=parser().parse_args();args.func(args)
