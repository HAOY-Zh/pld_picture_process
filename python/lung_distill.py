"""TRAIN-only R231 guidance for the unchanged full-field small U-Net.

Human targets remain authoritative. No target repair, teacher deployment,
external/test selection, or automatic original-model replacement.
"""
import argparse
from bisect import bisect_right
from contextlib import ExitStack
import copy
import importlib.metadata
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

from lung_body_crop import acceptance, check_hashes, selected_geometry, validate
from lung_segmentation import fresh_output, write_json
from lung_unet import (SliceDataset, file_sha256, load_cached_source, make_model,
    mapped_npy, model_summary, read_manifest, resize_stack, save_checkpoint,
    save_learned_preview, segmentation_loss, select_device, torch_module)

TEACHER_SHA256 = 'd5d2fc3df432115933491f115b283e94d1f5c841c0e82f6e7bd2d65d3ccde69f'
KIND = 'r231_train_soft_guidance_v1'


def train_records(manifest):
    if manifest['hu_clip'] != [-1000,400] or manifest.get('preprocessing',{}).get('kind','full_frame') != 'full_frame':
        raise ValueError('Only original full-field HU inputs are supported')
    rows = [r for r in manifest['cases'] if r['split']=='train']
    if not rows or any(not r['case'].startswith('LCTSC-Train-') for r in rows):
        raise ValueError('Teacher generation is restricted to original TRAIN cases')
    return rows


def restore_teacher(probability, box, native_yx):
    """Vendor endpoint-aligned zoom, but bilinear for SOFT probabilities.

    R231 box order is y0,x0,y1,x1, unlike the body-crop experiment.
    Teacher preprocessing uses scipy.zoom(grid_mode=False); preserve that
    convention before the student's separate pixel-center full-frame resize.
    """
    p = np.asarray(probability,dtype=np.float32)
    b = np.asarray(box)
    if (p.ndim!=2 or min(p.shape)<2 or not np.isfinite(p).all() or
            (p<0).any() or (p>1).any() or b.shape!=(4,) or
            not np.isfinite(b).all() or not np.array_equal(b,b.astype(int))):
        raise ValueError('Invalid teacher probability or crop box')
    y0,x0,y1,x1 = b.astype(int)
    h,w = native_yx
    if not (0<=y0<y1<=h and 0<=x0<x1<=w):raise ValueError('Teacher box outside native grid')
    restored = np.zeros((h,w),np.float32)
    restored[y0:y1,x0:x1] = ndi.zoom(p,np.array([y1-y0,x1-x0])/p.shape,order=1)
    return restored


def teacher_probabilities(inferer,hu,size):
    from lungmask import utils
    torch = torch_module()
    prepared,boxes = utils.preprocess(hu,resolution=[256,256])
    prepared = (np.clip(prepared,-1024,600)+1024)/1624
    result = np.empty((len(hu),size,size),np.float16)
    inferer.model.eval()
    with torch.inference_mode():
        for begin in range(0,len(hu),inferer.batch_size):
            tensor = torch.as_tensor(prepared[begin:begin+inferer.batch_size,None],device=inferer.device,dtype=torch.float32)
            logits = inferer.model(tensor)
            if logits.shape[1]!=3:raise ValueError('Teacher must be R231 background/right/left')
            # T=1: union soft probability, no argmax or volume postprocessing.
            probabilities = logits.softmax(1)[:,1:].sum(1).cpu().numpy()
            for i,p in enumerate(probabilities,begin):
                native = restore_teacher(np.clip(p,0,1),boxes[i],hu.shape[1:])
                result[i] = resize_stack(native[None],size,1)[0]
    return result,[np.asarray(b,dtype=int).tolist() for b in boxes]


def source_hashes(paths):
    return {str(Path(p).resolve()):file_sha256(p) for p in paths}


def prepare(args):
    import pydicom
    from lungmask.mask import LMInferer
    if args.batch_size<1 or args.threads<1:raise ValueError('Positive batch/threads required')
    if importlib.metadata.version('lungmask')!='0.2.20':raise ValueError('Pinned lungmask 0.2.20 required')
    root,manifest = read_manifest(args.dataset)
    records = train_records(manifest)
    if file_sha256(args.weights)!=TEACHER_SHA256:raise ValueError('Official R231 weight hash differs')
    tracked = source_hashes([args.dataset,args.weights,__file__])
    for r in records:
        tracked.update(source_hashes(r['source_ct_files_sorted']))
        tracked.update(source_hashes([root/r['cache_folder']/n for n in ['images.npy','labels.npy','reference_native.npy']]))
    device = select_device(args.device);torch = torch_module();torch.set_num_threads(args.threads)
    old = torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    output = fresh_output(args.outdir)
    report = dict(status='running',kind=KIND,dataset_fingerprint=manifest['fingerprint'],
        dataset_sha256=file_sha256(args.dataset),image_size=manifest['image_size'],temperature=1,
        teacher='R231',teacher_sha256=TEACHER_SHA256,package='lungmask==0.2.20',cases=[],input_sha256=tracked,
        preprocessing='Official HU [-1024,600], native crop 256, endpoint-aligned bilinear soft restore, '
            'then original full-field student pixel-center resize; no teacher mask postprocessing',
        warning='Only TRAIN CT evaluated by teacher; original human targets unchanged. Teacher may have '
            'different annotation semantics or unknown training overlap; this is not a blind accuracy test.')
    write_json(output/'teacher_dataset.json',report)
    try:
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        inferer = LMInferer(modelname='R231',modelpath=str(args.weights),force_cpu=device.type=='cpu',
            batch_size=args.batch_size,volume_postprocessing=False,tqdm_disable=True)
        for r in records:
            print(f'TRAIN teacher: {r["case"]}, {r["slices"]} slices',flush=True)
            hu = load_cached_source(r)
            for path in r['source_ct_files_sorted']:
                ds = pydicom.dcmread(path,stop_before_pixels=True,specific_tags=['ImageOrientationPatient'])
                if not np.allclose(ds.ImageOrientationPatient,[1,0,0,0,1,0],atol=1e-5,rtol=0):
                    raise ValueError('R231 numpy input requires canonical LPS; unsupported CT orientation')
            probabilities,boxes = teacher_probabilities(inferer,hu,manifest['image_size'])
            folder = output/r['case'];folder.mkdir()
            path = folder/'probability.npy';np.save(path,probabilities)
            with mapped_npy(root/r['cache_folder']/'labels.npy') as labels:
                if labels.shape!=probabilities.shape:raise ValueError('Teacher/student grid differs')
                positive = labels!=0
                stats = dict(teacher_human_disagreement_pixels=int(((probabilities>=.5)!=positive).sum()),
                    confident_agree_positive=int((positive & (probabilities>=.8)).sum()),
                    confident_agree_negative=int((~positive & (probabilities<=.2)).sum()))
            report['cases'].append(dict(case=r['case'],split='train',patient_key=r['patient_key'],
                frame_uid=r['frame_uid'],slices=r['slices'],native_shape=r['native_shape'],
                probability_file=str(path.relative_to(output)),probability_sha256=file_sha256(path),
                teacher_boxes_y0_x0_y1_x1=boxes,statistics=stats))
            write_json(output/'teacher_dataset.json',report)
        check_hashes(tracked)
        report.update(status='complete',protected_inputs_unchanged=True)
        write_json(output/'teacher_dataset.json',report)
        print(f'Teacher cache complete: {len(records)} TRAIN cases only',flush=True)
    except BaseException as error:
        report.update(status='aborted_incomplete',error=str(error));write_json(output/'teacher_dataset.json',report);raise
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old


def guidance_loss(logits,labels,teacher,confidence=.8):
    """Class-balanced Bernoulli KL(T=1), gated by human agreement/confidence.

    Wrong teacher regions NEVER guide the student. Human BCE+Dice remains
    active everywhere, including disagreement and legitimate expert holes.
    No teacher gradient, no replacement of human labels, no mask filling.
    """
    torch = torch_module()
    if not .5<confidence<1:raise ValueError('Confidence must be strictly between .5 and 1')
    if logits.shape!=labels.shape or logits.shape!=teacher.shape:raise ValueError('Guidance grids differ')
    q = teacher.detach().float().clamp(1e-6,1-1e-6)
    if not torch.isfinite(teacher).all() or (teacher<0).any() or (teacher>1).any():
        raise ValueError('Invalid teacher probability')
    pointwise = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),q,reduction='none')
    pointwise += q*q.log()+(1-q)*(1-q).log()
    groups = [(labels>=.5)&(teacher>=confidence),(labels<.5)&(teacher<=1-confidence)]
    losses = [pointwise[g].mean() for g in groups if g.any()]
    return torch.stack(losses).mean() if losses else logits.sum()*0


def checked_teacher(path,manifest):
    path = Path(path);root = path.parent
    cache = json.loads(path.read_text(encoding='utf-8'))
    records = train_records(manifest)
    identity = lambda r:(r['case'],r['split'],r['patient_key'],r['frame_uid'],r['slices'],r['native_shape'])
    if (cache['status']!='complete' or cache['kind']!=KIND or cache['teacher_sha256']!=TEACHER_SHA256 or
            cache['dataset_fingerprint']!=manifest['fingerprint'] or cache['image_size']!=manifest['image_size'] or
            cache['temperature']!=1 or list(map(identity,records))!=list(map(identity,cache['cases']))):
        raise ValueError('Teacher provenance/TRAIN-only split/grid differs or cache incomplete')
    check_hashes(cache['input_sha256'])
    for r in cache['cases']:
        probability_path = (root/r['probability_file']).resolve()
        if not probability_path.is_relative_to(root.resolve()):raise ValueError('Teacher path escapes cache')
        if file_sha256(probability_path)!=r['probability_sha256']:raise ValueError('Teacher cache hash differs')
        with mapped_npy(probability_path) as a:
            if (a.shape!=(r['slices'],manifest['image_size'],manifest['image_size']) or
                    not np.isfinite(a).all() or (a<0).any() or (a>1).any()):
                raise ValueError('Invalid teacher cache grid/values')
    return root,cache


class GuidedDataset(SliceDataset):
    def __init__(self,root,records,teacher_root,teacher_records,augment=True):
        super().__init__(root,records,augment=False,context_slices=1)
        if any(r['split']!='train' for r in records+teacher_records):raise ValueError('TRAIN only')
        if [r['case'] for r in records]!=[r['case'] for r in teacher_records]:raise ValueError('Case order differs')
        self.teacher_root,self.teacher_records = Path(teacher_root),teacher_records
        self.guide_arrays={};self.guided_augment=augment

    def __getitem__(self,index):
        image,label = super().__getitem__(index)
        k = bisect_right(self.ends,index);j = index-(self.ends[k-1] if k else 0)
        if k not in self.guide_arrays:
            a = self._mappings.enter_context(mapped_npy(self.teacher_root/self.teacher_records[k]['probability_file']))
            if a.shape!=self.arrays[k][1].shape:raise ValueError('Teacher grid differs')
            self.guide_arrays[k]=a
        teacher = torch_module().from_numpy(np.array(self.guide_arrays[k][j,None],dtype=np.float32))
        if self.guided_augment and random.random()<.5:
            image,label,teacher = [t.flip(-1) for t in [image,label,teacher]]
        return image,label,teacher

    def close(self):
        self.guide_arrays.clear();super().close()


def previews(root,manifest,output):
    paths=[]
    for case,index in [('LCTSC-Train-S2-006',54),('LCTSC-Train-S3-008',79)]:
        rows=[r for r in manifest['cases'] if r['case']==case]
        if not rows:continue
        record=rows[0];hu=load_cached_source(record)[index]
        with mapped_npy(root/record['cache_folder']/'reference_native.npy') as a:expert=a[index]!=0
        with mapped_npy(output/'distilled'/'selected'/case/'prediction_native.npy') as a:candidate=a[index]!=0
        for name,folder in [('original','baseline'),('paired_control','control/selected')]:
            with mapped_npy(output/folder/case/'prediction_native.npy') as a:previous=a[index]!=0
            path=output/f'comparison_{case}_slice_{index:03d}_vs_{name}.png'
            save_learned_preview(hu,expert,candidate,previous,path,case,index,
                baseline_title=f'{name} (yellow)',learned_title='Teacher-guided SMALL U-Net (green)')
            paths.append(str(path))
    return paths


def paired(args):
    if (args.epochs<1 or args.batch_size<1 or args.threads<1 or not 0<args.lr<1 or
            not 0<args.weight<=1 or not .5<args.confidence<1):raise ValueError('Invalid bounded training settings')
    root,manifest=read_manifest(args.dataset);records=train_records(manifest)
    teacher_root,teacher=checked_teacher(args.teacher_dataset,manifest)
    if teacher['dataset_sha256']!=file_sha256(args.dataset):raise ValueError('Original manifest bytes differ')
    torch=torch_module();original=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if (original['dataset_fingerprint']!=manifest['fingerprint'] or original.get('context_slices',1)!=1 or
            original['image_size']!=manifest['image_size'] or original['hu_clip']!=[-1000,400] or original['threshold']!=.5):
        raise ValueError('Use original unchanged full-field 2D initialization')
    tracked=source_hashes([args.dataset,args.teacher_dataset,args.checkpoint,__file__,
        Path(__file__).with_name('lung_unet.py'),Path(__file__).with_name('lung_body_crop.py')])
    tracked.update(teacher['input_sha256'])
    for r in manifest['cases']:
        tracked.update(source_hashes([root/r['cache_folder']/n for n in ['images.npy','labels.npy','reference_native.npy']]))
    for r in teacher['cases']:tracked.update(source_hashes([teacher_root/r['probability_file']]))
    device=select_device(args.device);torch.set_num_threads(args.threads)
    model=make_model(original['base_channels'],1);model.load_state_dict(original['model'])
    architecture=model_summary(model,manifest['image_size'])
    output=fresh_output(args.outdir)
    protocol=dict({k:v for k,v in vars(args).items() if k!='func'},architecture=architecture,input_sha256=tracked,
        teacher_train_cases=[r['case'] for r in records],temperature=1,original_epoch=original['epoch'],
        loss='Original BCE+Dice everywhere + weight * human-agree/confident class-balanced Bernoulli KL(T=1)',
        paired='Same initial weights, new Adam, seeds/order/horizontal flips; only teacher loss differs',
        selection='Highest validation Dice among fixed gate-passing epochs, otherwise exploratory best',
        target='Require native >=2mm deep FN reduction >=5% versus BOTH original and paired control',
        warning='Bounded development screen; no external/Test training or selection; no automatic replacement; '
            'teacher training overlap unknown; same small architecture/MACs, no FPGA or quantization change')
    write_json(output/'protocol.json',protocol)
    report=dict(status='running',original_model_retained=True,protocol=protocol,arms={})
    write_json(output/'evaluation.json',report)
    old=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    val=[r for r in manifest['cases'] if r['split']=='validation']
    try:
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        model.to(device)
        rows,baseline=validate(model,root,val,device,args.batch_size,outdir=output/'baseline')
        report['baseline']=baseline;write_json(output/'baseline.json',dict(cases=rows,summary=baseline))
        for arm in ['control','distilled']:
            model.load_state_dict(original['model']);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
            scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
            folder=output/arm;folder.mkdir();history=[];best=-1.;accepted=-1.
            with GuidedDataset(root,records,teacher_root,teacher['cases']) as dataset:
                for epoch in range(1,args.epochs+1):
                    torch.manual_seed(args.seed+epoch);random.seed(args.seed+epoch);np.random.seed(args.seed+epoch)
                    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,
                        pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(args.seed+epoch))
                    model.train();total=kd_total=count=0;start=perf_counter()
                    for step,(images,labels,probabilities) in enumerate(loader,1):
                        images,labels,probabilities=[t.to(device) for t in [images,labels,probabilities]]
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                            logits=model(images).float();supervised=segmentation_loss(logits,labels)
                            kd=guidance_loss(logits,labels,probabilities,args.confidence) if arm=='distilled' else logits.sum()*0
                            loss=supervised+args.weight*kd
                        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                        scaler.scale(loss).backward();scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update()
                        total+=float(loss.detach())*len(images);kd_total+=float(kd.detach())*len(images);count+=len(images)
                        if step%100==0:print(f'{arm} epoch {epoch} batch {step}/{len(loader)}',flush=True)
                    rows,summary=validate(model,root,val,device,args.batch_size);gate=acceptance(summary,baseline)
                    history.append(dict(epoch=epoch,loss=total/count,guidance_kl=kd_total/count,
                        cases=rows,summary=summary,gate=gate,seconds_including_validation=perf_counter()-start))
                    payload={k:copy.deepcopy(original[k]) for k in ['base_channels','image_size','hu_clip','threshold']}
                    payload.update(model=model.state_dict(),context_slices=1,epoch=epoch,history=history,seed=args.seed,
                        dataset_fingerprint=manifest['fingerprint'],preprocessing={'kind':'full_frame'},
                        distillation=dict(arm=arm,teacher_sha256=TEACHER_SHA256,temperature=1,
                            confidence=args.confidence,weight=args.weight,original_sha256=file_sha256(args.checkpoint)),
                        warning='Development screen; not automatically promoted or quantized/deployed')
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
            rows,summary=validate(model,root,val,device,args.batch_size,outdir=folder/'selected')
            report['arms'][arm]=dict(selected_epoch=selected['epoch'],selected_file=chosen.name,
                cases=rows,summary=summary,gate=acceptance(summary,baseline),history=history)
            write_json(output/'evaluation.json',report)
        report['selected_geometry']=selected_geometry(root,manifest,output,
            arms={'baseline':'baseline','control':'control/selected','distilled':'distilled/selected'})
        depths=report['selected_geometry']['totals']
        report['internal_target_passed']=all(depths[n]['deep_fn_pixels']>0 and
            depths['distilled']['deep_fn_pixels']<=depths[n]['deep_fn_pixels']*.95 for n in ['baseline','control'])
        report['beats_control_gates']=acceptance(report['arms']['distilled']['summary'],report['arms']['control']['summary'])
        report['previews']=previews(root,manifest,output)
        check_hashes(tracked)
        report.update(status='complete_paired_development_screen',protected_inputs_unchanged=True,
            decision='candidate_needs_further_validation' if report['internal_target_passed'] and
                report['arms']['distilled']['gate']['passed'] and report['beats_control_gates']['passed'] else 'do_not_replace_original')
        write_json(output/'evaluation.json',report)
        print(f'Completed paired screen: {report["decision"]}',flush=True)
    except BaseException as error:
        report.update(status='aborted_incomplete',error=str(error));write_json(output/'evaluation.json',report);raise
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old


def parser():
    result=argparse.ArgumentParser(description=__doc__);sub=result.add_subparsers(dest='command',required=True)
    for name,func in [('prepare',prepare),('paired',paired)]:
        p=sub.add_parser(name);p.add_argument('--dataset',default='data/lung_learning_20261009/dataset.json')
        p.add_argument('--outdir',required=True);p.add_argument('--batch-size',type=int,default=4 if name=='prepare' else 8)
        p.add_argument('--threads',type=int,default=4);p.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
        p.set_defaults(func=func)
        if name=='prepare':p.add_argument('--weights',default='data/reference_models/unet_r231-d5d2fc3d.pth')
        else:
            p.add_argument('--teacher-dataset',required=True)
            p.add_argument('--checkpoint',default='data/lung_training_20261011/early_stop_run/best.pt')
            p.add_argument('--epochs',type=int,default=4);p.add_argument('--lr',type=float,default=.0001)
            p.add_argument('--seed',type=int,default=20261013);p.add_argument('--weight',type=float,default=.25)
            p.add_argument('--confidence',type=float,default=.8)
    return result


if __name__=='__main__':
    args=parser().parse_args();args.func(args)
