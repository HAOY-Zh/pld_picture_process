"""Bounded train/development-only refinement after earlier test feedback.

Three predeclared HWC8 candidates; train-only activation-gradient importance,
teacher feature/logit distillation, unchanged inference and expert labels.
Previously observed test cohorts must not be described as a new blind test.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np

from lung_prune import (BASELINE,DATASET,ORIGINAL,WORKSPACE,evaluate,export_contract,
    prune_model,restore_candidate,restore_pruned_quantized,workload)
from lung_prune_moderate import development_rank
from lung_quantize import (BLOCKS,calibrate_training,convert_static,export_core,
    fold_batchnorm,make_quantizable,prepare_static)
from lung_int8_final import final_masks,score_mask,summarize_rows
from lung_segmentation import fresh_output,write_json
from lung_unet import (SliceDataset,_case_prediction,context_batch,file_sha256,
    load_cached_source,make_model,mapped_npy,read_manifest,segmentation_loss,torch_module)
from lung_fixed_network import FixedUNet
from verify_lung_axil import check_hashes

SEED=2026101701
PLANS={
    'fisher_moderate':{'enc3.0':24,'neck.0':48,'neck.3':48,'dec3.0':24,'dec3.3':24},
    'retain_encoder':{'neck.0':48,'neck.3':48,'dec3.0':24,'dec3.3':24},
    'retain_detail':{'neck.0':48,'neck.3':48,'dec3.0':24},
}
FEATURES={'enc3.5':'enc3.3','neck.5':'neck.3','dec3.5':'dec3.3',
          'dec2.5':'dec2.3','dec1.5':'dec1.3'}


def retained_channels(original,widths,importance):
    """Only real channel tiles are removed; every full-width layer is unchanged."""
    selected={}
    for block in BLOCKS:
        for number in (0,3):
            name=f'{block}.{number}';conv=original.get_submodule(name)
            count=widths.get(name,conv.out_channels)
            if count<8 or count%8 or count>conv.out_channels:raise ValueError('Need 8-aligned channel widths')
            if count==conv.out_channels:selected[name]=list(range(count));continue
            scores=np.asarray(importance[name],dtype=np.float64)
            if (scores.shape!=(conv.out_channels,) or not np.isfinite(scores).all()
                    or (scores<0).any() or scores.sum()<=0):
                raise ValueError('Invalid training importance')
            selected[name]=sorted(np.argsort(-scores,kind='stable')[:count].tolist())
    return selected


def importance_indices(root,records):
    """Eight slices per group and patient, drawn only from TRAIN patients."""
    rng=np.random.default_rng(SEED);indices=[];weights=[];rows=[];offset=0
    for r in records:
        if r['split']!='train' or 'Test-' in r['case'] or 'corona' in r['case'].lower():
            raise ValueError('Importance statistics require original training patients')
        with mapped_npy(root/r['cache_folder']/'reference_native.npy') as a:area=(a!=0).sum((1,2))
        groups=[np.flatnonzero(area==0),np.flatnonzero((area>0)&(area<=1000)),np.flatnonzero(area>1000)]
        chosen=[]
        for g in groups:
            if len(g):chosen.extend(sorted(rng.choice(g,size=min(8,len(g)),replace=False).tolist()))
        indices.extend(offset+z for z in chosen)
        weights.extend(np.where(area==0,3.,np.where(area<=1000,6.,1.)).tolist())
        rows.append(dict(case=r['case'],selected_slices=chosen,
            empty=int((area==0).sum()),small=int(((area>0)&(area<=1000)).sum())))
        offset+=r['slices']
    return indices,weights,rows


def collect_importance(original,root,train,indices,output):
    torch=torch_module();model=copy.deepcopy(original).cuda().eval()
    targets=sorted({name for p in PLANS.values() for name in p})
    totals={n:torch.zeros(model.get_submodule(n).out_channels,device='cuda') for n in targets}
    hooks=[]
    def hook(name):
        def forward(module,args,value):
            activation=value.detach()
            def backward(gradient):
                totals[name].add_((activation*gradient.detach()).square().sum((0,2,3)))
            value.register_hook(backward)
        return forward
    for n in targets:hooks.append(model.get_submodule(n).register_forward_hook(hook(n)))
    try:
        with SliceDataset(root,train,augment=False) as dataset:
            loader=torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,indices),
                batch_size=8,shuffle=False,num_workers=0)
            for i,(image,target) in enumerate(loader,1):
                model.zero_grad(set_to_none=True)
                logits=model(image.cuda());loss=segmentation_loss(logits,target.cuda());loss.backward()
                if i%10==0:print('FISHER TRAIN batches',i,'/',len(loader),flush=True)
        result={n:totals[n].cpu().double().numpy().tolist() for n in targets}
        write_json(output/'train_importance.json',dict(method='Sum squared activation times loss gradient; original TRAIN only',
            sampled_slices=len(indices),scores=result))
        return result
    finally:
        for h in hooks:h.remove()
        del model,totals;torch.cuda.empty_cache()


def feature_distillation(student,teacher,kept):
    torch=torch_module();loss=None
    for feature,conv in FEATURES.items():
        a=student[feature];b=teacher[feature][:,kept[conv]]
        if a.shape!=b.shape:raise ValueError('Feature channel provenance differs')
        scale=b.square().mean().sqrt().clamp(min=.1)
        current=torch.nn.functional.smooth_l1_loss(a/scale,b/scale,beta=1.)
        loss=current if loss is None else loss+current
    return loss/len(FEATURES)


def run(outdir,epochs=10):
    if not 1<=epochs<=12:raise ValueError('Bounded experiment: 1..12 epochs')
    torch=torch_module();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise ValueError('CUDA required; no CPU fallback')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    root,manifest=read_manifest(DATASET)
    train=[r for r in manifest['cases'] if r['split']=='train']
    validation=[r for r in manifest['cases'] if r['split']=='validation']
    if len(train)!=18 or len(validation)!=6:raise ValueError('Need original 18/6 patient split')
    baseline=json.loads((BASELINE/'evaluation.json').read_text(encoding='utf-8'))
    protected=dict(baseline['protocol']['input_sha256'])
    for r in train:
        for name in ['images.npy','labels.npy','reference_native.npy']:
            p=root/r['cache_folder']/name;protected[str(p)]=file_sha256(p)
    for p in [ORIGINAL,DATASET,BASELINE/'evaluation.json']:protected[str(p)]=file_sha256(p)
    check_hashes(protected)
    original=make_model(8,1).eval()
    original.load_state_dict(torch.load(ORIGINAL,map_location='cpu',weights_only=True)['model'])
    original_work=workload(original)
    indices,weights,sampling_rows=importance_indices(root,train)
    output=fresh_output(outdir).resolve()
    sources=[Path(__file__).resolve()]+[WORKSPACE/'python'/n for n in ['lung_prune_refine_fulltest.py','lung_prune.py',
        'lung_prune_moderate.py','lung_unet.py','lung_quantize.py','lung_fixed_network.py',
        'lung_int8_final.py','lung_body_gate.py','lung_island_cleanup.py']]
    protocol=dict(status='predeclared_refined_pruning_development_protocol',seed=SEED,epochs=epochs,
        candidates=PLANS,learning_rate=.00005,batch_size=12,
        ranking='Train-only activation-gradient squared Fisher proxy; 8-aligned output widths',
        importance_rows=sampling_rows,importance_sample_slices=len(indices),
        sampling=dict(empty_weight=3.,small_weight=6.,other_weight=1.,draws_per_epoch=len(weights)),
        loss='BCE+softDice + 0.5 teacher BCE + 0.35 teacher-logit SmoothL1 + 0.05 matched-feature SmoothL1',
        feature_sites=FEATURES,bn_running_stats='frozen; affine trainable',
        augmentation='Original horizontal flip; p=.35/image gain[.95,1.05], offset[-.02,.02], sigma[0,.007] in normalized input; labels unchanged',
        checkpoint_selection='Development FP32 all-case/small/empty gates then mean Dice, including epoch zero',
        max_mean_dice_drop=.01,max_each_case_dice_drop=.01,max_small_slice_dice_drop=.03,
        max_extra_empty_fp_pixels=1000,min_structural_work_reduction=.10,
        final_acceptance='All 27 available cases each exact-INT8 final Dice drop <=0.01; inspect empty/small/missed; actual board speed separately',
        train_cases=[r['case'] for r in train],validation_cases=[r['case'] for r in validation],
        validation_for_selection=True,test_or_covid_used=False,threshold=.5,image_size=256,
        source_checkpoint_sha256=file_sha256(ORIGINAL),input_sha256=protected,
        source_sha256={str(p):file_sha256(p) for p in sources},
        warning='Proposed after prior test feedback; repeated confirmation cohorts are no longer a new blind test. No test images/labels used for training, ranking, calibration or epoch selection.')
    write_json(output/'protocol.json',protocol);write_json(output/'baseline_workload.json',original_work)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    importance=collect_importance(original,root,train,indices,output)
    teacher=copy.deepcopy(original).cuda().eval().requires_grad_(False)
    teacher_features={};teacher_hooks=[]
    for name in FEATURES:
        teacher_hooks.append(teacher.get_submodule(name).register_forward_hook(
            lambda m,a,v,n=name:teacher_features.__setitem__(n,v.detach())))
    hu={r['case']:load_cached_source(r) for r in validation};results=[]
    for candidate,widths in PLANS.items():
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        folder=output/candidate;folder.mkdir()
        kept=retained_channels(original,widths,importance)
        model,kept=prune_model(original,widths,kept);work=workload(model)
        write_json(folder/'kept.json',kept);write_json(folder/'workload.json',work)
        student_features={};hooks=[]
        for name in FEATURES:
            hooks.append(model.get_submodule(name).register_forward_hook(
                lambda m,a,v,n=name:student_features.__setitem__(n,v)))
        model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.00005)
        scaler=torch.amp.GradScaler('cuda');history=[];best_rank=(False,-float('inf'))
        saved=dict(schema='lung_pruned_fp32_v1',widths=widths,kept=kept,
            source_checkpoint_sha256=protocol['source_checkpoint_sha256'],dataset_fingerprint=manifest['fingerprint'])
        sampler=torch.utils.data.WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),len(weights),
            replacement=True,generator=torch.Generator().manual_seed(SEED))
        with SliceDataset(root,train,augment=True) as dataset:
            loader=torch.utils.data.DataLoader(dataset,batch_size=12,sampler=sampler,num_workers=0,pin_memory=True)
            for epoch in range(epochs+1):
                start=time.perf_counter();total=0.;seen=0
                if epoch:
                    model.train()
                    for module in model.modules():
                        if isinstance(module,torch.nn.BatchNorm2d):module.eval()
                    for image,target in loader:
                        image=image.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
                        enabled=(torch.rand(len(image),1,1,1,device='cuda')<.35)
                        gain=1+(torch.rand(len(image),1,1,1,device='cuda')-.5)*.10
                        offset=(torch.rand(len(image),1,1,1,device='cuda')-.5)*.04
                        sigma=torch.rand(len(image),1,1,1,device='cuda')*.007
                        image=torch.where(enabled,(image*gain+offset+torch.randn_like(image)*sigma).clamp(0,1),image)
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):teacher_logits=teacher(image)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.float16):
                            logits=model(image)
                            loss=segmentation_loss(logits,target)
                            loss=loss+.5*torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.sigmoid(teacher_logits))
                            loss=loss+.35*torch.nn.functional.smooth_l1_loss(logits,teacher_logits,beta=1.)
                            loss=loss+.05*feature_distillation(student_features,teacher_features,kept)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite recovery loss; stopping candidate')
                        scaler.scale(loss).backward();scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                        scaler.step(optimizer);scaler.update();total+=loss.detach().item()*len(image);seen+=len(image)
                        student_features.clear();teacher_features.clear()
                model.eval();rows=[]
                for record in validation:
                    raw,_=_case_prediction(model,root,record,'cuda',8)
                    _,final,_,_,_=final_masks(hu[record['case']],raw,record['spacing_zyx_mm'])
                    with mapped_npy(root/record['cache_folder']/'reference_native.npy') as reference:
                        rows.append(dict(case=record['case'],scores={'final':score_mask(final,reference)}))
                student_features.clear()
                summary=summarize_rows(rows,['final'])['final'];rank,gate=development_rank(rows,summary,baseline)
                history.append(dict(epoch=epoch,loss=total/seen if seen else None,
                    wall_seconds=time.perf_counter()-start,summary=summary,cases=rows,**gate))
                write_json(folder/'training_history.json',history)
                if rank>best_rank:
                    best_rank=rank
                    torch.save(dict(saved,epoch=epoch,model={k:v.detach().cpu() for k,v in model.state_dict().items()}),folder/'best.pt')
                print(candidate,'epoch',epoch,'Dice',round(summary['mean_case_dice'],7),
                    'worst',round(gate['worst_case_dice_delta'],7),'emptyFP',gate['empty_false_positive_pixels'],
                    'gate',rank[0],'seconds',round(history[-1]['wall_seconds'],1),flush=True)
        for h in hooks:h.remove()
        del model,optimizer,scaler;torch.cuda.empty_cache()
        model,saved=restore_candidate(folder/'best.pt');model.cuda().eval()
        write_json(folder/'fp32_evaluation.json',evaluate(model,root,validation,'cuda',folder,'fp32'))
        prepared=prepare_static(fold_batchnorm(make_quantizable(model.cpu().eval())))
        write_json(folder/'calibration.json',calibrate_training(prepared,model,root,train,'cuda',8,256))
        quantized=convert_static(prepared);core=export_core(quantized,256,folder)
        torch.save(dict(schema='lung_pruned_ptq_v1',widths=widths,kept=kept,state_dict=quantized.state_dict(),
            source_checkpoint_sha256=file_sha256(folder/'best.pt'),original_checkpoint_sha256=protocol['source_checkpoint_sha256'],
            dataset_fingerprint=manifest['fingerprint']),folder/'int8_model.pt')
        restored,_=restore_pruned_quantized(folder/'int8_model.pt')
        with mapped_npy(root/train[0]['cache_folder']/'images.npy') as a,torch.inference_mode():
            probe=torch.from_numpy(context_batch(a,[0,len(a)//2,len(a)-1],1))
            torch.testing.assert_close(restored(probe),quantized(probe),rtol=0,atol=0)
        contract=export_contract(core,folder,file_sha256(folder/'best.pt'))
        del restored,prepared,model;torch.cuda.empty_cache()
        write_json(folder/'ptq_evaluation.json',evaluate(quantized,root,validation,'cpu',folder,'ptq'));del quantized
        with np.load(folder/'parameters.npz') as p:fixed=FixedUNet(contract,p,'cuda')
        report=evaluate(fixed,root,validation,'cpu',folder,'fixed');del fixed
        write_json(folder/'fixed_evaluation.json',report)
        base=baseline['summaries']['fixed_final']['mean_case_dice'];dice=report['summaries']['fixed_final']['mean_case_dice']
        pairs=[dict(case=r['case'],dice_delta=r['scores']['fixed_final']['metrics']['dice']-
            next(b for b in baseline['cases'] if b['case']==r['case'])['scores']['fixed_final']['metrics']['dice']) for r in report['cases']]
        reduction=1-work['padded_input_beats']/original_work['padded_input_beats']
        result=dict(candidate=candidate,best_epoch=saved['epoch'],mean_case_dice=dice,baseline_mean_case_dice=base,
            mean_dice_drop=base-dice,per_case=pairs,padded_work_reduction=reduction,
            passes_mean_and_work_screen=dice>=base-.01 and reduction>=.10,
            passes_each_case_dice_screen=min(r['dice_delta'] for r in pairs)>=-.01-1e-12,
            boundary_and_small_slice_review_required=True,board_tested=False)
        write_json(folder/'result.json',result);results.append(result);write_json(output/'results.json',results)
        print('REFINE RESULT',json.dumps(result),flush=True)
    for h in teacher_hooks:h.remove()
    del teacher,hu;torch.cuda.empty_cache()
    check_hashes(protected);check_hashes(protocol['source_sha256'])
    write_json(output/'verification.json',dict(status='complete_pruning_development_experiment',
        original_inputs_unchanged=True,protocol_sha256=file_sha256(output/'protocol.json'),results=results,
        test_or_covid_used=False,board_tested=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',default='data/lung_prune_refine_20261017');p.add_argument('--epochs',type=int,default=10)
    a=p.parse_args();run(a.outdir,a.epochs)
