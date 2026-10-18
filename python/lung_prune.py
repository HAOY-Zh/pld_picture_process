"""Hardware-aware channel pruning; train/development patients only, never Test/COVID.

Candidates are separate checkpoints. Dense zeroing is deliberately not counted as
acceleration: only removed HWC8 channel tiles change the hardware workload.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from lung_fixed_network import FixedUNet
from lung_fpga_contract import SCHEMA, multiplier
from lung_int8_final import final_masks, score_mask, summarize_rows
from lung_quantize import (BLOCKS, calibrate_training, convert_static, export_core,
                           fold_batchnorm, make_quantizable, prepare_static)
from lung_segmentation import fresh_output, write_json
from lung_unet import (SliceDataset, _case_prediction, aggregate_slice_quality,
    context_batch, file_sha256, load_cached_source, make_model, mapped_npy,
    read_manifest, segmentation_loss, torch_module, validate_model)
from verify_lung_axil import check_hashes

WORKSPACE=Path(__file__).resolve().parents[1]
ORIGINAL=WORKSPACE/'data/lung_training_20261011/early_stop_run/best.pt'
DATASET=WORKSPACE/'data/lung_learning_20261009/dataset.json'
BASELINE=WORKSPACE/'data/lung_fixed_network_20261014/screen'
PLANS={
    'decoder8':{'dec2.0':8,'dec2.3':8},
    'balanced':{'enc3.0':24,'neck.0':48,'neck.3':48,'dec3.0':24,'dec3.3':24,'dec2.0':8,'dec2.3':8},
}


def prune_model(source,widths,kept=None):
    """Propagate original channel indices across pool, resize and both skip edges."""
    torch=torch_module();result=copy.deepcopy(source).cpu().eval();selections={}
    original={name:module for name,module in source.named_modules()}
    previous=None
    for block_name in BLOCKS:
        for number in [0,3]:
            name=block_name+'.'+str(number);conv=original[name];bn=original[block_name+'.'+str(number+1)]
            count=widths.get(name,conv.out_channels)
            if count<8 or count%8 or count>conv.out_channels:raise ValueError('Prune in real 8-channel tiles')
            if kept is None:
                rank=(conv.weight.detach().cpu().abs()*
                    (bn.weight.detach().cpu().abs()/torch.sqrt(bn.running_var.cpu()+bn.eps))[:,None,None,None]).sum((1,2,3))
                outputs=sorted(torch.argsort(rank,descending=True,stable=True)[:count].tolist())
            else:
                outputs=list(kept[name])
                if len(outputs)!=count or len(set(outputs))!=count or outputs!=sorted(outputs) or min(outputs)<0 or max(outputs)>=conv.out_channels:
                    raise ValueError('Invalid retained original-channel indices')
            if previous is None:inputs=[0]
            elif number==0 and block_name.startswith('dec'):
                skip={'dec3':'enc3.3','dec2':'enc2.3','dec1':'enc1.3'}[block_name]
                offset=original[previous].out_channels
                inputs=selections[previous]+[offset+i for i in selections[skip]]
            else:inputs=selections[previous]
            replacement=torch.nn.Conv2d(len(inputs),count,conv.kernel_size,conv.stride,conv.padding,
                dilation=conv.dilation,bias=conv.bias is not None)
            with torch.no_grad():
                replacement.weight.copy_(conv.weight.cpu()[outputs][:,inputs])
                if conv.bias is not None:replacement.bias.copy_(conv.bias.cpu()[outputs])
            normalization=torch.nn.BatchNorm2d(count,eps=bn.eps,momentum=bn.momentum)
            with torch.no_grad():
                for key in ['weight','bias','running_mean','running_var']:
                    getattr(normalization,key).copy_(getattr(bn,key).cpu()[outputs])
                normalization.num_batches_tracked.copy_(bn.num_batches_tracked.cpu())
            getattr(result,block_name)[number]=replacement
            getattr(result,block_name)[number+1]=normalization
            selections[name]=outputs;previous=name
    head=source.head
    result.head=torch.nn.Conv2d(len(selections[previous]),1,1,bias=True)
    with torch.no_grad():
        result.head.weight.copy_(head.weight.cpu()[:,selections[previous]])
        result.head.bias.copy_(head.bias.cpu())
    return result.eval(),selections


def workload(model,size=256):
    torch=torch_module();layers=[];hooks=[]
    def hook(name):
        def collect(module,inputs,output):
            ci,co=module.in_channels,module.out_channels;h,w=output.shape[-2:];k=module.kernel_size[0]
            layers.append(dict(name=name,cin=ci,cout=co,height=h,width=w,kernel=k,
                macs=int(h*w*ci*co*k*k),output_tiles=(co+7)//8,
                padded_input_beats=int(h*w*((ci+7)//8)*((co+7)//8)*k*k)))
        return collect
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Conv2d):hooks.append(module.register_forward_hook(hook(name)))
    try:
        with torch.inference_mode():model.cpu().eval()(torch.zeros(1,1,size,size))
    finally:
        for handle in hooks:handle.remove()
    lookup={r['name']:r for r in layers}
    cats=[dict(name='cats.'+str(i),channels=lookup[a]['cout']+lookup[b]['cout'],
        words=lookup[b]['height']*lookup[b]['width']*((lookup[a]['cout']+lookup[b]['cout']+7)//8))
        for i,(a,b) in enumerate([('neck.3','enc3.3'),('dec3.3','enc2.3'),('dec2.3','enc1.3')])]
    return dict(parameters=sum(p.numel() for p in model.parameters()),layers=layers,concat=cats,
        macs=sum(r['macs'] for r in layers),padded_input_beats=sum(r['padded_input_beats'] for r in layers),
        commands=sum(r['output_tiles'] for r in layers)+6,
        board_tested=False,warning='Structural workload only; NOT measured latency or speedup')


def restore_candidate(path):
    torch=torch_module();saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['schema']!='lung_pruned_fp32_v1':raise ValueError('Wrong candidate checkpoint')
    model,_=prune_model(make_model(8,1).eval(),saved['widths'],saved['kept'])
    model.load_state_dict(saved['model'],strict=True)
    return model.eval(),saved


def restore_pruned_quantized(path):
    torch=torch_module();saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['schema']!='lung_pruned_ptq_v1':raise ValueError('Wrong pruned INT8 schema')
    model,_=prune_model(make_model(8,1).eval(),saved['widths'],saved['kept'])
    prepared=prepare_static(fold_batchnorm(make_quantizable(model)))
    with torch.inference_mode():prepared(torch.zeros(1,1,32,32))
    quantized=convert_static(prepared);quantized.load_state_dict(saved['state_dict'],strict=True)
    return quantized.eval(),saved


def export_contract(core,output,checkpoint_hash):
    rows=[];params={}
    with np.load(output/'int8_core_parameters.npz') as arrays:
        for layer in core['layers']:
            row=dict(layer);prefix=row['name'].replace('.','_')
            weight=arrays[prefix+'_weight_int8'];bias=arrays[prefix+'_proposed_bias_int32']
            if np.any(arrays[prefix+'_weight_zero_point']!=0):raise ValueError('Non-symmetric weights')
            m,s=multiplier(row['input_scale']*arrays[prefix+'_weight_scale']/row['output_scale'])
            for key,value in [('weight_int8',weight),('bias_int32',bias),('multiplier_int32',m),('right_shift_int32',s)]:
                params[prefix+'_'+key]=value
            row['conservative_accumulator_bound']=row['proposed_int32_accumulator_max_abs_bound'];rows.append(row)
    lookup={r['name']:r for r in rows};cats=[]
    for i,(a,b) in enumerate([('neck.3','enc3.3'),('dec3.3','enc2.3'),('dec2.3','enc1.3')]):
        dest=core['boundaries']['cats.'+str(i)];branches=[]
        for name in [a,b]:
            row=lookup[name];m,s=multiplier(row['output_scale']/dest['scale'])
            branches.append(dict(source=name,input_scale=row['output_scale'],input_zero_point=row['output_zero_point'],
                multiplier=int(m),right_shift=int(s)))
        cats.append(dict(name='cats.'+str(i),output_scale=dest['scale'],output_zero_point=dest['zero_point'],branches=branches))
    contract=dict(schema=SCHEMA,status='complete_proposed_integer_contract',checkpoint_sha256=checkpoint_hash,
        layers=rows,concat=cats,input_quantization=core['boundaries']['input_quant'],
        arithmetic='Original ACC32/Q31 signed ties-to-even contract; custom 8-aligned channel widths',
        hardware_ready=False,whole_network_fixed_point_validated=False)
    np.savez(output/'parameters.npz',**params);write_json(output/'contract.json',contract)
    return contract


def boundary_metrics(mask,reference,spacing):
    """Native in-plane symmetric boundary distances on EVERY annotated slice.

    This is deliberately labelled 2D boundary distance, never a 3D Hausdorff.
    Empty prediction/reference mismatches are counted separately.
    """
    total=0.;count=0;per_slice_p95=[];mismatch=0
    for predicted,truth in zip(mask,np.asarray(reference)!=0):
        if not truth.any():continue
        if not predicted.any():mismatch+=1;continue
        pb=predicted&~binary_erosion(predicted);tb=truth&~binary_erosion(truth)
        distances=np.concatenate([distance_transform_edt(~tb,sampling=spacing)[pb],
                                  distance_transform_edt(~pb,sampling=spacing)[tb]])
        total+=float(distances.sum());count+=len(distances);per_slice_p95.append(float(np.percentile(distances,95)))
    return dict(mean_symmetric_boundary_mm=total/count if count else None,
        mean_slice_boundary_p95_mm=float(np.mean(per_slice_p95)) if per_slice_p95 else None,
        annotated_slices_with_empty_prediction=mismatch,annotated_slices_compared=len(per_slice_p95),
        scope='Native 2D boundary per annotated slice; not 3D surface distance')


def evaluate(model,root,records,device,output,stage):
    rows=[]
    for record in records:
        print(stage+' VALIDATION '+record['case'],flush=True)
        raw,_=_case_prediction(model,root,record,device,8)
        hu=load_cached_source(record)
        _,final,_,_,_=final_masks(hu,raw,record['spacing_zyx_mm']);del hu
        folder=output/record['case'];folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(stage+'_native.npz'),raw=raw,final=final)
        with mapped_npy(root/record['cache_folder']/'reference_native.npy') as reference:
            row=dict(case=record['case'],scores={stage+'_raw':score_mask(raw,reference),stage+'_final':score_mask(final,reference)},
                boundary=boundary_metrics(final,reference,record['spacing_zyx_mm'][1:]),
                mask_sha256=file_sha256(folder/(stage+'_native.npz')))
        rows.append(row);print('  final Dice',row['scores'][stage+'_final']['metrics']['dice'],flush=True)
    return dict(cases=rows,summaries=summarize_rows(rows,[stage+'_raw',stage+'_final']))


def run(args):
    torch=torch_module();torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise ValueError('Explicit CUDA experiment; no silent CPU fallback')
    root,manifest=read_manifest(DATASET);train=[r for r in manifest['cases'] if r['split']=='train']
    validation=[r for r in manifest['cases'] if r['split']=='validation']
    if len(train)!=18 or len(validation)!=6:raise ValueError('Expected frozen 18/6 split')
    baseline=json.loads((BASELINE/'evaluation.json').read_text())
    protected=dict(baseline['protocol']['input_sha256']);check_hashes(protected)
    # Protect training labels/images too, including those not in the original screen.
    for record in train:
        for name in ['images.npy','labels.npy']:
            p=root/record['cache_folder']/name;protected[str(p)]=file_sha256(p)
    source=torch.load(ORIGINAL,map_location='cpu',weights_only=True)
    original=make_model(8,1).eval();original.load_state_dict(source['model'],strict=True)
    output=fresh_output(args.outdir)
    protocol=dict(status='predeclared_pruning_development_protocol',seed=20261017,
        max_mean_dice_drop=args.max_drop,min_structural_work_reduction=.10,
        epochs=args.epochs,learning_rate=.0001,batch_size=12,teacher_BCE_weight=.5,
        bn_running_stats='frozen during fine-tuning; affine parameters trainable',
        ranking='BN-folded output-filter L1, original weight indices propagated through skip concat',
        candidates=PLANS,train_cases=[r['case'] for r in train],validation_cases=[r['case'] for r in validation],
        validation_for_selection=True,test_or_covid_used=False,threshold=.5,image_size=256,
        source_checkpoint_sha256=file_sha256(ORIGINAL),dataset_sha256=file_sha256(DATASET),
        input_sha256=protected,source_sha256={str(Path(__file__).resolve()):file_sha256(__file__)},
        warning='Development selection, not independent generalization evidence. No candidate board timing yet.')
    write_json(output/'protocol.json',protocol)
    original_work=workload(original);write_json(output/'baseline_workload.json',original_work)
    teacher=copy.deepcopy(original).cuda().eval();results=[]
    try:
        for candidate,widths in PLANS.items():
            folder=output/candidate;folder.mkdir();random.seed(20261017);np.random.seed(20261017);torch.manual_seed(20261017);torch.cuda.manual_seed_all(20261017)
            model,kept=prune_model(original,widths);work=workload(model)
            write_json(folder/'workload.json',work);write_json(folder/'kept.json',kept)
            model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.0001);scaler=torch.amp.GradScaler('cuda')
            history=[];best=-1.;best_epoch=None
            with SliceDataset(root,train,augment=True) as dataset:
                loader=torch.utils.data.DataLoader(dataset,batch_size=12,shuffle=True,num_workers=0,pin_memory=True)
                for epoch in range(args.epochs+1):
                    start=time.perf_counter();loss_sum=0.;seen=0
                    if epoch:
                        model.train()
                        for module in model.modules():
                            if isinstance(module,torch.nn.BatchNorm2d):module.eval()
                        for image,target in loader:
                            image=image.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
                            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):soft=torch.sigmoid(teacher(image))
                            optimizer.zero_grad(set_to_none=True)
                            with torch.autocast('cuda',dtype=torch.float16):
                                logits=model(image)
                                loss=segmentation_loss(logits,target)+.5*torch.nn.functional.binary_cross_entropy_with_logits(logits,soft)
                            scaler.scale(loss).backward();scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update()
                            loss_sum+=float(loss)*len(image);seen+=len(image)
                    model.eval();rows,score=validate_model(model,root,validation,'cuda',8)
                    history.append(dict(epoch=epoch,raw_mean_case_dice=score,loss=loss_sum/seen if seen else None,
                        wall_seconds=time.perf_counter()-start,cases=rows))
                    write_json(folder/'training_history.json',history)
                    if score>best:
                        best=score;best_epoch=epoch
                        torch.save(dict(schema='lung_pruned_fp32_v1',widths=widths,kept=kept,
                            model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,
                            source_checkpoint_sha256=protocol['source_checkpoint_sha256'],dataset_fingerprint=manifest['fingerprint']),folder/'best.pt')
                    print(candidate,'epoch',epoch,'Dice',round(score,7),'seconds',round(history[-1]['wall_seconds'],1),flush=True)
            del model,optimizer,scaler;torch.cuda.empty_cache()
            model,saved=restore_candidate(folder/'best.pt');model.cuda().eval()
            fp32=evaluate(model,root,validation,'cuda',folder,'fp32');write_json(folder/'fp32_evaluation.json',fp32)
            prepared=prepare_static(fold_batchnorm(make_quantizable(model.cpu().eval())))
            calibration=calibrate_training(prepared,model,root,train,'cuda',8,256)
            write_json(folder/'calibration.json',calibration)
            quantized=convert_static(prepared);core=export_core(quantized,256,folder)
            torch.save(dict(schema='lung_pruned_ptq_v1',widths=widths,kept=kept,state_dict=quantized.state_dict(),
                source_checkpoint_sha256=file_sha256(folder/'best.pt'),original_checkpoint_sha256=protocol['source_checkpoint_sha256'],
                dataset_fingerprint=manifest['fingerprint']),folder/'int8_model.pt')
            restored,_=restore_pruned_quantized(folder/'int8_model.pt')
            with mapped_npy(root/train[0]['cache_folder']/'images.npy') as images,torch.inference_mode():
                sample=torch.from_numpy(context_batch(images,[0,len(images)//2,len(images)-1],1))
                torch.testing.assert_close(restored(sample),quantized(sample),rtol=0,atol=0)
            contract=export_contract(core,folder,file_sha256(folder/'best.pt'))
            del restored,prepared,model;torch.cuda.empty_cache()
            ptq=evaluate(quantized,root,validation,'cpu',folder,'ptq');write_json(folder/'ptq_evaluation.json',ptq)
            del quantized
            with np.load(folder/'parameters.npz') as parameters:fixed=FixedUNet(contract,parameters,'cuda')
            fixed_report=evaluate(fixed,root,validation,'cpu',folder,'fixed');write_json(folder/'fixed_evaluation.json',fixed_report)
            del fixed;torch.cuda.empty_cache()
            base_dice=baseline['summaries']['fixed_final']['mean_case_dice']
            dice=fixed_report['summaries']['fixed_final']['mean_case_dice'];drop=base_dice-dice
            reduction=1-work['padded_input_beats']/original_work['padded_input_beats']
            per_case=[dict(case=r['case'],dice_delta=r['scores']['fixed_final']['metrics']['dice']-
                next(b for b in baseline['cases'] if b['case']==r['case'])['scores']['fixed_final']['metrics']['dice']) for r in fixed_report['cases']]
            result=dict(candidate=candidate,best_epoch=best_epoch,mean_case_dice=dice,baseline_mean_case_dice=base_dice,
                mean_dice_drop=drop,per_case=per_case,padded_work_reduction=reduction,
                passes_mean_and_work_screen=drop<=args.max_drop and reduction>=.10,
                boundary_and_small_slice_review_required=True,board_tested=False,
                warning='Actual hardware speedup pending; development selection is not independent test evidence')
            results.append(result);write_json(folder/'result.json',result);write_json(output/'results.json',results)
            print('CANDIDATE RESULT',result,flush=True)
        check_hashes(protected)
        write_json(output/'verification.json',dict(status='complete_pruning_development_experiment',
            original_inputs_unchanged=True,protocol_sha256=file_sha256(output/'protocol.json'),results=results,
            test_or_covid_used=False,board_tested=False))
    except Exception as error:
        write_json(output/'failure.json',dict(status='aborted',error=str(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',default='data/lung_prune_20261017')
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--max-drop',type=float,default=.01)
    run(p.parse_args())
