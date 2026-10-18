"""One bounded recovery of balanced pruning, without touching the first experiment.

Uses only the original 18 TRAIN patients for updates/calibration. Six development
patients select checkpoints; this is iterative development, not blind testing.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np

from lung_prune import (BASELINE,DATASET,ORIGINAL,WORKSPACE,evaluate,export_contract,
    restore_candidate,restore_pruned_quantized,workload)
from lung_fixed_network import FixedUNet
from lung_int8_final import final_masks,score_mask,summarize_rows
from lung_quantize import (calibrate_training,convert_static,export_core,fold_batchnorm,
    make_quantizable,prepare_static)
from lung_segmentation import fresh_output,write_json
from lung_unet import (SliceDataset,_case_prediction,context_batch,file_sha256,load_cached_source,
    make_model,mapped_npy,read_manifest,segmentation_loss,torch_module)
from verify_lung_axil import check_hashes


def run(outdir,epochs):
    torch=torch_module();torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    random.seed(20261018);np.random.seed(20261018);torch.manual_seed(20261018);torch.cuda.manual_seed_all(20261018)
    cache,manifest=read_manifest(DATASET);train=[r for r in manifest['cases'] if r['split']=='train']
    validation=[r for r in manifest['cases'] if r['split']=='validation']
    if len(train)!=18 or len(validation)!=6 or not torch.cuda.is_available():raise ValueError('Need frozen split and CUDA')
    first=WORKSPACE/'data/lung_prune_20261017';old=json.loads((first/'protocol.json').read_text());check_hashes(old['input_sha256']);check_hashes(old['source_sha256'])
    start_path=first/'balanced/best.pt';model,saved=restore_candidate(start_path)
    baseline=json.loads((BASELINE/'evaluation.json').read_text())
    output=fresh_output(outdir);candidate='balanced_recovered';folder=output/candidate;folder.mkdir()
    teacher=make_model().eval();teacher.load_state_dict(torch.load(ORIGINAL,map_location='cpu',weights_only=True)['model']);teacher.cuda()
    baseline_work=json.loads((first/'baseline_workload.json').read_text());write_json(output/'baseline_workload.json',baseline_work)
    work=workload(model);write_json(folder/'workload.json',work);write_json(folder/'kept.json',saved['kept'])
    sampling=[];sampling_rows=[];protected=dict(old['input_sha256'])
    for record in train:
        path=cache/record['cache_folder']/'reference_native.npy';protected[str(path)]=file_sha256(path)
        with mapped_npy(path) as reference:
            area=reference.sum((1,2));weights=np.where(area==0,2.,np.where(area<=1000,4.,1.))
        sampling.extend(weights.tolist());sampling_rows.append(dict(case=record['case'],empty=int((area==0).sum()),small=int(((area>0)&(area<=1000)).sum())))
    protocol=dict(status='predeclared_bounded_pruning_recovery',source_checkpoint_sha256=file_sha256(ORIGINAL),
        warm_start_sha256=file_sha256(start_path),seed=20261018,epochs=epochs,learning_rate=.00005,batch_size=12,
        sampling=dict(empty_weight=2.,small_weight=4.,other_weight=1.,native_small_area_max=1000,rows=sampling_rows,
            draws_per_epoch=len(sampling),with_replacement=True),
        loss='Original BCE+softDice + 0.5 teacher soft-target BCE + 0.1 teacher-logit SmoothL1(beta=1)',
        candidates={candidate:saved['widths']},bn_running_stats='frozen; affine parameters trainable',
        checkpoint_selection='Prefer final-native mean Dice within 0.01, small-slice mean within 0.03, empty FP <= baseline+1000; then maximize mean Dice',
        max_mean_dice_drop=.01,min_structural_work_reduction=.10,
        train_cases=[r['case'] for r in train],validation_cases=[r['case'] for r in validation],
        validation_for_selection=True,test_or_covid_used=False,input_sha256=protected,
        source_sha256={str(Path(__file__).resolve()):file_sha256(__file__),str(WORKSPACE/'python/lung_prune.py'):file_sha256(WORKSPACE/'python/lung_prune.py')},
        warning='Extra development recovery after inspecting first validation failure; never independent test evidence')
    write_json(output/'protocol.json',protocol);check_hashes(protected)
    # Keep original CT geometry. The HU cache is read-only and never a target.
    hu={r['case']:load_cached_source(r) for r in validation}
    base=baseline['summaries']['fixed_final'];best_rank=(False,-float('inf'));history=[]
    model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.00005);scaler=torch.amp.GradScaler('cuda')
    generator=torch.Generator().manual_seed(20261018)
    sampler=torch.utils.data.WeightedRandomSampler(torch.tensor(sampling,dtype=torch.double),len(sampling),replacement=True,generator=generator)
    with SliceDataset(cache,train,augment=True) as dataset:
        loader=torch.utils.data.DataLoader(dataset,batch_size=12,sampler=sampler,num_workers=0,pin_memory=True)
        for epoch in range(1,epochs+1):
            start=time.perf_counter();model.train();total_loss=0.;seen=0
            for module in model.modules():
                if isinstance(module,torch.nn.BatchNorm2d):module.eval()
            for image,target in loader:
                image=image.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):teacher_logits=teacher(image)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(image)
                    loss=segmentation_loss(logits,target)+.5*torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.sigmoid(teacher_logits))
                    loss=loss+.1*torch.nn.functional.smooth_l1_loss(logits,teacher_logits,beta=1.)
                scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                scaler.step(optimizer);scaler.update();total_loss+=loss.detach().item()*len(image);seen+=len(image)
            model.eval();rows=[]
            for record in validation:
                raw,_=_case_prediction(model,cache,record,'cuda',8)
                _,final,_,_,_=final_masks(hu[record['case']],raw,record['spacing_zyx_mm'])
                with mapped_npy(cache/record['cache_folder']/'reference_native.npy') as reference:
                    rows.append(dict(case=record['case'],scores={'final':score_mask(final,reference)}))
            summary=summarize_rows(rows,['final'])['final'];quality=summary['slice_quality']
            passed=bool(summary['mean_case_dice']>=base['mean_case_dice']-.01 and
                quality['groups']['small']['mean_slice_dice']>=base['slice_quality']['groups']['small']['mean_slice_dice']-.03 and
                quality['empty_false_positive_pixels']<=base['slice_quality']['empty_false_positive_pixels']+1000)
            rank=(passed,summary['mean_case_dice'])
            history.append(dict(epoch=epoch,loss=total_loss/seen,wall_seconds=time.perf_counter()-start,
                passes_selection_gates=passed,summary=summary,cases=rows))
            write_json(folder/'training_history.json',history)
            if rank>best_rank:
                best_rank=rank
                torch.save(dict(saved,model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,
                    recovery_warm_start_sha256=protocol['warm_start_sha256']),folder/'best.pt')
            print('Recovery epoch',epoch,'Dice',summary['mean_case_dice'],'small',quality['groups']['small']['mean_slice_dice'],
                'emptyFP',quality['empty_false_positive_pixels'],'gate',passed,'seconds',round(history[-1]['wall_seconds'],1),flush=True)
    del hu,model,teacher,optimizer,scaler;torch.cuda.empty_cache()
    model,saved=restore_candidate(folder/'best.pt');model.cuda().eval()
    write_json(folder/'fp32_evaluation.json',evaluate(model,cache,validation,'cuda',folder,'fp32'))
    prepared=prepare_static(fold_batchnorm(make_quantizable(model.cpu().eval())))
    calibration=calibrate_training(prepared,model,cache,train,'cuda',8,256);write_json(folder/'calibration.json',calibration)
    quantized=convert_static(prepared);core=export_core(quantized,256,folder)
    torch.save(dict(schema='lung_pruned_ptq_v1',widths=saved['widths'],kept=saved['kept'],state_dict=quantized.state_dict(),
        source_checkpoint_sha256=file_sha256(folder/'best.pt'),original_checkpoint_sha256=protocol['source_checkpoint_sha256'],
        dataset_fingerprint=manifest['fingerprint']),folder/'int8_model.pt')
    restored,_=restore_pruned_quantized(folder/'int8_model.pt')
    with mapped_npy(cache/train[0]['cache_folder']/'images.npy') as images,torch.inference_mode():
        probe=torch.from_numpy(context_batch(images,[0,len(images)//2,len(images)-1],1))
        torch.testing.assert_close(restored(probe),quantized(probe),rtol=0,atol=0)
    contract=export_contract(core,folder,file_sha256(folder/'best.pt'))
    del restored,model,prepared;torch.cuda.empty_cache()
    write_json(folder/'ptq_evaluation.json',evaluate(quantized,cache,validation,'cpu',folder,'ptq'));del quantized
    with np.load(folder/'parameters.npz') as parameters:fixed=FixedUNet(contract,parameters,'cuda')
    fixed_report=evaluate(fixed,cache,validation,'cpu',folder,'fixed');write_json(folder/'fixed_evaluation.json',fixed_report)
    dice=fixed_report['summaries']['fixed_final']['mean_case_dice'];reduction=1-work['padded_input_beats']/baseline_work['padded_input_beats']
    result=dict(candidate=candidate,best_epoch=saved['epoch'],mean_case_dice=dice,baseline_mean_case_dice=base['mean_case_dice'],
        mean_dice_drop=base['mean_case_dice']-dice,padded_work_reduction=reduction,
        passes_mean_and_work_screen=dice>=base['mean_case_dice']-.01 and reduction>=.10,
        boundary_and_small_slice_review_required=True,board_tested=False)
    write_json(folder/'result.json',result);write_json(output/'results.json',[result]);check_hashes(protected);check_hashes(protocol['source_sha256'])
    write_json(output/'verification.json',dict(status='complete_pruning_development_experiment',original_inputs_unchanged=True,
        protocol_sha256=file_sha256(output/'protocol.json'),results=[result],test_or_covid_used=False,board_tested=False))
    print('RECOVERY RESULT',result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',default='data/lung_prune_recover_20261017')
    p.add_argument('--epochs',type=int,default=8);a=p.parse_args();run(a.outdir,a.epochs)
