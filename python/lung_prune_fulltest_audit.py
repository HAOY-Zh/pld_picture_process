"""Recompute all saved full-test masks and audit frozen inputs, coverage and failures."""
import argparse
import json
from pathlib import Path

import numpy as np

from lung_prune_fulltest import MODEL_NAMES,STAGES,REBUILT,summaries,paired_slices
from lung_segmentation import write_json
from lung_unet import file_sha256,mapped_npy
from verify_lung_axil import check_hashes


def independent_counts(prediction,reference):
    predicted=prediction!=0;truth=reference!=0
    true_positive=int(np.count_nonzero(predicted&truth))
    false_positive=int(predicted.sum())-true_positive
    false_negative=int(truth.sum())-true_positive
    denominator=2*true_positive+false_positive+false_negative
    return dict(true_positive=true_positive,false_positive=false_positive,false_negative=false_negative,
        dice=2*true_positive/denominator if denominator else None)


def check_slice_quality(mask,reference,saved):
    truth=reference!=0;predicted=mask!=0
    area=truth.sum((1,2));foreground=predicted.sum((1,2));intersection=(truth&predicted).sum((1,2))
    denominator=area+foreground
    dice=np.divide(2.*intersection,denominator,out=np.zeros(len(area),float),where=denominator!=0)
    for name,selected in [('small',(area>0)&(area<=1000)),('medium',(area>1000)&(area<=10000)),('large',area>10000)]:
        group=saved['groups'][name];mean=float(dice[selected].mean()) if selected.any() else None
        if group['slices']!=int(selected.sum()) or group['mean_slice_dice']!=mean:
            raise ValueError('Saved slice-area group statistics differ: '+name)
    empty=area==0
    checks=dict(empty_reference_slices=int(empty.sum()),empty_slices_with_prediction=int((empty&(foreground>0)).sum()),
        empty_false_positive_pixels=int(foreground[empty].sum()))
    if any(saved[k]!=v for k,v in checks.items()):raise ValueError('Saved empty-reference statistics differ')


def run(root):
    root=Path(root).resolve();destination=root/'returned_mask_audit.json'
    if destination.exists():raise FileExistsError('Preserve full-test mask audit')
    report=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    proof=json.loads((root/'verification.json').read_text(encoding='utf-8'))
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    if report['status']!='complete_all_27_frozen_pruning_test_cases' or proof['cases']!=27 or proof['slices']!=5286:
        raise ValueError('Incomplete all-case test')
    if proof['protocol_sha256']!=file_sha256(root/'protocol.json'):raise ValueError('Changed protocol')
    check_hashes({str(root/n):h for n,h in proof['outputs_sha256'].items()})
    check_hashes(protocol['input_sha256']);check_hashes(protocol['protected_development_sha256']);check_hashes(protocol['source_sha256'])
    expected={(r['cohort'],r['case'],r['slices']) for r in protocol['cases']}
    actual={(r['cohort'],r['case'],r['slices']) for r in report['cases']}
    if expected!=actual or len(report['cases'])!=27:raise ValueError('Missing, duplicate or extra cases')
    checked=0;newly_missed=[];recovered=[]
    cache=json.loads((REBUILT/'dataset.json').read_text(encoding='utf-8'));by_name={r['case']:r for r in cache['cases']}
    for row in report['cases']:
        name=row['case'];record=by_name[name];folder=root/name
        plans=json.loads((folder/'ct_only_envelope.json').read_text(encoding='utf-8'))['plans']
        with mapped_npy(REBUILT/record['cache_folder']/'reference_native.npy') as reference:
            envelope=np.zeros(reference.shape,bool)
            if len(plans)!=len(reference):raise ValueError('Missing body geometry')
            for z,plan in enumerate(plans):
                y0,y1,x0,x1=plan['box_y0_y1_x0_x1'];envelope[z,y0:y1,x0:x1]=True
            for model in MODEL_NAMES:
                path=folder/(model+'_native.npz')
                if file_sha256(path)!=row['mask_sha256'][path.name]:raise ValueError('Saved mask changed')
                with np.load(path) as arrays:
                    raw,final=arrays['raw'],arrays['final'];body=raw&envelope
                    if raw.dtype!=np.bool_ or final.dtype!=np.bool_ or raw.shape!=reference.shape or (final&~body).any():
                        raise ValueError('Invalid native mask or changed postprocessing semantics')
                    for stage,mask in [('raw',raw),('body',body),('final',final)]:
                        counts=independent_counts(mask,reference);saved=row['scores'][model+'_'+stage]['metrics']
                        if any(saved[k]!=v for k,v in counts.items()):raise ValueError('Saved-mask counts differ: '+name+'/'+model+'/'+stage)
                        check_slice_quality(mask,reference,row['scores'][model+'_'+stage]['slice_quality'])
                        checked+=1
            with np.load(folder/'original_fixed_native.npz') as a:old=a['final']
            with np.load(folder/'pruned_fixed_native.npz') as a:new=a['final']
            for change in paired_slices(old,new,reference):
                if change['newly_zero_overlap']:newly_missed.append(dict(cohort=row['cohort'],case=name,**change))
                if change['recovered_overlap']:recovered.append(dict(cohort=row['cohort'],case=name,**change))
        print('Independent saved-mask audit:',name,flush=True)
    if checked!=27*12 or report['cohorts']!=summaries(report['cases']):raise ValueError('Incomplete masks or wrong cohort summary')
    test_limits={cohort:dict(mean_dice_delta=data['pairs']['fixed_final']['mean_case_dice_delta'],
        within_user_one_percentage_point_mean_drop=data['pairs']['fixed_final']['mean_case_dice_delta']>=-.01,
        worst_case_dice_delta=data['pairs']['fixed_final']['worst_case_dice_delta'],
        cases_dropped_more_than_one_percentage_point=data['pairs']['fixed_final']['cases_dropped_more_than_one_percentage_point'])
        for cohort,data in report['cohorts'].items()}
    write_json(destination,dict(status='verified_all_27_frozen_test_cases_saved_masks',cases=27,slices=5286,
        independently_recomputed_mask_stages=checked,candidate_and_original_inputs_unchanged=True,
        evaluation_sha256=file_sha256(root/'evaluation.json'),protocol_sha256=file_sha256(root/'protocol.json'),
        test_limits=test_limits,newly_zero_overlap_slices=newly_missed,recovered_overlap_slices=recovered,
        source_sha256={str(Path(__file__).resolve()):file_sha256(__file__)},
        note='All saved raw/body/final TP/FP/FN/Dice, small/medium/large slice Dice and empty-reference errors checked; no model selection or retraining based on these results.'))
    print('All 27 cases / 5286 slices,',checked,'saved-mask stage checks PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='data/lung_prune_fulltest_20261017_v2')
    run(p.parse_args().root)
