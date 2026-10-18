"""Review all development cases including 2D boundaries and empty/small slices."""
import argparse
import json
from pathlib import Path

import numpy as np

from lung_prune import BASELINE, DATASET, boundary_metrics
from lung_segmentation import write_json
from lung_unet import file_sha256, mapped_npy, read_manifest
from verify_lung_axil import check_hashes


def run(root,baseline_review=None):
    root=Path(root).resolve();destination=root/'review.json'
    if destination.exists():raise FileExistsError('Preserve quality review')
    complete=json.loads((root/'verification.json').read_text())
    if complete['status']!='complete_pruning_development_experiment':raise ValueError('Need completed frozen experiment')
    protocol=json.loads((root/'protocol.json').read_text());check_hashes(protocol['input_sha256']);check_hashes(protocol['source_sha256'])
    cache_root,manifest=read_manifest(DATASET);baseline=json.loads((BASELINE/'evaluation.json').read_text())
    original_boundaries={};tracked={str(BASELINE/'evaluation.json'):file_sha256(BASELINE/'evaluation.json')}
    if baseline_review is not None:
        previous_path=Path(baseline_review).resolve();previous=json.loads(previous_path.read_text())
        if previous['status']!='complete_pruning_boundary_and_failure_review':raise ValueError('Wrong previous boundary review')
        check_hashes(previous['input_sha256'])
        original_boundaries={r['case']:r['baseline_boundary'] for r in previous['reviews'][0]['cases']}
        tracked[str(previous_path)]=file_sha256(previous_path)
    for record in [r for r in manifest['cases'] if r['split']=='validation']:
        name=record['case'];path=BASELINE/name/'predictions_native.npz'
        expected=next(r for r in baseline['cases'] if r['case']==name)['prediction_sha256']
        if file_sha256(path)!=expected:raise ValueError('Baseline mask changed')
        tracked[str(path)]=expected
        if name not in original_boundaries:
            print('Review original boundaries:',name,flush=True)
            with np.load(path) as arrays,mapped_npy(cache_root/record['cache_folder']/'reference_native.npy') as reference:
                original_boundaries[name]=boundary_metrics(arrays['fixed_final'],reference,record['spacing_zyx_mm'][1:])
    reviews=[]
    for result in complete['results']:
        name=result['candidate'];path=root/name/'fixed_evaluation.json';report=json.loads(path.read_text());rows=[]
        tracked[str(path)]=file_sha256(path)
        for row in report['cases']:
            old=next(r for r in baseline['cases'] if r['case']==row['case']);a=row['scores']['fixed_final'];b=old['scores']['fixed_final']
            old_boundary=original_boundaries[row['case']]
            small=a['slice_quality']['groups']['small']['mean_slice_dice'];old_small=b['slice_quality']['groups']['small']['mean_slice_dice']
            rows.append(dict(case=row['case'],dice_delta=a['metrics']['dice']-b['metrics']['dice'],
                recall_delta=a['metrics']['recall']-b['metrics']['recall'],
                false_negative_voxel_delta=a['metrics']['false_negative']-b['metrics']['false_negative'],
                small_slice_mean_dice_delta=small-old_small if small is not None and old_small is not None else None,
                empty_false_positive_pixels=a['slice_quality']['empty_false_positive_pixels'],
                baseline_empty_false_positive_pixels=b['slice_quality']['empty_false_positive_pixels'],
                boundary=row['boundary'],baseline_boundary=old_boundary,
                mean_boundary_delta_mm=row['boundary']['mean_symmetric_boundary_mm']-old_boundary['mean_symmetric_boundary_mm']))
        current=report['summaries']['fixed_final'];previous=baseline['summaries']['fixed_final']
        small_delta=current['slice_quality']['groups']['small']['mean_slice_dice']-previous['slice_quality']['groups']['small']['mean_slice_dice']
        # Conservative extra review gates; mean quality alone must not hide failures.
        boundary_drop=max(r['mean_boundary_delta_mm'] for r in rows)
        worst_drop=min(r['dice_delta'] for r in rows)
        empty=current['slice_quality']['empty_false_positive_pixels'];old_empty=previous['slice_quality']['empty_false_positive_pixels']
        passed=bool(result['passes_mean_and_work_screen'] and worst_drop>=-.01 and small_delta>=-.03
            and boundary_drop<=1. and empty<=old_empty+1000
            and all(r['boundary']['annotated_slices_with_empty_prediction']<=r['baseline_boundary']['annotated_slices_with_empty_prediction'] for r in rows))
        reviews.append(dict(candidate=name,passes_conservative_development_review=passed,
            mean_dice_drop=result['mean_dice_drop'],worst_case_dice_delta=worst_drop,small_slice_mean_dice_delta=small_delta,
            maximum_case_mean_boundary_increase_mm=boundary_drop,empty_false_positive_pixels=empty,
            baseline_empty_false_positive_pixels=old_empty,cases=rows))
    eligible=[r for r in complete['results'] if next(v for v in reviews if v['candidate']==r['candidate'])['passes_conservative_development_review']]
    selected=max(eligible,key=lambda r:r['padded_work_reduction'])['candidate'] if eligible else None
    write_json(destination,dict(status='complete_pruning_boundary_and_failure_review',selected_for_board_experiment=selected,
        reviews=reviews,input_sha256=tracked,
        extra_conservative_gates=dict(max_each_case_dice_drop=.01,max_small_slice_dice_drop=.03,
            max_case_mean_2d_boundary_increase_mm=1.,max_extra_empty_false_positive_pixels=1000,
            no_extra_annotated_slices_with_empty_prediction=True),
        note='Extra engineering review gates added after the fixed training schedule; not independent clinical acceptance.',
        test_or_covid_used=False,board_tested=False))
    print('Reviewed candidate for board experiment:',selected,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='data/lung_prune_20261017')
    parser.add_argument('--baseline-review');args=parser.parse_args();run(args.root,args.baseline_review)
