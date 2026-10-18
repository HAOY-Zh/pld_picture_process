# Standalone moderate-candidate adaptation; prior frozen runner remains unchanged.
"""One frozen all-case evaluation: 17 LCTSC hold-outs + 10 Coronacases HU volumes.

No training, calibration, model/threshold search, performance exclusions or board
timing. FP32 pairs and exact FPGA-integer-reference pairs are reported separately.
"""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import pydicom

from lung_body_gate import PARAMETERS as BODY_PARAMETERS,gate_volume
from lung_external import PRIMARY as EXTERNAL_PRIMARY,normalize_hu
from lung_fixed_network import FixedUNet,checked_contract
from lung_int8_final import score_mask,summarize_rows
from lung_island_cleanup import PRIMARY as ISLAND_PARAMETERS,clean_prediction
from lung_prune import (ORIGINAL,DATASET,WORKSPACE,boundary_metrics,restore_candidate)
from lung_segmentation import fresh_output,write_json
from lung_unet import (_case_prediction,file_sha256,load_cached_source,make_model,
    mapped_npy,read_manifest,torch_module)
from verify_lung_axil import check_hashes

HELDOUT=WORKSPACE/'data/lung_training_20261011/heldout_2d_evaluation'
EXTERNAL=WORKSPACE/'data/lung_external_20261012/prepared'
RECOVERY=WORKSPACE/'data/lung_prune_moderate_20261017'
CANDIDATE=RECOVERY/'moderate'
REBUILT=WORKSPACE/'data/lung_prune_test_cache_20261017'
ORIGINAL_CONTRACT=WORKSPACE/'data/lung_fpga_readiness_20261014/integer_contract_v3/contract.json'
PRIOR_TEST=WORKSPACE/'data/lung_prune_fulltest_20261017_v2'
MODEL_NAMES=['original_fp32','pruned_fp32','original_fixed','pruned_fixed']
STAGES=[name+'_'+stage for name in MODEL_NAMES for stage in ['raw','body','final']]


def postprocess(raw,envelope,spacing):
    """Exact shared CT-only envelope, then the unchanged frozen island cleanup."""
    if raw.dtype!=np.bool_ or envelope.dtype!=np.bool_ or raw.shape!=envelope.shape:
        raise ValueError('Expected matching native bool prediction and CT-only envelope')
    body=raw&envelope
    final,geometry=clean_prediction(body,spacing,**ISLAND_PARAMETERS)
    return body,final,geometry


def slice_counts(mask,reference):
    if mask.shape!=reference.shape or mask.ndim!=3:raise ValueError('Native grids differ')
    truth=np.asarray(reference)!=0;predicted=np.asarray(mask)!=0
    area=truth.sum((1,2));foreground=predicted.sum((1,2));tp=(truth&predicted).sum((1,2))
    return area,foreground,tp


def missed_summary(mask,reference):
    area,predicted,tp=slice_counts(mask,reference)
    return dict(positive_reference_slices=int((area>0).sum()),
        positive_slices_with_empty_prediction=int(((area>0)&(predicted==0)).sum()),
        positive_slices_with_zero_true_positive=int(((area>0)&(tp==0)).sum()))


def paired_slices(original,pruned,reference):
    area,old,old_tp=slice_counts(original,reference)
    _,new,new_tp=slice_counts(pruned,reference);rows=[]
    for z in range(len(area)):
        old_dice=2*old_tp[z]/(area[z]+old[z]) if area[z]+old[z] else None
        new_dice=2*new_tp[z]/(area[z]+new[z]) if area[z]+new[z] else None
        group='empty' if area[z]==0 else ('small' if area[z]<=1000 else ('medium' if area[z]<=10000 else 'large'))
        rows.append(dict(slice_index=z,area_group=group,expert_pixels=int(area[z]),
            original_prediction_pixels=int(old[z]),pruned_prediction_pixels=int(new[z]),
            original_true_positive_pixels=int(old_tp[z]),pruned_true_positive_pixels=int(new_tp[z]),
            original_dice=old_dice,pruned_dice=new_dice,
            dice_delta=new_dice-old_dice if old_dice is not None and new_dice is not None else None,
            newly_zero_overlap=bool(area[z]>0 and old_tp[z]>0 and new_tp[z]==0),
            recovered_overlap=bool(area[z]>0 and old_tp[z]==0 and new_tp[z]>0)))
    return rows


def summaries(rows):
    cohorts={}
    for cohort in ['lctsc_heldout','covid_external']:
        selected=[r for r in rows if r['cohort']==cohort]
        if not selected:continue
        summary=summarize_rows(selected,STAGES)
        pairs={}
        for kind in ['fp32','fixed']:
            for stage in ['raw','final']:
                a='original_'+kind+'_'+stage;b='pruned_'+kind+'_'+stage
                deltas=[r['scores'][b]['metrics']['dice']-r['scores'][a]['metrics']['dice'] for r in selected]
                old_small=summary[a]['slice_quality']['groups']['small']['mean_slice_dice']
                new_small=summary[b]['slice_quality']['groups']['small']['mean_slice_dice']
                small=new_small-old_small if old_small is not None and new_small is not None else None
                pairs[kind+'_'+stage]=dict(mean_case_dice_delta=float(np.mean(deltas)),worst_case_dice_delta=float(min(deltas)),
                    cases_improved=sum(d>0 for d in deltas),cases_worsened=sum(d<0 for d in deltas),
                    cases_dropped_more_than_one_percentage_point=sum(d<-.01 for d in deltas),
                    small_slice_mean_dice_delta=small,
                    original_empty_fp_pixels=summary[a]['slice_quality']['empty_false_positive_pixels'],
                    pruned_empty_fp_pixels=summary[b]['slice_quality']['empty_false_positive_pixels'])
        cohorts[cohort]=dict(cases=len(selected),slices=sum(r['slices'] for r in selected),summaries=summary,pairs=pairs)
    return cohorts


def write_case_table(output,rows):
    fields=['cohort','case','slices','original_fp32_raw','pruned_fp32_raw','original_fp32_final','pruned_fp32_final',
        'original_fixed_raw','pruned_fixed_raw','original_fixed_final','pruned_fixed_final','fixed_final_dice_delta',
        'original_small_slice_dice','pruned_small_slice_dice','original_empty_fp_pixels','pruned_empty_fp_pixels',
        'newly_zero_overlap_slices','recovered_overlap_slices']
    with (output/'case_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            a=row['scores']['original_fixed_final'];b=row['scores']['pruned_fixed_final']
            data={k:row[k] for k in ['cohort','case','slices']}
            data.update({k:row['scores'][k]['metrics']['dice'] for k in fields if k in row['scores']})
            data.update(fixed_final_dice_delta=b['metrics']['dice']-a['metrics']['dice'],
                original_small_slice_dice=a['slice_quality']['groups']['small']['mean_slice_dice'],
                pruned_small_slice_dice=b['slice_quality']['groups']['small']['mean_slice_dice'],
                original_empty_fp_pixels=a['slice_quality']['empty_false_positive_pixels'],
                pruned_empty_fp_pixels=b['slice_quality']['empty_false_positive_pixels'],
                newly_zero_overlap_slices=row['paired_slice_audit']['newly_zero_overlap_slices'],
                recovered_overlap_slices=row['paired_slice_audit']['recovered_overlap_slices'])
            writer.writerow(data)


def prepare(output):
    frozen=json.loads((RECOVERY/'freeze.json').read_text(encoding='utf-8'))
    if frozen['status']!='frozen_pruning_candidate_identity_verified':raise ValueError('Need frozen reviewed candidate')
    check_hashes(frozen['files_sha256']);check_hashes(frozen['source_sha256'])
    candidate_protocol=json.loads((RECOVERY/'protocol.json').read_text(encoding='utf-8'))
    check_hashes(candidate_protocol['input_sha256']);check_hashes(candidate_protocol['source_sha256'])
    held=json.loads((HELDOUT/'test_dataset.json').read_text(encoding='utf-8'))
    external=json.loads((EXTERNAL/'external_dataset.json').read_text(encoding='utf-8'))
    prior_external=json.loads((EXTERNAL/'protocol.json').read_text(encoding='utf-8'))
    _,development=read_manifest(DATASET)
    reconstructed=json.loads((REBUILT/'dataset.json').read_text(encoding='utf-8'))
    cache_proof=json.loads((REBUILT/'verification.json').read_text(encoding='utf-8'))
    if reconstructed['status']!='complete_hash_matched_reconstruction' or cache_proof['cases']!=27:
        raise ValueError('Need all historical test caches reconstructed and hash matched')
    check_hashes(cache_proof['source_sha256']);check_hashes(cache_proof['arrays_sha256'])
    cache_by_name={r['case']:r for r in reconstructed['cases']}
    if len(held['cases'])!=17 or sum(r['slices'] for r in held['cases'])!=2705 or external['primary_prepared']!=10:
        raise ValueError('Unexpected cached test coverage; inspect before inference')
    missing=[]
    for row in held['failures']:
        folder=Path('D:/CT2/lctsc')/row['case'];modalities=[]
        for path in sorted(folder.rglob('*.dcm')):
            ds=pydicom.dcmread(path,stop_before_pixels=True);modalities.append(str(ds.Modality))
        if 'CT' in modalities:raise ValueError('Previously missing CT has arrived; rebuild full audited cohort')
        missing.append(dict(case=row['case'],current_CT_files=modalities.count('CT'),current_RTSTRUCT_files=modalities.count('RTSTRUCT'),reason='No source CT'))
    all_dirs={p.name for p in Path('D:/CT2/lctsc').glob('LCTSC-Test-*') if p.is_dir()}
    if all_dirs!={r['case'] for r in held['cases']+held['failures']}:
        raise ValueError('New or missing test folders; update the all-case inventory explicitly')
    records=[];tracked=dict(frozen['files_sha256'])
    # Unchanged original predictions already have a complete independent audit.
    # Reuse those masks, but recompute every metric and postprocessing stage below.
    prior_report=json.loads((PRIOR_TEST/'evaluation.json').read_text(encoding='utf-8'))
    prior_audit=json.loads((PRIOR_TEST/'returned_mask_audit.json').read_text(encoding='utf-8'))
    if (prior_audit['status']!='verified_all_27_frozen_test_cases_saved_masks' or
        prior_audit['evaluation_sha256']!=file_sha256(PRIOR_TEST/'evaluation.json')):
        raise ValueError('Need independently audited original predictions')
    prior_protocol=json.loads((PRIOR_TEST/'protocol.json').read_text(encoding='utf-8'))
    if (prior_protocol['original_checkpoint_sha256']!=file_sha256(ORIGINAL) or
        prior_audit['protocol_sha256']!=file_sha256(PRIOR_TEST/'protocol.json')):
        raise ValueError('Original baseline identity changed')
    check_hashes(prior_protocol['source_sha256'])
    for row in prior_report['cases']:
        for model_name in ['original_fp32','original_fixed']:
            filename=model_name+'_native.npz';path=PRIOR_TEST/row['case']/filename
            if file_sha256(path)!=row['mask_sha256'][filename]:
                raise ValueError('Original baseline mask changed')
            tracked[str(path)]=row['mask_sha256'][filename]
    for name in ['evaluation.json','returned_mask_audit.json','protocol.json']:
        tracked[str(PRIOR_TEST/name)]=file_sha256(PRIOR_TEST/name)
    for row in held['cases']:
        if row['patient_key'] in {r['patient_key'] for r in development['cases']}:raise ValueError('Development patient overlap')
        restored=cache_by_name[row['case']]
        records.append(dict(row,cohort='lctsc_heldout',cache_root=str(REBUILT),cache_folder=restored['cache_folder'],
            spacing_zyx_mm=row['spacing_zyx_mm'],hu_cache_sha256=restored['hu_cache_sha256']))
    for row in external['cases']:
        if row['status']=='ready_primary_hu':
            records.append(dict(row,cohort='covid_external',cache_root=str(REBUILT),
                cache_folder=cache_by_name[row['case']]['cache_folder'],spacing_zyx_mm=row['reference_geometry']['spacing_zyx']))
    if {r['case'] for r in records if r['cohort']=='covid_external'}!=set(EXTERNAL_PRIMARY) or len(records)!=27 or sum(r['slices'] for r in records)!=5286:
        raise ValueError('Must evaluate all 27 cases / 5286 slices')
    for row in records:
        folder=Path(row['cache_root'])/row['cache_folder']
        for name,key in [('images.npy','images_sha256'),('reference_native.npy','reference_sha256')]:
            path=folder/name;digest=file_sha256(path)
            if digest!=row[key]:raise ValueError('Previously audited input/reference cache changed: '+str(path))
            tracked[str(path)]=digest
        path=folder/'hu_native.npy'
        if file_sha256(path)!=row['hu_cache_sha256']:raise ValueError('Native HU cache changed')
        tracked[str(path)]=row['hu_cache_sha256']
        if row['cohort']=='lctsc_heldout':
            for path in row['source_ct_files_sorted']:tracked[str(Path(path).resolve())]=file_sha256(path)
    for key,path in prior_external['source_paths'].items():
        if file_sha256(path)!=prior_external['source_sha256'][key]:raise ValueError('External original source changed: '+key)
        tracked[str(Path(path).resolve())]=prior_external['source_sha256'][key]
    extra=[ORIGINAL,DATASET,ORIGINAL_CONTRACT,ORIGINAL_CONTRACT.parent/'parameters.npz',
        RECOVERY/'freeze.json',RECOVERY/'review.json',HELDOUT/'test_dataset.json',HELDOUT/'protocol.json',
        EXTERNAL/'external_dataset.json',EXTERNAL/'protocol.json',WORKSPACE/'data/lung_prune_board_20261017/verification.json']
    extra.extend([REBUILT/'dataset.json',REBUILT/'verification.json',REBUILT/'protocol.json'])
    for path in extra:tracked[str(path)]=file_sha256(path)
    code=[Path(__file__).resolve()]+[WORKSPACE/'python'/n for n in ['lung_unet.py','lung_fixed_network.py',
        'lung_prune.py','lung_external.py','lung_quantize.py','lung_body_gate.py','lung_body_crop.py',
        'lung_island_cleanup.py','lung_int8_final.py','lung_segmentation.py','lungseg/dicom_io.py','lungseg/metrics.py']]
    protocol=dict(status='frozen_full_pruning_test_protocol',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        candidate='moderate',candidate_checkpoint_sha256=file_sha256(CANDIDATE/'best.pt'),
        candidate_contract_sha256=file_sha256(CANDIDATE/'contract.json'),original_checkpoint_sha256=file_sha256(ORIGINAL),
        expected_cases=27,expected_slices=5286,models=MODEL_NAMES,primary_comparison='original_fixed vs pruned_fixed',
        cases=[{k:r[k] for k in ['cohort','case','slices']} for r in records],
        missing_lctsc_cases=missing,excluded_windowed_external_cases=external['cohort_exclusions'],
        preprocessing='Unchanged HU[-1000,400], pixel-center linear 256, normalized FP16 cache',
        output='Sigmoid BEFORE native bilinear align_corners=False, threshold >=0.5',
        body_parameters=BODY_PARAMETERS,island_parameters=ISLAND_PARAMETERS,batch_size=8,
        reference='Unchanged native expert union masks; no label interpolation or edits',
        boundary_scope='All annotated native 2D slices for original/pruned FIXED final masks; not 3D Hausdorff',
        no_training=True,no_recalibration=True,no_threshold_or_postprocess_search=True,
        no_test_case_selection=True,previously_observed_original_test_data=True,
        original_prediction_reuse='Audited original masks; postprocessing and overlap metrics recomputed; original 2D boundaries reused after exact mask identity check',
        original_prediction_source=str(PRIOR_TEST),
        identity_limit=external['identity_warning'],
        scope='Frozen candidate confirmation on prior held-out/external cohorts; workstation numerical accuracy, not board performance',
        input_sha256=tracked,protected_development_sha256=candidate_protocol['input_sha256'],
        source_sha256={str(p):file_sha256(p) for p in code})
    write_json(output/'protocol.json',protocol)  # Before model inference / new test voxel scoring.
    write_json(output/'coverage.json',dict(available_cases=27,available_slices=5286,lctsc_folders_audited=24,
        lctsc_missing=missing,external_volumes_audited=20,external_windowed_exclusions=external['cohort_exclusions']))
    print('Protocol frozen: all 27 cases / 5286 slices; missing/excluded cohorts retained',flush=True)
    return records,protocol


def load_models():
    pruned,_=restore_candidate(CANDIDATE/'best.pt')
    candidate_contract=json.loads((CANDIDATE/'contract.json').read_text(encoding='utf-8'))
    with np.load(CANDIDATE/'parameters.npz') as p:pruned_fixed=FixedUNet(candidate_contract,p,'cuda')
    return [('original_fp32',None,None),('pruned_fp32',pruned.cuda(),'cuda'),
            ('original_fixed',None,None),('pruned_fixed',pruned_fixed,'cpu')]


def run(outdir):
    torch=torch_module();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise ValueError('Explicit CUDA accuracy run; no CPU fallback')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    output=fresh_output(outdir).resolve();report=dict(status='preparing',cases=[])
    try:
        records,protocol=prepare(output);models=load_models()
        original_rows={r['case']:r for r in json.loads((PRIOR_TEST/'evaluation.json').read_text(encoding='utf-8'))['cases']}
        report.update(status='evaluating_all_frozen_cases',expected_cases=27,expected_slices=5286,
            protocol_sha256=file_sha256(output/'protocol.json'),no_test_case_selection=True,
            scope=protocol['scope'],cohorts={})
        write_json(output/'evaluation.json',report)
        for index,record in enumerate(records,1):
            start=time.perf_counter();name=record['case'];root=Path(record['cache_root']);cache=root/record['cache_folder']
            folder=output/name;folder.mkdir();print(f'[{index}/27] {name}: ALL {record["slices"]} slices',flush=True)
            with mapped_npy(cache/'hu_native.npy') as a:hu=np.array(a)
            # Check cached normalized inputs against unchanged native HU before any predictions.
            expected=normalize_hu(hu,256)
            with mapped_npy(cache/'images.npy') as images:
                if not np.array_equal(images,expected):raise ValueError('Native HU no longer reproduces cached model input')
            del expected
            envelope,plans,gate_ms=gate_volume(hu,np.ones(hu.shape,dtype=bool),record['spacing_zyx_mm'][1:])
            write_json(folder/'ct_only_envelope.json',dict(plans=plans,source='CT HU only; no expert input',cpu_ms=gate_ms))
            del hu
            row=dict(cohort=record['cohort'],case=name,slices=record['slices'],scores={},boundaries={},missed={},
                numerical_inference_ms={},mask_sha256={},cleanup_geometry={})
            with mapped_npy(cache/'reference_native.npy') as reference:
                for model_name,model,device in models:
                    source_final=None
                    if model_name.startswith('original_'):
                        print('  '+model_name+' audited-mask reuse; recomputing metrics',flush=True)
                        with np.load(PRIOR_TEST/name/(model_name+'_native.npz')) as saved:
                            raw=saved['raw'];source_final=saved['final']
                        elapsed=None
                    else:
                        print('  '+model_name+' forward',flush=True)
                        raw,elapsed=_case_prediction(model,root,record,device,8)
                    body,final,geometry=postprocess(raw,envelope,record['spacing_zyx_mm'])
                    if source_final is not None and not np.array_equal(final,source_final):
                        raise ValueError('Original audited postprocessing no longer reproduces')
                    del source_final
                    for stage,mask in [('raw',raw),('body',body),('final',final)]:
                        row['scores'][model_name+'_'+stage]=score_mask(mask,reference)
                    row['missed'][model_name]=missed_summary(final,reference)
                    if model_name=='original_fixed':
                        row['boundaries'][model_name]=original_rows[name]['boundaries'][model_name]
                    elif model_name=='pruned_fixed':
                        row['boundaries'][model_name]=boundary_metrics(final,reference,record['spacing_zyx_mm'][1:])
                    path=folder/(model_name+'_native.npz')
                    with path.open('xb') as stream:np.savez_compressed(stream,raw=raw,final=final)
                    row['mask_sha256'][path.name]=file_sha256(path);row['numerical_inference_ms'][model_name]=elapsed
                    row['cleanup_geometry'][model_name]=geometry
                    print('    raw/final Dice {:.7f} / {:.7f}'.format(row['scores'][model_name+'_raw']['metrics']['dice'],
                        row['scores'][model_name+'_final']['metrics']['dice']),flush=True)
                    del raw,body,final
                with np.load(folder/'original_fixed_native.npz') as a:old=a['final']
                with np.load(folder/'pruned_fixed_native.npz') as a:new=a['final']
                slice_rows=paired_slices(old,new,reference)
                with (folder/'fixed_final_slice_metrics.csv').open('x',encoding='utf-8-sig',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(slice_rows[0]));writer.writeheader();writer.writerows(slice_rows)
                row['paired_slice_audit']=dict(newly_zero_overlap_slices=sum(r['newly_zero_overlap'] for r in slice_rows),
                    recovered_overlap_slices=sum(r['recovered_overlap'] for r in slice_rows),
                    newly_zero_overlap=[r for r in slice_rows if r['newly_zero_overlap']],
                    recovered_overlap=[r for r in slice_rows if r['recovered_overlap']],
                    original_fixed_positive_slices_with_zero_overlap=row['missed']['original_fixed']['positive_slices_with_zero_true_positive'],
                    pruned_fixed_positive_slices_with_zero_overlap=row['missed']['pruned_fixed']['positive_slices_with_zero_true_positive'])
                del old,new
            del envelope
            row['wall_seconds']=time.perf_counter()-start;write_json(folder/'evaluation.json',row)
            report['cases'].append(row);report['cohorts']=summaries(report['cases'])
            write_json(output/'evaluation.json',report);write_case_table(output,report['cases'])
            print('  COMPLETE',index,'/27; fixed final delta',row['scores']['pruned_fixed_final']['metrics']['dice']-
                row['scores']['original_fixed_final']['metrics']['dice'],'seconds',round(row['wall_seconds'],1),flush=True)
        if len(report['cases'])!=27 or sum(r['slices'] for r in report['cases'])!=5286:raise ValueError('Incomplete evaluation')
        check_hashes(protocol['input_sha256']);check_hashes(protocol['protected_development_sha256']);check_hashes(protocol['source_sha256'])
        report.update(status='complete_all_27_frozen_pruning_test_cases',evaluated_cases=27,evaluated_slices=5286,
            all_original_inputs_unchanged=True,candidate_unchanged=True,training_or_recalibration_performed=False,
            board_inference_performed=False,
            warning='Same previously observed cohorts: frozen-candidate confirmation, not a brand-new blind trial. No model changes after this run.')
        write_json(output/'evaluation.json',report)
        outputs={str(p.relative_to(output)):file_sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'verification.json',dict(status='complete_all_27_frozen_pruning_test_cases',
            cases=27,slices=5286,protocol_sha256=file_sha256(output/'protocol.json'),outputs_sha256=outputs,
            original_inputs_unchanged=True,candidate_unchanged=True,source_sha256=protocol['source_sha256'],
            scope=protocol['scope']))
        print('ALL 27 CASES COMPLETE',json.dumps(report['cohorts'],ensure_ascii=False),flush=True)
    except Exception as error:
        report['status']='aborted_incomplete';write_json(output/'evaluation.json',report)
        write_json(output/'failure.json',dict(status='aborted_incomplete',error=str(error),completed_cases=len(report['cases'])))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--outdir',default='data/lung_prune_moderate_fulltest_20261017')
    run(parser.parse_args().outdir)
