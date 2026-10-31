"""Use all 90 development scans while preserving the sealed 45-scan holdout.

Separate GPU-mask and CPU-descriptor commands avoid holding Torch and large
candidate arrays at the same time on a 16 GiB workstation. Completed caches
are identity checked and resumed rather than recomputed.
"""
import argparse
from dataclasses import asdict,replace
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from lung_candidate_data import read_split,load_development_case,sha_file,write_json,stratified_pick
from lung_candidate_rules_run import ROOT,SPLIT,EXCLUDED
from lung_candidate_cached import load,PRIOR,LOCAL_COLUMNS
from lung_candidate_lowfp import FEATURE_NAMES as BASE_NAMES
from lung_candidate_shell import FEATURE_NAMES as SHELL_NAMES,measure
from lung_candidate_svm import from_local_table
from lung_candidate_rules import RulesConfig,merge_regions,rule_scores
from lung_candidate_refined import _measure_with_bound
from lung_candidates import CandidateConfig,detect_candidates,point_metrics
from lung_candidate_structure import hessian_features

FEATURE_NAMES=BASE_NAMES+SHELL_NAMES
CHECKPOINT=ROOT/'data/lung_training_20261011/early_stop_run/best.pt'
PRIMARY=PRIOR/'evaluation_scale2/balanced_rank_model.json'
PREVIOUS=ROOT/'data/lung_candidate_lowfp_20261018/frozen_check'


def verify(out):
    plan_path=out/'plan.json'
    if sha_file(plan_path)!=(out/'plan.sha256').read_text().strip():raise ValueError('Expansion plan changed')
    plan=json.loads(plan_path.read_text('utf-8'))
    for path,digest in plan['protected_files_sha256'].items():
        if sha_file(path)!=digest:raise ValueError('Protected input/source changed: '+path)
    return plan


def freeze(out):
    if out.exists():raise FileExistsError('Preserve previous expansion')
    split=read_split(SPLIT)
    initial=json.loads((PRIOR/'features_scale2/protocol.json').read_text('utf-8'))
    previous_check=json.loads((PRIOR/'new_development_check/report.json').read_text('utf-8'))
    old_train=[c['series_uid'] for c in initial['cases']];old_check=previous_check['selected_uids']
    remaining=[c for c in split['cases'] if c['split']=='development' and c['series_uid'] not in set(old_train+old_check)]
    if len(remaining)!=60:raise ValueError('Need exactly 60 unused development scans')
    seed=2026101802
    fresh=set()
    for positive,count in ((True,8),(False,4)):
        pool=[r for r in remaining if bool(r['nodules'])==positive]
        fresh|=stratified_pick(pool,count,seed,set())
    added=sorted(r['series_uid'] for r in remaining if r['series_uid'] not in fresh)
    training=old_train+added
    if len(training)!=72 or len(fresh)!=12 or set(training)&fresh:raise ValueError('Invalid development allocation')
    records={r['series_uid']:r for r in split['cases']}
    # Fixed four folds, distribute positive/negative scans before inference.
    buckets=[[] for _ in range(4)]
    for positive in (True,False):
        group=[u for u in training if bool(records[u]['nodules'])==positive]
        group.sort(key=lambda u:hashlib.sha256(f'expand-fold:{seed}:{u}'.encode()).hexdigest())
        for i,uid in enumerate(group):buckets[i%4].append(uid)
    folds=[dict(fold=i,validation_scans=sorted(v),training_scans=sorted(set(training)-set(v))) for i,v in enumerate(buckets)]
    protected={str(p):sha_file(p) for p in (SPLIT,EXCLUDED,CHECKPOINT,PRIMARY,PREVIOUS/'model.json',
        ROOT/'data/lung_candidate_svm_20261018_v2/selected_svm.json',
        PRIOR/'features_scale2/protocol.json',PRIOR/'new_development_check/report.json')}
    for uid in old_train:
        for p in (PRIOR/'features_scale2'/(uid+'.npz'),PRIOR/'oof_comparisons'/uid/'candidates.json',
                  ROOT/'data/lung_candidate_lowfp_20261018/shell_features'/(uid+'.npz')):
            protected[str(p)]=sha_file(p)
    for uid in old_check:
        for p in (PRIOR/'new_development_check'/uid/'candidates.json',PREVIOUS/'six_features'/(uid+'.npz')):
            protected[str(p)]=sha_file(p)
    sources=['lung_candidate_expand.py','lung_candidate_expand_train.py','lung_candidates.py','lung_candidate_structure.py',
        'lung_candidate_rules.py','lung_candidate_refined.py','lung_candidate_shell.py','lung_candidate_paper.py',
        'lung_candidate_lowfp.py','lung_candidate_cached.py','lung_candidate_froc.py','lung_candidate_svm.py',
        'lung_candidate_data.py','lung_candidate_run.py','lung_body_gate.py','lung_unet.py']
    out.mkdir(parents=True);snapshots=out/'executed_sources';snapshots.mkdir()
    for name in sources:
        p=ROOT/'python'/name
        protected[str(p)]=sha_file(p);(snapshots/name).write_bytes(p.read_bytes())
    summary={}
    for name,uids in [('training_72',training),('fresh_development_12',sorted(fresh)),('reused_development_6',old_check)]:
        summary[name]=dict(scans=len(uids),reference_nodules=sum(len(records[u]['nodules']) for u in uids),
            with_reference=sum(bool(records[u]['nodules']) for u in uids))
    plan=dict(version='expanded_development_v1',seed=seed,original_split_sha256=sha_file(SPLIT),
        original_training_24=old_train,added_training_48=added,training_72=training,
        fresh_development_12=sorted(fresh),reused_development_6=old_check,folds=folds,summary=summary,
        protected_files_sha256=protected,heldout_scans_run=0,
        fresh_check_policy='do not generate fresh 12 images/features or inspect results before expanded model freeze',
        previous_cases='reuse existing 24+6 descriptors; no repeated CNN or LoG',
        method='unchanged rule v4 primary + 41 descriptors + RSFS-inspired 16-feature weighted SVM',
        grid=[dict(C=c,negative_ratio=ratio) for ratio in (None,10,20) for c in (.1,1.,10.)],
        selection='maximum training-72 OOF CPM; tie smaller high-recall FP then fixed grid order',
        thresholds='training OOF at 1/4/8 FP per scan and retaining all primary-detected references',
        exact_paper_reproduction=False,patient_independence_proven=False)
    write_json(out/'plan.json',plan);(out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def selected(out,cohort):
    plan=verify(out)
    key='added_training_48' if cohort=='training' else 'fresh_development_12'
    if cohort=='validation':
        frozen=out/'training/model_freeze.json'
        if not frozen.exists():raise ValueError('Freeze expanded model before fresh validation access')
        item=json.loads(frozen.read_text('utf-8'))
        if sha_file(out/'training/model.json')!=item['model_sha256']:raise ValueError('Expanded model changed')
    split=read_split(SPLIT);records={r['series_uid']:r for r in split['cases']}
    if any(records[u]['split']!='development' for u in plan[key]):raise ValueError('Heldout refused')
    return plan,split,[records[u] for u in plan[key]]


def masks(out,cohort):
    from lung_unet import torch_module,make_model,select_device
    from lung_candidate_run import predict_lungs
    plan,split,records=selected(out,cohort);folder=out/'cases';folder.mkdir(exist_ok=True)
    missing=[]
    for record in records:
        case=folder/record['series_uid'];meta=case/'mask.json'
        if meta.exists():
            item=json.loads(meta.read_text('utf-8'))
            if item['input_raw_sha256']!=record['raw_sha256'] or sha_file(case/'lung_mask.npz')!=item['packed_sha256']:
                raise ValueError('Changed cached mask')
        else:missing.append(record)
    if not missing:print('All mask caches already complete',flush=True);return
    torch=torch_module();torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=select_device('cuda');cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    if cp['image_size']!=256 or cp.get('context_slices',1)!=1 or cp['epoch']!=12 or cp.get('threshold',.5)!=.5:
        raise ValueError('Need unchanged original FP32 lung model')
    model=make_model(cp['base_channels'],1).to(device).eval();model.load_state_dict(cp['model'])
    for i,r in enumerate(missing,1):
        uid=r['series_uid'];start=perf_counter();hu,geometry=load_development_case(split,r)
        raw,body,elapsed=predict_lungs(hu,geometry,model,device,16);del raw
        case=folder/uid;case.mkdir(exist_ok=True);path=case/'lung_mask.npz'
        np.savez_compressed(path,shape_zyx=np.array(hu.shape),lung_body=np.packbits(body))
        write_json(case/'mask.json',dict(series_uid=uid,input_raw_sha256=r['raw_sha256'],geometry=geometry,
            lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),packed_sha256=sha_file(path),
            checkpoint_sha256=plan['protected_files_sha256'][str(CHECKPOINT)],lung_ms=elapsed,
            elapsed_ms=(perf_counter()-start)*1000,backend='unchanged original FP32 CUDA; not board INT8'))
        print(f'masks {i}/{len(missing)} {uid} slices={len(hu)} seconds={perf_counter()-start:.2f}',flush=True)
        del hu,body;gc.collect()
    verify(out)


def memory_bounded_primary(hu,body,geometry,primary):
    """Same v4 formulas/order; discard full seed arrays before Hessian passes."""
    cfg=RulesConfig(**primary['rules_config']);params=CandidateConfig();mode=primary['center_mode']
    start=perf_counter();stages={};coarse,seed=detect_candidates(hu,body,geometry,params);del seed
    stages['primary_log_seed']=coarse['timings_ms']['seed']
    stages['primary_regions']=coarse['timings_ms']['postprocess'];watch=perf_counter()
    hessian=hessian_features(hu,coarse['candidates'],geometry['spacing_zyx_mm'])
    stages['primary_hessian']=(perf_counter()-watch)*1000;watch=perf_counter()
    table,pool,skipped=_measure_with_bound(hu,coarse['candidates'],hessian,geometry,cfg,primary['threshold'],True)
    stages['primary_local_measurements']=(perf_counter()-watch)*1000
    total_pool=len(pool);original_count=len(coarse['candidates']);complement=primary['complementary_sigma_mm']
    single,single_seed=detect_candidates(hu,body,geometry,replace(params,sigma_mm=(float(complement),)));del single_seed
    stages['complementary_log_seed']=single['timings_ms']['seed']
    stages['complementary_regions']=single['timings_ms']['postprocess'];watch=perf_counter()
    existing={tuple(coarse['candidates'][i]['center_index_zyx']) for i in pool
        if np.isclose(hessian[i]['sigma_mm'],complement,atol=1e-5)}
    extra=[dict(c,candidate_id='scale2_'+c['candidate_id']) for c in single['candidates']
        if tuple(c['center_index_zyx']) not in existing];del single
    extra_features=hessian_features(hu,extra,geometry['spacing_zyx_mm'])
    stages['complementary_hessian_and_identity']=(perf_counter()-watch)*1000;watch=perf_counter()
    added,extra_pool,extra_skipped=_measure_with_bound(hu,extra,extra_features,geometry,cfg,primary['threshold'],True)
    stages['complementary_local_measurements']=(perf_counter()-watch)*1000;watch=perf_counter()
    total_pool+=len(extra_pool);skipped+=extra_skipped;added['source_indices']+=original_count
    table={k:np.concatenate((table[k],added[k]),axis=0) for k in table};del added
    coarse['candidates'].extend(extra);hessian.extend(extra_features)
    scores=rule_scores(table['features'].astype(float));indices=np.flatnonzero(scores>=primary['threshold'])
    indices=indices[merge_regions(table[mode][indices],table['boxes_zyx'][indices],scores[indices],geometry['spacing_zyx_mm'],cfg)]
    candidates=[]
    for j,k in enumerate(indices):
        source=int(table['source_indices'][k]);candidate=dict(coarse['candidates'][source]);point=table[mode][k]
        candidate.update(candidate_id=f'refined_{j+1:05d}',source_candidate_id=candidate['candidate_id'],
            original_peak_index_zyx=candidate['center_index_zyx'],center_index_zyx=point.tolist(),
            center_lps_xyz_mm=(np.array(geometry['origin_lps_mm'])+np.array(geometry['basis_lps_zyx'])@point).tolist(),
            bbox_zyx=table['boxes_zyx'][k].tolist(),score=float(scores[k]),
            score_kind='local_feature_rank_not_calibrated_probability',structure_features=hessian[source])
        candidates.append(candidate)
    x=from_local_table(table,indices,scores[indices],geometry['spacing_zyx_mm'])
    local=np.column_stack((x,table['features'][indices][:,LOCAL_COLUMNS]))
    stages['ranking_merge_packet_and_local_encoding']=(perf_counter()-watch)*1000;watch=perf_counter()
    shell=np.array([measure(hu,c['center_index_zyx'],geometry['spacing_zyx_mm'],np.expm1(a[3]))
        for c,a in zip(candidates,x)],np.float32).reshape(-1,len(SHELL_NAMES))
    stages['shell_descriptors']=(perf_counter()-watch)*1000
    result=dict(schema_version='lung_candidate_result_v1',algorithm_id='lung_candidate_refined_v4_experimental',
        backend='CPU_REFERENCE',status='success',source_shape_zyx=list(hu.shape),geometry=geometry,candidates=candidates,
        parameters=coarse['parameters'],rules_parameters=asdict(cfg),rank_model_kind='rule',rank_threshold=primary['threshold'],
        center_mode=mode,coarse_candidate_count=len(coarse['candidates']),broad_pool_count=total_pool,
        local_feature_candidates_measured=len(scores),local_feature_candidates_skipped_by_bound=skipped,
        safe_rule_bound_enabled=True,original_coarse_count=original_count,complementary_sigma_mm=complement,
        rank_model_sha256=hashlib.sha256(json.dumps(primary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        timings_ms=dict(candidate_and_shell_total=(perf_counter()-start)*1000),
        stage_timings_ms=stages,
        execution='serial volume; original v4 seed arrays discarded after finalization')
    return result,np.column_stack((local,shell))


def features(out,cohort):
    plan,split,records=selected(out,cohort);primary=json.loads(PRIMARY.read_text('utf-8'))
    for i,r in enumerate(records,1):
        uid=r['series_uid'];case=out/'cases'/uid;complete=case/'features_complete.json'
        if complete.exists():
            item=json.loads(complete.read_text('utf-8'))
            for name,digest in item['outputs_sha256'].items():
                if sha_file(case/name)!=digest:raise ValueError('Corrupt completed case cache')
            print(f'features {i}/{len(records)} cached {uid}',flush=True);continue
        meta=json.loads((case/'mask.json').read_text('utf-8'))
        if sha_file(case/'lung_mask.npz')!=meta['packed_sha256']:raise ValueError('Mask cache changed')
        start=perf_counter();hu,geometry=load_development_case(split,r)
        with np.load(case/'lung_mask.npz',allow_pickle=False) as a:
            if tuple(a['shape_zyx'])!=hu.shape:raise ValueError('Mask/native HU shape mismatch')
            body=np.unpackbits(a['lung_body'],count=hu.size).reshape(hu.shape).astype(bool)
        if hashlib.sha256(body.tobytes()).hexdigest()!=meta['lung_mask_sha256']:raise ValueError('Mask bytes changed')
        print(f'features {i}/{len(records)} START {uid} slices={len(hu)}',flush=True)
        result,x=memory_bounded_primary(hu,body,geometry,primary)
        result.update(series_uid=uid,input_raw_sha256=r['raw_sha256'],lung_mask_sha256=meta['lung_mask_sha256'])
        write_json(case/'candidates.json',result)
        np.savez_compressed(case/'features.npz',features=x,
            candidate_ids=np.array([c['candidate_id'] for c in result['candidates']]))
        item=dict(series_uid=uid,candidates=len(x),feature_names=FEATURE_NAMES,elapsed_ms=(perf_counter()-start)*1000,
            input_raw_sha256=r['raw_sha256'],outputs_sha256={name:sha_file(case/name) for name in ('candidates.json','features.npz','mask.json','lung_mask.npz')})
        write_json(complete,item)
        # References deliberately unused in feature generation.
        print(f'features {i}/{len(records)} DONE {uid} candidates={len(x)} seconds={perf_counter()-start:.2f}',flush=True)
        del hu,body,result,x;gc.collect()
    verify(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['plan','masks','features','train','evaluate','audit'])
    p.add_argument('--outdir',type=Path,required=True);p.add_argument('--cohort',choices=['training','validation'],default='training')
    a=p.parse_args()
    if a.command=='plan':freeze(a.outdir)
    elif a.command=='masks':masks(a.outdir,a.cohort)
    elif a.command=='features':features(a.outdir,a.cohort)
    else:
        from lung_candidate_expand_train import run_train,run_evaluate,run_audit
        {'train':run_train,'evaluate':run_evaluate,'audit':run_audit}[a.command](a.outdir)
