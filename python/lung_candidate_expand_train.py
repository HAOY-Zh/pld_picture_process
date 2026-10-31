"""Cached training72, frozen checks12+6 and all-development90 audit."""
import json
from pathlib import Path
import numpy as np
from lung_candidate_data import read_split,sha_file,write_json
from lung_candidate_cached import load as load_original,PRIOR
from lung_candidate_rules_run import ROOT,SPLIT,EXCLUDED,labels_for,reference_weights
from lung_candidate_run import excluded_findings
from lung_candidate_lowfp_experiment import training
from lung_candidate_paper import fit,score,select_features
from lung_candidate_froc import evaluate
from lung_candidate_svm_experiment import aggregate,preserve_threshold
from lung_candidate_svm import decision_scores as old_score
from lung_candidate_lowfp_apply import decision_scores as prior_score
from lung_candidates import point_metrics


def load_rows(out,uids):
    from lung_candidate_expand import verify,FEATURE_NAMES
    plan=verify(out);split=read_split(SPLIT);records={r['series_uid']:r for r in split['cases']}
    ignored=excluded_findings(EXCLUDED);original=set(plan['original_training_24']);old_check=set(plan['reused_development_6'])
    existing={r['uid']:r for r in load_original()[0]} if original&set(uids) else {}
    rows=[]
    for uid in uids:
        if records[uid]['split']!='development':raise ValueError('Heldout refused')
        if uid in original:
            row=existing[uid]
            with np.load(ROOT/'data/lung_candidate_lowfp_20261018/shell_features'/(uid+'.npz'),allow_pickle=False) as a:
                np.testing.assert_array_equal(a['candidate_ids'],[c['candidate_id'] for c in row['packet']['candidates']])
                x=np.column_stack((row['features_local'],a['features']))
        elif uid in old_check:
            packet=json.loads((PRIOR/'new_development_check'/uid/'candidates.json').read_text('utf-8'))
            with np.load(ROOT/'data/lung_candidate_lowfp_20261018/frozen_check/six_features'/(uid+'.npz'),allow_pickle=False) as a:
                np.testing.assert_array_equal(a['candidate_ids'],[c['candidate_id'] for c in packet['candidates']]);x=a['features']
            row=dict(uid=uid,packet=packet)
        else:
            folder=out/'cases'/uid;meta=json.loads((folder/'features_complete.json').read_text('utf-8'))
            for name,digest in meta['outputs_sha256'].items():
                if sha_file(folder/name)!=digest:raise ValueError('Expanded candidate cache changed')
            packet=json.loads((folder/'candidates.json').read_text('utf-8'))
            with np.load(folder/'features.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['candidate_ids'],[c['candidate_id'] for c in packet['candidates']]);x=a['features']
            row=dict(uid=uid,packet=packet)
        if x.shape!=(len(row['packet']['candidates']),len(FEATURE_NAMES)):raise ValueError('Wrong descriptor shape')
        row.update(features_local=x,features=x[:,:13],record=records[uid],excluded=ignored.get(uid,[]),
            world=np.array([c['center_lps_xyz_mm'] for c in row['packet']['candidates']]).reshape(-1,3))
        row['labels']=labels_for(row['world'],row['record'],row['excluded'])
        row['positive_weights']=reference_weights(row['world'],row['record'])
        row['baseline']=point_metrics(row['world'],row['record']['nodules'],row['excluded']);rows.append(row)
    return rows


def groups_for(rows):
    return np.concatenate([np.repeat(r['uid'],len(r['labels']))[r['labels']>=0] for r in rows])


def run_train(out):
    from lung_candidate_expand import verify,FEATURE_NAMES
    plan=verify(out);folder=out/'training'
    if folder.exists():raise FileExistsError('Preserve frozen training; new output required')
    rows=load_rows(out,plan['training_72']);by_uid={r['uid']:r for r in rows};folder.mkdir()
    datasets={}
    for fold in plan['folds']:
        if set(fold['training_scans'])&set(fold['validation_scans']):raise ValueError('Training fold overlap')
        subset=[by_uid[u] for u in fold['training_scans']];x,y,w=training(subset,len(FEATURE_NAMES))
        selection=select_features(x,y,groups_for(subset),w,FEATURE_NAMES)
        write_json(folder/f"selection_fold_{fold['fold']}.json",selection)
        datasets[fold['fold']]=(x,y,w,selection)
        print('TRAIN feature selection fold',fold['fold'],'candidates',len(y),'positives',int((y==1).sum()),flush=True)
    trials=[];best=None;best_scores=None
    for i,params in enumerate(plan['grid']):
        trial=folder/f'trial_{i:02d}';trial.mkdir();predictions={}
        for fold in plan['folds']:
            x,y,w,selection=datasets[fold['fold']];model=fit(x,y,w,params,FEATURE_NAMES,selection)
            model.update(training_scans=fold['training_scans'],fold=fold['fold'])
            write_json(trial/f"fold_{fold['fold']}.json",model)
            for uid in fold['validation_scans']:predictions[uid]=score(by_uid[uid]['features_local'],model)
        froc=evaluate(rows,predictions);threshold=preserve_threshold(rows,predictions)
        high=aggregate(rows,predictions,threshold)
        item=dict(trial=i,parameters=params,froc=froc,high_recall_threshold=threshold,high_recall=high['summary'])
        np.savez_compressed(trial/'oof_scores.npz',**predictions);write_json(trial/'report.json',item);trials.append(item)
        key=(froc['cpm_interpolated'],-high['summary']['after']['fp_per_scan'])
        if best is None or key>(best['froc']['cpm_interpolated'],-best['high_recall']['after']['fp_per_scan']):best=item;best_scores=predictions
        write_json(folder/'trials.json',dict(status='RUNNING',trials=trials))
        print('TRAIN trial',i,params,'CPM',round(froc['cpm_interpolated'],4),
            'sens@1/4/8',[(p['fp_budget'],round(p['sensitivity'],4)) for p in froc['operating_points'] if p['fp_budget'] in (1,4,8)],
            'preserve FP',round(high['summary']['after']['fp_per_scan'],3),flush=True)
    x,y,w=training(rows,len(FEATURE_NAMES));selection=select_features(x,y,groups_for(rows),w,FEATURE_NAMES)
    model=fit(x,y,w,best['parameters'],FEATURE_NAMES,selection)
    model.update(training_scans=plan['training_72'],selected_trial=best['trial'],formal_deployment=False,
        high_recall_threshold=best['high_recall_threshold'],
        operating_thresholds={str(p['fp_budget']):p['threshold'] for p in best['froc']['operating_points']},
        threshold_scope='training-72 OOF; fixed before fresh development 12 images/results')
    write_json(folder/'model.json',model);np.savez_compressed(folder/'selected_oof_scores.npz',**best_scores)
    write_json(folder/'report.json',dict(status='COMPLETE',selected=best,trials=trials,
        training_scans=len(rows),reference_nodules=sum(len(r['record']['nodules']) for r in rows),
        descriptor_candidates=sum(len(r['features_local']) for r in rows),model_bytes=(folder/'model.json').stat().st_size,
        support_vectors=len(model['svm']['support_vectors']),heldout_scans_run=0,
        validation_12_loaded=False,reused_6_used_for_training=False,
        limitation='OOF used for hyperparameter selection; training-development results, not unbiased final test'))
    verify(out)
    write_json(folder/'model_freeze.json',dict(model_sha256=sha_file(folder/'model.json'),
        plan_sha256=sha_file(out/'plan.json'),training_report_sha256=sha_file(folder/'report.json'),
        selected_oof_scores_sha256=sha_file(folder/'selected_oof_scores.npz'),
        fresh_validation_not_opened=True,heldout_scans_run=0))
    print('TRAIN COMPLETE; model frozen before fresh validation',flush=True)


def profiles(rows,predictions,model,previous=False):
    result={}
    for name in ('1.0','4.0','8.0','high_recall'):
        threshold=model.get('threshold') if previous and name=='high_recall' else (
            model['high_recall_threshold'] if name=='high_recall' else model['operating_thresholds'][name])
        if threshold is not None:result[name]=dict(threshold=threshold,metrics=aggregate(rows,predictions,threshold))
    return result


def run_evaluate(out):
    from lung_candidate_expand import verify,selected,PREVIOUS
    plan=verify(out);selected(out,'validation');folder=out/'evaluation'
    if folder.exists():raise FileExistsError('Preserve first expanded validation result')
    folder.mkdir();model=json.loads((out/'training/model.json').read_text('utf-8'))
    legacy_path=ROOT/'data/lung_candidate_svm_20261018_v2/selected_svm.json'
    legacy=json.loads(legacy_path.read_text('utf-8'));prior=json.loads((PREVIOUS/'model.json').read_text('utf-8'))
    # Old operating thresholds also come only from their original 24-scan OOF.
    old_froc=json.loads((PREVIOUS/'report.json').read_text('utf-8'))['development_froc']['previous_svm']
    legacy['operating_thresholds']={str(p['fp_budget']):p['threshold'] for p in old_froc['operating_points']}
    cohorts={};all_rows=[];all_scores={};all_legacy={};all_prior={}
    with np.load(out/'training/selected_oof_scores.npz',allow_pickle=False) as a:oof={k:a[k] for k in a.files}
    for name,key in [('training_72_oof','training_72'),('fresh_development_12','fresh_development_12'),('reused_development_6','reused_development_6')]:
        rows=load_rows(out,plan[key]);scores={};old={};prev={}
        for row in rows:
            uid=row['uid'];scores[uid]=oof[uid] if name=='training_72_oof' else score(row['features_local'],model)
            old[uid]=old_score(row['features'],legacy);prev[uid]=prior_score(row['features_local'],prior)
        # Avoid in-sample baselines on the original 24 training scans.
        old_24=set(plan['original_training_24'])&set(scores)
        if old_24:
            with np.load(ROOT/'data/lung_candidate_svm_20261018_v2/development_oof_scores.npz') as a:
                for uid in old_24:old[uid]=a[uid]
            old_selected=json.loads((PREVIOUS/'selection_freeze.json').read_text('utf-8'))
            path=Path(old_selected['selected_search'])/f"trial_{old_selected['selected_trial']:02d}"/'oof_scores.npz'
            with np.load(path) as a:
                for uid in old_24:prev[uid]=a[uid]
        cohorts[name]=dict(scans=len(rows),primary_detectable=aggregate(rows,{r['uid']:np.array([c['score'] for c in r['packet']['candidates']]) for r in rows},-1e30)['summary']['after'],
            froc=dict(old_svm=evaluate(rows,old),previous_paper=evaluate(rows,prev),expanded=evaluate(rows,scores)),
            fixed_profiles=dict(old_svm=profiles(rows,old,legacy,True),previous_paper=profiles(rows,prev,prior),expanded=profiles(rows,scores,model)))
        np.savez_compressed(folder/(name+'_scores.npz'),**scores)
        np.savez_compressed(folder/(name+'_old_scores.npz'),**old)
        np.savez_compressed(folder/(name+'_previous_paper_scores.npz'),**prev)
        all_rows+=rows;all_scores.update(scores);all_legacy.update(old);all_prior.update(prev)
        print('EVALUATE',name,'primary',cohorts[name]['primary_detectable'],'CPM',
            {k:round(v['cpm_interpolated'],4) for k,v in cohorts[name]['froc'].items()},flush=True)
    cohorts['all_development_90_descriptive']=dict(scans=len(all_rows),
        froc=dict(old_svm=evaluate(all_rows,all_legacy),previous_paper=evaluate(all_rows,all_prior),expanded=evaluate(all_rows,all_scores)),
        fixed_profiles=dict(old_svm=profiles(all_rows,all_legacy,legacy,True),previous_paper=profiles(all_rows,all_prior,prior),expanded=profiles(all_rows,all_scores,model)),
        interpretation='Mix of training OOF and development checks; not independent heldout test')
    write_json(folder/'report.json',dict(status='COMPLETE_ALL_DEVELOPMENT_90',cohorts=cohorts,
        training_72=plan['training_72'],fresh_12=plan['fresh_development_12'],reused_6=plan['reused_development_6'],
        expanded_model_sha256=sha_file(out/'training/model.json'),heldout_scans_run=0,new_cases_processed=60,
        prior_30_full_pipeline_reruns=0,old_ldct_unet_web_or_bit_modified=False,FPGA_validation=False,
        limitation='Fresh12 validation not used for fitting; previously inspected24+6 remain development. No official challenge or patient-level independence claim.'))
    verify(out)


def run_audit(out):
    from lung_candidate_expand import verify,selected
    from lung_candidate_lowfp_audit import check_froc
    plan=verify(out);selected(out,'validation');folder=out/'evaluation'
    report=json.loads((folder/'report.json').read_text('utf-8'));model=json.loads((out/'training/model.json').read_text('utf-8'))
    if set(model['training_scans'])!=set(plan['training_72']):raise ValueError('Wrong training identity')
    count=0;seen=set()
    for name,key in [('training_72_oof','training_72'),('fresh_development_12','fresh_development_12'),('reused_development_6','reused_development_6')]:
        rows=load_rows(out,plan[key])
        with np.load(folder/(name+'_scores.npz')) as a:scores={k:a[k] for k in a.files}
        check_froc(rows,scores,report['cohorts'][name]['froc']['expanded'])
        for row in rows:
            uid=row['uid'];actual=model
            if name=='training_72_oof':
                f=next(f for f in plan['folds'] if uid in f['validation_scans'])
                actual=json.loads((out/'training'/f"trial_{model['selected_trial']:02d}"/f"fold_{f['fold']}.json").read_text('utf-8'))
                inner=set(u for group in actual['selection']['inner_validation_scans'] for u in group)
                if inner!=set(actual['training_scans']):raise ValueError('Feature selection leakage')
            if uid in actual['training_scans']:raise ValueError('Evaluation scan trained in its scoring model')
            np.testing.assert_allclose(score(row['features_local'],actual),scores[uid],atol=1e-10,rtol=1e-10)
            seen.add(uid);count+=1
    if len(seen)!=90 or count!=90:raise ValueError('Incomplete development coverage')
    original=json.loads((PRIOR/'final_summary.json').read_text('utf-8'))
    for path,digest in original['current_source_sha256'].items():
        if sha_file(path)!=digest:raise ValueError('Original module changed')
    target=folder/'audit.json'
    if target.exists():raise FileExistsError('Preserve completed audit')
    write_json(target,dict(status='PASS',development_scans_checked=90,new_scans=60,
        previous_full_pipeline_reused=30,heldout_scans_run=0,model_frozen_before_fresh_12=True,
        per_scan_OOF_or_nontraining_scores_verified=True,FROC_independently_verified=True,
        original_candidate_sources_unchanged=len(original['current_source_sha256']),protected_sources_unchanged=len(plan['protected_files_sha256'])))
    print('AUDIT PASS: 90 development scans, 45 heldout untouched',flush=True)
