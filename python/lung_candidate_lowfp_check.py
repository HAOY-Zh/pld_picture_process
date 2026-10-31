"""Freeze selection using only 24-scan reports, then check six cached CT results.

Reads native HU only for genuinely new local features at existing candidates.
Does not rerun U-Net, LoG/Hessian or access sealed heldout images. Six scans have
been viewed in earlier development: this is NOT an independent final test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
from lung_candidate_cached import load,ROOT,PRIOR
from lung_candidate_data import read_split,load_development_case,sha_file,write_json
from lung_candidate_rules_run import SPLIT,EXCLUDED
from lung_candidate_run import excluded_findings
from lung_candidate_froc import evaluate
from lung_candidate_svm import from_packet,decision_scores as old_score
from lung_candidate_lowfp_apply import extract_features,filter_features,decision_scores
from lung_candidate_svm_experiment import aggregate
from lung_candidates import point_metrics


def canonical_sha(model):
    return hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def run(out,searches,shell_features):
    if out.exists():raise FileExistsError('Preserve prior check; choose new output')
    rows,folds,protected=load();summaries=[]
    baseline_path=ROOT/'data/lung_candidate_svm_20261018_v2/development_oof.json'
    protected[str(baseline_path)]=sha_file(baseline_path)
    fp_ceiling=json.loads(baseline_path.read_text('utf-8'))['summary']['after']['fp_per_scan']
    for folder in searches:
        report_path=folder/'report.json';report=json.loads(report_path.read_text('utf-8'))
        protected[str(report_path)]=sha_file(report_path)
        for trial_report in report['trials']:
            if trial_report['high_recall']['after']['fp_per_scan']<=fp_ceiling:
                summaries.append((trial_report['froc']['cpm_interpolated'],folder,trial_report))
    # No check reports/images loaded above this line.
    _,winner,trial_report=max(summaries,key=lambda t:t[0]);out.mkdir(parents=True)
    source=winner/'selected_model.json';prototype=json.loads(source.read_text('utf-8'))
    winner_report=json.loads((winner/'report.json').read_text('utf-8'))
    if sha_file(source)!=winner_report['selected_model_sha256']:raise ValueError('Source model changed')
    if len(prototype['feature_names'])>20:
        meta_path=shell_features/'protocol.json';protected[str(meta_path)]=sha_file(meta_path)
        meta=json.loads(meta_path.read_text('utf-8'))
        by_uid={r['series_uid']:r for r in meta['cases']}
        for row in rows:
            path=shell_features/(row['uid']+'.npz')
            if sha_file(path)!=by_uid[row['uid']]['sha256']:raise ValueError('Shell feature changed')
            protected[str(path)]=sha_file(path)
            with np.load(path,allow_pickle=False) as a:
                np.testing.assert_array_equal(a['candidate_ids'],[c['candidate_id'] for c in row['packet']['candidates']])
                row['features_local']=np.column_stack((row['features_local'],a['features']))
    from lung_candidate_lowfp_experiment import training
    from lung_candidate_paper import fit as fit_paper
    from lung_candidate_lowfp import fit as fit_local
    names=prototype['feature_names'];x,y,w=training(rows,len(names))
    if prototype['kind']=='paper_candidate_ranker_v1':
        model=fit_paper(x,y,w,trial_report['parameters'],names,prototype['selection'])
    else:model=fit_local(x,y,w,trial_report['parameters'],names)
    trial=trial_report['trial']
    model.update(training_scans=[r['uid'] for r in rows],selected_trial=trial,formal_deployment=False,
        high_recall_threshold=trial_report['preserve_threshold'],
        operating_thresholds={str(p['fp_budget']):p['threshold'] for p in trial_report['froc']['operating_points']},
        threshold_scope='development OOF estimates; no guarantee on new scans')
    write_json(out/'model.json',model)
    selected_sha=sha_file(out/'model.json')
    fold_sources=winner/f'trial_{trial:02d}';(out/'fold_models').mkdir()
    for fold in folds:shutil.copyfile(fold_sources/f"fold_{fold['fold']}.json",out/'fold_models'/f"fold_{fold['fold']}.json")
    with np.load(fold_sources/'oof_scores.npz',allow_pickle=False) as a:dev_scores={k:a[k] for k in a.files}
    primary_path=PRIOR/'evaluation_scale2/balanced_rank_model.json'
    primary=json.loads(primary_path.read_text('utf-8'))
    contract=dict(primary_file_sha256=sha_file(primary_path),primary_canonical_sha256=canonical_sha(primary),
        model_file_sha256=selected_sha,model_canonical_sha256=canonical_sha(model),
        feature_names=model['feature_names'],web_default_changed=False,FPGA_implemented=False)
    write_json(out/'runtime_contract.json',contract)
    write_json(out/'selection_freeze.json',dict(selected_search=str(winner),selected_trial=trial,
        selection='highest 24-scan development OOF CPM with high-recall FP/scan no worse than previous SVM',
        objective_changed_after_exploratory_24_before_six=True,high_recall_fp_ceiling=fp_ceiling,
        comparisons=[dict(search=str(folder),trial=t['trial'],cpm=cpm) for cpm,folder,t in summaries],
        model_file_sha256=selected_sha,heldout_scans_run=0,checks_opened=False))
    snapshots=out/'executed_sources';snapshots.mkdir()
    for filename in ('lung_candidate_lowfp_check.py','lung_candidate_lowfp_apply.py',
                     'lung_candidate_lowfp.py','lung_candidate_paper.py','lung_candidate_shell.py',
                     'lung_candidate_cached.py','lung_candidate_froc.py'):
        p=ROOT/'python'/filename;protected[str(p)]=sha_file(p);(snapshots/filename).write_bytes(p.read_bytes())
    # Selection is frozen before loading the six previously viewed checks.
    split=read_split(SPLIT);records={r['series_uid']:r for r in split['cases']};ignored=excluded_findings(EXCLUDED)
    path=PRIOR/'new_development_check/report.json';protected[str(path)]=sha_file(path)
    uids=json.loads(path.read_text('utf-8'))['selected_uids']
    if set(uids)&{r['uid'] for r in rows} or len(uids)!=6:raise ValueError('Check cohort overlap')
    check=[];scores={};previous_scores={};timings=[];(out/'six_features').mkdir()
    old_path=ROOT/'data/lung_candidate_svm_20261018_v2/selected_svm.json'
    protected[str(old_path)]=sha_file(old_path);old_model=json.loads(old_path.read_text('utf-8'))
    for uid in uids:
        record=records[uid]
        if record['split']!='development':raise ValueError('Heldout refused')
        path=PRIOR/'new_development_check'/uid/'candidates.json';protected[str(path)]=sha_file(path)
        packet=json.loads(path.read_text('utf-8'))
        hu,geom=load_development_case(split,record)
        if geom!=packet['geometry']:raise ValueError('Candidate/CT geometry mismatch')
        start=perf_counter();x=extract_features(hu,packet,model['feature_names'])
        feature_ms=(perf_counter()-start)*1000;start=perf_counter();scores[uid]=decision_scores(x,model)
        score_ms=(perf_counter()-start)*1000
        ids=[c['candidate_id'] for c in packet['candidates']]
        np.savez_compressed(out/'six_features'/(uid+'.npz'),features=x,candidate_ids=np.array(ids))
        previous_scores[uid]=old_score(from_packet(packet['candidates'],geom['spacing_zyx_mm']),old_model)
        xyz=np.array([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1,3)
        check.append(dict(uid=uid,record=record,packet=packet,world=xyz,excluded=ignored.get(uid,[]),features=x,
            baseline=point_metrics(xyz,record['nodules'],ignored.get(uid,[]))))
        timings.append(dict(uid=uid,candidates=len(x),feature_ms=feature_ms,score_ms=score_ms))
        print('check',uid,'candidate patches',len(x),'feature ms',round(feature_ms,1),flush=True);del hu
    np.savez_compressed(out/'six_scores.npz',**scores)
    np.savez_compressed(out/'six_previous_scores.npz',**previous_scores)
    with np.load(ROOT/'data/lung_candidate_svm_20261018_v2/development_oof_scores.npz',allow_pickle=False) as a:
        old_dev={k:a[k] for k in a.files}
    dev_new=evaluate(rows,dev_scores);dev_old=evaluate(rows,old_dev)
    check_new=evaluate(check,scores);check_old=evaluate(check,previous_scores)
    profiles={};old_profiles={}
    for profile in ('1.0','4.0','8.0','high_recall'):
        threshold=model['high_recall_threshold'] if profile=='high_recall' else model['operating_thresholds'][profile]
        if threshold is None:continue
        profiles[profile]=dict(threshold=threshold,development_24=aggregate(rows,dev_scores,threshold)['summary'],
            six_check_fixed_threshold=aggregate(check,scores,threshold)['summary'])
        old_threshold=(old_model['threshold'] if profile=='high_recall' else next(
            p['threshold'] for p in dev_old['operating_points'] if str(p['fp_budget'])==profile))
        old_profiles[profile]=dict(threshold=old_threshold,six_check_fixed_threshold=
            aggregate(check,previous_scores,old_threshold)['summary'])
    fold_by_uid={uid:f['fold'] for f in folds for uid in f['validation_scans']}
    for cohort,source_rows,predictions in [('development_oof',rows,dev_scores),('six_check',check,scores)]:
        folder=out/cohort;folder.mkdir()
        for row in source_rows:
            if cohort=='development_oof':
                path=out/'fold_models'/f"fold_{fold_by_uid[row['uid']]}.json"
                actual_sha=sha_file(path);actual_model=json.loads(path.read_text('utf-8'))
                if row['uid'] in actual_model['training_scans']:raise ValueError('OOF training leakage')
            else:actual_sha=selected_sha
            # Two operating profiles, same center/boxes. No per-scan top-K.
            for profile in ('4.0','high_recall'):
                threshold=profiles[profile]['threshold'];candidates=[]
                for c,s in zip(row['packet']['candidates'],predictions[row['uid']]):
                    if s>=threshold:candidates.append(dict(c,primary_rule_score=c['score'],score=float(s),
                        score_kind='uncalibrated_local_classifier_margin'))
                packet=dict(row['packet'],candidates=candidates,algorithm_id='rule_v4_plus_local_classifier_experimental',
                    second_stage_model_sha256=actual_sha,second_stage_profile=profile,second_stage_threshold=threshold,
                    validation_scope=cohort+'; not final heldout or PL validation')
                write_json(folder/(row['uid']+'_'+profile+'.json'),packet)
    for path,digest in protected.items():
        if sha_file(path)!=digest:raise ValueError('Protected file changed: '+path)
    if selected_sha!=sha_file(out/'model.json'):raise ValueError('Model changed during checks')
    result=dict(status='COMPLETE_FROZEN_LOCAL_CLASSIFIER_CHECK',selected_search=str(winner),selected_trial=trial,
        parameters=model['parameters'],feature_count=len(model['feature_names']),model_bytes=(out/'model.json').stat().st_size,
        development_froc=dict(previous_svm=dev_old,new=dev_new),six_check_froc=dict(previous_svm=check_old,new=check_new),
        fixed_development_threshold_profiles=profiles,previous_fixed_profiles=old_profiles,timings=timings,
        heldout_scans_run=0,ct_volumes_read_for_new_local_features=6,network_forwards=0,log_or_hessian_extractions=0,
        primary_model_modified=False,web_default_modified=False,FPGA_implemented=False,
        protected_files_sha256=protected,model_frozen_before_six=True,
        limitations='24 scans used for model selection; six checks were already viewed in prior development. Curves sweep thresholds on each cohort; fixed-threshold results are separate. No top-K. Not independent final test or official LUNA16 CPM.')
    write_json(out/'report.json',result)
    print('COMPLETE',json.dumps(dict(parameters=result['parameters'],development_cpm=dev_new['cpm_interpolated'],
        six_cpm=check_new['cpm_interpolated'],profiles=profiles)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--search',type=Path,action='append',required=True)
    p.add_argument('--shell-features',type=Path,required=True);a=p.parse_args();run(a.outdir,a.search,a.shell_features)
