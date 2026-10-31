"""Independent score/packet/FROC audit of a completed frozen local-ranker check."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve
from lung_candidate_cached import load,ROOT,PRIOR
from lung_candidate_data import read_split,sha_file,write_json
from lung_candidate_rules_run import SPLIT,EXCLUDED,labels_for
from lung_candidate_run import excluded_findings
from lung_candidate_lowfp_apply import decision_scores,filter_features
from lung_candidate_froc import evaluate,CPM_POINTS
from lung_candidates import point_metrics


def check_froc(rows,scores,expected):
    positive=[];negative=[];references=0
    for row in rows:
        prediction=scores[row['uid']]
        metric=point_metrics(row['world'],row['record']['nodules'],row['excluded'])
        for detail in metric['reference_details']:
            references+=1;indices=detail['matching_candidate_indices']
            if indices:positive.append(float(prediction[indices].max()))
        # Label 0 excludes all duplicate-reference hits and excluded findings.
        negative.extend(prediction[labels_for(row['world'],row['record'],row['excluded'])==0].tolist())
    fpr,tpr,_=roc_curve([1]*len(positive)+[0]*len(negative),positive+negative,drop_intermediate=False)
    official_style=np.interp(CPM_POINTS,fpr*len(negative)/len(rows),tpr*len(positive)/references)
    np.testing.assert_allclose(official_style,[p['interpolated_sensitivity'] for p in expected['operating_points']],atol=1e-12)
    # Recompute every deployable empirical operating point through a separate
    # direct thresholding/matching path, not through the curve construction.
    for point in expected['operating_points']:
        detected=fp=0
        for row in rows:
            keep=(scores[row['uid']]>=point['threshold']) if point['threshold'] is not None else np.zeros(len(row['world']),bool)
            m=point_metrics(row['world'][keep],row['record']['nodules'],row['excluded'])
            detected+=m['detected'];fp+=m['false_positive_candidates']
        if detected!=point['detected'] or fp/len(rows)!=point['actual_fp_per_scan']:raise AssertionError('Empirical FROC mismatch')


def run(out,shell_features):
    destination=out/'output_audit.json'
    if destination.exists():raise FileExistsError('Preserve previous audit')
    report=json.loads((out/'report.json').read_text('utf-8'))
    for path,digest in report['protected_files_sha256'].items():
        if sha_file(path)!=digest:raise ValueError('Protected source/cache changed: '+path)
    rows,folds,_=load();model=json.loads((out/'model.json').read_text('utf-8'))
    freeze=json.loads((out/'selection_freeze.json').read_text('utf-8'))
    if sha_file(out/'model.json')!=freeze['model_file_sha256']:raise ValueError('Model mutated after freeze')
    for row in rows:
        if len(model['feature_names'])>20:
            with np.load(shell_features/(row['uid']+'.npz'),allow_pickle=False) as a:
                row['features_local']=np.column_stack((row['features_local'],a['features']))
    split=read_split(SPLIT);records={r['series_uid']:r for r in split['cases']};ignored=excluded_findings(EXCLUDED)
    with np.load(out/'six_scores.npz',allow_pickle=False) as a:six_scores={k:a[k] for k in a.files}
    check=[]
    for uid in six_scores:
        if records[uid]['split']!='development' or uid in model['training_scans']:raise ValueError('Check leakage')
        packet=json.loads((PRIOR/'new_development_check'/uid/'candidates.json').read_text('utf-8'))
        with np.load(out/'six_features'/(uid+'.npz'),allow_pickle=False) as a:
            np.testing.assert_array_equal(a['candidate_ids'],[c['candidate_id'] for c in packet['candidates']]);features=a['features']
        np.testing.assert_allclose(decision_scores(features,model),six_scores[uid],atol=1e-10,rtol=1e-10)
        check.append(dict(uid=uid,record=records[uid],excluded=ignored.get(uid,[]),features=features,
            packet=packet,world=np.array([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1,3)))
    source=Path(freeze['selected_search'])/f"trial_{freeze['selected_trial']:02d}"/'oof_scores.npz'
    with np.load(source,allow_pickle=False) as a:dev_scores={k:a[k] for k in a.files}
    fold_by_uid={uid:f['fold'] for f in folds for uid in f['validation_scans']}
    packets=0
    for cohort,items,scores in [('development_oof',rows,dev_scores),('six_check',check,six_scores)]:
        for row in items:
            actual=model;actual_sha=sha_file(out/'model.json')
            if cohort=='development_oof':
                p=out/'fold_models'/f"fold_{fold_by_uid[row['uid']]}.json";actual=json.loads(p.read_text('utf-8'));actual_sha=sha_file(p)
                if row['uid'] in actual['training_scans']:raise ValueError('OOF leakage')
                if actual['kind']=='paper_candidate_ranker_v1':
                    inner=set(uid for fold in actual['selection']['inner_validation_scans'] for uid in fold)
                    if inner!=set(actual['training_scans']):raise ValueError('Feature-selection scan leakage')
                np.testing.assert_allclose(decision_scores(row['features_local'],actual),scores[row['uid']],atol=1e-10,rtol=1e-10)
            for profile in ('4.0','high_recall'):
                packet=json.loads((out/cohort/(row['uid']+'_'+profile+'.json')).read_text('utf-8'))
                if packet['second_stage_model_sha256']!=actual_sha:raise ValueError('Wrong per-packet model identity')
                original={c['candidate_id']:c for c in row['packet']['candidates']}
                threshold=report['fixed_development_threshold_profiles'][profile]['threshold']
                indices=np.flatnonzero(scores[row['uid']]>=threshold)
                if [c['candidate_id'] for c in packet['candidates']]!=[row['packet']['candidates'][i]['candidate_id'] for i in indices]:
                    raise ValueError('Threshold selection mismatch')
                for c in packet['candidates']:
                    before=original[c['candidate_id']]
                    for key in ('center_index_zyx','center_lps_xyz_mm','bbox_zyx'):
                        if c[key]!=before[key]:raise ValueError('Geometry changed')
                if cohort=='six_check':
                    runtime=filter_features(row['packet'],row['features'],model,profile,actual_sha)
                    np.testing.assert_allclose([c['score'] for c in runtime['candidates']],
                        [c['score'] for c in packet['candidates']],atol=1e-10,rtol=1e-10)
                packets+=1
    check_froc(rows,dev_scores,report['development_froc']['new'])
    check_froc(check,six_scores,report['six_check_froc']['new'])
    old=json.loads((PRIOR/'final_summary.json').read_text('utf-8'))
    for path,digest in old['current_source_sha256'].items():
        if sha_file(path)!=digest:raise ValueError('Original module changed')
    result=dict(status='PASS',packets_checked=packets,protected_files_unchanged=len(report['protected_files_sha256']),
        original_source_files_unchanged=len(old['current_source_sha256']),geometry_unchanged=True,
        candidate_scores_reproduced=True,feature_selection_grouped_training_only=True,
        official_style_roc_conversion_agrees=True,empirical_operating_points_independently_matched=True,
        heldout_scans_run=0,ct_volumes_read=0,network_forwards=0)
    write_json(destination,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--shell-features',type=Path,required=True);a=p.parse_args();run(a.experiment,a.shell_features)
