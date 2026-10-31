"""Cached-only second-stage experiment: no CT, U-Net, LoG or local extraction.

Adapt weighted RBF SVM classification from classical pulmonary CAD literature.
All hyperparameters/operating thresholds are selected on the original 24
development scans. The six already-viewed development checks are replayed once
after freezing the selected model; the 45 heldout scans are refused.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from lung_candidate_data import read_split,sha_file,write_json
from lung_candidate_rules_run import SPLIT,EXCLUDED,world,labels_for,reference_weights
from lung_candidate_run import excluded_findings
from lung_candidate_rules import RulesConfig,rule_scores,merge_regions
from lung_candidates import point_metrics
from lung_candidate_svm import from_local_table,from_packet,fit,decision_scores

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'data/lung_candidate_rules_20261018'
GRID=[dict(C=c,gamma=g) for c in (.1,1.,10.) for g in ('scale',.1)]


def metrics(row,scores,threshold):
    kept=np.flatnonzero(scores>=threshold)
    return point_metrics(row['world'][kept],row['record']['nodules'],row['excluded'])


def aggregate(rows,scores,threshold):
    cases=[]
    for row in rows:
        before=row['baseline'];after=metrics(row,scores[row['uid']],threshold)
        cases.append(dict(series_uid=row['uid'],before=before,after=after,
            lost=sum(a['detected'] and not b['detected'] for a,b in zip(before['reference_details'],after['reference_details'])),
            gained=sum(not a['detected'] and b['detected'] for a,b in zip(before['reference_details'],after['reference_details']))))
    summary={}
    for kind in ('before','after'):
        total=sum(c[kind]['reference_nodules'] for c in cases)
        detected=sum(c[kind]['detected'] for c in cases)
        fp=sum(c[kind]['false_positive_candidates'] for c in cases)
        summary[kind]=dict(scans=len(rows),reference_nodules=total,detected=detected,
            false_negatives=total-detected,sensitivity=detected/total if total else None,
            false_positive_candidates=fp,fp_per_scan=fp/len(rows),
            candidates=sum(c[kind]['candidates'] for c in cases),
            candidates_per_scan=sum(c[kind]['candidates'] for c in cases)/len(rows))
    return dict(summary=summary,lost=sum(c['lost'] for c in cases),gained=sum(c['gained'] for c in cases),cases=cases)


def preserve_threshold(rows,scores):
    best_hits=[]
    for row in rows:
        for detail in row['baseline']['reference_details']:
            indices=detail['matching_candidate_indices']
            if indices:best_hits.append(float(scores[row['uid']][indices].max()))
    if not best_hits:raise ValueError('No positive development detections')
    return min(best_hits)


def run(out):
    if out.exists():raise FileExistsError('Keep previous experiment; use a new output')
    out.mkdir(parents=True)
    split=read_split(SPLIT);records={r['series_uid']:r for r in split['cases']}
    ignored=excluded_findings(EXCLUDED)
    protocol_path=PRIOR/'features_scale2/protocol.json'
    fold_path=PRIOR/'evaluation_scale2/report.json'
    primary_path=PRIOR/'evaluation_scale2/balanced_rank_model.json'
    protocol=json.loads(protocol_path.read_text('utf-8'))
    old_eval=json.loads(fold_path.read_text('utf-8'))
    primary=json.loads(primary_path.read_text('utf-8'))
    protected={str(p):sha_file(p) for p in (SPLIT,EXCLUDED,protocol_path,fold_path,primary_path,
        Path(__file__),ROOT/'python/lung_candidate_svm.py',ROOT/'python/lung_candidates.py',
        ROOT/'python/lung_candidate_rules.py',ROOT/'python/lung_candidate_refined.py',ROOT/'python/lung_unet.py')}
    rows=[]
    for meta in protocol['cases']:
        uid=meta['series_uid'];record=records[uid]
        if record['split']!='development':raise ValueError('Heldout refused')
        path=PRIOR/'features_scale2'/(uid+'.npz')
        if sha_file(path)!=meta['feature_sha256']:raise ValueError('Saved feature identity changed')
        protected[str(path)]=sha_file(path)
        with np.load(path,allow_pickle=False) as a:table={k:a[k] for k in a.files}
        score=rule_scores(table['features'].astype(float))
        keep=np.flatnonzero(score>=primary['threshold'])
        keep=keep[merge_regions(table[primary['center_mode']][keep],table['boxes_zyx'][keep],score[keep],
            meta['geometry']['spacing_zyx_mm'],RulesConfig(**primary['rules_config']))]
        path=PRIOR/'oof_comparisons'/uid/'candidates.json'
        protected[str(path)]=sha_file(path);packet=json.loads(path.read_text('utf-8'))
        points=table[primary['center_mode']][keep]
        np.testing.assert_array_equal(points,np.array([c['center_index_zyx'] for c in packet['candidates']]))
        np.testing.assert_array_equal(score[keep],np.array([c['score'] for c in packet['candidates']]))
        pts=world(points,meta['geometry'])
        row=dict(uid=uid,record=record,excluded=ignored.get(uid,[]),world=pts,packet=packet,
            features=from_local_table(table,keep,score[keep],meta['geometry']['spacing_zyx_mm']))
        row['labels']=labels_for(pts,record,row['excluded'])
        row['positive_weights']=reference_weights(pts,record)
        row['baseline']=point_metrics(pts,record['nodules'],row['excluded'])
        rows.append(row)
    if len(rows)!=24 or len({r['uid'] for r in rows})!=24:raise ValueError('Need original 24 scans')
    by_uid={r['uid']:r for r in rows}
    plan=dict(status='FROZEN_BEFORE_NEW_SVM_FITS',method='weighted RBF SVM second stage',
        literature='https://doi.org/10.1371/journal.pone.0210551',
        reproduction='Adapt classification principle with existing descriptors; not full paper reproduction',
        grid=GRID,normalization='training fold only',class_weight='balanced',
        positive_weight='equal total weight per detected reference within training fold',
        folds=old_eval['folds'],threshold='highest OOF margin retaining every primary-rule detected reference',
        selection='lowest OOF FP while retaining all 25 references; ties follow fixed grid order',
        check_six_used_for_selection=False,heldout_scans_run=0,ct_volumes_read=0,
        network_forwards=0,log_or_hessian_extractions=0,new_cnn=False,new_bit=False,
        protected_files_sha256=protected)
    write_json(out/'protocol.json',plan)
    trials=[];winning_scores=None;winning_models=None;winner=None
    for trial,params in enumerate(GRID):
        predictions={};fold_models={};start=perf_counter();supports=[]
        for fold in old_eval['folds']:
            train=set(fold['training_scans']);val=set(fold['validation_scans'])
            if train&val or train|val!=set(by_uid):raise ValueError('Invalid scan-grouped fold')
            subset=[r for r in rows if r['uid'] in train]
            x=np.concatenate([r['features'] for r in subset]);y=np.concatenate([r['labels'] for r in subset])
            weights=np.concatenate([r['positive_weights'] for r in subset]);pos=y==1
            weights=np.where(pos,weights,1.)
            weights[pos]*=pos.sum()/weights[pos].sum()
            eligible=y>=0
            model=fit(x[eligible],y[eligible],weights[eligible],params['C'],params['gamma'])
            model.update(training_scans=fold['training_scans'],fold=fold['fold'])
            fold_models[fold['fold']]=model
            supports.append(len(model['support_vectors']))
            for uid in fold['validation_scans']:
                predictions[uid]=decision_scores(by_uid[uid]['features'],model)
        threshold=preserve_threshold(rows,predictions)
        result=aggregate(rows,predictions,threshold)
        if result['lost']:raise AssertionError('Preserving threshold lost a reference')
        item=dict(trial=trial,parameters=params,threshold=threshold,summary=result['summary'],
            lost=result['lost'],gained=result['gained'],fold_support_vectors=supports,
            fit_and_replay_ms=(perf_counter()-start)*1000)
        trials.append(item);write_json(out/'trials.json',dict(status='RUNNING',trials=trials))
        if winner is None or item['summary']['after']['false_positive_candidates']<winner['summary']['after']['false_positive_candidates']:
            winner=item;winning_scores=predictions;winning_models=fold_models
        print('trial',trial,params,'FP/scan',item['summary']['after']['fp_per_scan'],
            'detected',item['summary']['after']['detected'],'lost',item['lost'],flush=True)
    x=np.concatenate([r['features'] for r in rows]);y=np.concatenate([r['labels'] for r in rows])
    weights=np.concatenate([r['positive_weights'] for r in rows]);pos=y==1
    weights=np.where(pos,weights,1.);weights[pos]*=pos.sum()/weights[pos].sum();eligible=y>=0
    best=winner['parameters'];model=fit(x[eligible],y[eligible],weights[eligible],best['C'],best['gamma'])
    model.update(threshold=winner['threshold'],training_scans=[r['uid'] for r in rows],
        source_primary_model_sha256=sha_file(primary_path),formal_deployment=False)
    write_json(out/'selected_svm.json',model)
    frozen_model_sha=sha_file(out/'selected_svm.json')
    fold_folder=out/'selected_fold_models';fold_folder.mkdir()
    fold_identities={}
    for fold in old_eval['folds']:
        f=fold['fold'];fold_model=winning_models[f]
        fold_model.update(threshold=winner['threshold'],formal_deployment=False,
            source_primary_model_sha256=sha_file(primary_path))
        path=fold_folder/f'fold_{f}.json';write_json(path,fold_model)
        for uid in fold['validation_scans']:
            fold_identities[uid]=dict(sha256=sha_file(path),fold=f,path=path.relative_to(out).as_posix())
    development=aggregate(rows,winning_scores,winner['threshold'])
    write_json(out/'development_oof.json',development)
    np.savez_compressed(out/'development_oof_scores.npz',**winning_scores)
    # Only now load the six check packets. No volume or local feature access.
    check_report_path=PRIOR/'new_development_check/report.json'
    protected[str(check_report_path)]=sha_file(check_report_path)
    check_report=json.loads(check_report_path.read_text('utf-8'))
    check=[];check_scores={};score_times=[]
    for uid in check_report['selected_uids']:
        if uid in by_uid or records[uid]['split']!='development':raise ValueError('Check/heldout leakage')
        path=PRIOR/'new_development_check'/uid/'candidates.json'
        protected[str(path)]=sha_file(path);packet=json.loads(path.read_text('utf-8'))
        cands=packet['candidates'];pts=np.array([c['center_lps_xyz_mm'] for c in cands]).reshape(-1,3)
        features=from_packet(cands,packet['geometry']['spacing_zyx_mm'])
        row=dict(uid=uid,record=records[uid],excluded=ignored.get(uid,[]),world=pts,
            packet=packet,features=features,baseline=point_metrics(pts,records[uid]['nodules'],ignored.get(uid,[])))
        start=perf_counter();check_scores[uid]=decision_scores(features,model)
        score_times.append((perf_counter()-start)*1000);check.append(row)
    checked=aggregate(check,check_scores,model['threshold'])
    write_json(out/'six_development_check.json',checked)
    for subset,predictions,name in ((rows,winning_scores,'development_oof'),(check,check_scores,'six_check')):
        folder=out/name;folder.mkdir()
        for row in subset:
            candidates=[]
            for c,score in zip(row['packet']['candidates'],predictions[row['uid']]):
                if score>=model['threshold']:
                    candidates.append(dict(c,primary_rule_score=c['score'],score=float(score),
                        score_kind='uncalibrated_second_stage_svm_margin'))
            identity=(fold_identities[row['uid']] if name=='development_oof' else
                dict(sha256=frozen_model_sha,fold=None,path='selected_svm.json'))
            write_json(folder/(row['uid']+'.json'),dict(row['packet'],candidates=candidates,
                algorithm_id='rule_v4_plus_weighted_rbf_svm_experimental',
                second_stage_model_sha256=identity['sha256'],second_stage_model_path=identity['path'],
                second_stage_fold=identity['fold'],second_stage_threshold=model['threshold'],
                validation_scope=name+'; not final heldout or FPGA validation'))
    for path,expected in protected.items():
        if sha_file(path)!=expected:raise ValueError('Protected input/source changed: '+path)
    if sha_file(out/'selected_svm.json')!=frozen_model_sha:raise ValueError('Model changed during checks')
    before=development['summary']['before'];after=development['summary']['after']
    significant=after['false_positive_candidates']<=.9*before['false_positive_candidates']
    report=dict(status='COMPLETE_CACHED_ONLY_EXPERIMENT',selected=winner,all_trials=trials,
        development_24_oof=development['summary'],development_24_lost=development['lost'],
        development_check_6=checked['summary'],development_check_lost=checked['lost'],
        passes_local_candidate_gate=bool(significant and not development['lost'] and not checked['lost']),
        model_bytes=(out/'selected_svm.json').stat().st_size,support_vectors=len(model['support_vectors']),
        cached_six_score_only_ms=score_times,timing_scope='PC NumPy SVM scoring only; not end-to-end or PL',
        ct_volumes_read=0,network_forwards=0,log_or_hessian_extractions=0,heldout_scans_run=0,
        old_web_or_models_modified=False,model_frozen_before_six=True,
        limits='24 development scans used to choose parameters/threshold; six checks already viewed earlier; no new independent test. Second stage cannot recover upstream misses.',
        new_cnn=False,new_fpga_bit=False,protected_files_sha256=protected)
    write_json(out/'report.json',report)
    print('COMPLETE',json.dumps({k:report[k] for k in ['development_24_oof','development_check_6','passes_local_candidate_gate']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',type=Path,required=True)
    run(p.parse_args().outdir)
