"""Scan-grouped development evaluation; small feature rankers, no added CNN.

Uses separate source from long-running feature extraction so evaluation precision
and ranker comparisons can evolve while preserving exact extraction snapshots.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from lung_candidate_rules_run import (ROOT,EXCLUDED,expected_records,frozen_sources,
    world,labels_for,metric_key,export_linear,reference_weights,training_weights)
from lung_candidate_data import sha_file,write_json
from lung_candidate_run import excluded_findings
from lung_candidates import point_metrics
from lung_candidate_rules import (RulesConfig,FEATURE_NAMES,rule_scores,linear_scores,merge_regions)
from lung_candidate_feature_rank import fit_feature_rank,feature_rank_scores


def evaluate(features_dir,out):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)
    if out.exists():raise ValueError('Use fresh evaluation output')
    split,records=expected_records();by_uid={r['series_uid']:r for r in records}
    protocol=json.loads((features_dir/'protocol.json').read_text(encoding='utf-8'))
    if protocol['status']!='COMPLETE' or set(protocol['selected_uids'])!=set(by_uid):
        raise ValueError('Need all original 24 development feature tables')
    if protocol['rules']!=asdict(RulesConfig()) or tuple(protocol['feature_names'])!=FEATURE_NAMES:
        raise ValueError('Feature configuration differs')
    excluded=excluded_findings(EXCLUDED);out.mkdir(parents=True)
    # Deterministic scan groups. No patches from the same CT cross folds.
    folds={}
    for positive in [False,True]:
        subset=[r for r in records if bool(r['nodules'])==positive]
        subset.sort(key=lambda r:hashlib.sha256(('rules-cv:'+r['series_uid']).encode()).hexdigest())
        folds.update({r['series_uid']:i%4 for i,r in enumerate(subset)})
    tables=[];metas=[];groups=[];all_labels=[];all_weights=[];baseline_hits=set();baseline_fp=0
    for meta in protocol['cases']:
        uid=meta['series_uid'];path=features_dir/(uid+'.npz')
        if sha_file(path)!=meta['feature_sha256']:raise ValueError('Feature bytes changed')
        with np.load(path,allow_pickle=False) as p: table={k:p[k] for k in p.files}
        labels=labels_for(world(table['refined_centers_zyx'],meta['geometry']),by_uid[uid],excluded.get(uid,[]))
        all_weights.append(reference_weights(world(table['refined_centers_zyx'],meta['geometry']),by_uid[uid]))
        tables.append(table);metas.append(meta);all_labels.append(labels);groups.extend([folds[uid]]*len(labels))
        baseline=json.loads(Path(meta['baseline_path']).read_text(encoding='utf-8'))
        if sha_file(meta['baseline_path'])!=meta['baseline_candidate_sha256']:raise ValueError('Baseline changed')
        m=point_metrics([c['center_lps_xyz_mm'] for c in baseline['candidates']],by_uid[uid]['nodules'],excluded.get(uid,[]))
        baseline_fp+=m['false_positive_candidates']
        baseline_hits.update(metric_key(uid,d) for d in m['reference_details'] if d['detected'])
    x=np.concatenate([t['features'] for t in tables]).astype(np.float64);y=np.concatenate(all_labels);groups=np.array(groups);raw_weights=np.concatenate(all_weights)
    predictions=np.empty(len(y),float);boost_predictions=np.empty(len(y),float);fold_report=[]
    for fold in range(4):
        train=(groups!=fold)&(y>=0);valid=groups==fold
        scaler=StandardScaler().fit(x[train]);clf=LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,random_state=20261018)
        clf.fit(scaler.transform(x[train]),y[train],sample_weight=training_weights(y[train],raw_weights[train]));model=export_linear(scaler,clf)
        predictions[valid]=linear_scores(x[valid],model)
        np.testing.assert_allclose(predictions[valid],clf.decision_function(scaler.transform(x[valid])),atol=1e-10,rtol=1e-10)
        fold_report.append(dict(fold=fold,training_scans=[uid for uid,f in folds.items() if f!=fold],
            validation_scans=[uid for uid,f in folds.items() if f==fold],training_positive_candidates=int((y[train]==1).sum()),
            training_negative_candidates=int((y[train]==0).sum())))
        write_json(out/f'linear_fold_{fold}.json',model)
        boosted,boost_clf=fit_feature_rank(x[train],y[train],training_weights(y[train],raw_weights[train]))
        boost_predictions[valid]=feature_rank_scores(x[valid],boosted)
        np.testing.assert_allclose(boost_predictions[valid],boost_clf.decision_function(x[valid]),atol=1e-10,rtol=1e-10)
        write_json(out/f'boost_fold_{fold}.json',boosted)
        print('Fitted scan fold',fold,flush=True)
    offsets=np.cumsum([0]+[len(t['features']) for t in tables])
    results=[];cfg=RulesConfig();summaries={}
    for mode in ['original_centers_zyx','refined_centers_zyx']:
        for method,scores in [('rule',rule_scores(x)),('linear_oof',predictions),('boosted_rules_oof',boost_predictions)]:
            # A fixed percentile sweep creates a development tradeoff curve, not
            # a formal test score or per-case threshold tuning.
            quantiles=np.unique(np.r_[np.linspace(0,.98,50),np.linspace(.98,1,81)])
            thresholds=np.unique(np.r_[-1e20,np.quantile(scores,quantiles)])
            # Greedy suppression is score ordered: removing lower-scored points
            # cannot change the kept higher-scored prefix. Merge once per curve.
            merged=[]
            for i,(meta,table) in enumerate(zip(metas,tables)):
                s=scores[offsets[i]:offsets[i+1]]
                merged.append(merge_regions(table[mode],table['boxes_zyx'],s,meta['geometry']['spacing_zyx_mm'],cfg))
            for threshold in thresholds:
                hits=set();fp=0;count=0
                for i,(meta,table) in enumerate(zip(metas,tables)):
                    uid=meta['series_uid'];s=scores[offsets[i]:offsets[i+1]]
                    indices=merged[i][s[merged[i]]>=threshold]
                    m=point_metrics(world(table[mode][indices],meta['geometry']),by_uid[uid]['nodules'],excluded.get(uid,[]))
                    hits.update(metric_key(uid,d) for d in m['reference_details'] if d['detected'])
                    fp+=m['false_positive_candidates'];count+=len(indices)
                results.append(dict(center_mode=mode,method=method,threshold=float(threshold),detected=len(hits),
                    fp_per_scan=fp/24,false_positive_candidates=fp,candidates=count,
                    lost_baseline_nodules=len(baseline_hits-hits),gained_nodules=len(hits-baseline_hits)))
            print('Evaluated',mode,method,flush=True)
    safe=[r for r in results if r['lost_baseline_nodules']==0]
    selected=min(safe,key=lambda r:(r['false_positive_candidates'],-r['detected'])) if safe else None
    best_by_detected={str(n):min([r for r in results if r['detected']>=n],key=lambda r:r['false_positive_candidates'])
                      for n in range(24,29) if any(r['detected']>=n for r in results)}
    train=y>=0;scaler=StandardScaler().fit(x[train]);clf=LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,random_state=20261018)
    clf.fit(scaler.transform(x[train]),y[train],sample_weight=training_weights(y[train],raw_weights[train]));final_model=export_linear(scaler,clf)
    final_model.update(training_scans=list(by_uid),scope='24-development fit; new independent development checks required',
                       heldout_scans_run=0,classification_reference='LUNA annotated/excluded matching; not clinical benign labels')
    write_json(out/'linear_all_development.json',final_model)
    full_boost,_=fit_feature_rank(x[train],y[train],training_weights(y[train],raw_weights[train]))
    write_json(out/'boost_all_development.json',full_boost)
    write_json(out/'tradeoffs.json',results)
    report=dict(status='COMPLETE',scope='development parameter exploration and scan-grouped 4-fold OOF; not independent test',
        frozen_sources=dict(frozen_sources(),**{str(Path(__file__).resolve()):sha_file(__file__),str(ROOT/'python/lung_candidate_feature_rank.py'):sha_file(ROOT/'python/lung_candidate_feature_rank.py'),str(ROOT/'python/lung_candidate_rank_runtime.py'):sha_file(ROOT/'python/lung_candidate_rank_runtime.py')}),feature_protocol_sha256=sha_file(features_dir/'protocol.json'),
        scans=24,reference_nodules=28,baseline_detected=len(baseline_hits),baseline_fp_per_scan=baseline_fp/24,
        candidate_pool=len(y),positive_candidates=int((y==1).sum()),ignored_candidates=int((y<0).sum()),
        folds=fold_report,selected_preserving_all_baseline_nodules=selected,best_by_detected=best_by_detected,
        heldout_scans_run=0,new_neural_network=False,new_fpga_implementation=False,
        threshold_selection_uses_these_development_labels=True)
    balanced_options=[r for r in results if r['detected']>=len(baseline_hits)+1 and
                      r['lost_baseline_nodules']<=1 and r['fp_per_scan']<baseline_fp/24]
    balanced=min(balanced_options,key=lambda r:(r['false_positive_candidates'],-r['detected'])) if balanced_options else None
    report['balanced_experimental_point']=balanced
    if balanced:
        kind={'linear_oof':'linear','boosted_rules_oof':'boosted_rules','rule':'rule'}[balanced['method']]
        data={'linear':final_model,'boosted_rules':full_boost,'rule':{}}[kind]
        model=dict(data,kind=kind,center_mode=balanced['center_mode'],threshold=balanced['threshold'],
            rules_config=asdict(RulesConfig()),complementary_sigma_mm=protocol.get('complementary_sigma_mm'),
            feature_names=FEATURE_NAMES,training_scans=list(by_uid),heldout_scans_run=0,
            scope='experimental tradeoff selected on development OOF; up to one previously detected reference lost')
        write_json(out/'balanced_rank_model.json',model)
    if selected:
        common=dict(kind={'linear_oof':'linear','boosted_rules_oof':'boosted_rules','rule':'rule'}[selected['method']],
            center_mode=selected['center_mode'],threshold=selected['threshold'],rules_config=asdict(RulesConfig()),
            complementary_sigma_mm=protocol.get('complementary_sigma_mm'),
            scope='development-selected operating point; needs new development and heldout validation')
        if common['kind']=='boosted_rules':
            final_model=full_boost
        selected_model=dict(final_model,**common) if common['kind'] in ('linear','boosted_rules') else common
        selected_model['training_scans']=list(by_uid)
        selected_model['heldout_scans_run']=0
        write_json(out/'selected_rank_model.json',selected_model)
        for fold in range(4):
            prefix='boost' if common['kind']=='boosted_rules' else 'linear'
            fold_path=out/f'{prefix}_fold_{fold}.json';fold_model=json.loads(fold_path.read_text(encoding='utf-8'))
            write_json(out/f'selected_fold_{fold}.json',dict(fold_model,**common) if common['kind'] in ('linear','boosted_rules') else common)
    if selected:
        scores={'linear_oof':predictions,'boosted_rules_oof':boost_predictions,'rule':rule_scores(x)}[selected['method']]
        delivery=out/'development_oof_candidates';delivery.mkdir()
        for i,(meta,table) in enumerate(zip(metas,tables)):
            uid=meta['series_uid'];s=scores[offsets[i]:offsets[i+1]];inds=np.flatnonzero(s>=selected['threshold'])
            inds=inds[merge_regions(table[selected['center_mode']][inds],table['boxes_zyx'][inds],s[inds],meta['geometry']['spacing_zyx_mm'])]
            candidates=[]
            for j,k in enumerate(inds):
                p=table[selected['center_mode']][k]
                candidates.append(dict(candidate_id=f'rules_{j+1:05d}',center_index_zyx=p.tolist(),
                    center_lps_xyz_mm=world([p],meta['geometry'])[0].tolist(),bbox_zyx=table['boxes_zyx'][k].tolist(),
                    score=float(s[k]),score_kind='development_rank_not_calibrated_probability',
                    source_candidate_index=int(table['source_indices'][k])))
            packet=dict(schema_version='lung_candidate_result_v1',algorithm_id='lung_candidate_rules_v4_development',
                backend='CPU_REFERENCE',status='success',series_uid=uid,source_shape_zyx=meta['source_shape_zyx'],
                geometry=meta['geometry'],input_raw_sha256=meta['input_raw_sha256'],lung_mask_sha256=meta['lung_mask_sha256'],
                candidates=candidates,validation_scope=report['scope'],
                operating_point={k:selected[k] for k in ('method','center_mode','threshold')},fold=folds[uid])
            write_json(delivery/(uid+'.json'),packet)
    write_json(out/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    args=p.parse_args();evaluate(args.features,args.outdir)
