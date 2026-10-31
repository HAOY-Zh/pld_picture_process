"""Read-only whole-background score, geometry, training identity and curve audit."""
import argparse
from pathlib import Path

import numpy as np
from lung_candidate_data import sha_file
from lung_candidate_froc import evaluate
from lung_candidate_lowfp_audit import check_froc
from lung_nodule_cnn_balanced import read,save
from lung_nodule_cnn_review18 import summarize_fixed,independent_counts
from lung_nodule_continuity import feature_names,rerank_scores
from lung_nodule_whole_background import (OUT,PREVIOUS,CFG,cached_rows,training_rows,
    predictions,frozen_ranker)
import lung_nodule_cnn_experiment as first


def reference_changes(rows,before,after,t0,t1):
    details=[]
    for r in rows:
        for i,n in enumerate(r['record']['nodules']):
            hit=np.linalg.norm(r['world']-n['center_lps_xyz_mm'],axis=1)<n['diameter_mm']/2
            a=float(before[r['uid']][hit].max()) if hit.any() else None
            b=float(after[r['uid']][hit].max()) if hit.any() else None
            details.append(dict(series_uid=r['uid'],reference_index=i,diameter_mm=n['diameter_mm'],
                before_score=a,after_score=b,before_detected=a is not None and t0 is not None and a>=t0,
                after_detected=b is not None and t1 is not None and b>=t1))
    return details


def verify_affine(row,ranker):
    score=rerank_scores(row['scores'],row['features'],ranker)
    weights=np.asarray(ranker['coef'])/np.asarray(ranker['scale'])
    bias=ranker['intercept']-weights@np.asarray(ranker['mean'])
    expected=row['scores'].copy()
    mask=expected>=CFG.score_floor
    if ranker['alpha'] and mask.any():
        p=np.clip(expected[mask],1e-7,1-1e-7)
        delta=np.clip(row['features'][mask]@weights+bias,-CFG.correction_limit,CFG.correction_limit)
        z=np.log(p/(1-p))+ranker['alpha']*delta
        expected[mask]=1/(1+np.exp(-z))
    np.testing.assert_allclose(score,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(score[~mask],row['scores'][~mask])
    return score


def run(out):
    plan=first.verify(out)
    for p,digest in plan.get('reused_artifacts_sha256',{}).items():
        if sha_file(p)!=digest:
            raise ValueError('Changed source of reused complete training scan')
    ranker=frozen_ranker(out)
    freeze=read(out/'ranker_freeze.json')
    for p,digest in freeze['training_artifacts_sha256'].items():
        if sha_file(p)!=digest:
            raise ValueError('Changed training or selection artifact')
    if set(ranker['training_scans'])!=set(plan['fit_scans']):
        raise ValueError('Wrong training scans')
    selection=read(out/'selection.json')
    validation=[]
    for fold in selection['folds']:
        a,b=set(fold['training_scans']),set(fold['validation_scans'])
        if a&b or a|b!=set(plan['fit_scans']):
            raise ValueError('Group leakage')
        validation.extend(b)
    if len(validation)!=len(plan['fit_scans']) or set(validation)!=set(plan['fit_scans']):
        raise ValueError('Incomplete grouped validation')
    train=training_rows(out)
    chosen=selection['selected']
    name=f"C{chosen['C']}_{chosen['negative_weight']}_a{chosen['alpha']}"
    with np.load(out/'selection'/(name+'.npz'),allow_pickle=False) as a:
        oof={r['uid']:a[r['uid']] for r in train}
    trial=next(t for t in selection['trials'] if all(t[k]==v for k,v in chosen.items()))
    check_froc(train,oof,trial['curve'])
    # Reconstruct labels/feature gates independently from compressed input evidence.
    for r in train:
        if r['features'].shape!=(len(r['scores']),len(feature_names())) or not np.isfinite(r['features']).all():
            raise ValueError('Invalid whole-volume features')
        mask=r['scores']>=CFG.score_floor
        if ((r['features'][~mask])!=0).any():
            raise ValueError('Below-floor evidence unexpectedly used')
        if int(((r['labels']==0)&mask).sum())!=r['done']['background_training_candidates']:
            raise ValueError('Mining count mismatch')
    cal=read(out/'calibration.json')
    if sha_file(out/'calibration.json')!=read(out/'calibration_freeze.json')['calibration_sha256']:
        raise ValueError('Changed calibration')
    review=read(out/'review.json')
    old_cal=read(PREVIOUS/'calibration.json')
    old_review=read(PREVIOUS/'review.json')
    for name,parent_name in [('cnn','baseline'),('previous','continuity')]:
        for actual,expected in zip(cal['curves'][name]['operating_points'],
                                   old_cal['curves'][parent_name]['operating_points']):
            if (actual['detected'],actual['actual_fp_per_scan'])!=(expected['detected'],expected['actual_fp_per_scan']):
                raise ValueError('Prior calibration baseline failed to reproduce')
            if actual['threshold'] is None or expected['threshold'] is None:
                if actual['threshold']!=expected['threshold']:
                    raise ValueError('Prior baseline threshold mismatch')
            else:
                np.testing.assert_allclose(actual['threshold'],expected['threshold'],rtol=0,atol=1e-12)
    old=read(PREVIOUS/'ranker.json')
    models={'cnn':dict(old,alpha=0.),'previous':old,'whole_background':ranker}
    folder=out/'effect_review'
    folder.mkdir(exist_ok=True)
    summaries={};curves={};pooled=[];packets=0
    for cohort in ('calibration_scans','review_12','reused_6'):
        rows=cached_rows(out,cohort)
        scores={n:predictions(rows,m) for n,m in models.items()}
        counts={}
        for r in rows:
            for n,m in models.items():
                verify_affine(r,m)
            packet=r['packet']
            basis=np.asarray(packet['geometry']['basis_lps_zyx'])
            origin=np.asarray(packet['geometry']['origin_lps_mm'])
            index=np.array([c['center_index_zyx'] for c in packet['candidates']]).reshape(-1,3)
            np.testing.assert_allclose(index@basis.T+origin,r['world'],rtol=0,atol=1e-8)
            packets+=1
        for n in models:
            used=[dict(r,scores=scores[n][r['uid']]) for r in rows]
            counts[n]={}
            for profile,threshold in cal['thresholds'][n].items():
                metric=summarize_fixed(used,threshold)
                for k,v in independent_counts(used,threshold).items():
                    if metric[k]!=v:
                        raise AssertionError('Independent fixed count mismatch')
                counts[n][profile]={k:v for k,v in metric.items() if k!='cases'}
                if cohort!='calibration_scans':
                    recorded=review['results'][cohort][n][profile]
                    for k in ('detected','reference_nodules','false_positive_candidates','actual_fp_per_scan'):
                        if metric[k]!=recorded[k]:
                            raise ValueError('Published review count mismatch')
                    if n in ('cnn','previous'):
                        parent_name='baseline' if n=='cnn' else 'continuity'
                        expected=old_review['results'][cohort][parent_name][profile]
                        for k in ('detected','false_positive_candidates'):
                            if metric[k]!=expected[k]:
                                raise ValueError('Prior frozen review baseline failed to reproduce')
        t0=cal['thresholds']['previous']['4.0'];t1=cal['thresholds']['whole_background']['4.0']
        details=reference_changes(rows,scores['previous'],scores['whole_background'],t0,t1)
        profile_changes={}
        for profile in ('1.0','4.0','8.0'):
            changes=reference_changes(rows,scores['previous'],scores['whole_background'],
                cal['thresholds']['previous'][profile],cal['thresholds']['whole_background'][profile])
            profile_changes[profile]=dict(
                gained=[d for d in changes if not d['before_detected'] and d['after_detected']],
                lost=[d for d in changes if d['before_detected'] and not d['after_detected']])
        sizes={}
        for name,lo,hi in [('under6',0,6),('6to10',6,10),('10plus',10,float('inf'))]:
            refs=[d for d in details if lo<=d['diameter_mm']<hi]
            sizes[name]=dict(references=len(refs),before=sum(d['before_detected'] for d in refs),
                            after=sum(d['after_detected'] for d in refs))
        summaries[cohort]=dict(profiles=counts,reference_details=details,size_groups=sizes,
            reference_changes_all_profiles=profile_changes,
            gained=sum(not d['before_detected'] and d['after_detected'] for d in details),
            lost=sum(d['before_detected'] and not d['after_detected'] for d in details))
        if cohort=='calibration_scans':
            curves['calibration19']={n:evaluate(rows,s) for n,s in scores.items()}
            for n in models:
                check_froc(rows,scores[n],cal['curves'][n])
                np.testing.assert_allclose(curves['calibration19'][n]['cpm_interpolated'],cal['curves'][n]['cpm_interpolated'])
        else:
            pooled.extend(rows)
        np.savez_compressed(folder/(cohort+'_scores.npz'),**{
            n+'__'+uid:value for n,values in scores.items() for uid,value in values.items()})
    curves['review18_posthoc_diagnostic']={}
    for n,m in models.items():
        s=predictions(pooled,m)
        curves['review18_posthoc_diagnostic'][n]=evaluate(pooled,s)
        check_froc(pooled,s,curves['review18_posthoc_diagnostic'][n])
    save(folder/'ranking_diagnostic.json',dict(curves=curves,
        purpose='review18 curves are post-hoc development diagnostics; no new thresholds selected/exported',
        primary_fixed_threshold_results=str(out/'review.json')))
    summary=dict(selected=selection['selected'],training_scans=len(train),
        training_candidates=selection['training_candidates'],used_background=selection['used_background'],
        used_matched=selection['used_matched'],cohorts=summaries,
        recorded_case_mining_seconds_mixed_sampling=sum(r['done']['total_seconds'] for r in train),
        local_feature_algorithm_and_eligible_count_unchanged=True,feature_count=len(feature_names()),
        meaning='whole-volume mining and fitting are offline; deployed local feature algorithm unchanged',
        heldout_scans_run=0,expert_nodule_contours_available=False,fpga_validated=False)
    save(folder/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,(cohort,values) in zip(axes,curves.items()):
        for n,color in [('cnn','#7a8a9e'),('previous','#c58b44'),('whole_background','#087f8c')]:
            c=values[n]['curve']
            ax.plot(c['fp_per_scan'],c['sensitivity'],label=n,color=color)
        ax.set(xlim=(0,8),ylim=(0,1),xlabel='False positives / CT',ylabel='Sensitivity',title=cohort)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Frozen CNN and native-slice features; different local-ranker training data')
    fig.tight_layout();fig.savefig(folder/'froc.png',dpi=160);plt.close(fig)
    save(folder/'audit.json',dict(status='PASS',training_scans=len(train),development_evidence_packets=packets,
        protected_identities=len(plan['protected_sha256']),training_artifacts=len(freeze['training_artifacts_sha256']),
        completed_training_scans_reused=len(plan.get('reused_training_scans',[])),
        selected_oof_empirical_counts_independently_checked=True,ranker_groups_disjoint=True,
        parent_cnn_is_not_oof=True,affine_scores_and_low_score_preservation_checked=True,
        prior_calibration_and_frozen_review_baselines_reproduced=True,
        world_coordinates_unchanged=True,fixed_threshold_counts_independently_checked=True,
        review_used_to_retune=False,new_calibration_review_ct_reads=0,new_calibration_review_cnn_forwards=0,
        heldout_scans_run=0,report_source_sha256=sha_file(Path(__file__).resolve())))
    print('AUDIT_PASS',packets,'development packets, fit53 whole-volume mining',flush=True)
    for cohort,v in summaries.items():
        print(cohort,{n:(p['4.0']['detected'],p['4.0']['actual_fp_per_scan']) for n,p in v['profiles'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,default=OUT)
    run(p.parse_args().outdir)
