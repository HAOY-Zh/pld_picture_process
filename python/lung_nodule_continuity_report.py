"""Read-only score/geometry audit and visual diagnostics for continuity v1."""
import argparse
from pathlib import Path

import numpy as np

from lung_candidate_data import sha_file,load_development_case
from lung_candidate_rules_run import labels_for
from lung_candidate_froc import evaluate
from lung_candidate_lowfp_audit import check_froc
from lung_nodule_cnn_balanced import read,save
from lung_nodule_cnn_review18 import summarize_fixed, independent_counts
from lung_nodule_continuity import feature_names,rerank_scores
from lung_nodule_continuity_experiment import (OUT,COHORTS,CFG,eligible,rows_for,
    verify_ranker,parent_packet)
import lung_nodule_cnn_experiment as first


def run(out):
    plan = first.verify(out)
    ranker = verify_ranker(out)
    freeze = read(out/'ranker_freeze.json')
    for path,digest in freeze['training_cache_sha256'].items():
        if sha_file(path) != digest:
            raise ValueError('Changed frozen ranker training cache')
    cal = read(out/'calibration.json')
    if sha_file(out/'calibration.json') != read(out/'calibration_freeze.json')['calibration_sha256']:
        raise ValueError('Changed calibration')
    fit = set(plan['fit_scans'])
    if set(ranker['training_scans']) != fit:
        raise ValueError('Wrong ranker training cohort')
    selection = read(out/'training_selection.json')
    validation = []
    for fold in selection['folds']:
        a,b = set(fold['training_scans']),set(fold['validation_scans'])
        if a & b or a | b != fit:
            raise ValueError('Ranker CV group overlap')
        validation.extend(fold['validation_scans'])
    if len(validation) != len(fit) or set(validation) != fit:
        raise ValueError('Invalid grouped validation coverage')
    folder = out/'effect_review'
    folder.mkdir(exist_ok=True)
    summaries = {}
    identities = {}
    examples = []
    review_rows = []
    total = 0
    cohorts = ['calibration_scans']
    if (out/'review.json').exists():
        cohorts += ['review_12','reused_6']
    for cohort in cohorts:
        rows = rows_for(out,cohort)
        if cohort != 'calibration_scans':
            review_rows.extend(rows)
        old_threshold = cal['thresholds']['baseline']['4.0']
        new_threshold = cal['thresholds']['continuity']['4.0']
        refs = []
        feature_times = []
        for row in rows:
            uid = row['uid']
            packet = row['packet']
            source = parent_packet(cohort,uid)
            original = read(source)
            if packet['candidates'] != original['candidates']:
                raise ValueError('Candidate position/size/order/content changed')
            if packet['geometry'] != original['geometry']:
                raise ValueError('Native geometry changed')
            selected = np.flatnonzero(row['scores'] >= CFG.score_floor).tolist()
            if selected != packet['evidence']['selected_indices']:
                raise ValueError('Feature index mismatch')
            scores = rerank_scores(row['scores'],row['features'],ranker)
            # Separate affine export path, without the runtime helper's
            # standardization/matmul expression, checks score application.
            weights = np.asarray(ranker['coef'])/np.asarray(ranker['scale'])
            bias = ranker['intercept']-weights@np.asarray(ranker['mean'])
            expected = row['scores'].copy()
            if ranker['alpha'] and selected:
                idx = np.asarray(selected)
                p = np.clip(expected[idx],1e-7,1-1e-7)
                correction = np.clip(row['features'][idx]@weights+bias,-CFG.correction_limit,CFG.correction_limit)
                logits = np.log(p/(1-p))+ranker['alpha']*correction
                expected[idx] = 1/(1+np.exp(-logits))
            np.testing.assert_allclose(scores,expected,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(scores[row['scores'] < CFG.score_floor],
                                          row['scores'][row['scores'] < CFG.score_floor])
            feature_times.append(packet['evidence']['feature_seconds'])
            for i,n in enumerate(row['record']['nodules']):
                dist = np.linalg.norm(row['world']-n['center_lps_xyz_mm'],axis=1)
                hits = np.flatnonzero(dist < n['diameter_mm']/2)
                old = float(row['scores'][hits].max()) if len(hits) else None
                new = float(scores[hits].max()) if len(hits) else None
                before = old is not None and old_threshold is not None and old >= old_threshold
                after = new is not None and new_threshold is not None and new >= new_threshold
                detail = dict(series_uid=uid,reference_index=i,diameter_mm=n['diameter_mm'],
                    baseline_score=old,continuity_score=new,before_detected=before,after_detected=after,
                    matching_candidates=len(hits))
                refs.append(detail)
                if before != after:
                    q = int(hits[np.argmax(scores[hits])])
                    examples.append(dict(cohort=cohort,kind='gained' if after else 'lost',
                        **detail,candidate=packet['candidates'][q],evidence=next((d for d in packet['evidence']['details']
                        if d['candidate_index'] == q),None)))
            labels = labels_for(row['world'],row['record'],row['excluded'])
            background = np.flatnonzero((labels==0) & (row['scores'] >= CFG.score_floor))
            if len(background):
                q = int(background[np.argmin(scores[background]-row['scores'][background])])
                examples.append(dict(cohort=cohort,kind='lowered_reference_standard_false_positive',
                    series_uid=uid,baseline_score=float(row['scores'][q]),continuity_score=float(scores[q]),
                    candidate=packet['candidates'][q],evidence=next(d for d in packet['evidence']['details'] if d['candidate_index']==q)))
            identities[str(out/cohort/(uid+'.json'))] = sha_file(out/cohort/(uid+'.json'))
            total += 1
        profiles = {}
        for name in ('baseline','continuity'):
            used = [dict(r,scores=rerank_scores(r['scores'],r['features'],ranker)) if name=='continuity' else r for r in rows]
            profiles[name] = {}
            for profile,threshold in cal['thresholds'][name].items():
                m = summarize_fixed(used,threshold)
                for k,v in independent_counts(used,threshold).items():
                    if m[k] != v:
                        raise ValueError('Independent metric mismatch')
                profiles[name][profile] = {k:v for k,v in m.items() if k!='cases'}
        size = {}
        for name,lo,hi in [('under6',0,6),('6to10',6,10),('10plus',10,float('inf'))]:
            subset = [r for r in refs if lo <= r['diameter_mm'] < hi]
            size[name] = dict(references=len(subset),before=sum(r['before_detected'] for r in subset),
                             after=sum(r['after_detected'] for r in subset))
        summaries[cohort] = dict(profiles=profiles,reference_details=refs,size_groups=size,
            gained=sum(not r['before_detected'] and r['after_detected'] for r in refs),
            lost=sum(r['before_detected'] and not r['after_detected'] for r in refs),
            feature_seconds=dict(mean=float(np.mean(feature_times)),median=float(np.median(feature_times)),
                maximum=max(feature_times)),
            candidates_examined=sum(r['packet']['evidence']['candidates_examined'] for r in rows))
    importance = sorted(zip(feature_names(),ranker['coef']),key=lambda p:-abs(p[1]))
    save(folder/'summary.json',dict(selected_alpha=ranker['alpha'],cohorts=summaries,
        largest_standardized_coefficients=importance[:15],examples=examples,
        limits=['development only','candidate-local HU components are not nodule voxel masks',
            'same centers and candidate coverage; no localization improvement claim',
            'extra feature time excludes CT/CNN/transport/rendering','original FP32 lung masks; no board validation']))
    if review_rows:
        diagnostic = {}
        for name in ('baseline','continuity'):
            predictions = {r['uid']:(r['scores'] if name=='baseline' else
                rerank_scores(r['scores'],r['features'],ranker)) for r in review_rows}
            diagnostic[name] = evaluate(review_rows,predictions)
            check_froc(review_rows,predictions,diagnostic[name])
        save(folder/'review18_ranking_diagnostic.json',dict(curves=diagnostic,
            purpose='post-hoc development ranking diagnostic only; no thresholds exported or used for profile selection',
            fixed_threshold_primary_report=str(out/'review.json'),
            ranker_and_calibration_unchanged=True,heldout_scans_run=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for name,color in [('baseline','#6d809c'),('continuity','#df8b22')]:
        c=cal['curves'][name]
        ax.plot(c['curve']['fp_per_scan'],c['curve']['sensitivity'],color=color,label=name)
    ax.set(xlim=(0,8),ylim=(0,1),xlabel='False positives / CT',ylabel='Sensitivity',
           title='Calibration19: frozen CNN + local native-slice evidence')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(folder/'froc.png',dpi=160);plt.close(fig)
    # At most three real CT examples; annotations only choose audit examples.
    chosen=[]
    for kind in ('gained','lost','lowered_reference_standard_false_positive'):
        found=next((e for e in examples if e['kind']==kind),None)
        if found:chosen.append(found)
    if chosen:
        fig,axes=plt.subplots(len(chosen),3,figsize=(10,3.4*len(chosen)),squeeze=False)
        for j,e in enumerate(chosen):
            split,records=eligible(plan,e['cohort'])
            r=next(r for r in records if r['series_uid']==e['series_uid'])
            hu,geom=load_development_case(split,r)
            c=e['candidate'];center=np.array(c['center_index_zyx']);s=np.array(geom['spacing_zyx_mm'])
            z=int(round(center[0]));gap=max(1,int(round(2/s[0])))
            radius=12/s[1:]
            lo=np.maximum(0,np.floor(center[1:]-radius).astype(int));hi=np.minimum(hu.shape[1:],np.ceil(center[1:]+radius).astype(int)+1)
            for k,zz in enumerate([max(0,z-gap),z,min(hu.shape[0]-1,z+gap)]):
                ax=axes[j,k];ax.imshow(hu[zz,lo[0]:hi[0],lo[1]:hi[1]],cmap='gray',vmin=-1000,vmax=400,
                    aspect=s[1]/s[2]);ax.plot(center[2]-lo[1],center[1]-lo[0],'+',color='#ffab39',ms=10)
                label = f'native Z={zz}; {(zz-center[0])*s[0]:+.1f} mm'
                if k == 1:
                    label = f"{e['kind']} | {e['baseline_score']:.3f} -> {e['continuity_score']:.3f}\n"+label
                ax.set_title(label,fontsize=10);ax.axis('off')
            axes[j,0].set_ylabel(e['kind'])
            axes[j,1].set_xlabel(f"{e['kind']} | score {e['baseline_score']:.3f} -> {e['continuity_score']:.3f}")
            del hu
        fig.suptitle('Real CT neighboring planes; orange = projected candidate center, not contour')
        fig.tight_layout();fig.savefig(folder/'neighbor_examples.png',dpi=150);plt.close(fig)
    save(folder/'audit.json',dict(status='PASS',evidence_packets=total,
        protected_identities=len(plan['protected_sha256']),training_cache_identities=len(freeze['training_cache_sha256']),
        candidate_geometry_size_order_unchanged=True,below_floor_scores_unchanged=True,
        fixed_threshold_counts_independently_checked=True,ranker_cv_groups_disjoint=True,
        affine_score_export_independently_checked=True,
        ranker_oof_does_not_apply_to_parent_cnn=True,heldout_scans_run=0,
        development_cnn_rerun=False,feature_cache_sha256=identities,
        report_source_sha256=sha_file(Path(__file__).resolve())))
    print('AUDIT_PASS',total,'evidence packets', {k:{n:(v['4.0']['detected'],v['4.0']['actual_fp_per_scan'])
        for n,v in c['profiles'].items()} for k,c in summaries.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,default=OUT)
    run(p.parse_args().outdir)
