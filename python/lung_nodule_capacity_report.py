"""Independent counts and geometry audit for the paired width-capacity study."""
import argparse
from pathlib import Path

import numpy as np

from lung_candidate_data import sha_file
from lung_candidate_froc import evaluate
from lung_nodule_capacity_froc_audit import check_froc_stable
from lung_nodule_cnn_balanced import read, save
from lung_nodule_cnn_review18 import summarize_fixed, independent_counts
from lung_nodule_capacity import OUT, PREVIOUS, VARIANTS, COHORTS, rows_for
from lung_nodule_continuity import feature_names
import lung_nodule_cnn_experiment as first


def counts(rows, threshold, raw=False):
    used = [dict(r,scores=r['raw_scores']) if raw else r for r in rows]
    result = summarize_fixed(used, threshold)
    for k,v in independent_counts(used,threshold).items():
        if result[k] != v:
            raise ValueError('Independent threshold count mismatch')
    compact = {k:v for k,v in result.items() if k!='cases'}
    fp = [c['metric']['false_positive_candidates'] for c in result['cases']]
    compact['false_positive_distribution'] = dict(median=float(np.median(fp)),
        p90=float(np.percentile(fp,90)),maximum=max(fp),zero_fp_scans=fp.count(0))
    return compact


def reference_details(rows, threshold, raw=False):
    result = []
    for row in rows:
        scores = row['raw_scores'] if raw else row['scores']
        for i,n in enumerate(row['record']['nodules']):
            distance = np.linalg.norm(row['world']-n['center_lps_xyz_mm'],axis=1)
            hits = np.flatnonzero(distance < n['diameter_mm']/2)
            best = int(hits[np.argmax(scores[hits])]) if len(hits) else None
            detected = best is not None and threshold is not None and scores[best]>=threshold
            result.append(dict(series_uid=row['uid'],reference_index=i,diameter_mm=n['diameter_mm'],
                detected=bool(detected), best_score=float(scores[best]) if best is not None else None,
                center_error_mm=float(distance[best]) if detected else None,
                center_lps_xyz_mm=n['center_lps_xyz_mm']))
    return result


def changes(before,after):
    a = {(r['series_uid'],r['reference_index']):r for r in before}
    b = {(r['series_uid'],r['reference_index']):r for r in after}
    if a.keys()!=b.keys():
        raise ValueError('Reference identity mismatch')
    return dict(gained=[b[k] for k in a if b[k]['detected'] and not a[k]['detected']],
                lost=[a[k] for k in a if a[k]['detected'] and not b[k]['detected']])


def audit_row(row,ranker):
    p = row['packet']
    features, raw = row['features'], row['raw_scores']
    if features.shape!=(len(raw),len(feature_names())) or not np.isfinite(features).all():
        raise ValueError('Feature dimension/value mismatch')
    selected = raw >= .1
    if np.any(features[~selected]!=0):
        raise ValueError('Unexpected below-floor local feature use')
    if p['evidence']['selected_indices']!=np.flatnonzero(selected).tolist():
        raise ValueError('Local evidence index mismatch')
    # Algebraically independent affine export path, including exact floor rule.
    weights = np.asarray(ranker['coef'])/np.asarray(ranker['scale'])
    bias = ranker['intercept']-weights@np.asarray(ranker['mean'])
    expected = raw.copy()
    prob = np.clip(raw[selected],1e-7,1-1e-7)
    delta = np.clip(features[selected]@weights+bias,-2,2)
    z = np.log(prob/(1-prob))+ranker['alpha']*delta
    expected[selected] = 1/(1+np.exp(-z))
    np.testing.assert_allclose(expected,row['scores'],rtol=1e-12,atol=1e-12)
    shape = np.asarray(row['record']['shape_zyx'])
    for c in p['candidates']:
        box = np.asarray(c['bbox_zyx']).reshape(3,2)
        if np.any(box[:,0]<0) or np.any(box[:,1]>shape) or np.any(box[:,1]<box[:,0]):
            raise ValueError('Candidate box outside native CT')


def run(out):
    plan = first.verify(out)
    cal = read(out/'calibration.json')
    frozen = read(out/'calibration_freeze.json')
    if sha_file(out/'calibration.json')!=frozen['calibration_sha256']:
        raise ValueError('Changed calibration')
    train = read(out/'training_complete.json')
    history = train['epochs']
    if len(history)!=plan['training']['epochs'] or sum(h['batches'] for h in history)!=6060:
        raise ValueError('Unequal training budget')
    if any(any(h['skipped_updates'].values()) for h in history):
        raise ValueError('Skipped optimizer updates')
    import torch
    for v in VARIANTS:
        f = read(out/v/'model_freeze.json')
        if f['model_sha256']!=sha_file(out/v/'model.pt') or frozen['model_sha256'][v]!=f['model_sha256']:
            raise ValueError('Changed trained model')
        if f['shared_sample_augmentation_sha256']!=[h['shared_sample_augmentation_sha256'] for h in history]:
            raise ValueError('Unequal training stream')
        if f['training_batches']!=6060 or f['epochs']!=30 or f['skipped_updates']!=0:
            raise ValueError('Wrong effective optimizer budget')
        ckpt = torch.load(out/v/'model.pt',map_location='cpu',weights_only=True)
        if set(ckpt['training_scans'])!=set(plan['fit_scans']):
            raise ValueError('Wrong training cohort')
        if any(not torch.isfinite(t).all() for t in ckpt['state'].values()):
            raise ValueError('Nonfinite trained weights')
        if sum(t.numel() for t in ckpt['state'].values())!=plan['architectures'][v]['parameters']:
            raise ValueError('Wrong trained width')
    wide = torch.load(out/VARIANTS[1]/'model.pt',map_location='cpu',weights_only=True)['state']
    diversity = {}
    for key,value in wide.items():
        if key.endswith('.weight') and not key.startswith('head.'):
            diversity[key] = float((value[0::2]-value[1::2]).abs().max())
    if not all(d>0 for d in diversity.values()):
        raise ValueError('Widened neurons remained trapped in duplicate symmetry')
    ranker = read(PREVIOUS/'ranker.json')
    if sha_file(PREVIOUS/'ranker.json')!=frozen['ranker_sha256'] or ranker['alpha']!=.25:
        raise ValueError('Changed first cross-slice rule')
    folder = out/'effect_review'
    folder.mkdir(exist_ok=True)
    pooled = {v:[] for v in ('historical',*VARIANTS)}
    summaries, curves, identities = {}, {}, {}
    curve_audits = {}
    review = read(out/'review.json')
    old_review = read(PREVIOUS/'review.json')
    all_names = [v+'_'+mode for v in pooled for mode in ('raw','cross')]
    for cohort in COHORTS:
        summaries[cohort] = {}
        curves[cohort] = {}
        refs = {}
        timing = {}
        for v in pooled:
            rows = rows_for(out,cohort,v)
            for r in rows:
                audit_row(r,ranker)
                path = (PREVIOUS if v=='historical' else out/v)/cohort/(r['uid']+'.json')
                identities[str(path)] = sha_file(path)
            if cohort!='calibration_scans':
                pooled[v].extend(rows)
            for mode in ('raw','cross'):
                name = v+'_'+mode
                scores = {r['uid']:r['raw_scores' if mode=='raw' else 'scores'] for r in rows}
                curve = evaluate(rows,scores)
                curve_audits[cohort+'__'+name]=check_froc_stable(rows,scores,curve)
                curves[cohort][name] = curve
                profiles = {profile:counts(rows,t,mode=='raw') for profile,t in cal['thresholds'][name].items()}
                if cohort=='calibration_scans':
                    np.testing.assert_allclose(curve['cpm_interpolated'],cal['curves'][name]['cpm_interpolated'],atol=1e-12,rtol=0)
                else:
                    for profile,m in profiles.items():
                        expected = review['results'][cohort][name][profile]
                        for k in ('detected','false_positive_candidates','actual_fp_per_scan'):
                            if m[k]!=expected[k]:
                                raise ValueError('Published review mismatch')
                        if v=='historical':
                            key = 'baseline' if mode=='raw' else 'continuity'
                            old = old_review['results'][cohort][key][profile]
                            if (m['detected'],m['false_positive_candidates'])!=(old['detected'],old['false_positive_candidates']):
                                raise ValueError('First cross-slice reference failed to reproduce')
                summaries[cohort][name] = profiles
                refs[name] = {profile:reference_details(rows,t,mode=='raw') for profile,t in cal['thresholds'][name].items()}
            if v!='historical':
                timing[v] = dict(forward_transfer_mean_seconds=float(np.mean([
                    r['packet']['timing']['forward_transfer_seconds'] for r in rows])),
                    local_feature_mean_seconds=float(np.mean([r['packet']['evidence']['feature_seconds'] for r in rows])),
                    mean_tiles=float(np.mean([r['packet']['schedule']['lung_tiles'] for r in rows])),
                    scope='paired CUDA/CPU component observations; not independent end-to-end or FPGA benchmark')
        comparisons = {}
        for before,after in [('historical_cross','small_continued_cross'),
                             ('historical_cross','wide_continued_cross'),
                             ('small_continued_cross','wide_continued_cross')]:
            comparisons[after+'_vs_'+before] = {profile:changes(refs[before][profile],refs[after][profile])
                                                 for profile in ('1.0','4.0','8.0')}
        summaries[cohort]['reference_changes'] = comparisons
        summaries[cohort]['reference_details'] = refs
        summaries[cohort]['component_timing'] = timing
    pooled_summary, pooled_refs = {}, {}
    curves['review18_posthoc_diagnostic'] = {}
    for v,rows in pooled.items():
        for mode in ('raw','cross'):
            name = v+'_'+mode
            scores = {r['uid']:r['raw_scores' if mode=='raw' else 'scores'] for r in rows}
            curve = evaluate(rows,scores)
            curve_audits['review18__'+name]=check_froc_stable(rows,scores,curve)
            curves['review18_posthoc_diagnostic'][name] = curve
            pooled_summary[name] = {profile:counts(rows,t,mode=='raw') for profile,t in cal['thresholds'][name].items()}
            pooled_refs[name] = {profile:reference_details(rows,t,mode=='raw') for profile,t in cal['thresholds'][name].items()}
    pooled_summary['reference_changes'] = {}
    for before in ('historical_cross','small_continued_cross'):
        pooled_summary['reference_changes']['wide_continued_cross_vs_'+before] = {
            profile:changes(pooled_refs[before][profile],pooled_refs['wide_continued_cross'][profile])
            for profile in ('1.0','4.0','8.0')}
    pooled_summary['size_groups_at_frozen4'] = {}
    pooled_summary['matched_center_error_at_frozen4'] = {}
    for name in all_names:
        detail = pooled_refs[name]['4.0']
        errors = [r['center_error_mm'] for r in detail if r['detected']]
        pooled_summary['matched_center_error_at_frozen4'][name] = dict(
            detected=len(errors), mean_mm=float(np.mean(errors)) if errors else None,
            median_mm=float(np.median(errors)) if errors else None,
            scope='only detected reference nodules; changing detected sets prevents an all-nodule localization claim')
        pooled_summary['size_groups_at_frozen4'][name] = {}
        for group,lo,hi in [('under6',0,6),('6to10',6,10),('10plus',10,float('inf'))]:
            items = [r for r in detail if lo<=r['diameter_mm']<hi]
            pooled_summary['size_groups_at_frozen4'][name][group] = dict(
                references=len(items),detected=sum(r['detected'] for r in items))
    summaries['review18_pooled_fixed_thresholds'] = pooled_summary
    save(folder/'summary.json',dict(cohorts=summaries,architectures=plan['architectures'],
        historical_model_sha256=plan['initial_checkpoint_sha256'],
        matched_extra_epochs=30, matched_optimizer_updates=6060,
        widened_neuron_max_pair_difference=diversity, initialization=train['initialization'],
        heldout_scans_run=0, fpga_validated=False,
        conclusion_scope='single-seed warm-start width ablation on repeatedly viewed development cohorts',
        causal_comparison='wide_continued versus small_continued; historical reference has fewer updates',
        unchanged='data split, cached catalogue, sampling, augmented tensors, update count, loss, decoder, original ranker',
        not_controlled='parameter/FLOP budget differs; wider gradients/optimization geometry inherently differ'))
    save(folder/'ranking_diagnostic.json',dict(curves=curves,
        warning='review18 curves are post-hoc development diagnostics; their thresholds are NOT deployment settings'))
    save(folder/'curve_audit.json',dict(curves=curve_audits,
        explanation='Independent sklearn ROC recovers integer FP/TP counts before per-scan division; avoids floating boundary side changes',
        frozen_model_thresholds_and_empirical_counts_unchanged=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,4.6))
    colors = {'historical':'#8a8a8a','small_continued':'#b77b29','wide_continued':'#087f8c'}
    for ax,key,title in zip(axes,('calibration_scans','review18_posthoc_diagnostic'),
                            ('Calibration19 (19 reference nodules)','Review18 (28 references; post-hoc curves)')):
        for v in pooled:
            for mode,style in [('raw',':'),('cross','-')]:
                c = curves[key][v+'_'+mode]['curve']
                ax.plot(c['fp_per_scan'],c['sensitivity'],color=colors[v],linestyle=style,
                        label=v+' '+mode)
        ax.set(xlim=(0,8),ylim=(0,1),xlabel='False positives / complete CT',ylabel='Sensitivity',title=title)
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,loc='lower right')
    fig.suptitle('Width capacity: same paired batches and updates; unchanged first cross-slice rule')
    fig.tight_layout()
    fig.savefig(folder/'capacity_froc.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,index,title in zip(axes,range(3),('Total training loss','Center focal loss','Offset / size loss')):
        for v in VARIANTS:
            ax.plot([h['epoch'] for h in history],[h['loss'][v][index] for h in history],
                    label=v,color=colors[v])
        ax.set(xlabel='Matched extra epoch',ylabel='Minibatch mean',title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Identical sampled tensors; training loss is not independent detection performance')
    fig.tight_layout()
    fig.savefig(folder/'training_loss.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.7,4.3))
    for v in pooled:
        data=[pooled_summary[v+'_cross'][p] for p in ('1.0','4.0','8.0')]
        ax.plot([m['actual_fp_per_scan'] for m in data],[m['sensitivity'] for m in data],
                'o-',color=colors[v],label=v)
        for profile,m in zip(('cal1','cal4','cal8'),data):
            ax.annotate(profile,(m['actual_fp_per_scan'],m['sensitivity']),fontsize=8,xytext=(4,4),
                        textcoords='offset points',color=colors[v])
    ax.set(xlabel='Measured false positives / CT on review18',ylabel='Sensitivity (28 references)',
           ylim=(0,1.05),title='Frozen calibration19 thresholds transferred to review18')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder/'fixed_profile_tradeoff.png',dpi=160)
    plt.close(fig)
    first.verify(out)
    save(out/'audit.json',dict(status='PASS',independent_froc_and_fixed_counts=True,
        original_ranker_unchanged=True,historical_reference_reproduced=True,
        geometries_and_native_boxes_checked=True,training_minibatches_shared=True,
        optimizer_updates_per_arm=6060,protected_files_unchanged=len(plan['protected_sha256']),
        inference_packet_sha256=identities,report_source_sha256=sha_file(__file__),
        curve_audit_source_sha256=sha_file(Path(__file__).with_name('lung_nodule_capacity_froc_audit.py')),
        curve_audit_tests_sha256=sha_file(Path(__file__).parent/'tests/test_lung_nodule_capacity_froc_audit.py'),
        heldout_scans_run=0,fpga_validated=False))
    for name in all_names:
        fixed = pooled_summary[name]
        curve = curves['review18_posthoc_diagnostic'][name]
        print('REVIEW18',name,'CPM',round(curve['cpm_interpolated'],4),
              [(p,m['detected'],round(m['actual_fp_per_scan'],3)) for p,m in fixed.items()],flush=True)
    print('CAPACITY AUDIT PASS',len(identities),'cached packets',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=OUT)
    run(parser.parse_args().outdir.resolve())
