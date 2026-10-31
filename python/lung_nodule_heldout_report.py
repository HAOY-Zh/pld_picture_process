"""All45 fixed-threshold metrics and independent cache audit; never tune on tests."""
import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path

import numpy as np

from lung_candidate_data import read_split,sha_file
from lung_candidate_rules_run import SPLIT,EXCLUDED
from lung_candidate_run import excluded_findings
from lung_candidate_froc import evaluate
from lung_nodule_cnn_balanced import read,save
from lung_nodule_cnn_review18 import summarize_fixed,independent_counts
from lung_nodule_continuity import feature_names,rerank_scores
from lung_nodule_capacity_froc_audit import check_froc_stable
from lung_nodule_test_data import test_records,require_complete
from lung_nodule_heldout_eval import OUT,load_packet,verify_case
import lung_nodule_cnn_experiment as first


def bootstrap_fixed(cases,draws=2000,seed=2026102003):
    refs=np.array([r['metric']['reference_nodules'] for r in cases])
    detected=np.array([r['metric']['detected'] for r in cases])
    fp=np.array([r['metric']['false_positive_candidates'] for r in cases])
    indices=np.random.default_rng(seed).integers(0,len(cases),(draws,len(cases)))
    denominator=refs[indices].sum(axis=1)
    valid=denominator>0
    sensitivity=detected[indices].sum(axis=1)[valid]/denominator[valid]
    false_positives=fp[indices].mean(axis=1)
    return dict(case_bootstrap_draws=draws,seed=seed,nonzero_reference_draws=int(valid.sum()),
        sensitivity_95_percentile_interval=np.percentile(sensitivity,[2.5,97.5]).tolist() if valid.any() else None,
        mean_fp_per_scan_95_percentile_interval=np.percentile(false_positives,[2.5,97.5]).tolist(),
        scope='fixed thresholds, resample whole scans; not calibration/model-selection uncertainty')


def references_at(row,threshold,scores):
    details=[]
    for i,n in enumerate(row['record']['nodules']):
        distance=np.linalg.norm(row['world']-n['center_lps_xyz_mm'],axis=1)
        hits=np.flatnonzero(distance<n['diameter_mm']/2)
        best=int(hits[np.argmax(scores[hits])]) if len(hits) else None
        detected=best is not None and scores[best]>=threshold
        if detected:
            reason='detected'
        elif len(hits):
            reason='matching_candidate_exists_below_frozen_threshold'
        elif row['reference_center_in_search'][i]:
            reason='center_inside_search_no_matching_candidate_at_decode_floor'
        else:
            reason='reference_center_outside_predicted_search_no_matching_candidate'
        details.append(dict(reference_index=i,diameter_mm=n['diameter_mm'],detected=bool(detected),
            center_lps_xyz_mm=n['center_lps_xyz_mm'],best_matching_score=float(scores[best]) if best is not None else None,
            best_matching_candidate_id=row['packet']['candidates'][best]['candidate_id'] if best is not None else None,
            best_center_error_mm=float(distance[best]) if best is not None else None,
            reference_center_in_search=row['reference_center_in_search'][i],status=reason))
    return details


def fixed_metrics(rows,threshold,mode,seed):
    used=[dict(r,scores=r['raw_scores'] if mode=='raw' else r['cross_scores']) for r in rows]
    metric=summarize_fixed(used,threshold)
    for key,value in independent_counts(used,threshold).items():
        if metric[key]!=value:
            raise ValueError('Independent fixed-threshold count mismatch')
    for row,case in zip(used,metric['cases']):
        case['reference_details']=references_at(row,threshold,row['scores'])
    fp=np.array([c['metric']['false_positive_candidates'] for c in metric['cases']])
    metric['false_positive_distribution']=dict(median=float(np.median(fp)),p90=float(np.percentile(fp,90)),
        maximum=int(fp.max()),zero_fp_scans=int((fp==0).sum()))
    metric['uncertainty']=bootstrap_fixed(metric['cases'],seed=seed)
    groups={}
    details=[d for c in metric['cases'] for d in c['reference_details']]
    for name,lo,hi in [('under6',0,6),('6to10',6,10),('10plus',10,float('inf'))]:
        group=[d for d in details if lo<=d['diameter_mm']<hi]
        found=sum(d['detected'] for d in group)
        groups[name]=dict(reference_nodules=len(group),detected=found,
            sensitivity=found/len(group) if group else None)
    metric['size_groups']=groups
    metric['missed_reference_reasons']=dict(Counter(d['status'] for d in details if not d['detected']))
    return metric


def verified_rows(out,plan):
    split=read_split(SPLIT)
    records=test_records(split,plan['test_uids'])
    identity=sha_file(out/'plan.json')
    completed=read(out/'inference_complete.json')
    require_complete(plan['test_uids'],completed['completed_uids'])
    require_complete(plan['test_uids'],[p.parent.name for p in (out/'cases').glob('*/complete.json')])
    excluded=excluded_findings(EXCLUDED)
    ranker=read(Path(plan['ranker']))
    rows=[];identities={}
    for r in records:
        uid=r['series_uid'];folder=out/'cases'/uid
        done=verify_case(out,uid,identity)
        for name,digest in done['artifacts_sha256'].items():
            identities[str(folder/name)]=digest
        packet=load_packet(folder/'candidates.json.gz')
        if (packet['input_raw_sha256']!=r['raw_sha256'] or packet['model_sha256']!=plan['cnn_sha256'] or
                packet['ranker_sha256']!=plan['ranker_sha256'] or packet['lung_model_sha256']!=plan['lung_sha256'] or
                packet['plan_sha256']!=identity or packet['split']!='heldout' or packet['parameters']!=plan['config']):
            raise ValueError('Wrong heldout packet/model identity')
        geometry={k:r[k] for k in ('spacing_zyx_mm','origin_lps_mm','basis_lps_zyx')}
        if packet['geometry']!=geometry or packet['source_shape_zyx']!=r['shape_zyx']:
            raise ValueError('Wrong native CT geometry')
        candidates=packet['candidates']
        index=np.array([c['center_index_zyx'] for c in candidates]).reshape(-1,3)
        world=np.array([c['center_lps_xyz_mm'] for c in candidates]).reshape(-1,3)
        np.testing.assert_allclose(world,index@np.asarray(r['basis_lps_zyx']).T+r['origin_lps_mm'],rtol=0,atol=1e-6)
        shape=np.array(r['shape_zyx'])
        q=np.rint(index).astype(int)
        if np.any(q<0) or np.any(q>=shape):
            raise ValueError('Center outside original nearest-voxel gate')
        for c in candidates:
            box=np.asarray(c['bbox_zyx']).reshape(3,2)
            if np.any(box[:,0]<0) or np.any(box[:,1]>shape) or np.any(box[:,1]<box[:,0]):
                raise ValueError('Unclipped candidate bbox')
        with np.load(folder/'scores_features.npz',allow_pickle=False) as a:
            features=a['features'];raw=a['raw_scores'];cross=a['cross_scores'];ids=a['candidate_ids']
        np.testing.assert_array_equal(ids,[c['candidate_id'] for c in candidates])
        np.testing.assert_array_equal(raw,[c['score'] for c in candidates])
        if features.shape!=(len(raw),len(feature_names())) or not np.isfinite(features).all():
            raise ValueError('Invalid local feature matrix')
        selected=raw>=.1
        if np.any(features[~selected]!=0) or packet['evidence']['selected_indices']!=np.flatnonzero(selected).tolist():
            raise ValueError('Unexpected local evidence gate')
        recomputed=rerank_scores(raw,features,ranker)
        np.testing.assert_allclose(cross,recomputed,rtol=0,atol=1e-12)
        # Independent affine application, without the runtime standardization path.
        weights=np.asarray(ranker['coef'])/np.asarray(ranker['scale'])
        bias=ranker['intercept']-weights@np.asarray(ranker['mean'])
        expected=raw.copy()
        p=np.clip(raw[selected],1e-7,1-1e-7)
        z=np.log(p/(1-p))+ranker['alpha']*np.clip(features[selected]@weights+bias,-2,2)
        expected[selected]=1/(1+np.exp(-z))
        np.testing.assert_allclose(cross,expected,rtol=1e-12,atol=1e-12)
        with np.load(folder/'masks_packed.npz',allow_pickle=False) as a:
            if a['shape_zyx'].tolist()!=r['shape_zyx']:
                raise ValueError('Wrong predicted mask geometry')
            body=np.unpackbits(a['body'],count=int(shape.prod())).reshape(shape).astype(bool)
            search=np.unpackbits(a['search'],count=int(shape.prod())).reshape(shape).astype(bool)
        if (hashlib.sha256(body.tobytes()).hexdigest()!=packet['lung_mask_sha256'] or
                hashlib.sha256(search.tobytes()).hexdigest()!=packet['search_mask_sha256']):
            raise ValueError('Predicted mask hash mismatch')
        coverage=[]
        for n in r['nodules']:
            center=np.rint(n['center_index_zyx']).astype(int)
            coverage.append(bool(search[tuple(center)]) if ((center>=0)&(center<shape)).all() else False)
        for profile,threshold in plan['thresholds']['cross'].items():
            exported=read(folder/('profile_'+profile+'.json'))
            wanted=np.flatnonzero(cross>=threshold)
            if (exported['threshold']!=threshold or exported['plan_sha256']!=identity or
                    [c['candidate_id'] for c in exported['candidates']]!=ids[wanted].tolist()):
                raise ValueError('Wrong frozen-threshold output')
            for c,j in zip(exported['candidates'],wanted):
                for key in ('center_index_zyx','center_lps_xyz_mm','bbox_zyx','estimated_scale_diameter_mm'):
                    if c[key]!=candidates[j][key]:
                        raise ValueError('Filtered output changed candidate geometry')
                if c['score']!=cross[j] or c['cnn_score']!=raw[j]:
                    raise ValueError('Filtered output score mismatch')
        rows.append(dict(uid=uid,record=r,excluded=excluded.get(uid,[]),world=world,raw_scores=raw,
            cross_scores=cross,features=features,packet=packet,reference_center_in_search=coverage))
        del body,search
        print('FINAL AUDIT',len(rows),'/',len(records),flush=True)
    return rows,identities


def run(out):
    plan=first.verify(out)
    rows,identities=verified_rows(out,plan)
    require_complete(plan['test_uids'],[r['uid'] for r in rows])
    if sum(len(r['record']['nodules']) for r in rows)!=56:
        raise ValueError('Wrong reference count')
    metrics={};curves={};curve_audits={}
    for mode in ('cross','raw'):
        metrics[mode]={}
        for profile,threshold in plan['thresholds'][mode].items():
            metrics[mode][profile]=fixed_metrics(rows,threshold,mode,plan['seed'])
        scores={r['uid']:r[mode+'_scores'] for r in rows}
        curve=evaluate(rows,scores)
        curve_audits[mode]=check_froc_stable(rows,scores,curve)
        curves[mode]=curve
    primary=metrics[plan['primary']['score']][plan['primary']['profile']]
    timing={}
    for key in ('load_seconds','lung_wall_seconds','detector_wall_seconds','load_lung_detector_wall_seconds'):
        values=[r['packet']['timing'][key] for r in rows]
        timing[key]=dict(mean=float(np.mean(values)),median=float(np.median(values)),
            p90=float(np.percentile(values,90)),maximum=float(np.max(values)))
    evaluation=dict(status='COMPLETE_ALL45_FIXED_TEST',protocol_sha256=sha_file(out/'plan.json'),
        selected_model=plan['selected_model'],model_sha256=plan['cnn_sha256'],
        test_scans=45,reference_nodules=56,with_reference=29,without_reference=16,
        primary_score=plan['primary']['score'],primary_profile=plan['primary']['profile'],
        primary={k:v for k,v in primary.items() if k!='cases'},profiles=metrics,
        inference_timing_seconds=timing,parameters=13157,conv_macs_per_patch=19038208,
        raw_cnn='predeclared secondary ablation with its own original19 calibration thresholds',
        not_official_luna16_tenfold=True,patient_independence_proven=False,
        model_or_threshold_tuning_performed=False,fpga_validated=False,
        test_now_exposed=True)
    save(out/'evaluation.json',evaluation)
    save(out/'froc_diagnostic.json',dict(curves=curves,independent_audits=curve_audits,
        policy='descriptive whole-test curves only; no new operating thresholds selected/exported'))
    with (out/'case_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['score_mode','calibration_profile','series_uid','references','detected','sensitivity','false_positives'])
        for mode,profiles in metrics.items():
            for profile,metric in profiles.items():
                for c in metric['cases']:
                    m=c['metric']
                    writer.writerow([mode,profile,c['series_uid'],m['reference_nodules'],m['detected'],
                                     m['sensitivity'],m['false_positive_candidates']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for mode,color in [('raw','#8a8a8a'),('cross','#087f8c')]:
        data=[metrics[mode][p] for p in ('1.0','4.0','8.0')]
        axes[0].plot([m['actual_fp_per_scan'] for m in data],[m['sensitivity'] for m in data],'o-',
                     label=mode,color=color)
        c=curves[mode]['curve']
        axes[1].plot(c['fp_per_scan'],c['sensitivity'],label=mode,color=color)
    axes[0].set(title='Frozen calibration19 profiles on all45 test CTs',xlabel='Measured false positives / complete CT',
                ylabel='Reference nodule sensitivity',ylim=(0,1.05))
    axes[1].set(title='Descriptive test FROC; no threshold export',xlabel='False positives / complete CT',
                ylabel='Reference nodule sensitivity',xlim=(0,8),ylim=(0,1.05))
    for ax in axes:
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Continued13k CNN, unchanged first cross-slice rule; 45 test scans / 56 references')
    fig.tight_layout();fig.savefig(out/'test_results.png',dpi=160);plt.close(fig)
    lines=['# 肺结节小模型续训版：全部45例测试结果','',
        '模型及阈值在测试CT读取前冻结。三个档位来自原19例开发校准，未使用测试结果重新调参。','',
        '| 原校准档位 | 检出率 | 检出/参考结节 | 平均误报/整卷CT |',
        '|---|---:|---:|---:|']
    for profile,m in metrics['cross'].items():
        lines.append(f"| {profile} | {m['sensitivity']:.1%} | {m['detected']}/{m['reference_nodules']} | {m['actual_fp_per_scan']:.2f} |")
    lines += ['',f"主档位为原校准4档：检出率{primary['sensitivity']:.1%}，平均每例{primary['actual_fp_per_scan']:.2f}个误报。",
        '', '全部45例纳入，包括29例有参考结节、16例无本参考标准收录的结节；不是仅测试阳性病例。',
        '误报单位是一整套CT。中心/直径标注用于检测匹配，不是结节专家轮廓分割。',
        '当前为原FP32肺预测＋FP32候选CNN的电脑CUDA/CPU流程，未进行INT8或FPGA实板验证。',
        '该45例已经用于本次测试，后续根据结果改模型时，不能再称同批数据为新的盲测。','',
        '[完整逐病例/逐结节结果](evaluation.json) · [逐病例CSV](case_metrics.csv) · [结果图](test_results.png) · [冻结协议](plan.json)']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    first.verify(out)
    save(out/'audit.json',dict(status='PASS',all_test_scans=45,reference_nodules=56,
        independent_fixed_threshold_counts=True,independent_roc_integer_counts=True,
        native_geometry_masks_scores_exports_verified=True,
        protected_files_unchanged=len(plan['protected_sha256']),artifacts_sha256=identities,
        protocol_sha256=sha_file(out/'plan.json'),evaluation_sha256=sha_file(out/'evaluation.json'),
        report_source_sha256=sha_file(__file__),model_or_threshold_tuning_performed=False,
        fpga_validated=False))
    for profile,m in metrics['cross'].items():
        print('FINAL TEST RESULT',profile,'recall',round(m['sensitivity'],6),
            str(m['detected'])+'/'+str(m['reference_nodules']),'FP/CT',round(m['actual_fp_per_scan'],6),flush=True)
    print('ALL45 AUDIT PASS; no tuning; supplementary test CPM',curves['cross']['cpm_interpolated'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=OUT)
    run(parser.parse_args().outdir.resolve())
