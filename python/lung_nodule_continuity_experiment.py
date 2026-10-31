"""Training-only local-continuity ranker, then cached-CNN development checks.

No CNN retraining, no inference rerun on calibration/review, no heldout entrypoint.
Training proposals come from existing cached training patches (not full scans).
The CNN already saw those scans; cross-fitting applies ONLY to this new ranker.
"""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from lung_candidate_data import load_development_case, sha_file
from lung_candidate_rules_run import ROOT, EXCLUDED, labels_for, reference_weights
from lung_candidate_run import excluded_findings
from lung_candidate_froc import evaluate
from lung_candidate_lowfp_audit import check_froc
import lung_nodule_cnn_experiment as first
from lung_nodule_cnn_balanced import OUT as V3, read, save
from lung_nodule_cnn_hardneg import BASE, DEFAULT as V2, setup_torch
from lung_nodule_cnn import config_from_dict, model, decode, consolidate
from lung_nodule_cnn_review18 import summarize_fixed, independent_counts
from lung_nodule_continuity import (ContinuityConfig, feature_names, evidence_for_packet,
                                    rerank_scores)

OUT = ROOT/'data/lung_nodule_continuity_20261019_v1'
CNN = V3/'balanced_orientation'
REVIEW = ROOT/'data/lung_nodule_cnn_review18_20261019'
CFG = ContinuityConfig()
COHORTS = ('fit_scans', 'calibration_scans', 'review_12', 'reused_6')


def eligible(plan, cohort):
    if cohort not in COHORTS:
        raise ValueError('No heldout entrypoint')
    sets = [set(plan[k]) for k in COHORTS]
    if any(len(plan[k]) != len(set(plan[k])) for k in COHORTS):
        raise ValueError('Duplicate scan')
    if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]):
        raise ValueError('Cohort overlap')
    return first.records_for(plan, cohort)


def parent_packet(cohort, uid):
    if cohort == 'calibration_scans':
        return CNN/cohort/(uid+'.json')
    if cohort in ('review_12', 'reused_6'):
        return REVIEW/cohort/uid/'v3_balanced.json'
    raise ValueError('Fit proposals have their own sampled-cache pipeline')


def freeze(out):
    if out.exists():
        raise FileExistsError('Preserve prior experiments; resume stages instead')
    old = first.verify(V3)
    protected = dict(old['protected_sha256'])
    extra = [CNN/'model.pt', CNN/'model_freeze.json', CNN/'calibration.json',
        V3/'plan.json', V3/'plan.sha256', Path(__file__).resolve(),
        ROOT/'python/lung_nodule_continuity.py', ROOT/'python/tests/test_lung_nodule_continuity.py',
        ROOT/'python/lung_nodule_cnn_review18.py', ROOT/'python/lung_candidate_froc.py',
        ROOT/'python/lung_candidate_lowfp_audit.py']
    for cohort in COHORTS[1:]:
        for uid in old[cohort]:
            extra.append(parent_packet(cohort, uid))
    for p in extra:
        protected[str(p)] = sha_file(p)
    plan = {k:old[k] for k in ('config', *COHORTS)}
    for k in COHORTS:
        eligible(plan, k)
    plan.update(version='candidate_local_native_slice_continuity_v1', seed=2026101905,
        continuity=asdict(CFG), feature_names=feature_names(),
        cnn_model_sha256=read(CNN/'model_freeze.json')['model_sha256'],
        protected_sha256=protected,
        training=dict(proposals='v3 applied to existing v1 and v2 training patches only',
            positives='up to four matched predicted centers per reference; excluded findings ignored',
            negatives='highest 32 plus seeded random 16 remaining per scan',
            regularization_C=.1, balance='each reference total weight one; negatives equal total weight',
            alpha_options=[0., .25, .5, 1.], selection='maximum ranker-only fivefold grouped OOF empirical CPM; ties prefer smaller alpha',
            warning='CNN saw all fit53; OOF concerns new ranker only, sampled-proposal FROC is not whole-volume performance'),
        score_policy='original center logit plus clipped local linear score times training-selected alpha; below .1 unchanged',
        geometry_policy='candidate centers, estimated diameters, bounds and candidate coverage unchanged',
        scope='development experiment; original FP32 lung masks; not PL/INT8/default web',
        heldout_scans_run=0, expert_voxel_masks_used=False, cnn_retrained=False)
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in extra:
        if p.suffix == '.py':
            (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json',plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    print('FROZEN native-slice continuity: fit53, cached calibration19; sealed45 refused',flush=True)


def select_training(candidates, record, excluded, rng):
    world = np.array([c['center_lps_xyz_mm'] for c in candidates]).reshape(-1,3)
    labels = labels_for(world,record,excluded)
    scores = np.array([c['score'] for c in candidates])
    positive = set()
    for n in record['nodules']:
        hits = np.flatnonzero(np.linalg.norm(world-n['center_lps_xyz_mm'],axis=1) < n['diameter_mm']/2)
        positive.update(hits[np.argsort(-scores[hits],kind='stable')[:4]].tolist())
    negative = np.flatnonzero(labels == 0)
    negative = negative[np.argsort(-scores[negative],kind='stable')]
    chosen = negative[:32].tolist()
    if len(negative) > 32:
        chosen += rng.choice(negative[32:],min(16,len(negative)-32),replace=False).tolist()
    selected = sorted(positive | set(chosen))
    return [candidates[j] for j in selected], labels[selected]


def train_cache(out):
    plan = first.verify(out)
    torch = setup_torch(plan['seed'])
    cfg = replace(config_from_dict(plan['config']),decode_floor=CFG.score_floor)
    ckpt = torch.load(CNN/'model.pt',map_location='cpu',weights_only=True)
    if set(ckpt['training_scans']) != set(plan['fit_scans']):
        raise ValueError('CNN training identity differs')
    net = model(cfg).cuda().eval()
    net.load_state_dict(ckpt['state'])
    split, records = eligible(plan,'fit_scans')
    ignored = excluded_findings(EXCLUDED)
    folder = out/'fit_scans'
    folder.mkdir(exist_ok=True)
    identity = sha_file(out/'plan.json')
    with threadpool_limits(limits=1), torch.inference_mode():
        for i, r in enumerate(records,1):
            uid = r['series_uid']
            dest = folder/(uid+'.json')
            if dest.exists():
                old = read(dest)
                if old['plan_sha256'] != identity or old['input_raw_sha256'] != r['raw_sha256']:
                    raise ValueError('Changed completed training cache')
                continue
            tick = perf_counter()
            geom = {k:r[k] for k in ('spacing_zyx_mm','basis_lps_zyx','origin_lps_mm')}
            accumulated = [[],[],[]]
            for cache in (BASE/'patches',V2/'hard_patches'):
                metadata = read(cache/(uid+'.json'))
                images = np.load(cache/(uid+'.npy'),mmap_mode='r',allow_pickle=False)
                for begin in range(0,len(images),48):
                    image = np.array(images[begin:begin+48],copy=True)
                    tensor = torch.from_numpy(image).cuda().float().clamp(-1000,400).add_(1000).div_(1400)
                    jobs = [d['job'] for d in metadata['labels'][begin:begin+48]]
                    pred = net(tensor).cpu().numpy()
                    for store, values in zip(accumulated,decode(pred,jobs,geom,cfg)):
                        store.append(values)
            values = [np.concatenate(a) for a in accumulated]
            inside = ((values[0] >= 0) & (values[0] <= np.asarray(r['shape_zyx'])-1)).all(axis=1)
            candidates = consolidate(*(v[inside] for v in values),geom,cfg)
            rng = np.random.default_rng(int(hashlib.sha256((str(plan['seed'])+uid).encode()).hexdigest()[:16],16))
            candidates, labels = select_training(candidates,r,ignored.get(uid,[]),rng)
            hu, actual = load_development_case(split,r)
            if actual != geom:
                raise ValueError('Changed native geometry')
            features, evidence = evidence_for_packet(hu,geom['spacing_zyx_mm'],candidates)
            save(dest,dict(series_uid=uid,plan_sha256=identity,input_raw_sha256=r['raw_sha256'],
                model_sha256=plan['cnn_model_sha256'],candidates=candidates,features=features.tolist(),
                labels=labels.tolist(),evidence=evidence,geometry=geom,
                total_seconds=perf_counter()-tick,sampled_training_proposals=True))
            print('TRAIN_EVIDENCE',i,'/',len(records),'positive',int((labels==1).sum()),
                'negative',int((labels==0).sum()),'seconds',round(perf_counter()-tick,2),flush=True)
            del hu,images,tensor,accumulated,features
            gc.collect()


def rows_for(out, cohort):
    plan = first.verify(out)
    _, records = eligible(plan,cohort)
    ignored = excluded_findings(EXCLUDED)
    rows = []
    for r in records:
        uid = r['series_uid']
        packet = read(out/cohort/(uid+'.json'))
        if (packet['plan_sha256'] != sha_file(out/'plan.json') or
                packet['input_raw_sha256'] != r['raw_sha256'] or packet['model_sha256'] != plan['cnn_model_sha256']):
            raise ValueError('Changed completed evidence identity')
        candidates = packet['candidates']
        rows.append(dict(uid=uid,record=r,excluded=ignored.get(uid,[]),packet=packet,
            features=np.asarray(packet['features']).reshape(len(candidates),len(feature_names())),
            world=np.array([c['center_lps_xyz_mm'] for c in candidates]).reshape(-1,3),
            scores=np.array([c['score'] for c in candidates])))
    return rows


def fit_linear(rows):
    x = np.concatenate([r['features'] for r in rows])
    y = np.concatenate([np.asarray(r['packet']['labels']) for r in rows])
    w = np.concatenate([reference_weights(r['world'],r['record']) for r in rows])
    if not (y==1).any() or not (y==0).any():
        raise ValueError('Need both matched references and background')
    w[y==0] = w[y==1].sum()/(y==0).sum()
    w *= len(w)/w.sum()
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=.1,max_iter=2000,solver='lbfgs')
    clf.fit(scaler.transform(x),y,sample_weight=w)
    return dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
        coef=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0]),alpha=1.)


def fit(out):
    plan = first.verify(out)
    if (out/'ranker.json').exists():
        return
    rows = rows_for(out,'fit_scans')
    alphas = plan['training']['alpha_options']
    oof = {str(a):{} for a in alphas}
    folds = []
    # One row per scan; never split patches from the same scan across folds.
    for train_ids, val_ids in GroupKFold(5).split(np.zeros(len(rows)),groups=[r['uid'] for r in rows]):
        train_rows = [rows[i] for i in train_ids]
        val_rows = [rows[i] for i in val_ids]
        ranker = fit_linear(train_rows)
        folds.append(dict(training_scans=[r['uid'] for r in train_rows],
                          validation_scans=[r['uid'] for r in val_rows]))
        for alpha in alphas:
            ranker['alpha'] = alpha
            for r in val_rows:
                oof[str(alpha)][r['uid']] = rerank_scores(r['scores'],r['features'],ranker)
    curves = {str(a):evaluate(rows,oof[str(a)]) for a in alphas}
    # Discrete empirical rates: no selecting through interpolation artifacts.
    selected = max(alphas,key=lambda a:(curves[str(a)]['mean_empirical_sensitivity'],-a))
    ranker = fit_linear(rows)
    ranker.update(alpha=selected,feature_names=feature_names(),training_scans=plan['fit_scans'],
        cnn_model_sha256=plan['cnn_model_sha256'],plan_sha256=sha_file(out/'plan.json'),
        score_kind='bounded_local_continuity_ranking_not_calibrated_probability')
    save(out/'training_selection.json',dict(folds=folds,curves=curves,selected_alpha=selected,
        selected_on='sampled fit53 ranker-only cross-fitting; CNN is in-sample, not a detector OOF result',
        matched_training_candidates=int(sum(sum(np.asarray(r['packet']['labels'])==1) for r in rows)),
        background_training_candidates=int(sum(sum(np.asarray(r['packet']['labels'])==0) for r in rows))))
    save(out/'ranker.json',ranker)
    save(out/'ranker_freeze.json',dict(ranker_sha256=sha_file(out/'ranker.json'),
        plan_sha256=sha_file(out/'plan.json'),calibration_not_read=True,review_not_read=True,
        training_cache_sha256={str(out/'fit_scans'/(r['uid']+'.json')):sha_file(out/'fit_scans'/(r['uid']+'.json')) for r in rows}))
    print('RANKER_FROZEN alpha',selected,'OOF sampled empirical CPM',
          {a:round(c['mean_empirical_sensitivity'],5) for a,c in curves.items()},flush=True)


def verify_ranker(out):
    f = read(out/'ranker_freeze.json')
    if sha_file(out/'ranker.json') != f['ranker_sha256'] or sha_file(out/'plan.json') != f['plan_sha256']:
        raise ValueError('Changed frozen local ranker')
    return read(out/'ranker.json')


def cached_evidence(out, cohort):
    plan = first.verify(out)
    verify_ranker(out)  # freeze all train-only choices before reading calibration CT
    if cohort not in COHORTS[1:]:
        raise ValueError('Only declared cached development cohorts')
    if cohort != 'calibration_scans' and not (out/'calibration.json').exists():
        raise ValueError('Freeze calibration thresholds before review')
    split, records = eligible(plan,cohort)
    folder = out/cohort
    folder.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for i,r in enumerate(records,1):
            uid = r['series_uid']
            dest = folder/(uid+'.json')
            if dest.exists():
                packet = read(dest)
                if packet['plan_sha256'] != sha_file(out/'plan.json') or packet['parent_sha256'] != sha_file(parent_packet(cohort,uid)):
                    raise ValueError('Changed cached local evidence')
                continue
            tick = perf_counter()
            source = parent_packet(cohort,uid)
            original = read(source)
            if original['model_sha256'] != plan['cnn_model_sha256'] or original['input_raw_sha256'] != r['raw_sha256']:
                raise ValueError('Wrong parent prediction/CT')
            hu, geom = load_development_case(split,r)
            if geom != original['geometry'] or list(hu.shape) != original['source_shape_zyx']:
                raise ValueError('Native geometry mismatch')
            features, evidence = evidence_for_packet(hu,geom['spacing_zyx_mm'],original['candidates'])
            save(dest,dict(series_uid=uid,plan_sha256=sha_file(out/'plan.json'),
                model_sha256=plan['cnn_model_sha256'],input_raw_sha256=r['raw_sha256'],
                parent_path=str(source),parent_sha256=sha_file(source),geometry=geom,
                candidates=original['candidates'],features=features.tolist(),evidence=evidence,
                total_seconds=perf_counter()-tick,cnn_forward_performed=False))
            print('CACHED_EVIDENCE',cohort,i,'/',len(records),'examined',evidence['candidates_examined'],
                  'feature_seconds',round(evidence['feature_seconds'],3),'total_seconds',round(perf_counter()-tick,2),flush=True)
            del hu,features
            gc.collect()


def score_rows(rows, ranker):
    return {r['uid']:rerank_scores(r['scores'],r['features'],ranker) for r in rows}


def calibrate(out):
    first.verify(out)
    if (out/'calibration.json').exists():
        return
    ranker = verify_ranker(out)
    rows = rows_for(out,'calibration_scans')
    baseline = {r['uid']:r['scores'] for r in rows}
    after = score_rows(rows,ranker)
    curves = dict(baseline=evaluate(rows,baseline),continuity=evaluate(rows,after))
    for name,scores in [('baseline',baseline),('continuity',after)]:
        check_froc(rows,scores,curves[name])
    # Verify baseline reproduces pre-existing v3, before comparing new scores.
    old = read(CNN/'calibration.json')['cnn']
    if [(p['detected'],p['actual_fp_per_scan']) for p in old['operating_points']] != [
            (p['detected'],p['actual_fp_per_scan']) for p in curves['baseline']['operating_points']]:
        raise ValueError('Existing v3 calibration did not reproduce')
    timing = [r['packet']['evidence']['feature_seconds'] for r in rows]
    calibration = dict(curves=curves,ranker_sha256=sha_file(out/'ranker.json'),
        selected_alpha=ranker['alpha'],thresholds={name:{str(p['fp_budget']):p['threshold'] for p in c['operating_points']
        if p['fp_budget'] in (1.,4.,8.)} for name,c in curves.items()},
        extra_feature_seconds=dict(mean=float(np.mean(timing)),median=float(np.median(timing)),maximum=max(timing)),
        calibration_scans=19,cnn_rerun=False,localization_unchanged=True,candidate_coverage_unchanged=True,
        no_hard_slice_count_filter=True,heldout_scans_run=0,
        feature_cache_sha256={str(out/'calibration_scans'/(r['uid']+'.json')):sha_file(out/'calibration_scans'/(r['uid']+'.json')) for r in rows})
    save(out/'calibration.json',calibration)
    save(out/'calibration_freeze.json',dict(calibration_sha256=sha_file(out/'calibration.json'),
        ranker_sha256=sha_file(out/'ranker.json'),review_not_run=True))
    print('CALIBRATION_COMPLETE', {name:[p['detected'] for p in c['operating_points'] if p['fp_budget'] in (1.,4.,8.)]
        for name,c in curves.items()}, 'extra mean seconds',round(np.mean(timing),3),flush=True)


def review_report(out):
    first.verify(out)
    ranker = verify_ranker(out)
    cal = read(out/'calibration.json')
    if sha_file(out/'calibration.json') != read(out/'calibration_freeze.json')['calibration_sha256']:
        raise ValueError('Calibration changed before review')
    if (out/'review.json').exists():
        return
    report = {}
    for cohort in COHORTS[2:]:
        rows = rows_for(out,cohort)
        predictions = score_rows(rows,ranker)
        result = {}
        for name in ('baseline','continuity'):
            used = [dict(r,scores=predictions[r['uid']]) if name == 'continuity' else r for r in rows]
            result[name] = {}
            for profile,threshold in cal['thresholds'][name].items():
                metric = summarize_fixed(used,threshold)
                counts = independent_counts(used,threshold)
                for k,v in counts.items():
                    if metric[k] != v:
                        raise AssertionError('Independent fixed-threshold count disagreement')
                result[name][profile] = metric
        report[cohort] = result
    save(out/'review.json',dict(results=report,ranker_sha256=sha_file(out/'ranker.json'),
        calibration_sha256=sha_file(out/'calibration.json'),thresholds_recalibrated_on_review=False,
        review_cnn_forwards=0,heldout_scans_run=0))
    print('REVIEW_COMPLETE', {k:{n:dict(detected=v['4.0']['detected'],fp=v['4.0']['actual_fp_per_scan'])
        for n,v in value.items()} for k,value in report.items()},flush=True)


def run(out,stage):
    if stage in ('all-dev','freeze') and not out.exists():
        freeze(out)
    if stage in ('all-dev','train'):
        train_cache(out)
        fit(out)
    if stage in ('all-dev','calibrate'):
        cached_evidence(out,'calibration_scans')
        calibrate(out)
    if stage == 'review':
        for cohort in COHORTS[2:]:
            cached_evidence(out,cohort)
        review_report(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=OUT)
    parser.add_argument('--stage',choices=['all-dev','freeze','train','calibrate','review'],default='all-dev')
    args = parser.parse_args()
    run(args.outdir,args.stage)
