"""Paired calibration47 evaluation; no training or sealed-test image entrypoint."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from lung_candidate_data import sha_file, read_split
from lung_candidate_rules_run import ROOT, SPLIT, EXCLUDED
from lung_candidate_run import predict_lungs, excluded_findings
from lung_nodule_cnn import config_from_dict, model, search_mask, tiles, decode, consolidate
from lung_nodule_cnn_balanced import read, save
from lung_nodule_cnn_hardneg import setup_torch
from lung_nodule_sampling_cache import sample_patch_reuse
from lung_nodule_whole_background import FloatPlaneCache
from lung_nodule_continuity_boundary import evidence_for_packet
from lung_nodule_continuity import rerank_scores
from lung_nodule_heldout_eval import save_npz, save_packet, load_packet
from lung_nodule_expanded_training import OUT as TRAIN, INITIAL, LUNG
from luna16_multibundle import OUT as DATA, read_manifest, records_for, load_development_volume

OUT = ROOT/'data/lung_nodule_expanded_calibration_20261020_v1'
BASE_CACHE = INITIAL.parent/'calibration_scans'
RANKER = ROOT/'data/lung_nodule_continuity_20261019_boundary_v2/ranker.json'
LABELS = ('old53', 'expanded247')


def calibration_records(manifest):
    records = records_for(manifest, 'calibration')
    uids = [r['series_uid'] for r in records]
    train = {r['series_uid'] for r in records_for(manifest, 'train')}
    sealed = {r['series_uid'] for r in records_for(manifest, 'sealed_test')}
    if len(uids) != len(set(uids)) or set(uids) & (train | sealed):
        raise ValueError('Calibration UID isolation failed')
    return records


def validate_cached(packet, record, digest, body_digest=None):
    if (packet['series_uid'] != record['series_uid'] or packet['input_raw_sha256'] != record['raw_sha256'] or
            packet['model_sha256'] != digest or packet['source_shape_zyx'] != record['shape_zyx'] or
            body_digest is not None and packet['lung_mask_sha256'] != body_digest):
        raise ValueError('Cached inference identity mismatch')
    features = np.asarray(packet['features'], dtype=float).reshape(-1, 38)
    if len(features) != len(packet['candidates']) or not np.isfinite(features).all():
        raise ValueError('Invalid cached features')
    return features


def freeze(out):
    if out.exists():
        raise FileExistsError('Preserve frozen calibration; run to resume')
    manifest = read_manifest(DATA/'manifest.json')
    records = calibration_records(manifest)
    if len(records) != 47 or sum(len(r['nodules']) for r in records) != 56:
        raise ValueError('Expected all calibration47/56')
    f = read(TRAIN/'model_freeze.json')
    t = read(TRAIN/'plan.json')
    if (sha_file(TRAIN/'model.pt') != f['model_sha256'] or f['optimizer_updates'] != f['expected_updates'] or
            len(f['epochs']) != 30 or f['amp_skipped_updates'] != 0):
        raise ValueError('Expanded training not successfully completed')
    paths = [DATA/'manifest.json', TRAIN/'model.pt', TRAIN/'model_freeze.json', TRAIN/'plan.json',
             INITIAL, LUNG, RANKER, EXCLUDED]
    sources = [Path(__file__).resolve(), ROOT/'python/tests/test_lung_nodule_expanded_evaluation.py']
    sources += [ROOT/'python'/n for n in ('lung_nodule_cnn.py', 'lung_nodule_sampling_cache.py',
        'lung_nodule_continuity.py', 'lung_nodule_continuity_boundary.py', 'lung_nodule_heldout_eval.py',
        'lung_candidate_froc.py', 'lung_nodule_capacity_froc_audit.py', 'lung_nodule_cnn_review18.py',
        'lung_candidate_run.py', 'lung_body_gate.py', 'lung_candidates.py', 'lung_nodule_cnn_experiment.py')]
    reused = {}
    for r in records:
        path = BASE_CACHE/(r['series_uid']+'.json')
        if r['origin_cohort'] == 'original135':
            validate_cached(read(path), r, sha_file(INITIAL))
            reused[r['series_uid']] = str(path)
            paths.append(path)
    if len(reused) != 19:
        raise ValueError('Expected original19 reusable baseline outputs')
    protected = {str(p):sha_file(p) for p in paths+sources}
    plan = dict(version='paired_calibration47_expanded13k_v1', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest=str(DATA/'manifest.json'), manifest_sha256=sha_file(DATA/'manifest.json'),
        calibration_uids=[r['series_uid'] for r in records], calibration_references=56,
        models={'old53':dict(path=str(INITIAL), sha256=sha_file(INITIAL)),
                'expanded247':dict(path=str(TRAIN/'model.pt'), sha256=f['model_sha256'])},
        training_uids=t['training_scans'], config=t['config'], reused_baseline=reused,
        lung_checkpoint=str(LUNG), lung_sha256=sha_file(LUNG),
        ranker=str(RANKER), ranker_sha256=sha_file(RANKER), ranker_refitted=False,
        scoring='raw CNN and identical frozen first 38-feature cross-slice ranker, alpha=.25',
        primary='recall at <=4 false positives per full CT on calibration47; also 1 and 8 profiles',
        inference='same input batches, predicted lungs, physical grid, decode and NMS; no reference positions',
        calibration_is_independent_test=False, interpretation='development calibration, not new56 test or official ten-fold',
        attribution_limit='More independent fit cases plus additional continuation and new-background mining teacher',
        sealed_test_scans_accessed=0, fpga_validated=False, seed=2026102006, protected_sha256=protected)
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in sources:
        (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json', plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n')
    print('FROZEN paired47 / refs56 / baseline19 cached / new56 SEALED', flush=True)


def verify(out):
    identity = sha_file(out/'plan.json')
    if identity != (out/'plan.sha256').read_text().strip():
        raise ValueError('Calibration plan changed')
    plan = read(out/'plan.json')
    manifest = read_manifest(Path(plan['manifest']))
    if {r['series_uid'] for r in calibration_records(manifest)} != set(plan['calibration_uids']):
        raise ValueError('Calibration cohort changed')
    for path, digest in plan['protected_sha256'].items():
        if sha_file(path) != digest:
            raise ValueError('Protected calibration artifact changed: '+path)
    return plan, manifest


def paired_inference(hu, body, geometry, nets, cfg, torch):
    """Native images, masks and geometry only: annotations are not accepted."""
    spacing = np.asarray(geometry['spacing_zyx_mm'])
    search = search_mask(body, spacing, cfg)
    jobs, schedule = tiles(search, spacing, cfg)
    cached = FloatPlaneCache(hu)
    accum = {name:[[], [], []] for name in nets}
    for begin in range(0, len(jobs), 48):
        group = jobs[begin:begin+48]
        tensor = torch.from_numpy(np.stack([sample_patch_reuse(cached, spacing, j, cfg) for j in group])).cuda().float()
        tensor = tensor.clamp(-1000, 400).add_(1000).div_(1400)
        for name, net in nets.items():
            prediction = net(tensor).cpu().numpy()
            for dest, value in zip(accum[name], decode(prediction, group, geometry, cfg, search)):
                dest.append(value)
    result = {}
    for name in nets:
        candidates = consolidate(*(np.concatenate(a) for a in accum[name]), geometry, cfg) if jobs else []
        for c in candidates:
            box = np.asarray(c['bbox_zyx']).reshape(3, 2)
            box[:, 0] = np.maximum(box[:, 0], 0)
            box[:, 1] = np.minimum(box[:, 1], hu.shape)
            c['bbox_zyx'] = box.ravel().tolist()
        features, evidence = evidence_for_packet(hu, spacing, candidates)
        result[name] = (candidates, features, evidence)
    return result, schedule


def verified_done(folder, identity):
    done = read(folder/'complete.json')
    if done['plan_sha256'] != identity:
        raise ValueError('Wrong completed inference plan')
    for name, digest in done['artifacts_sha256'].items():
        if Path(name).name != name or sha_file(folder/name) != digest:
            raise ValueError('Completed inference modified')
    return done


def infer(out):
    plan, manifest = verify(out)
    identity = sha_file(out/'plan.json')
    records = calibration_records(manifest)
    torch = setup_torch(plan['seed'])
    from lung_unet import make_model
    import lung_nodule_cnn_experiment as first
    cfg = config_from_dict(plan['config'])
    nets = {}
    for name, info in plan['models'].items():
        cp = torch.load(info['path'], map_location='cpu', weights_only=True)
        if config_from_dict(cp['config']) != cfg or set(cp['training_scans']) & set(plan['calibration_uids']):
            raise ValueError('Checkpoint configuration/training isolation mismatch')
        if name == 'expanded247' and set(cp['training_scans']) != set(plan['training_uids']):
            raise ValueError('Expanded model wrong training cohort')
        nets[name] = model(cfg).cuda().eval()
        nets[name].load_state_dict(cp['state'])
    cp = torch.load(LUNG, map_location='cpu', weights_only=True)
    lung = make_model(cp['base_channels'], cp.get('context_slices', 1)).cuda().eval()
    lung.load_state_dict(cp['model'])
    ranker = read(Path(plan['ranker']))
    split = read_split(SPLIT)
    completed = []
    with threadpool_limits(limits=1), torch.inference_mode():
        for i, r in enumerate(records, 1):
            tick = perf_counter()
            uid = r['series_uid']
            folder = out/'cases'/uid
            folder.mkdir(parents=True, exist_ok=True)
            if (folder/'complete.json').exists():
                verified_done(folder, identity)
                completed.append(uid)
                print('REUSE EVALUATED', i, '/', len(records), flush=True)
                continue
            hu, geometry = load_development_volume(manifest, uid, 'calibration')
            old_packet = None
            if uid in plan['reused_baseline']:
                old_packet = read(Path(plan['reused_baseline'][uid]))
                body = first.load_mask(r, split)
                print('CALIBRATION', i, '/47', 'reuse old baseline and lungs', flush=True)
            else:
                raw, body, _ = predict_lungs(hu, geometry, lung, 'cuda', 16)
                del raw
                print('CALIBRATION', i, '/47', 'shared new lungs and paired CNNs', flush=True)
            body_sha = hashlib.sha256(body.tobytes()).hexdigest()
            active = {'expanded247':nets['expanded247']} if old_packet else nets
            result, schedule = paired_inference(hu, body, geometry, active, cfg, torch)
            if old_packet:
                features = validate_cached(old_packet, r, plan['models']['old53']['sha256'], body_sha)
                result['old53'] = (old_packet['candidates'], features, old_packet['evidence'])
            names, counts = [], {}
            for name in LABELS:
                candidates, features, evidence = result[name]
                raw_scores = np.asarray([c['score'] for c in candidates])
                cross = rerank_scores(raw_scores, features, ranker)
                packet = dict(series_uid=uid, plan_sha256=identity, model_sha256=plan['models'][name]['sha256'],
                    input_raw_sha256=r['raw_sha256'], lung_mask_sha256=body_sha,
                    source_shape_zyx=list(hu.shape), geometry=geometry, candidates=candidates,
                    evidence=evidence, baseline_output_reused=name == 'old53' and old_packet is not None,
                    schedule=schedule, backend='FP32_CUDA_NOT_FPGA', annotations_used_by_inference=False)
                a, b = name+'.json.gz', name+'.npz'
                save_packet(folder/a, packet)
                save_npz(folder/b, raw_scores=raw_scores, cross_scores=cross, features=features)
                names.extend((a, b))
                counts[name] = len(candidates)
            save(folder/'complete.json', dict(series_uid=uid, plan_sha256=identity,
                artifacts_sha256={n:sha_file(folder/n) for n in names}, shared_seconds=perf_counter()-tick))
            completed.append(uid)
            save(out/'inference_progress.json', dict(status='RUNNING', completed=len(completed), expected=47,
                completed_uids=completed, sealed_test_scans_accessed=0))
            print('CAL DONE', i, '/47', counts, 'shared seconds', round(perf_counter()-tick, 2), flush=True)
            del hu, body, result, features, evidence, candidates, packet
            gc.collect()
    save(out/'inference_progress.json', dict(status='COMPLETE', completed=47, expected=47,
        completed_uids=completed, sealed_test_scans_accessed=0))


def summarize(out):
    from lung_candidate_froc import evaluate
    from lung_nodule_cnn_review18 import summarize_fixed, independent_counts
    from lung_nodule_capacity_froc_audit import check_froc_stable
    plan, manifest = verify(out)
    identity = sha_file(out/'plan.json')
    records = calibration_records(manifest)
    excluded = excluded_findings(EXCLUDED)
    all_rows = {name:[] for name in LABELS}
    predictions = {name:{mode:{} for mode in ('raw', 'cross')} for name in LABELS}
    for r in records:
        uid = r['series_uid']
        folder = out/'cases'/uid
        verified_done(folder, identity)
        mask_shas = []
        for name in LABELS:
            packet = load_packet(folder/(name+'.json.gz'))
            if packet['model_sha256'] != plan['models'][name]['sha256'] or packet['input_raw_sha256'] != r['raw_sha256']:
                raise ValueError('Wrong evaluated model/input')
            mask_shas.append(packet['lung_mask_sha256'])
            all_rows[name].append(dict(uid=uid, world=np.array([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1, 3),
                record=r, excluded=excluded.get(uid, [])))
            with np.load(folder/(name+'.npz'), allow_pickle=False) as a:
                for mode in ('raw', 'cross'):
                    predictions[name][mode][uid] = a[mode+'_scores'].copy()
        if mask_shas[0] != mask_shas[1]:
            raise ValueError('Paired lung masks differ')
    reports, thresholds, audits = {}, {}, {}
    for name in LABELS:
        for mode in ('raw', 'cross'):
            key = name+'_'+mode
            scores = predictions[name][mode]
            rows = [dict(r, scores=scores[r['uid']]) for r in all_rows[name]]
            report = evaluate(rows, scores)
            audit = check_froc_stable(rows, scores, report)
            selected, fixed = {}, {}
            for point in report['operating_points']:
                if point['fp_budget'] in (1., 4., 8.):
                    profile = str(point['fp_budget'])
                    selected[profile] = point['threshold']
                    fixed[profile] = summarize_fixed(rows, point['threshold'])
                    counts = independent_counts(rows, point['threshold'])
                    for metric in ('detected', 'reference_nodules', 'false_positive_candidates'):
                        if counts[metric] != fixed[profile][metric]:
                            raise AssertionError('Independent fixed count mismatch')
            reports[key] = dict(froc=report, profiles=fixed)
            thresholds[key] = selected
            audits[key] = audit
            main = fixed['4.0']
            print('RESULT', key, str(main['detected'])+'/'+str(main['reference_nodules']),
                'recall', round(main['sensitivity'], 4), 'FP/CT', round(main['actual_fp_per_scan'], 4), flush=True)
    report = dict(status='COMPLETE', cohort='calibration47_not_independent_test', scans=47, reference_nodules=56,
        plan_sha256=identity, models=plan['models'], results=reports, thresholds=thresholds,
        independent_audits=audits, sealed_test_scans_accessed=0, fpga_validated=False)
    save(out/'evaluation.json', report)
    save(out/'calibration_freeze.json', dict(plan_sha256=identity, evaluation_sha256=sha_file(out/'evaluation.json'),
        models=plan['models'], ranker_sha256=plan['ranker_sha256'], thresholds=thresholds,
        threshold_source='complete calibration47 only; fixed before any new56 image', sealed_test_scans_accessed=0))
    lines = ['# 扩充训练后47例开发校准评价', '', '本表是开发校准结果，不是新56例测试或官方十折评价。', '',
        '|模型与评分|误报预算/CT|检出/参考|检出率|实际误报/CT|', '|---|---:|---:|---:|---:|']
    for key, result in reports.items():
        for profile, value in result['profiles'].items():
            lines.append(f"|{key}|{profile}|{value['detected']}/{value['reference_nodules']}|{value['sensitivity']:.1%}|{value['actual_fp_per_scan']:.2f}|")
    lines += ['', '所有47例均计入，包括无参考结节病例。原始CNN及相同冻结跨层评分分别报告；无Top-K截断。',
              '旧模型已存在的19例输出复用，新28例新旧网络共享输入批次和预测肺范围。新56例仍封存。',
              '若要判断泛化改善，应固定本次阈值后运行独立测试；不可将校准结果当作测试结果。', '']
    (out/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('COMPLETE all47 paired calibration and independent count audit; new56 NOT RUN', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('freeze', 'infer', 'summarize', 'run'))
    p.add_argument('--out', type=Path, default=OUT)
    args = p.parse_args()
    out = args.out.resolve()
    if args.stage == 'run':
        if not out.exists():
            freeze(out)
        infer(out)
        summarize(out)
    else:
        globals()[args.stage](out)


if __name__ == '__main__':
    main()
