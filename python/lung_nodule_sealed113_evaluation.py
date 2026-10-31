"""One user-authorized frozen test; no training or test threshold selection.

The development reader stays sealed. This separate reader accepts only all113
UIDs from an immutable, authorized protocol frozen before any test CT access.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
from io import BytesIO
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from threadpoolctl import threadpool_limits

from lung_candidate_data import sha_file
from lung_candidate_rules_run import ROOT, EXCLUDED
from lung_candidate_run import predict_lungs, excluded_findings
from lung_nodule_cnn import model, config_from_dict
from lung_nodule_cnn_balanced import read, save
from lung_nodule_cnn_hardneg import setup_torch
from lung_nodule_five_bundle_training import verify as verify_training, OUT as TRAIN, LUNG
from lung_nodule_five_bundle_evaluation import RANKER
from lung_nodule_expanded_evaluation import paired_inference, verified_done
from lung_nodule_heldout_eval import save_packet, load_packet, save_npz
from lung_nodule_continuity import rerank_scores
from lung_nodule_cnn_review18 import summarize_fixed, independent_counts
from luna16_archive_audit import parse_header
from luna16_multibundle import read_manifest, records_for

OUT = ROOT/'data/lung_nodule_sealed113_20261022_v1'
PROFILES = (.125, .25, .5, 1., 2., 4., 8.)


def utc():
    return datetime.now(timezone.utc).isoformat()


def test_records(manifest, training, calibration, ranker_training):
    records = records_for(manifest, 'sealed_test')
    uids = [r['series_uid'] for r in records]
    if len(uids) != 113 or len(set(uids)) != 113 or sum(len(r['nodules']) for r in records) != 175:
        raise ValueError('Expected complete sealed113/175, no case selection')
    forbidden = set(training) | set(calibration) | set(ranker_training)
    if set(uids) & forbidden:
        raise ValueError('Sealed test overlaps detector fit, calibration or ranker fit')
    other = {r['series_uid'] for r in manifest['cases'] if r['role'] != 'sealed_test'}
    if set(uids) & other:
        raise ValueError('Test UID has multiple roles')
    if {r['raw_sha256'] for r in records} & {r['raw_sha256'] for r in manifest['cases'] if r['role'] != 'sealed_test'}:
        raise ValueError('Test pixels duplicate development pixels')
    return records


def freeze(out):
    if out.exists():
        raise FileExistsError('Preserve existing protocol; use run to resume')
    parent, manifest = verify_training(TRAIN)
    frozen = read(TRAIN/'model_freeze.json')
    evaluation = read(TRAIN/'evaluation.json')
    ranker = read(RANKER)
    model_path = TRAIN/'model.pt'
    if frozen['status'] != 'COMPLETE' or sha_file(model_path) != frozen['model_sha256']:
        raise ValueError('Detector not a verified completed model')
    ep = read(TRAIN/'evaluation_protocol.json')
    if (evaluation['status'] != 'COMPLETE' or evaluation['models']['expanded']['sha256'] != frozen['model_sha256']
            or ep['ranker_sha256'] != sha_file(RANKER) or ep['ranker_refitted']):
        raise ValueError('Development model/scoring provenance changed')
    calibration = {r['series_uid'] for r in records_for(manifest, 'calibration')}
    if len(calibration) != 75 or set(ep['calibration_uids']) != calibration:
        raise ValueError('Wrong threshold calibration cohort')
    records = test_records(manifest, parent['training_scans'], calibration, ranker['training_scans'])
    thresholds = {}
    for mode in ('raw', 'cross'):
        curve = evaluation['results']['all_calibration']['expanded_'+mode]
        if curve['scans'] != 75 or curve['reference_nodules'] != 94:
            raise ValueError('Incomplete calibration')
        thresholds[mode] = {str(p['fp_budget']): p['threshold'] for p in curve['operating_points']}
        if set(thresholds[mode]) != {str(p) for p in PROFILES}:
            raise ValueError('Missing frozen operating profiles')
    paths = [model_path, LUNG, RANKER, EXCLUDED, TRAIN/'model_freeze.json',
             TRAIN/'evaluation.json', TRAIN/'evaluation_protocol.json', TRAIN/'plan.json',
             TRAIN/'plan.sha256', Path(parent['manifest']), Path(parent['manifest']).with_suffix('.sha256'),
             ROOT/'python/tests/test_lung_nodule_sealed113_evaluation.py']
    paths += [Path(m.__file__).resolve() for m in list(sys.modules.values())
              if getattr(m, '__file__', None) and Path(m.__file__).resolve().parent == ROOT/'python'
              and str(m.__file__).endswith('.py')]
    from lung_unet import make_model  # Include the unchanged segmentation source in the freeze.
    paths += [ROOT/'python/lung_unet.py']
    protected = dict(parent['protected_sha256'])
    protected.update({str(p): sha_file(p) for p in set(paths)})
    plan = dict(version='frozen_sealed113_20261022_v1', frozen_at_utc=utc(),
        user_authorization='2026-10-22: user accepted the proposed fixed-model, fixed-threshold113 sealed test: 做吧',
        training_output=str(TRAIN), manifest=parent['manifest'],
        model=str(model_path), model_sha256=frozen['model_sha256'], lung_model=str(LUNG),
        ranker=str(RANKER), ranker_refitted=False, config=parent['config'],
        training_uids=parent['training_scans'], calibration_uids=sorted(calibration),
        ranker_training_uids=ranker['training_scans'], test_uids=[r['series_uid'] for r in records],
        scans=113, reference_nodules=175, reference_negative_scans=sum(not r['nodules'] for r in records),
        thresholds=thresholds, threshold_source='all75 development CT/94 references only; copied before test images',
        primary=dict(mode='cross', profile='4.0'), secondary_profiles='all seven pre-existing development profiles; raw CNN ablation',
        policy='No model selection, refitting, threshold adjustment, Top-K cap, failed-case exclusion or further training',
        matching='strict center distance < reference diameter/2; reference duplicates and excluded findings ignored',
        inference_annotations=False, device='cuda', lung_batch=16, cnn_batch=48, seed=2026102201,
        bootstrap=dict(seed=2026102202, replicates=2000, unit='whole CT; resample negative scans too'),
        independence='scan UID/raw-pixel independence checked; PatientID independence not proven; not official tenfold',
        test_reuse='After this access these113 are exposed; never describe later tuning on them as blind validation',
        fpga_validated=False, default_web_model_changed=False, protected_sha256=protected)
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for p in set(paths):
        if p.suffix == '.py':
            (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json', plan)
    (out/'plan.sha256').write_text(sha_file(out/'plan.json')+'\n', encoding='utf-8')
    save(out/'progress.json', dict(status='FROZEN', completed=0, expected=113, accessed_uids=[]))
    print('SEALED113 FROZEN: 113CT/175 nodules; fixed primary threshold', thresholds['cross']['4.0'], flush=True)


def verify(out):
    plan = read(out/'plan.json')
    if sha_file(out/'plan.json') != (out/'plan.sha256').read_text().strip():
        raise ValueError('Test plan modified')
    for p, digest in plan['protected_sha256'].items():
        if sha_file(p) != digest:
            raise ValueError('Frozen test input/source modified: '+p)
    manifest = read_manifest(Path(plan['manifest']))
    records = test_records(manifest, plan['training_uids'], plan['calibration_uids'], plan['ranker_training_uids'])
    if [r['series_uid'] for r in records] != plan['test_uids']:
        raise ValueError('Frozen test membership/order changed')
    return plan, manifest, records


def load_test_volume(manifest, uid, plan):
    if plan['version'] != 'frozen_sealed113_20261022_v1' or uid not in plan['test_uids']:
        raise ValueError('Not an authorized sealed-test UID')
    records = {r['series_uid']: r for r in records_for(manifest, 'sealed_test')}
    if set(records) != set(plan['test_uids']) or uid not in records:
        raise ValueError('Sealed membership mismatch')
    r = records[uid]
    source = manifest['archives'][r['archive_id']]
    stat = Path(source['path']).stat()
    if (stat.st_size, stat.st_mtime_ns) != (source['bytes'], source['mtime_ns']):
        raise ValueError('Test archive changed')
    with ZipFile(source['path']) as outer, ZipFile(BytesIO(outer.read(r['source_member']))) as inner:
        header = parse_header(inner.read(uid+'.mhd'), uid)
        raw = inner.read(uid+'.raw')
    if (len(raw) != r['raw_bytes'] or hashlib.sha256(raw).hexdigest() != r['raw_sha256']
            or header['shape_zyx'] != r['shape_zyx'] or header['dtype'] != r['dtype']
            or any(not np.allclose(header[k], r[k], atol=1e-9, rtol=0)
                   for k in ('spacing_zyx_mm', 'origin_lps_mm', 'basis_lps_zyx'))):
        raise ValueError('Sealed image pixel/geometry identity mismatch')
    hu = np.frombuffer(raw, dtype=header['dtype']).reshape(header['shape_zyx'])
    return hu, {k: header[k] for k in ('spacing_zyx_mm', 'origin_lps_mm', 'basis_lps_zyx')}


def checked_summary(rows, threshold):
    result = summarize_fixed(rows, threshold)
    audit = independent_counts(rows, threshold)
    for key in ('detected', 'reference_nodules', 'false_positive_candidates'):
        if result[key] != audit[key]:
            raise ValueError('Independent fixed-threshold count mismatch: '+key)
    return result


def bootstrap(summary, protocol):
    values = np.array([[c['metric']['detected'], c['metric']['reference_nodules'],
                        c['metric']['false_positive_candidates']] for c in summary['cases']])
    rng = np.random.default_rng(protocol['seed'])
    totals = values[rng.integers(len(values), size=(protocol['replicates'], len(values)))].sum(axis=1)
    recall = totals[:, 0]/np.maximum(totals[:, 1], 1)
    fp = totals[:, 2]/len(values)
    return dict(method='percentile whole-CT bootstrap, exploratory95% interval',
                sensitivity_95=np.quantile(recall, [.025, .975]).tolist(),
                fp_per_scan_95=np.quantile(fp, [.025, .975]).tolist(), **protocol)


def evaluate(out, plan, records):
    identity = sha_file(out/'plan.json')
    excluded = excluded_findings(EXCLUDED)
    rows = {mode: [] for mode in ('raw', 'cross')}
    timings = []
    for r in records:
        uid = r['series_uid']; case = out/'cases'/uid
        done = verified_done(case, identity)
        packet = load_packet(case/'prediction.json.gz')
        if (packet['series_uid'] != uid or packet['model_sha256'] != plan['model_sha256']
                or packet['input_raw_sha256'] != r['raw_sha256'] or packet['annotations_used_by_inference']):
            raise ValueError('Wrong test prediction identity')
        timings.append(done['seconds'])
        world = np.array([c['center_lps_xyz_mm'] for c in packet['candidates']]).reshape(-1, 3)
        with np.load(case/'scores.npz', allow_pickle=False) as a:
            for mode in rows:
                scores = a[mode+'_scores']
                if scores.shape != (len(world),) or not np.isfinite(scores).all() or not np.isfinite(world).all():
                    raise ValueError('Invalid candidate scores/coordinates')
                rows[mode].append(dict(uid=uid, world=world, scores=scores.copy(), record=r, excluded=excluded.get(uid, [])))
    results = {mode: {profile: checked_summary(rows[mode], threshold)
                     for profile, threshold in plan['thresholds'][mode].items()} for mode in rows}
    primary = results[plan['primary']['mode']][plan['primary']['profile']]
    report = dict(status='COMPLETE', completed_at_utc=utc(), plan_sha256=identity,
        model_sha256=plan['model_sha256'], scans=len(records), reference_nodules=175,
        reference_negative_scans=plan['reference_negative_scans'], results=results,
        primary=plan['primary'], primary_bootstrap=bootstrap(primary, plan['bootstrap']),
        thresholds_selected_on='development75 only; no test threshold search',
        independent_counting='PASS', model_refitted=False, threshold_tuning=False,
        inference_seconds=dict(total=sum(timings), median=float(np.median(timings)),
                               scope='archive read+U-Net+sampling+CNN+decode+evidence+score+output; GPU PC, not FPGA/web'),
        fpga_validated=False, default_model_changed=False,
        independence=plan['independence'], now_exposed_test_uids=plan['test_uids'])
    save(out/'evaluation.json', report)
    lines = ['# 113例封存测试结果', '',
        '固定445例训练模型及跨层评分；阈值全部来自75例开发集，测试集不调参。', '',
        '| 开发集档位（误报/套CT） | 固定阈值 | 测试检出率 | 检出/参考结节 | 测试实际平均误报/套CT |',
        '|---:|---:|---:|---:|---:|']
    for profile, r in results['cross'].items():
        lines.append(f"| {profile} | {r['threshold']:.9f} | {r['sensitivity']:.1%} | {r['detected']}/{r['reference_nodules']} | {r['actual_fp_per_scan']:.3f} |")
    lines += ['', f"主档位：检出{primary['detected']}/175（{primary['sensitivity']:.1%}），平均误报{primary['actual_fp_per_scan']:.3f}个/整套CT。",
        '包含全部113例，其中36例无参考结节；没有排除低表现病例，没有Top-K截断。',
        '误报按整套CT计算；采用参考中心距离严格小于半径匹配，重复命中与excluded findings按原规则处理。',
        '这是扫描UID及像素去重后的留出测试，并非官方十折或已证实患者级独立的外部临床验证。',
        '本次为电脑GPU上的FP32结果，尚未验证INT8或FPGA速度，也没有替换网页模型。',
        '这113例现已揭示结果，后续若利用它们调参，不再将其称为未见测试集。', '']
    (out/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    save(out/'progress.json', dict(status='COMPLETE', completed=113, expected=113,
        accessed_uids=plan['test_uids'], evaluation_sha256=sha_file(out/'evaluation.json')))
    print('SEALED113 COMPLETE', primary['detected'], '/175', 'FP/CT', primary['actual_fp_per_scan'], flush=True)


def run(out):
    if not out.exists():
        freeze(out)
    plan, manifest, records = verify(out)
    if (out/'evaluation.json').exists():
        if read(out/'progress.json')['evaluation_sha256'] != sha_file(out/'evaluation.json'):
            raise ValueError('Completed test report changed')
        print('Already complete; no inference repeated', flush=True)
        return
    identity = sha_file(out/'plan.json')
    torch = setup_torch(plan['seed'])
    cfg = config_from_dict(plan['config'])
    cp = torch.load(plan['model'], map_location='cpu', weights_only=True)
    if set(cp['training_scans']) != set(plan['training_uids']) or config_from_dict(cp['config']) != cfg:
        raise ValueError('Detector fit/config changed')
    net = model(cfg).cuda().eval(); net.load_state_dict(cp['state']); del cp
    from lung_unet import make_model
    cp = torch.load(plan['lung_model'], map_location='cpu', weights_only=True)
    lung = make_model(cp['base_channels'], cp.get('context_slices', 1)).cuda().eval()
    lung.load_state_dict(cp['model']); del cp
    ranker = read(Path(plan['ranker']))
    accessed = read(out/'progress.json')['accessed_uids']
    failures = []
    completed = []
    with threadpool_limits(limits=1), torch.inference_mode():
        for i, r in enumerate(records, 1):
            uid = r['series_uid']; case = out/'cases'/uid; case.mkdir(parents=True, exist_ok=True)
            if (case/'complete.json').exists():
                verified_done(case, identity); completed.append(uid); continue
            if uid not in accessed:
                accessed.append(uid)
            # Persist exposure BEFORE opening a sealed image, even if interrupted.
            save(out/'progress.json', dict(status='RUNNING', completed=len(completed), expected=113,
                current=uid, accessed_uids=accessed, failures=failures, updated_at_utc=utc()))
            try:
                tick = perf_counter()
                hu, geometry = load_test_volume(manifest, uid, plan)
                raw, body, _ = predict_lungs(hu, geometry, lung, 'cuda', 16); del raw
                output, schedule = paired_inference(hu, body, geometry, {'expanded': net}, cfg, torch)
                candidates, features, evidence = output['expanded']
                scores = np.array([c['score'] for c in candidates])
                cross = rerank_scores(scores, features, ranker)
                packet = dict(series_uid=uid, plan_sha256=identity, model_sha256=plan['model_sha256'],
                    input_raw_sha256=r['raw_sha256'], lung_mask_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
                    geometry=geometry, candidates=candidates, evidence=evidence, schedule=schedule,
                    backend='FP32_CUDA_NOT_FPGA', annotations_used_by_inference=False)
                save_packet(case/'prediction.json.gz', packet)
                save_npz(case/'scores.npz', raw_scores=scores, cross_scores=cross, features=features)
                save(case/'complete.json', dict(series_uid=uid, plan_sha256=identity,
                    artifacts_sha256={n: sha_file(case/n) for n in ('prediction.json.gz', 'scores.npz')},
                    seconds=perf_counter()-tick))
                completed.append(uid)
                print('SEALED113', i, '/113', 'seconds', round(perf_counter()-tick, 1), flush=True)
                del hu, body, output, candidates, features, evidence, packet
            except Exception as error:
                failures.append(dict(uid=uid, error=repr(error)))
                save(case/'error.json', failures[-1])
                print('SEALED113 CASE FAILED', uid, repr(error), flush=True)
                torch.cuda.empty_cache()
            gc.collect()
            save(out/'progress.json', dict(status='RUNNING', completed=len(completed), expected=113,
                accessed_uids=accessed, failures=failures, updated_at_utc=utc()))
    if failures or set(completed) != set(plan['test_uids']):
        raise RuntimeError('Incomplete113 test; no aggregate performance report emitted')
    verify(out)
    evaluate(out, plan, records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(OUT)
    else:
        try:
            run(OUT)
        except Exception as error:
            if (OUT/'progress.json').exists():
                p = read(OUT/'progress.json'); p.update(status='FAILED', error=repr(error), updated_at_utc=utc())
                save(OUT/'progress.json', p)
            raise
