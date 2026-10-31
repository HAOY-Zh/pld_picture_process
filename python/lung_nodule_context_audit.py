"""Read-only independent provenance/geometry/score audit; never opens CT images."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import lung_nodule_context_comparison as e


def reference_hits(candidates, scores, record, threshold):
    xyz = np.array([c['center_lps_xyz_mm'] for c in candidates]).reshape(-1, 3)
    selected = np.asarray(scores) >= threshold if threshold is not None else np.zeros(len(scores), bool)
    return {j for j, n in enumerate(record['nodules'])
            if np.any(np.linalg.norm(xyz[selected]-n['center_lps_xyz_mm'], axis=1) < n['diameter_mm']/2)}


def audit(out, variant):
    plan, manifest = e.verify(out)
    folder = out/variant
    report = e.read(folder/'evaluation.json')
    frozen = e.read(folder/'model_freeze.json')
    identity = e.sha_file(out/'plan.json')
    if report['status'] != 'COMPLETE' or frozen['optimizer_updates'] != 29070 or len(frozen['epochs']) != 30:
        raise ValueError('Incomplete training/evaluation')
    if e.sha_file(folder/'model.pt') != frozen['model_sha256']:
        raise ValueError('Changed context model')
    import torch
    cp = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    if (cp['initialized_from_sha256'] != plan['initial_sha256'] or cp['epochs'] != 30 or
            set(cp['training_scans']) != set(plan['training_uids'])):
        raise ValueError('Unmatched initialization/training budget')
    old_cat, new_cat = e.read(e.TRAIN/'training_catalogue.json'), e.read(out/'training_catalogue.json')
    if len(old_cat['items']) != len(new_cat['items']) or len(new_cat['items']) != 23240:
        raise ValueError('Wrong patch count')
    for before, after in zip(old_cat['items'], new_cat['items']):
        if any(before[k] != after[k] for k in ('series_uid', 'label', 'group')):
            raise ValueError('Patch jobs, labels, order or sampling groups changed')
    worker = e.read(out/'worker_identity.json') if (out/'worker_identity.json').exists() else None
    if worker:
        if (worker['plan_sha256'] != identity or
                worker['source_sha256'] != e.sha_file(e.ROOT/'python/lung_nodule_context_eval_worker.py')):
            raise ValueError('Changed worker provenance')
        for p, digest in worker['extra_sources_sha256'].items():
            if e.sha_file(p) != digest:
                raise ValueError('Changed companion dependency')
    baseline_report = e.read(e.CAL/'evaluation.json')
    threshold = report['results']['cross']['profiles']['4.0']['threshold']
    baseline_points = baseline_report['results']['expanded247_cross']['froc']['operating_points']
    baseline_threshold = next(p['threshold'] for p in baseline_points if p['fp_budget'] == 4.)
    ranker, cases, gained, lost, workers, timing_uids = e.read(e.RANKER), [], [], [], 0, []
    for r in e.calibration_records(manifest):
        uid = r['series_uid']
        path = folder/'cases'/uid
        done = e.verified_done(path, identity)
        packet = e.load_packet(path/'prediction.json.gz')
        old_dir = e.CAL/'cases'/uid
        e.verified_done(old_dir, e.sha_file(e.CAL/'plan.json'))
        old = e.load_packet(old_dir/'expanded247.json.gz')
        if (packet['series_uid'] != uid or packet['lung_mask_sha256'] != old['lung_mask_sha256'] or
                packet['model_sha256'] != frozen['model_sha256'] or packet['schedule'] != old['schedule'] or
                packet['geometry'] != old['geometry'] or packet['input_raw_sha256'] != r['raw_sha256']):
            raise ValueError('Inference identity/geometry mismatch')
        if 'worker_source_sha256' in done:
            workers += 1
            if (worker is None or done['worker_source_sha256'] != worker['source_sha256'] or
                    packet.get('companion_source_sha256') != worker['source_sha256']):
                raise ValueError('Wrong worker artifact')
        if packet['timing_probe'] is not None:
            timing_uids.append(uid)
            if 'worker_source_sha256' in done:
                raise ValueError('Companion executed a timing CT')
        candidates = packet['candidates']
        with np.load(path/'scores.npz', allow_pickle=False) as a:
            raw, cross, features = a['raw_scores'].copy(), a['cross_scores'].copy(), a['features'].copy()
        if features.shape != (len(candidates), 38) or not np.isfinite(features).all():
            raise ValueError('Invalid feature array')
        np.testing.assert_array_equal(raw, [c['score'] for c in candidates])
        np.testing.assert_allclose(cross, e.rerank_scores(raw, features, ranker), atol=1e-12, rtol=0)
        geometry = packet['geometry']
        shape = np.array(r['shape_zyx'])
        for c in candidates:
            point = np.asarray(c['center_index_zyx'])
            nearest = np.rint(point).astype(int)
            world = np.asarray(geometry['origin_lps_mm'])+np.asarray(geometry['basis_lps_zyx'])@point
            np.testing.assert_allclose(world, c['center_lps_xyz_mm'], atol=1e-8, rtol=0)
            box = np.asarray(c['bbox_zyx']).reshape(3, 2)
            if (not np.isfinite(point).all() or np.any(nearest < 0) or np.any(nearest >= shape) or
                    np.any(box[:, 0] < 0) or np.any(box[:, 1] > shape) or np.any(box[:, 0] >= box[:, 1])):
                raise ValueError('Out-of-bounds center or box')
        with np.load(old_dir/'expanded247.npz', allow_pickle=False) as a:
            previous_hits = reference_hits(old['candidates'], a['cross_scores'], r, baseline_threshold)
        new_hits = reference_hits(candidates, cross, r, threshold)
        for dest, indices in ((gained, new_hits-previous_hits), (lost, previous_hits-new_hits)):
            dest.extend(dict(series_uid=uid, reference_index=j, diameter_mm=r['nodules'][j]['diameter_mm']) for j in sorted(indices))
        cases.append(dict(series_uid=uid, candidates=len(candidates), cross4_hits=len(new_hits)))
    if set(timing_uids) != set(plan['timing_uids']):
        raise ValueError('Missing predeclared matched timing CT')
    expected = baseline_report['results']['expanded247_cross']['profiles']['4.0']['detected']+len(gained)-len(lost)
    if expected != report['results']['cross']['profiles']['4.0']['detected']:
        raise ValueError('Reference-level gain/loss count mismatch')
    result = dict(status='PASS', variant=variant, plan_sha256=identity,
        audit_source_sha256=e.sha_file(Path(__file__)),
        audit_tests_sha256=e.sha_file(e.ROOT/'python/tests/test_lung_nodule_context_audit.py'),
        evaluation_sha256=e.sha_file(folder/'evaluation.json'), scans=len(cases), references=56,
        exact_patch_jobs_labels_groups_and_order=True, matched_initial_checkpoint=True,
        model_sha256=frozen['model_sha256'], optimizer_updates=29070, worker_cases_verified=workers,
        raw_and_cross_scores_verified=True, geometry_and_boxes_verified=True,
        timing_cts_verified=timing_uids, cross4_gained=gained, cross4_lost=lost,
        cases=cases, sealed_test_scans_accessed=0, ct_images_read=0, fpga_validated=False)
    e.save(folder/'audit.json', result)
    def brief_profiles(profiles):
        return {k:{f:v[f] for f in ('detected', 'reference_nodules', 'sensitivity',
                    'false_positive_candidates', 'actual_fp_per_scan')} for k, v in profiles.items()}
    timing = report['timing_probes']
    baseline_seconds = [t['mean_sample_pipeline_seconds']['baseline5'] for t in timing]
    variant_seconds = [t['mean_sample_pipeline_seconds']['variant'] for t in timing]
    brief = dict(status='AUDITED_COMPLETE', variant=variant, scans=47, references=56,
        results={m:dict(profiles=brief_profiles(r['profiles']),
            cpm=r['froc']['cpm_interpolated'], detectable=r['froc']['detectable'])
            for m, r in report['results'].items()},
        baseline={m:dict(profiles=brief_profiles(baseline_report['results']['expanded247_'+m]['profiles']),
            cpm=baseline_report['results']['expanded247_'+m]['froc']['cpm_interpolated']) for m in ('raw', 'cross')},
        parameters=report['parameters'], training_minutes=report['training_seconds']/60,
        cross4_gained=len(gained), cross4_lost=len(lost),
        matched_pipeline_ratio=sum(variant_seconds)/sum(baseline_seconds),
        matched_pipeline_ratio_per_ct=[t['pipeline_ratio'] for t in timing],
        estimated_full_ct_pipeline_extra_seconds=[t['estimated_full_ct_pipeline_extra_seconds'] for t in timing],
        resident_forward_ratios=[t['resident_forward_seconds_per_patch']['variant']/
            t['resident_forward_seconds_per_patch']['baseline5'] for t in timing],
        timing_scope='192 regular jobs x four declared CTs, two orders; pipeline difference extrapolated; not total end-to-end or FPGA',
        evaluation_sha256=result['evaluation_sha256'], audit_sha256=e.sha_file(folder/'audit.json'),
        fpga_validated=False, sealed_test_scans_accessed=0)
    e.save(folder/'result_brief.json', brief)
    print('AUDIT PASS', variant, '47 CT, gained', len(gained), 'lost', len(lost), 'worker', workers, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=e.OUT)
    parser.add_argument('--variant', choices=tuple(e.VARIANTS), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        audit(args.out.resolve(), args.variant)
