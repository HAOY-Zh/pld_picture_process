"""Select a development operating point from identity-checked cached features."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from lung_candidate_data import load_development_case, read_split, sha_file, write_json
from lung_candidate_run import aggregate, excluded_findings, preview
from lung_candidate_structure import BALANCED_PILOT_CONFIG, StructureConfig, select_structural_candidates
from lung_candidates import point_metrics


def run(args):
    split = read_split(args.split)
    records = {r['series_uid']: r for r in split['cases']}
    source_report = json.loads((args.features/'report.json').read_text())
    if source_report['status'] != 'COMPLETE' or source_report['summary']['heldout_scans_run'] != 0:
        raise ValueError('Need completed development structure results')
    cfg = StructureConfig(**json.loads(args.params.read_text())) if args.params else BALANCED_PILOT_CONFIG
    cfg.checked()
    output = args.outdir.resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory')
    selected = [r['series_uid'] for r in source_report['cases']]
    if any(records[uid]['split'] != 'development' for uid in selected):
        raise ValueError('Held-out selection refused')
    paths = [args.split, args.features/'report.json', args.baseline/'report.json', args.excluded,
             Path(__file__), Path(__file__).with_name('lung_candidate_structure.py')]
    frozen = {str(p.resolve(strict=True)): sha_file(p) for p in paths}
    output.mkdir(parents=True)
    write_json(output/'structure_params.json', asdict(cfg))
    protocol = dict(scope='development-only operating point after pilot parameter selection; not independent test',
        selected_uids=selected, structure_parameters=asdict(cfg), frozen_sources=frozen,
        baseline=str(args.baseline.resolve()), features=str(args.features.resolve()),
        feature_reuse='No recomputation of unchanged Hessian features; IDs and source candidates hashes checked',
        timing='Cached selection wall time is not full extraction latency; full kernel estimates from original measurements',
        heldout_execution_allowed=False, parameters_selected_on_this_cohort=True)
    write_json(output/'protocol.json', protocol)
    report = dict(status='RUNNING', expected_scans=len(selected), cases=[], failures=[],
                  protocol_sha256=sha_file(output/'protocol.json'), scope=protocol['scope'])
    excluded = excluded_findings(args.excluded)
    start = perf_counter()
    aggregate(report, output)
    try:
        for i, old in enumerate(source_report['cases'], 1):
            uid = old['series_uid']
            record = records[uid]
            original_path = args.baseline/uid/'candidates.json'
            original = json.loads(original_path.read_text())
            feature_result = json.loads((args.features/uid/'candidates.json').read_text())
            cached = json.loads((args.features/uid/'all_structure_features.json').read_text())
            if (sha_file(original_path) != feature_result['baseline_candidates_sha256'] or
                    cached['baseline_candidate_ids'] != [c['candidate_id'] for c in original['candidates']]):
                raise ValueError('Cached features refer to different candidate identity/order')
            t = perf_counter()
            candidates = select_structural_candidates(original['candidates'], cached['features'], original['geometry'], cfg)
            selection_ms = (perf_counter()-t)*1000
            result = {k: v for k, v in feature_result.items() if k != 'candidates'}
            result.update(algorithm_id='lung_suspicious_regions_log_hessian_pilot_v3', candidates=candidates,
                structure_parameters=asdict(cfg), validation_scope=protocol['scope'],
                structure_source_sha256=frozen[str(Path(__file__).with_name('lung_candidate_structure.py').resolve())])
            result['timings_ms']['cached_selection_only'] = selection_ms
            result['timings_ms']['warning'] = 'Full path latency not rerun at selected parameters; cached selection is not FPGA time'
            folder = output/uid
            folder.mkdir()
            write_json(folder/'candidates.json', result)
            metrics = point_metrics([c['center_lps_xyz_mm'] for c in candidates], record['nodules'], excluded.get(uid, ()))
            write_json(folder/'metrics.json', dict(metrics=metrics, original_metrics=old['baseline_metrics']))
            hu, geometry = load_development_case(split, record)
            previews = preview(hu, result, record['nodules'], geometry, folder)
            report['cases'].append(dict(series_uid=uid, metrics=metrics, previews=previews,
                lung_ms=old['lung_ms'], candidate_ms=old['candidate_ms'],
                timing_warning='Component measurements reused from prior run, not new full-path measurement',
                cached_selection_ms=selection_ms))
            aggregate(report, output)
            print(f'[{i}/{len(selected)}] {uid}: {metrics["detected"]}/{metrics["reference_nodules"]}, '
                  f'{len(candidates)} candidates, {metrics["false_positive_candidates"]} FP', flush=True)
            del hu, original, feature_result, cached, candidates, result
        for path, expected in frozen.items():
            if sha_file(path) != expected:
                raise ValueError('Frozen source changed: '+path)
        report['status'] = 'COMPLETE'
    except Exception as error:
        report['status'] = 'FAIL'
        report['failures'].append(dict(error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        report['elapsed_seconds'] = perf_counter()-start
        aggregate(report, output)
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--excluded', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--params', type=Path)
    run(parser.parse_args())
