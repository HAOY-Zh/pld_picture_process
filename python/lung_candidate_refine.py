"""Apply annotation-free 3-D structure refinement to saved development candidates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from lung_candidate_data import load_development_case, read_split, sha_file, write_json
from lung_candidate_run import aggregate, excluded_findings, preview
from lung_candidate_structure import StructureConfig, refine_candidates
from lung_candidates import point_metrics


def run(args):
    split = read_split(args.split)
    records = {r['series_uid']: r for r in split['cases']}
    baseline = json.loads((args.baseline/'report.json').read_text())
    if baseline['status'] != 'COMPLETE' or baseline['summary']['heldout_scans_run'] != 0:
        raise ValueError('Need completed development baseline')
    for row in baseline['cases']:
        if records[row['series_uid']]['split'] != 'development':
            raise ValueError('Held-out candidate refinement refused')
    cfg = StructureConfig(**json.loads(args.params.read_text()) if args.params else {}).checked()
    excluded = excluded_findings(args.excluded)
    output = args.outdir.resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True)
    sources = [args.split, args.excluded, args.baseline/'report.json',
               Path(__file__), Path(__file__).with_name('lung_candidate_structure.py'),
               Path(__file__).with_name('lung_candidates.py')]
    frozen = {str(p.resolve(strict=True)): sha_file(p) for p in sources}
    protocol = dict(scope='development structure refinement, CPU_REFERENCE; no training or FPGA',
        frozen_sources=frozen, parameters=asdict(cfg), heldout_execution_allowed=False,
        candidate_input='baseline candidates only; reference annotations used after inference for scoring',
        baseline=str(args.baseline.resolve()))
    write_json(output/'protocol.json', protocol)
    report = dict(status='RUNNING', expected_scans=len(baseline['cases']), cases=[], failures=[],
                  protocol_sha256=sha_file(output/'protocol.json'), scope=protocol['scope'])
    aggregate(report, output)
    start = perf_counter()
    try:
        for i, old_row in enumerate(baseline['cases'], 1):
            uid = old_row['series_uid']
            record = records[uid]
            source = args.baseline/uid/'candidates.json'
            baseline_result = json.loads(source.read_text())
            hu, geometry = load_development_case(split, record)
            print(f'[{i}/{len(baseline["cases"])}] 3-D structure: {uid}', flush=True)
            candidates, features, elapsed = refine_candidates(hu, baseline_result['candidates'], geometry, cfg)
            result = {k: v for k, v in baseline_result.items() if k not in ('candidates', 'stage_centers_index_zyx')}
            result.update(algorithm_id='lung_candidates_2d_log_3d_hessian_v2', candidates=candidates,
                structure_parameters=asdict(cfg), structure_source_sha256=frozen[str(Path(__file__).with_name('lung_candidate_structure.py').resolve())],
                baseline_candidates_sha256=sha_file(source), backend='CPU_REFERENCE',
                timings_ms=dict(**baseline_result['timings_ms'], structure=elapsed,
                    candidate_total=baseline_result['timings_ms']['total']+elapsed))
            folder = output/uid
            folder.mkdir()
            write_json(folder/'candidates.json', result)
            write_json(folder/'all_structure_features.json', dict(
                baseline_candidate_ids=[c['candidate_id'] for c in baseline_result['candidates']], features=features))
            metrics = point_metrics([c['center_lps_xyz_mm'] for c in candidates], record['nodules'], excluded.get(uid, ()))
            write_json(folder/'metrics.json', dict(metrics=metrics, baseline_metrics=old_row['metrics']))
            previews = preview(hu, result, record['nodules'], geometry, folder)
            report['cases'].append(dict(series_uid=uid, metrics=metrics, baseline_metrics=old_row['metrics'],
                lung_ms=old_row['lung_ms'], candidate_ms=result['timings_ms']['candidate_total'],
                structure_ms=elapsed, previews=previews))
            aggregate(report, output)
            print(f'  detected={metrics["detected"]}/{metrics["reference_nodules"]}; '
                  f'candidates={len(candidates)}; FP={metrics["false_positive_candidates"]}; '
                  f'structure_s={elapsed/1000:.2f}', flush=True)
            del hu, features, result, candidates, baseline_result
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
    parser.add_argument('--excluded', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--params', type=Path)
    run(parser.parse_args())
