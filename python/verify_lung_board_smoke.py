"""Validate a user-returned board report against the immutable build package."""
import argparse
import json
import math
from pathlib import Path

from lung_unet import file_sha256
from lung_segmentation import write_json
from verify_lung_axil import check_hashes


def validate_report(report, hashes):
    if report.get('status') != 'passed_real_board_single_block_smoke' or report.get('full_network') is not False:
        raise ValueError('Not a single-block board PASS report')
    for suffix in ['bit', 'hwh']:
        if report.get(suffix+'_sha256') != hashes['lung_axil.'+suffix]:
            raise ValueError('Board artifact identity mismatch: '+suffix)
    expected = [(0, 0, 1), ('same_address_update', None, 1), (1, 0, 63),
                (2, 0, 63), (2, 1, 63), (3, 0, 15), (3, 1, 15)]
    results = report.get('results', [])
    if len(results) != len(expected):
        raise ValueError('Missing or extra board commands')
    for completed, (row, (case, block, words)) in enumerate(zip(results, expected), 1):
        if row.get('case') != case or row.get('output_block') != block or row.get('same') is not True:
            raise ValueError('Wrong board case/block/result order')
        for field, value in [('different_words', 0), ('completed', completed), ('written_words', words)]:
            if type(row.get(field)) is not int or row[field] != value:
                raise ValueError('Wrong board counter: '+field)
        for field in ['cycles', 'cache_hits', 'cache_misses']:
            if type(row.get(field)) is not int or row[field] < (1 if field == 'cycles' else 0):
                raise ValueError('Invalid board statistic: '+field)
        if case != 'same_address_update':
            elapsed = row.get('cpu_call_ms')
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('Invalid host call duration')
    return dict(commands=7, written_words=sum(r['written_words'] for r in results),
                total_engine_cycles=sum(r['cycles'] for r in results), different_words=0)


def audit(root):
    root = Path(root).resolve()
    output = root/'board_smoke_verification.json'
    if output.exists():
        raise ValueError('Do not overwrite board verification')
    build_path = root/'verification.json'; package_path = root/'deploy/package.json'
    build = json.loads(build_path.read_text(encoding='utf-8'))
    if build != json.loads(package_path.read_text(encoding='utf-8')) or not build.get('vendor_reset_and_id_simulated'):
        raise ValueError('Need matching reset-corrected build proof/package')
    check_hashes({**build['evidence_sha256'], **build['packaged_sources_sha256'], **build['reset_evidence_sha256']})
    workspace = Path(__file__).resolve().parents[1]
    for filename, key in [('fpga/build_lung_axil_board.tcl', 'builder_sha256'),
                          ('python/verify_lung_axil_board.py', 'board_auditor_sha256'),
                          ('python/verify_lung_axil_resetfix.py', 'reset_auditor_sha256')]:
        check_hashes({str(workspace/filename):build[key]})
    check_hashes({str(root/'deploy'/name):digest for name, digest in build['deploy_sha256'].items()})
    report_path = root/'lung_axil_board_result_resetfix.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    stats = validate_report(report, build['deploy_sha256'])
    result = dict(status='verified_returned_real_board_single_block_smoke', **stats,
                  report_sha256=file_sha256(report_path), build_proof_sha256=file_sha256(build_path),
                  package_sha256=file_sha256(package_path), verifier_sha256=file_sha256(__file__),
                  bit_sha256=report['bit_sha256'], hwh_sha256=report['hwh_sha256'],
                  full_network=False, original_build_proof_unchanged=True,
                  provenance='User-transferred board JSON matched local artifacts; not a live remote board attestation',
                  warning='Synthetic small cases only; not CT segmentation accuracy or full-network speed')
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board', default='data/lung_axil_board_20261015_resetfix')
    audit(parser.parse_args().board)
