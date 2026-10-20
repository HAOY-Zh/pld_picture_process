"""Independently audit collected borrowed-API outputs and every timing record."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ctdenoise.cnn_fixed import infer_fixed, load_quantized
from ctdenoise.pl_reference import fixed_filter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    bundles = dict(v.split('=',1) for v in a.bundle)
    bundles = {k:Path(v) for k,v in bundles.items()}
    result = json.loads((a.results/'result.json').read_text())
    assert result['status'] == 'PASS' and result['actual_FPGA_execution']
    assert set(result['sequence']) == set(bundles)
    assert result['per_patient_measured_per_version'] >= 20
    checks, statistics = [], []
    hashes = set()
    for label, bundle in bundles.items():
        manifest = json.loads((bundle/'manifest.json').read_text())
        for path, sha in manifest['files_sha256'].items():
            assert digest(bundle/path) == sha
        arrays, spec = load_quantized(bundle/'model')
        hashes.add(spec['weights_sha256'])
        fixtures = json.loads((bundle/'fixtures/cases.json').read_text())
        actual_checks = [c for c in result['checks'] if c['label'] == label]
        assert {c['case'] for c in actual_checks} == {c['name'] for c in fixtures}
        expected_by_case = {}
        for case in fixtures:
            image = np.rint(np.load(bundle/case['input'])).astype(np.int16)
            mode = manifest['methods'][case['method']]
            expected = infer_fixed(image,arrays,spec)[0] if mode == 6 else fixed_filter(image,mode)
            np.testing.assert_array_equal(expected,np.load(bundle/case['expected']))
            expected_by_case[case['name']] = expected
            record = next(c for c in actual_checks if c['case'] == case['name'])
            actual_path = a.results/record['output']
            actual = np.load(actual_path)
            assert actual.dtype == np.int16 and actual.shape == image.shape
            assert digest(actual_path) == record['output_sha256']
            np.testing.assert_array_equal(actual,expected)
            assert record['state_after'] == dict(busy=False,faulted=False,dma_may_access=False,error=None)
            checks.append(dict(label=label,case=case['name'],max_integer_error_hu=0,output=str(actual_path),sha256=digest(actual_path)))
        for phase in [p for p in result['phases'] if p['label'] == label]:
            assert phase['bit_sha256'] == digest(bundle/'overlay/ct_denoise_tile.bit')
            assert phase['hwh_sha256'] == digest(bundle/'overlay/ct_denoise_tile.hwh')
            assert phase['weights_sha256'] == spec['weights_sha256']
            assert phase['model_id'] == spec['checkpoint_sha256']
            assert abs(phase['actual_pl_mhz']-manifest['clock_mhz']) <= .5
            assert phase['cpu_mhz'] == 650.
            assert phase['state'] == dict(busy=False,faulted=False,dma_may_access=False,error=None)
            for patient in ['C067','C077']:
                path = a.results/f"phase_{phase['phase']:02d}_{label}"/(patient+'_timed_actual.npy')
                np.testing.assert_array_equal(np.load(path),expected_by_case[patient])
                checks.append(dict(label=label,case=patient+'_timed_phase'+str(phase['phase']),
                                   max_integer_error_hu=0,output=str(path),sha256=digest(path)))
        for row in [r for r in result['rows'] if r['label'] == label]:
            assert row['bit_exact'] and row['model_id'] == spec['checkpoint_sha256'] and row['tiles'] == 81
            assert row['total_ms'] > 0
            for key in ['actual_pl_mhz_before','actual_pl_mhz_after']:
                assert abs(row[key]-manifest['clock_mhz']) <= .5
        for patient in ['C067','C077']:
            values = [r['total_ms'] for r in result['rows'] if r['label']==label and r['patient']==patient and not r['warmup']]
            assert len(values) == result['per_patient_measured_per_version']
            reported = next(v for v in result['summary'] if v['label']==label and v['patient']==patient)
            assert abs(reported['mean_ms']-np.mean(values)) < 1e-8
            assert abs(reported['median_ms']-np.median(values)) < 1e-8
            assert abs(reported['p95_ms']-np.percentile(values,95)) < 1e-8
        values = [r['total_ms'] for r in result['rows'] if r['label']==label and not r['warmup']]
        statistics.append(dict(label=label,measured_frames=len(values),mean_ms=float(np.mean(values)),
            median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),clock_mhz=manifest['clock_mhz']))
    assert len(hashes) == 1
    payload = dict(status='PASS',actual_FPGA_execution=True,independent_output_checks=checks,
        pooled_timing_statistics=statistics,summary_per_patient=result['summary'],
        every_frame_frequency_verified=True,identical_integer_weights=True,
        resources_clean_after_every_phase=True,board_environment=result['environment'],
        timing_scope=result['timing_scope'],raw_result_sha256=digest(a.results/'result.json'))
    if a.out.exists():
        raise ValueError('Choose a new audit file')
    a.out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(statistics,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--bundle',action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
