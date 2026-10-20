"""Audit returned LDCT startup arrays and segmentation after the LDCT bit switch."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq'))
from ct_studio_service import validate_package
from ldct_contract import sha
from lung_continuous import unpack_words


def audit(folder,report):
    folder=Path(folder).resolve(); report=Path(report).resolve()
    manifest=validate_package(folder)
    result=json.loads(report.read_text(encoding='utf-8'))
    if (result.get('passed') is not True or result.get('error') is not None
            or result['platform_manifest_sha256']!=sha(folder/'platform_manifest.json')
            or result['ldct']!=manifest['ldct']):
        raise ValueError('Startup failed or deployment identity differs')
    arrays=(report.parent/result['arrays_file']).resolve()
    if arrays.parent!=report.parent or sha(arrays)!=result['arrays_sha256']:
        raise ValueError('Returned array identity differs')
    cases=json.loads((folder/'ldct/fixtures/cases.json').read_text())
    checks={row['case']:row['stats'] for row in result['checks']}
    if len(checks)!=len(cases) or set(checks)!={c['name'] for c in cases}:
        raise ValueError('Incomplete LDCT startup cases')
    with np.load(arrays,allow_pickle=False) as actual:
        if set(actual.files)!={c['name'] for c in cases}|{'segmentation_after_ldct'}:
            raise ValueError('Incomplete returned outputs')
        for case in cases:
            np.testing.assert_array_equal(actual[case['name']],np.load(folder/'ldct'/case['expected'],allow_pickle=False))
            for key in ('actual_pl_mhz_before','actual_pl_mhz_after'):
                if not np.isfinite(checks[case['name']][key]) or abs(checks[case['name']][key]-manifest['ldct']['clock_mhz'])>.5:
                    raise ValueError('Wrong recorded LDCT clock')
        model=json.loads((folder/'segmentation/model.json').read_text())
        with np.load(folder/'segmentation/reference.npz',allow_pickle=False) as reference:
            np.testing.assert_array_equal(actual['segmentation_after_ldct'],unpack_words(reference['head'],model['nodes'][-1]['shape']))
    identity=result['segmentation_after_ldct']
    if (len(identity['startup_nodes_checked'])!=21 or identity['startup_command_count']!=51
            or identity['bit_sha256']!=sha(folder/'segmentation/lung_network.bit')
            or identity['model_sha256']!=sha(folder/'segmentation/model.json')):
        raise ValueError('Missing segmentation after-switch checks')
    return dict(status='PASS_RECORDED_STARTUP_ARRAYS',ldct_cases=len(cases),
        segmentation_after_ldct_bit_exact=True,platform_manifest_sha256=result['platform_manifest_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.package,a.report),indent=2))
