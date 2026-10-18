"""Independently verify returned actual startup arrays for six basic PL modules."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq'))
from ct_platform_service import validate_package, sha, startup_cases, BASIC_OPERATIONS
from ct_basic_processing import process, parameters
from lung_continuous import unpack_words


def verify(package, report):
    package=Path(package).resolve();report=Path(report).resolve()
    destination=report.with_name(report.stem+'_verification.json')
    if destination.exists():raise FileExistsError('Preserve existing board acceptance')
    manifest=validate_package(package);proof=json.loads(report.read_text(encoding='utf-8'))
    if proof.get('schema')!='ct_platform_startup_v1' or proof.get('passed') is not True:
        raise ValueError('Startup did not pass')
    if proof['platform_manifest_sha256']!=sha((package/'platform_manifest.json').read_bytes()):
        raise ValueError('Different actual board package')
    arrays=report.parent/proof['actual_arrays']
    if arrays.parent!=report.parent or sha(arrays.read_bytes())!=proof['actual_arrays_sha256']:
        raise ValueError('Actual output arrays changed')
    expected={op+'_'+tag:(op,tag,hu) for op in BASIC_OPERATIONS for tag,hu in startup_cases()}
    if len(proof['basic'])!=12 or {r['key'] for r in proof['basic']}!=set(expected):
        raise ValueError('Missing/duplicate basic hardware cases')
    checked=[]
    with np.load(arrays,allow_pickle=False) as saved:
        if set(saved.files)!=set(expected)|{'segmentation_head'}:raise ValueError('Missing actual output')
        for row in proof['basic']:
            op,tag,hu=expected[row['key']];p=parameters(dict(operation=op))
            if op=='window' and tag=='patterned':p.update(wl=0,ww=5)
            actual=saved[row['key']]
            if (actual.shape!=(512,512) or actual.dtype!=np.uint8 or row['parameters']!=p or
                row['input_sha256']!=sha(hu.astype('<i2').tobytes()) or
                row['output_sha256']!=sha(actual.tobytes()) or
                row['bit_sha256']!=manifest['basic'][op]['bit_sha256']):
                raise ValueError('Different fixture/output/bit: '+row['key'])
            np.testing.assert_array_equal(actual,process(hu,p))
            stats=row['stats']
            for name in ('dma_ms','preparation_ms','overlay_switch_ms','fclk0_mhz','cpu_mhz'):
                if not isinstance(stats.get(name),(int,float)) or not np.isfinite(stats[name]) or stats[name]<0:
                    raise ValueError('Invalid clock/time evidence')
            if stats['guard_verified'] is not True or stats['input_bytes']!=512*512*4 or stats['output_bytes']!=512*512*4:
                raise ValueError('Incomplete DMA/guard evidence')
            checked.append(row['key'])
        model=json.loads((package/'segmentation/model.json').read_text(encoding='utf-8'))
        with np.load(package/'segmentation/reference.npz',allow_pickle=False) as refs:
            np.testing.assert_array_equal(saved['segmentation_head'],unpack_words(refs['head'],model['nodes'][-1]['shape']))
    identity=proof['segmentation_after_switch']
    if (identity['bit_sha256']!=manifest['segmentation_bit_sha256'] or
        identity['model_sha256']!=manifest['model_sha256'] or identity['startup_command_count']!=51 or
        identity['startup_nodes_checked']!=[n['name'] for n in model['nodes']]):
        raise ValueError('Segmentation after-switch identity/coverage differs')
    result=dict(status='verified_actual_basic_pl_and_after_switch_segmentation_outputs',
        actual_basic_cases=checked,basic_output_pixels=12*512*512,segmentation_head_codes=65536,
        actual_outputs_bit_exact=True,returned_report_sha256=sha(report.read_bytes()),
        returned_arrays_sha256=sha(arrays.read_bytes()),platform_manifest_sha256=proof['platform_manifest_sha256'],
        verifier_sha256=sha(Path(__file__).read_bytes()),
        scope='Synthetic startup test and one bound segmentation reference; not webpage real-CT/whole-system performance')
    with destination.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True)
    p.add_argument('--report',required=True);a=p.parse_args();verify(a.package,a.report)
