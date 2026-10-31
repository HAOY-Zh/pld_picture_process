"""Audit returned physical nodule heads/timings against the frozen local package."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq'))
from nodule_plan import load_package,sha


def verify(package,result):
    folder,manifest,model,_=load_package(package);result=Path(result).resolve()
    d=json.loads((result/'result.json').read_text(encoding='utf-8'))
    if (d['status']!='physical_board_exact_all_nodes' or not d['board_tested']
            or d['machine'] not in ['armv7l','armv7'] or abs(d['clock_mhz']-100)>.1
            or d['manifest_sha256']!=sha(folder/'nodule_manifest.json')
            or d['bit_sha256']!=sha(folder/'lung_network.bit')
            or d['actual_heads_sha256']!=sha(result/'actual_heads.npz')):
        raise ValueError('Wrong/incomplete physical-board identity')
    with np.load(result/'actual_heads.npz',allow_pickle=False) as a, np.load(folder/'nodule_reference.npz',allow_pickle=False) as ref:
        for i in range(4):
            for prefix in ['', 'cpu_']:
                actual=a[prefix+f'case{i}_head']
                if actual.dtype!=np.uint8 or not np.array_equal(actual,ref[f'case{i}_head']):
                    raise ValueError('Returned head differs: '+prefix+str(i))
    names=[n['name'] for n in model['nodes'] for _ in range((n['shape'][0]+7)//8 if n['kind']=='conv' else 1)]
    if len(d['details'])!=d['runs']:raise ValueError('Missing measured repeats')
    for row in d['details']:
        if ([c['node'] for c in row['commands']]!=names or row['command_count']!=15
                or row['pl_command_cycles']!=sum(c['cycles'] for c in row['commands'])
                or row['buffer_allocations_per_frame']!=0 or row['weight_packets_repacked_per_frame']!=0):
            raise ValueError('Incomplete/changed nodule command schedule')
    for key in ['cpu_int8_two_threads','pl_resident_network','pl_patch_with_input_pack','pl_engine_active_cycles_ms']:
        a=np.asarray(d[key]['samples_ms'],float)
        if len(a)!=d['runs'] or not np.isfinite(a).all() or (a<=0).any() or not np.isclose(a.mean(),d[key]['mean_ms']):
            raise ValueError('Invalid timing samples: '+key)
    ratio=d['cpu_int8_two_threads']['mean_ms']/d['pl_resident_network']['mean_ms']
    if not np.isclose(ratio,d['speedup_same_lowered_integer_graph']):raise ValueError('Speedup calculation mismatch')
    proof=dict(status='verified_returned_nodule_patch_benchmark',actual_heads_bit_exact=True,
        manifest_sha256=sha(folder/'nodule_manifest.json'),result_sha256=sha(result/'result.json'),
        mean_cpu_ms=d['cpu_int8_two_threads']['mean_ms'],mean_pl_ms=d['pl_resident_network']['mean_ms'],
        mean_patch_with_pack_ms=d['pl_patch_with_input_pack']['mean_ms'],speedup_same_integer_graph=ratio,
        scope='Four patch heads plus complete measured graph schedule; not full-CT latency or INT8 detection-quality acceptance')
    target=result/'verification.json'
    with target.open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True);p.add_argument('--result-dir',required=True)
    a=p.parse_args();print(json.dumps(verify(a.package,a.result_dir),indent=2))
