"""Check the returned four-way same-plane acceleration experiment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq'))
from nodule_plan import load_package,sha


def verify(package,addon,result):
    package,_,_,_=load_package(package);addon=Path(addon).resolve();result=Path(result).resolve()
    a=json.loads((addon/'fast_manifest.json').read_text(encoding='utf-8'))
    for n,h in a['files_sha256'].items():
        p=(addon/n).resolve()
        if addon not in p.parents or sha(p)!=h:raise ValueError('Addon changed')
    d=json.loads((result/'result.json').read_text(encoding='utf-8'))
    if (d['status']!='physical_same_plane_acceleration_checked' or not d['all_tile_outputs_equal']
            or d['weights_bit_config_changed'] or abs(d['clock_mhz']-100.)>.1
            or d['parent_manifest_sha256']!=sha(package/'nodule_manifest.json')
            or d['addon_manifest_sha256']!=sha(addon/'fast_manifest.json')
            or d['plane_predictions_sha256']!=sha(result/'plane_predictions.npz')):
        raise ValueError('Wrong experiment identity')
    phases=['baseline_ARM','optimized_ARM','baseline_PL','optimized_PL'];means={}
    with np.load(result/'plane_predictions.npz',allow_pickle=False) as z:
        ref=z['baseline_ARM']
        for name in phases:
            if not np.array_equal(ref,z[name]):raise ValueError('Returned tile maps differ: '+name)
            rows=d['results'][name]['samples'];xs=np.asarray([r['slice_sampling_to_candidates_ms'] for r in rows])
            if len(xs)==0 or not np.isfinite(xs).all() or (xs<=0).any() or any(r['tiles']!=len(ref) for r in rows):
                raise ValueError('Invalid same-plane timing samples')
            means[name]=float(xs.mean())
            if not np.isclose(means[name],d['results'][name]['timing']['slice_sampling_to_candidates_ms']['mean_ms']):raise ValueError('Wrong timing summary')
    gain=means['baseline_PL']/means['optimized_PL']
    if not np.isclose(gain,d['gains']['pl_before_after']):raise ValueError('Wrong speedup')
    proof=dict(status='verified_physical_same_plane_four_way_outputs',mean_ms=means,pl_before_after_speedup=gain,
        output_maps_equal=True,result_sha256=sha(result/'result.json'),
        scope='Single-plane INT8 acceleration comparison, not whole-case speed or FP32 detection-quality evaluation')
    with (result/'verification.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True);p.add_argument('--addon',required=True)
    p.add_argument('--result-dir',required=True);a=p.parse_args();print(json.dumps(verify(a.package,a.addon,a.result_dir),indent=2))
