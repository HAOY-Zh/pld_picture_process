"""Audit a returned physical same-plane canvas benchmark."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'pynq'))
from nodule_plan import sha


def verify(folder):
    folder=Path(folder).resolve();p=json.loads((folder/'result.json').read_text())
    if p['status']!='physical_canvas_same_plane_pass' or p['weights_bit_threshold_changed']:
        raise ValueError('Not a valid unchanged-model physical run')
    if abs(p['clock_mhz']-100.)>.1 or p['canvas_hw']!=[448,448] or p['selected_tiles']!=32:
        raise ValueError('Wrong clock/plane')
    addon=ROOT/'data/nodule_canvas_20261023_v1/nodule_canvas/canvas_manifest.json'
    if p['addon_manifest_sha256']!=sha(addon):raise ValueError('Different addon')
    a=json.loads(addon.read_text())
    if p['parent_manifest_sha256']!=a['parent_manifest_sha256']:raise ValueError('Wrong parent')
    if sha(folder/'required_heads.npz')!=p['required_heads_sha256']:raise ValueError('Changed output receipt')
    phases=['previous_ARM','canvas_ARM','previous_PL','canvas_PL']
    with np.load(folder/'required_heads.npz') as z:
        if set(z.files)!=set(phases):raise ValueError('Incomplete comparison')
        first=z[phases[0]]
        if first.shape!=(32,5,34,34):raise ValueError('Wrong comparison region')
        for name in phases:np.testing.assert_array_equal(first,z[name])
    means={}
    for name in phases:
        phase=json.loads((folder/(name+'.json')).read_text())
        if phase!=p['results'][name]:raise ValueError('Phase receipt differs')
        values=np.asarray([r['slice_sampling_to_candidates_ms'] for r in phase['samples']])
        mean=float(values.mean())
        if not len(values) or not np.isfinite(values).all() or min(values)<=0 or abs(mean-p['means_ms'][name])>1e-6:
            raise ValueError('Invalid timing '+name)
        means[name]=mean
    gains=dict(pl_before_after=means['previous_PL']/means['canvas_PL'],arm_before_after=means['previous_ARM']/means['canvas_ARM'],
        same_canvas_arm_vs_pl=means['canvas_ARM']/means['canvas_PL'])
    if any(abs(v-p['gains'][k])>1e-9 for k,v in gains.items()):raise ValueError('Wrong gain')
    print(json.dumps(dict(status='verified_physical_required_heads_equal',mean_ms=means,gains=gains),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);a=p.parse_args();verify(a.result)
