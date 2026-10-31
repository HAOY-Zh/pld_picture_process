"""Verify returned five-way measurements without hardware access or reruns."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'pynq'))
from nodule_plan import sha


def verify(folder):
    folder=Path(folder).resolve();r=json.loads((folder/'result.json').read_text())
    addon=ROOT/'data/nodule_direct_lut_20261023_v1/nodule_direct_lut/direct_manifest.json'
    m=json.loads(addon.read_text())
    if (r['status']!='physical_direct_input_and_lut_checked' or not r['same_model_weights_quantization_threshold']
            or r['sampling_algorithm_changed'] or r['lut_all_node_physical_checks']!=4 or abs(r['clock_mhz']-100)>.1):
        raise ValueError('Wrong run/clock/precision checks')
    if (r['addon_manifest_sha256']!=sha(addon) or r['parent_manifest_sha256']!=m['parent_manifest_sha256']
            or r['lut_bit_sha256']!=m['lut_bit_sha256'] or r['flow_bit_sha256']!=m['flow_bit_sha256']):raise ValueError('Wrong code/model/bit')
    if sha(folder/'full_heads.npz')!=r['full_heads_sha256']:raise ValueError('Changed physical outputs')
    names=['canvas_ARM','native_ARM','canvas_PL','direct_PL','direct_LUT_PL']
    with np.load(folder/'full_heads.npz') as z:
        if set(z.files)!=set(names) or z[names[0]].shape!=(5,224,224):raise ValueError('Incomplete canvas comparison')
        for n in names:np.testing.assert_array_equal(z[n],z[names[0]])
    means={}
    for n in names:
        p=json.loads((folder/(n+'.json')).read_text())
        if p!=r['results'][n]:raise ValueError('Phase packet differs')
        v=np.asarray([a['slice_sampling_to_candidates_ms'] for a in p['samples']])
        if not len(v) or not np.isfinite(v).all() or min(v)<=0:raise ValueError('Invalid timing')
        means[n]=float(v.mean())
        if abs(means[n]-r['means_ms'][n])>1e-6:raise ValueError('Wrong mean')
    g=dict(direct_input=means['canvas_PL']/means['direct_PL'],lut_only=means['direct_PL']/means['direct_LUT_PL'],
        total_pl=means['canvas_PL']/means['direct_LUT_PL'],same_native_arm_vs_lut_pl=means['native_ARM']/means['direct_LUT_PL'])
    if any(abs(v-r['gains'][k])>1e-9 for k,v in g.items()):raise ValueError('Wrong gain')
    print(json.dumps(dict(status='verified_full_heads_and_five_way_timing',mean_ms=means,gains=g),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);a=p.parse_args();verify(a.result)
