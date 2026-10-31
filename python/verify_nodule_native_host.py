"""Verify returned four-way host optimization data without rerunning inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from nodule_plan import load_package,sha
from lung_nodule_cnn import consolidate,config_from_dict


def verify(folder,package):
    folder=Path(folder).resolve();package=Path(package).resolve()
    m=json.loads((package/'host_manifest.json').read_text());r=json.loads((folder/'result.json').read_text())
    for n,h in m['files_sha256'].items():
        p=(package/n).resolve()
        if package not in p.parents or sha(p)!=h:raise ValueError('Changed host package '+n)
    if (r['status']!='physical_exact_native_host_four_way_pass' or not r['same_bit_model_sampling_values_quantization_threshold']
        or not r['synthetic_float_planes_exact'] or not r['real_context_exact'] or r['all_node_physical_checks']!=4
        or abs(r['clock_mhz']-100)>.1):raise ValueError('Incomplete run/clock/precision checks')
    if (r['manifest_sha256']!=sha(package/'host_manifest.json') or r['parent_manifest_sha256']!=m['parent_manifest_sha256']
        or r['bit_sha256']!=sha(package/'lung_network.bit')):raise ValueError('Wrong bit/model/runtime')
    for key,n in [('contexts_sha256','contexts.npz'),('full_heads_sha256','full_heads.npz'),('raw_decode_sha256','raw_decode.npz'),
                  ('all_nodes_sha256','physical_all_nodes.npz'),('candidates_sha256','candidates.json')]:
        if r[key]!=sha(folder/n):raise ValueError('Changed physical output '+n)
    with np.load(folder/'contexts.npz') as z:
        if set(z.files)!={'previous','optimized'} or z['previous'].shape!=(5,448,448) or z['previous'].dtype!=np.int16:
            raise ValueError('Incomplete raw input comparison')
        np.testing.assert_array_equal(z['previous'],z['optimized'])
        if hashlib.sha256(z['previous'].tobytes()).hexdigest()!=r['context_sha256']:raise ValueError('Wrong context identity')
    names=['previous_ARM','optimized_ARM','previous_PL','optimized_PL']
    with np.load(folder/'full_heads.npz') as z:
        if set(z.files)!=set(names) or z[names[0]].shape!=(5,224,224):raise ValueError('Missing full heads')
        for n in names:
            if not np.isfinite(z[n]).all():raise ValueError('Nonfinite head')
            np.testing.assert_array_equal(z[n],z[names[0]])
    with np.load(folder/'raw_decode.npz') as z:
        keys=['points','scores','diameters']
        if set(z.files)!={n+'_'+k for n in names for k in keys}:raise ValueError('Incomplete raw decode outputs')
        for n in names:
            for k in keys:np.testing.assert_array_equal(z[n+'_'+k],z[names[0]+'_'+k])
        raw=tuple(z[names[0]+'_'+k].copy() for k in keys)
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';_,_,model,_=load_package(parent)
    if sha(parent/'nodule_manifest.json')!=m['parent_manifest_sha256']:raise ValueError('Wrong canonical reference')
    with np.load(parent/'nodule_reference.npz') as ref,np.load(folder/'physical_all_nodes.npz') as z:
        expected={'case'+str(i)+'_'+n['name'] for i in range(4) for n in model['nodes']}
        if set(z.files)!=expected:raise ValueError('Incomplete physical all-node capture')
        for n in expected:np.testing.assert_array_equal(z[n],ref[n])
    cfg=config_from_dict(model['detector_config']);g=json.loads((package/'slice_case.json').read_text())['local_geometry']
    candidates=json.loads((folder/'candidates.json').read_text())
    if candidates!=consolidate(*raw,g,cfg):raise ValueError('Raw decode does not match final candidates')
    means={};network={};active={}
    for n in names:
        p=json.loads((folder/(n+'.json')).read_text())
        if p!=r['results'][n] or p['status']!='exact_context_head_raw_decode_candidates' or len(p['samples'])!=r['runs']:raise ValueError('Wrong phase packet')
        for key,value in p['mean_ms'].items():
            values=[s['details']['pl_command_cycles']/100000. if key=='hardware_active_ms' else s[key] for s in p['samples']]
            if not np.isfinite(values).all() or min(values)<0 or abs(np.mean(values)-value)>1e-6:raise ValueError('Wrong phase mean '+key)
        for s in p['samples']:
            if s['plane_candidate_count']!=len(candidates):raise ValueError('Wrong timed candidate count')
            if s['decoder']!=('exact_core_vectorized' if n.startswith('optimized') else 'original_scalar'):raise ValueError('Wrong decoder')
            if s['sampler']!=('native_xy_same_z_numpy' if n.startswith('optimized') else 'original_scipy'):raise ValueError('Wrong sampler')
            if n.endswith('_PL'):
                commands=s['details']['commands']
                if len(commands)!=15 or sum(v['cycles'] for v in commands)!=s['details']['pl_command_cycles']:raise ValueError('Incomplete PL graph')
        means[n]=p['mean_ms']['slice_sampling_to_candidates_ms'];network[n]=p['mean_ms']['network_ms']
        if means[n]<=0 or network[n]<=0 or abs(means[n]-r['means_ms'][n])>1e-6 or abs(network[n]-r['resident_network_call_ms'][n])>1e-6:
            raise ValueError('Wrong overall mean')
        if n.endswith('_PL'):
            active[n]=p['mean_ms']['hardware_active_ms']
            if abs(active[n]-r['hardware_active_ms'][n])>1e-6:raise ValueError('Wrong hardware active mean')
    gains=dict(arm_host_before_after=means['previous_ARM']/means['optimized_ARM'],pl_host_before_after=means['previous_PL']/means['optimized_PL'],
               same_optimized_arm_vs_pl=means['optimized_ARM']/means['optimized_PL'],resident_call_arm_vs_pl=network['optimized_ARM']/network['optimized_PL'])
    if any(abs(v-r['gains'][k])>1e-9 for k,v in gains.items()):raise ValueError('Wrong gains')
    if not all(flag in r['sampling_build']['command'] for flag in ['-fopenmp','-ffp-contract=off','-fno-fast-math']):raise ValueError('Wrong sampling compiler flags')
    if r['sampling_build']['source_sha256']!=sha(package/'nodule_sampling_native.cpp'):raise ValueError('Wrong native sampling source')
    result=dict(status='independently_verified_native_host_four_way',means_ms=means,resident_network_call_ms=network,
        hardware_active_ms=active,gains=gains,candidates=len(candidates),all_node_arrays_checked=44,result_sha256=sha(folder/'result.json'))
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--package',default=str(ROOT/'data/nodule_native_host_20261024_v1/nodule_host_fast'))
    a=p.parse_args();verify(a.result,a.package)
