"""Independently verify returned physical three-way pixel-overlap evidence."""
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
    m=json.loads((package/'overlap_manifest.json').read_text());r=json.loads((folder/'result.json').read_text())
    for n,h in m['files_sha256'].items():
        p=(package/n).resolve()
        if package not in p.parents or not p.is_file() or sha(p)!=h:raise ValueError('Changed package '+n)
    if (r['status']!='physical_exact_pixel_overlap_three_way_pass' or not r['model_sampling_quantization_decode_threshold_unchanged']
        or not r['online_sampling_each_run'] or r['all_node_physical_checks_per_bit']!=4 or not 1<=r['runs']<=20
        or not 1<=r['warmup']<=3):raise ValueError('Incomplete precision/run checks')
    if r['manifest_sha256']!=sha(package/'overlap_manifest.json') or r['parent_manifest_sha256']!=m['parent_manifest_sha256']:
        raise ValueError('Wrong runtime/model')
    for n,sub in [('prefetch_PL','baseline'),('overlap_PL','')]:
        if r['bit_sha256'][n]!=sha(package/sub/'lung_network.bit'):raise ValueError('Wrong bit '+n)
        if abs(r['clocks'][n]['fclk_mhz']-100)>.1:raise ValueError('Wrong FCLK')
    if r['clocks']['prefetch_PL']['cpu_mhz']!=r['clocks']['overlap_PL']['cpu_mhz']:raise ValueError('CPU clock changed')
    for n in ['contexts.npz','full_heads.npz','raw_decode.npz','physical_all_nodes.npz','candidates.json']:
        if r['evidence_sha256'][n]!=sha(folder/n):raise ValueError('Changed evidence '+n)
    with np.load(folder/'contexts.npz') as z:
        if set(z.files)!={'previous','optimized'} or z['previous'].shape!=(5,448,448) or z['previous'].dtype!=np.int16:
            raise ValueError('Incomplete input context')
        np.testing.assert_array_equal(z['previous'],z['optimized'])
        if hashlib.sha256(z['previous'].tobytes()).hexdigest()!=r['context_sha256']:raise ValueError('Wrong context')
    names=['native_ARM','prefetch_PL','overlap_PL'];keys=['points','scores','diameters']
    with np.load(folder/'full_heads.npz') as z:
        if set(z.files)!=set(names) or z[names[0]].shape!=(5,224,224):raise ValueError('Missing full heads')
        for n in names:
            if not np.isfinite(z[n]).all():raise ValueError('Nonfinite head')
            np.testing.assert_array_equal(z[n],z[names[0]])
    with np.load(folder/'raw_decode.npz') as z:
        if set(z.files)!={n+'_'+k for n in names for k in keys}:raise ValueError('Incomplete raw decode')
        for n in names:
            for k in keys:np.testing.assert_array_equal(z[n+'_'+k],z[names[0]+'_'+k])
        raw=tuple(z[names[0]+'_'+k].copy() for k in keys)
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';_,_,model,_=load_package(parent)
    if sha(parent/'nodule_manifest.json')!=m['parent_manifest_sha256']:raise ValueError('Wrong canonical references')
    with np.load(parent/'nodule_reference.npz') as ref,np.load(folder/'physical_all_nodes.npz') as z:
        expected={phase+'_case'+str(i)+'_'+n['name'] for phase in names[1:] for i in range(4) for n in model['nodes']}
        if set(z.files)!=expected:raise ValueError('Incomplete physical all-node capture')
        for n in expected:
            prefix='prefetch_PL_' if n.startswith('prefetch_PL_') else 'overlap_PL_'
            np.testing.assert_array_equal(z[n],ref[n[len(prefix):]])
    cfg=config_from_dict(model['detector_config']);g=json.loads((package/'slice_case.json').read_text())['local_geometry']
    candidates=json.loads((folder/'candidates.json').read_text())
    if candidates!=consolidate(*raw,g,cfg):raise ValueError('Final candidates disagree with raw decode')
    means={};network={};active={}
    for n in names:
        p=json.loads((folder/(n+'.json')).read_text())
        if p!=r['results'][n] or p['status']!='exact_context_head_raw_decode_candidates' or len(p['samples'])!=r['runs']:
            raise ValueError('Wrong phase packet '+n)
        for key,value in p['mean_ms'].items():
            vs=[s['details']['pl_command_cycles']/100000. if key=='hardware_active_ms' else s[key] for s in p['samples']]
            if not np.isfinite(vs).all() or min(vs)<0 or abs(np.mean(vs)-value)>1e-6:raise ValueError('Wrong mean '+key)
        for s in p['samples']:
            if s['plane_candidate_count']!=len(candidates) or s['decoder']!='exact_core_vectorized' or s['sampler']!='native_xy_same_z_numpy':
                raise ValueError('Different online pipeline')
            if n.endswith('_PL'):
                commands=s['details']['commands']
                if len(commands)!=15 or sum(c['cycles'] for c in commands)!=s['details']['pl_command_cycles']:raise ValueError('Incomplete PL commands')
        means[n]=p['mean_ms']['slice_sampling_to_candidates_ms'];network[n]=p['mean_ms']['network_ms']
        if means[n]<=0 or network[n]<=0 or abs(means[n]-r['means_ms'][n])>1e-6 or abs(network[n]-r['resident_network_call_ms'][n])>1e-6:
            raise ValueError('Wrong overall timing')
        if n.endswith('_PL'):
            active[n]=p['mean_ms']['hardware_active_ms']
            if active[n]<=0 or abs(active[n]-r['hardware_active_ms'][n])>1e-6:raise ValueError('Wrong hardware counter')
    gains=dict(complete_plane_pl_before_after=means['prefetch_PL']/means['overlap_PL'],
        hardware_active_pl_before_after=active['prefetch_PL']/active['overlap_PL'],same_native_arm_vs_overlap_pl=means['native_ARM']/means['overlap_PL'],
        resident_call_arm_vs_overlap_pl=network['native_ARM']/network['overlap_PL'])
    if any(abs(v-r['gains'][k])>1e-9 for k,v in gains.items()):raise ValueError('Wrong speedup')
    if not all(v in r['sampling_build']['command'] for v in ['-fopenmp','-ffp-contract=off','-fno-fast-math']):raise ValueError('Wrong sampling flags')
    if r['sampling_build']['source_sha256']!=sha(package/'nodule_sampling_native.cpp'):raise ValueError('Wrong sampling source')
    result=dict(status='independently_verified_physical_pixel_overlap',means_ms=means,resident_network_call_ms=network,
        hardware_active_ms=active,gains=gains,candidates=len(candidates),all_node_arrays_checked=88,result_sha256=sha(folder/'result.json'))
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--package',default=str(ROOT/'data/nodule_overlap_release_20261024/nodule_overlap'))
    a=p.parse_args();verify(a.result,a.package)
