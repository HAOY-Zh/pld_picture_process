"""Independently verify returned prefetch results without rerunning the board."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'pynq'))
from nodule_plan import sha,load_package


def verify(folder,package):
    folder=Path(folder).resolve();package=Path(package).resolve()
    m=json.loads((package/'prefetch_manifest.json').read_text());r=json.loads((folder/'result.json').read_text())
    for n,h in m['files_sha256'].items():
        p=(package/n).resolve()
        if package not in p.parents or sha(p)!=h:raise ValueError('Changed package '+n)
    if (r['status']!='physical_prefetch_all_nodes_and_full_canvas_exact' or not r['model_sampling_quantization_threshold_unchanged']
        or r['all_node_physical_checks_per_bit']!=4 or abs(r['clock_mhz']-100)>.1):raise ValueError('Incomplete precision/clock checks')
    if r['manifest_sha256']!=sha(package/'prefetch_manifest.json') or r['parent_manifest_sha256']!=m['parent_manifest_sha256']:
        raise ValueError('Wrong addon/model')
    for k,f in [('baseline_bit_sha256','baseline/lung_network.bit'),('prefetch_bit_sha256','lung_network.bit')]:
        if r[k]!=sha(package/f):raise ValueError('Wrong physical bit identity')
    for key,file in [('full_heads_sha256','full_heads.npz'),('all_nodes_sha256','physical_all_nodes.npz')]:
        if r[key]!=sha(folder/file):raise ValueError('Changed physical output')
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';_,_,model,_=load_package(parent)
    if sha(parent/'nodule_manifest.json')!=m['parent_manifest_sha256']:raise ValueError('Wrong original references')
    names=['native_ARM','baseline_LUT_PL','prefetch_PL'];bits=names[1:]
    with np.load(folder/'full_heads.npz') as z:
        if set(z.files)!=set(names) or z[names[0]].shape!=(5,224,224):raise ValueError('Incomplete full-head comparison')
        if not all(np.isfinite(z[n]).all() for n in names):raise ValueError('Nonfinite physical head')
        for n in names:np.testing.assert_array_equal(z[n],z[names[0]])
    with np.load(parent/'nodule_reference.npz') as ref,np.load(folder/'physical_all_nodes.npz') as actual:
        nodes=[n['name'] for n in model['nodes']]
        expected={name+'_case'+str(i)+'_'+node for name in bits for i in range(4) for node in nodes}
        if len(nodes)!=11 or set(actual.files)!=expected:raise ValueError('Incomplete physical all-node capture')
        for name in bits:
            for i in range(4):
                for node in nodes:np.testing.assert_array_equal(actual[name+'_case'+str(i)+'_'+node],ref['case'+str(i)+'_'+node])
    means={};active={}
    for name in names:
        p=json.loads((folder/(name+'.json')).read_text())
        if p!=r['results'][name] or len(p['samples'])!=r['runs'] or p['status']!='same_full_canvas_and_candidates':raise ValueError('Incomplete phase packet')
        for key,value in p['mean_ms'].items():
            if key=='hardware_active_ms':v=[s['details']['pl_command_cycles']/100000. for s in p['samples']]
            elif key=='execute_wait_ms':v=[s['details']['timing_ms']['execute_wait_ms'] for s in p['samples']]
            else:v=[s[key] for s in p['samples']]
            if not np.isfinite(v).all() or min(v)<0 or abs(np.mean(v)-value)>1e-6:raise ValueError('Wrong phase mean '+key)
        means[name]=p['mean_ms']['slice_sampling_to_candidates_ms']
        if abs(means[name]-r['means_ms'][name])>1e-6 or means[name]<=0:raise ValueError('Wrong plane mean')
        if name in bits:
            active[name]=p['mean_ms']['hardware_active_ms']
            if abs(active[name]-r['hardware_active_ms'][name])>1e-6 or active[name]<=0:raise ValueError('Wrong active cycles')
            for sample in p['samples']:
                commands=sample['details']['commands']
                if len(commands)!=15 or sum(x['cycles'] for x in commands)!=sample['details']['pl_command_cycles']:raise ValueError('Wrong hardware command count')
    gains=dict(complete_plane=means[bits[0]]/means[bits[1]],hardware_active=active[bits[0]]/active[bits[1]],same_native_arm_vs_prefetch_pl=means[names[0]]/means[bits[1]])
    if any(abs(v-r['gains'][k])>1e-9 for k,v in gains.items()):raise ValueError('Wrong gains')
    result=dict(status='independently_verified_physical_prefetch',means_ms=means,hardware_active_ms=active,gains=gains,
        all_node_arrays_checked=88,result_sha256=sha(folder/'result.json'))
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--package',default=str(ROOT/'data/nodule_prefetch_20261024_v2/nodule_prefetch'))
    a=p.parse_args();verify(a.result,a.package)
