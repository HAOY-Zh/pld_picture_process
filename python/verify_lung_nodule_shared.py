"""Independent full-size physical shared-bit acceptance, no inference rerun."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from lung_nodule_shared import checked_package,load_model
from lung_continuous import unpack_words
from lung_axil_smoke import feature_words
from nodule_plan import sha


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def verify(folder,package):
    folder=Path(folder).resolve();package,manifest=checked_package(package)
    r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if (r['status']!='physical_shared_bit_all_nodes_switching_exact' or r['overlay_downloads_by_driver']!=1
        or not 1<=r['cycles']<=10 or abs(r['clock_mhz']-100.)>.1):raise ValueError('Incomplete shared validation')
    if r['manifest_sha256']!=sha(package/'shared_manifest.json') or r['bit_sha256']!=sha(package/'lung_network.bit'):raise ValueError('Wrong model/bit/runtime')
    if r['all_nodes_sha256']!=sha(folder/'all_nodes.npz') or r['heads_sha256']!=sha(folder/'heads.npz'):raise ValueError('Changed returned arrays')
    models={};refs={};inputs={}
    for label in ['unet_original','unet_pruned','nodule']:
        m,p=load_model(package,label);models[label]=m;src=package/'models'/label
        with np.load(src/'reference.npz') as z:
            if label=='nodule':
                for c in range(4):
                    refs[label,c]={n['name']:z['case'+str(c)+'_'+n['name']] for n in m['nodes']};inputs[label,c]=z['case'+str(c)+'_input']
            else:
                refs[label,0]={n['name']:unpack_words(z[n['name']],n['shape']) for n in m['nodes']}
                with np.load(src/'input.npz') as a:inputs[label,0]=a['codes']
    sequence=[('unet_original',0,'baseline'),('unet_pruned',0,'baseline')]+[('nodule',c,'baseline') for c in range(4)]
    pattern=[('unet_original',0),('nodule',0),('unet_pruned',0),('nodule',1),('unet_original',0),('nodule',2),('unet_pruned',0),('nodule',3)]
    for i in range(r['cycles']):sequence += [(label,c,'switch'+str(i)) for label,c in pattern]
    if len(r['stages'])!=len(sequence):raise ValueError('Missing model switching calls')
    expected_array_keys=set();head_keys=set();last_label=None;residents={};count=0;retained_count=0
    with np.load(folder/'all_nodes.npz') as arrays,np.load(folder/'heads.npz') as heads:
        for i,(label,c,phase) in enumerate(sequence):
            s=r['stages'][i];stage='stage'+str(i).zfill(3);model=models[label];expected=refs[label,c]
            if (s['stage'],s['label'],s['case'],s['phase'])!=(stage,label,c,phase):raise ValueError('Wrong switching order')
            if json.loads((folder/(stage+'.json')).read_text(encoding='utf-8'))!=s:raise ValueError('Changed stage packet')
            if abs(s['clock_mhz']-100.)>.1:raise ValueError('FCLK changed')
            if digest(inputs[label,c])!=s['input_sha256']:raise ValueError('Wrong model input')
            if len(s['node_checks'])!=len(model['nodes']):raise ValueError('Missing node diagnostics')
            for n,check in zip(model['nodes'],s['node_checks']):
                key=stage+'__'+n['name'];expected_array_keys.add(key)
                if check['array']!=key or check['node']!=n['name']:raise ValueError('Wrong node identity')
                np.testing.assert_array_equal(arrays[key],expected[n['name']])
                if digest(arrays[key])!=check['sha256']:raise ValueError('Wrong node digest')
                count+=1
            head_keys.add(stage);np.testing.assert_array_equal(heads[stage],expected['head'])
            if digest(heads[stage])!=s['head_sha256']:raise ValueError('Wrong head digest')
            if label.startswith('unet'):
                key=stage+'__mask';expected_array_keys.add(key)
                np.testing.assert_array_equal(arrays[key],expected['head']>model['nodes'][-1]['zero'])
            inactive=set(residents)-{label}
            if set(s['retained_payloads_unchanged'])!=inactive or set(s['retained_payload_sha256'])!=inactive:raise ValueError('Missing inactive-payload validation')
            for other in inactive:
                if s['retained_payload_sha256'][other]!=residents[other]:raise ValueError('Inactive model device payload changed')
                retained_count+=len(residents[other])
            residents[label]={'input':digest(feature_words(inputs[label,c],model['input_zero']))}
            residents[label].update({n['name']:digest(feature_words(expected[n['name']],n['zero'])) for n in model['nodes']})
            stats=s['stats'];commands=stats['commands']
            if (stats['selected_model']!=label or stats['previous_model']!=last_label or stats['model_changed']!=(label!=last_label)
                or stats['overlay_reloaded'] or stats['command_count']!=model['commands'] or len(commands)!=model['commands']
                or stats['pl_command_cycles']!=sum(c['cycles'] for c in commands)
                or stats['buffer_allocations_per_frame']!=0 or stats['weight_packets_repacked_per_frame']!=0):raise ValueError('Invalid shared scheduler stats')
            expected_commands=[]
            for n in model['nodes']:
                for block in (range((n['shape'][0]+7)//8) if n['kind']=='conv' else [None]):
                    expected_commands.append((n,block))
            for actual,(n,block) in zip(commands,expected_commands):
                words=n['shape'][1]*n['shape'][2] if block is not None else n['shape'][1]*n['shape'][2]*((n['shape'][0]+7)//8)
                if actual['node']!=n['name'] or actual['block']!=block or actual['written_words']!=words or actual['cycles']<=0:raise ValueError('Incomplete graph command')
            last_label=label
        if set(arrays.files)!=expected_array_keys or set(heads.files)!=head_keys:raise ValueError('Incomplete array bank')
    summary={}
    for label in models:
        rows=[s for s in r['stages'] if s['label']==label and s['phase']!='baseline']
        summary[label]=dict(calls=len(rows),host_mean_ms=float(np.mean([s['stats']['host_total_ms'] for s in rows])),
            hardware_mean_ms=float(np.mean([s['stats']['pl_command_cycles']/100000. for s in rows])))
        if summary[label]!=r['summary'][label]:raise ValueError('Wrong timing summary')
    if sum(v['cma_bytes'] for v in r['registrations'].values())!=r['total_persistent_cma_bytes']:raise ValueError('Wrong shared memory sum')
    d=dict(status='independently_verified_physical_single_bit_switching',nodes_checked=count,
        retained_payload_hashes_checked=retained_count,model_calls=len(sequence),summary=summary,
        result_sha256=sha(folder/'result.json'),scope=r['scope'])
    print(json.dumps(d,indent=2));return d


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--package',default=str(ROOT/'data/lung_nodule_shared_20261024_v1/lung_nodule_shared'))
    a=p.parse_args();verify(a.result,a.package)
