"""One bit, full-size U-Nets and synthetic nodule patches, interleaved validation.

Standalone board acceptance only. Does not change the web app or clinical flow.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from lung_nodule_shared import open_shared,load_model
from lung_continuous import unpack_words
from nodule_plan import sha


def write(path,data):
    with Path(path).open('x',encoding='utf-8') as s:json.dump(data,s,indent=2)


def array_digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run(args):
    if platform.machine().lower() not in ('armv7','armv7l'):raise RuntimeError('Run on PYNQ-Z2')
    if not 1<=args.cycles<=10:raise ValueError('Use 1..10 switching cycles')
    here=Path(__file__).resolve().parent;out=Path(args.out).resolve();out.mkdir();owner=None
    try:
        owner,manifest=open_shared(here,args.timeout)
        models={};params={};references={};inputs={};registrations={}
        for label in ['unet_original','unet_pruned','nodule']:
            model,parameters=load_model(here,label);models[label]=model;params[label]=parameters
            registrations[label]=owner.register(label,model,parameters,'nodule' if label=='nodule' else 'unet')
            p=here/'models'/label
            with np.load(p/'reference.npz',allow_pickle=False) as z:ref={n:z[n] for n in z.files}
            if label=='nodule':
                for case in range(4):
                    inputs[label,case]=ref['case'+str(case)+'_input']
                    references[label,case]={n['name']:ref['case'+str(case)+'_'+n['name']] for n in model['nodes']}
            else:
                with np.load(p/'input.npz',allow_pickle=False) as z:inputs[label,0]=z['codes']
                references[label,0]={n['name']:unpack_words(ref[n['name']],n['shape']) for n in model['nodes']}
        print('Loaded one 100MHz bit; original U-Net, pruned U-Net and nodule share the SAME drivers',flush=True)
        stages=[];arrays={};saved_heads={};residents={}
        # Each timed call runs the real whole graph without an observer. All
        # device snapshots/comparisons below are outside the reported timing.
        def step(label,case,phase):
            expected=references[label,case]
            head,stats=owner.predict_codes(label,inputs[label,case])
            np.testing.assert_array_equal(head,expected['head'])
            actual=owner.snapshot(label);checks=[];stage='stage'+str(len(stages)).zfill(3)
            for n in models[label]['nodes']:
                value=unpack_words(actual[n['name']],n['shape']);np.testing.assert_array_equal(value,expected[n['name']])
                key=stage+'__'+n['name'];arrays[key]=value
                checks.append(dict(node=n['name'],array=key,sha256=array_digest(value)))
            np.testing.assert_array_equal(unpack_words(actual['input'],models[label]['input_shape']),inputs[label,case])
            # Check every inactive model's previously produced device payload,
            # not just its detached CPU head copy. Never flush these snapshots.
            retained=[];retained_hashes={}
            for other,bank in residents.items():
                if other==label:continue
                now=owner.snapshot(other)
                for n,digest in bank.items():
                    if array_digest(now[n])!=digest:raise AssertionError('Inactive device payload changed: '+other+'/'+n)
                retained_hashes[other]={n:array_digest(a) for n,a in now.items()}
                retained.append(other)
            residents[label]={n:array_digest(v) for n,v in actual.items()}
            if label.startswith('unet'):
                zero=models[label]['nodes'][-1]['zero'];mask=head>zero
                np.testing.assert_array_equal(mask,expected['head']>zero)
                arrays[stage+'__mask']=mask
            saved_heads[stage]=head.copy()
            row=dict(stage=stage,label=label,case=case,phase=phase,stats=stats,node_checks=checks,
                input_sha256=array_digest(inputs[label,case]),head_sha256=array_digest(head),
                retained_payloads_unchanged=retained,retained_payload_sha256=retained_hashes,
                clock_mhz=float(owner.clocks.fclk0_mhz),
                cpu_mhz=float(owner.clocks.cpu_mhz))
            stages.append(row);write(out/(stage+'.json'),row)
            print('{} {} case{}: {:.3f} ms, {} nodes EXACT, retained {} unchanged; NO bit reload'.format(
                stage,label,case,stats['host_total_ms'],len(checks),','.join(retained) or 'none'),flush=True)
        # Baselines before formal switching: preserve separate model identities.
        step('unet_original',0,'baseline');step('unet_pruned',0,'baseline')
        for case in range(4):step('nodule',case,'baseline')
        pattern=[('unet_original',0),('nodule',0),('unet_pruned',0),('nodule',1),
                 ('unet_original',0),('nodule',2),('unet_pruned',0),('nodule',3)]
        for cycle in range(args.cycles):
            for label,case in pattern:step(label,case,'switch'+str(cycle))
        np.savez_compressed(out/'all_nodes.npz',**arrays);np.savez_compressed(out/'heads.npz',**saved_heads)
        summary={}
        for label in models:
            rows=[s for s in stages if s['label']==label and s['phase']!='baseline']
            summary[label]=dict(calls=len(rows),host_mean_ms=float(np.mean([s['stats']['host_total_ms'] for s in rows])),
                hardware_mean_ms=float(np.mean([s['stats']['pl_command_cycles']/100000. for s in rows])))
        result=dict(status='physical_shared_bit_all_nodes_switching_exact',board_tested=True,
            manifest_sha256=sha(here/'shared_manifest.json'),bit_sha256=sha(here/'lung_network.bit'),
            overlay_downloads_by_driver=1,cycles=args.cycles,stages=stages,summary=summary,
            registrations=registrations,total_persistent_cma_bytes=owner.cma_bytes,
            all_nodes_sha256=sha(out/'all_nodes.npz'),heads_sha256=sha(out/'heads.npz'),io_build=owner.build,
            full_unet_shape=[1,256,256],nodule_patch_shape=[45,64,64],clock_mhz=float(owner.clocks.fclk0_mhz),
            scope='Fixed frozen full-size U-Net input and four frozen synthetic nodule patches; one overlay load, no reset/reconfiguration between graphs. Times include per-command weights and guards; snapshots/file writes excluded. Not end-to-end case speed, candidate scoring, accuracy against annotation, or web integration. All three model arenas remain resident for isolation checks.')
        write(out/'result.json',result);print(json.dumps(dict(status=result['status'],summary=summary,
            overlay_downloads=1,cma_bytes=owner.cma_bytes),indent=2),flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(status='shared_acceptance_failed_discard_timings',error=repr(exc)));raise
    finally:
        if owner is not None:owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cycles',type=int,default=2);p.add_argument('--timeout',type=float,default=30.)
    p.add_argument('--out',required=True);run(p.parse_args())
