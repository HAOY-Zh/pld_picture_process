"""Freeze three existing models and one already accepted physical bit."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from nodule_plan import sha,load_package
from lung_continuous import validate_model,word_count,unpack_words


def resize_unet(model,size):
    m=copy.deepcopy(model);m['input_shape']=[1,size,size];shapes={'input':m['input_shape']}
    for n in m['nodes']:
        source=shapes[n['sources'][0]];c=n['shape'][0]
        hw=[v//2 for v in source[1:]] if n['kind']=='pool' else [v*2 for v in source[1:]] if n['kind']=='fused_concat' else source[1:]
        n['shape']=[c,*hw];shapes[n['name']]=n['shape']
        if n['kind']=='conv':n['layer']['input_shape']=source;n['layer']['output_shape']=n['shape']
    return m


def checked(root):
    root=Path(root).resolve();p=json.loads((root/'preparation.json').read_text())
    for group in ['protected_sha256','prepared_sha256']:
        for f,h in p[group].items():
            if sha(f)!=h:raise ValueError('Changed shared input '+f)
    return root,p


def prepare(root):
    root=Path(root).resolve()
    if root.parent!=ROOT/'data' or root.exists():raise ValueError('Need fresh data child')
    accepted=ROOT/'data/nodule_overlap_release_20261024/nodule_overlap'
    manifest=json.loads((accepted/'overlap_manifest.json').read_text())
    for n,h in manifest['files_sha256'].items():
        if sha(accepted/n)!=h:raise ValueError('Changed accepted physical package '+n)
    proof=ROOT/'data/nodule_overlap_speed_01/independent_verification.json';d=json.loads(proof.read_text())
    if d['status']!='independently_verified_physical_pixel_overlap':raise ValueError('Need physical nodule acceptance')
    if sha(proof.parent/'result.json')!=d['result_sha256']:raise ValueError('Changed physical evidence')
    canonical=ROOT/'data/nodule_board_20261023_v1/nodule_board';load_package(canonical)
    root.mkdir();target=root/'lung_nodule_shared';target.mkdir();models=target/'models';models.mkdir()
    protected={str(proof):sha(proof),str(accepted/'overlap_manifest.json'):sha(accepted/'overlap_manifest.json')}
    sources={};stats={}
    for label,tag in [('unet_original','original'),('unet_pruned','pruned')]:
        source=ROOT/'data/lung_all_accel_20261017_v2/lung_all_accel'/tag
        fast=json.loads((source/'fast_manifest.json').read_text())
        for n,h in fast['files_sha256'].items():
            if sha(source/n)!=h:raise ValueError('Changed frozen U-Net '+label+'/'+n)
        dst=models/label;dst.mkdir()
        for n in ['model.json','parameters.npz','input.npz','reference.npz']:
            shutil.copy2(source/n,dst/n);protected[str(source/n)]=sha(source/n)
        protected[str(source/'fast_manifest.json')]=sha(source/'fast_manifest.json');sources[label]=str(source)
    dst=models/'nodule';dst.mkdir()
    for n,out in [('nodule_model.json','model.json'),('nodule_parameters.npz','parameters.npz'),('nodule_reference.npz','reference.npz')]:
        shutil.copy2(canonical/n,dst/out);protected[str(canonical/n)]=sha(canonical/n)
    sources['nodule']=str(canonical)
    # Compatible driver/runtime sources come from frozen packages, not current
    # mutable training directories. There is ONE pair of hardware drivers.
    for n in ['lung_axil.py','lung_axil_smoke.py','lung_continuous.py','lung_model_layers_test.py','lung_tensor.py',
              'lung_fast.py','nodule_pl.py','nodule_plan.py','nodule_cpu.py','lung_cpu_native.cpp','lung_cpu_kernel.h',
              'lung_nodule_cnn.py','lung_nodule_continuity.py','lung_nodule_continuity_boundary.py','lung_nodule_sampling_cache.py']:
        shutil.copy2(canonical/n,target/n);protected[str(canonical/n)]=sha(canonical/n)
    source=Path(sources['unet_original'])
    shutil.copy2(source/'lung_cpu.py',target/'lung_cpu.py');protected[str(source/'lung_cpu.py')]=sha(source/'lung_cpu.py')
    for n in ['lung_network.bit','lung_network.hwh','lung_fast_io.cpp','lung_fast_io.h']:
        shutil.copy2(accepted/n,target/n);protected[str(accepted/n)]=sha(accepted/n)
    shutil.copytree(accepted/'rtl',target/'rtl')
    for n in ['lung_nodule_shared.py','lung_nodule_shared_benchmark.py']:
        f=ROOT/'pynq'/n
        if f.exists():shutil.copy2(f,target/n)
    for label in ['unet_original','unet_pruned','nodule']:
        p=models/label;m=json.loads((p/'model.json').read_text());params=dict(np.load(p/'parameters.npz'))
        if label=='nodule':
            from nodule_plan import validate
        else:validate=validate_model
        shapes=validate(m,params)
        stats[label]=dict(nodes=len(m['nodes']),commands=m['commands'],input_shape=m['input_shape'],
            max_channels=max(s[0][0] for s in shapes.values()),
            max_weight_beats=max(((params[n['layer']['prefix']+'_weight'].shape[1]+7)//8)*n['layer']['kernel']**2 for n in m['nodes'] if n['kind']=='conv'),
            cma_bytes=sum(((word_count(s[0])+7)//8*8+16)*8 for s in shapes.values()))
        if stats[label]['max_channels']>96 or stats[label]['max_weight_beats']>128:raise ValueError('Hardware capacity exceeded')
    prep=dict(status='three_frozen_models_one_accepted_bit_prepared',model_changed=False,rtl_changed=False,
        sources=sources,compatibility=stats,bit_sha256=sha(target/'lung_network.bit'),
        protected_sha256=protected,prepared_sha256={str(p):sha(p) for p in target.rglob('*') if p.is_file()})
    (root/'preparation.json').write_text(json.dumps(prep,indent=2));print(json.dumps(stats,indent=2),flush=True)


def oracle(root,compiler):
    root,prep=checked(root);out=root/'oracle';out.mkdir();target=root/'lung_nodule_shared'
    from lung_cpu import compile_native,NativeUNet
    from nodule_cpu import NativeNodule
    from lung_fpga_contract import conv_accumulator,requantize
    from lung_tensor_vectors import reference as tensor_ref
    builddir=out/'build';builddir.mkdir()
    for n in ['lung_cpu_native.cpp','lung_cpu_kernel.h']:shutil.copy2(target/n,builddir/n)
    lib,build=compile_native(builddir,compiler)
    fixtures={};proof={}
    for label in ['unet_original','unet_pruned']:
        p=target/'models'/label;m=json.loads((p/'model.json').read_text());params=dict(np.load(p/'parameters.npz'))
        with np.load(p/'input.npz') as z:image=z['codes']
        with np.load(p/'reference.npz') as z:refs={n:z[n] for n in z.files}
        print('Independent native full 256 U-Net oracle:',label,flush=True)
        with NativeUNet(lib,m,params,2) as net:
            net.forward(image)
            for i,n in enumerate(m['nodes']):np.testing.assert_array_equal(net.node_codes(i),unpack_words(refs[n['name']],n['shape']))
        m=resize_unet(m,32);image=np.ascontiguousarray(image[:,:32,:32]);validate_model(m,params)
        trace={'input':image.copy()}
        with NativeUNet(lib,m,params,2) as net:
            net.forward(image)
            for i,n in enumerate(m['nodes']):trace[n['name']]=net.node_codes(i)
        # Separate dense integer implementation checks every reduced geometry
        # node. Native outputs must not serve as their own only oracle.
        dense={'input':image.copy()}
        for n in m['nodes']:
            a=dense[n['sources'][0]]
            if n['kind']=='conv':
                row=n['layer'];prefix=row['prefix'];acc=conv_accumulator(a,params[prefix+'_weight'],params[prefix+'_bias'],row['input_zero'],row['padding'])
                value=requantize(acc,params[prefix+'_multiplier'],params[prefix+'_shift'],row['output_zero'],row['relu'])
            else:
                q=n['quant'];b=None if n['kind']=='pool' else dense[n['sources'][1]]
                value=tensor_ref(0 if b is None else 3,a,b,q['za'],q.get('zb',0),q['zo'],q.get('ma',1),q.get('sa',0),q.get('mb',1),q.get('sb',0))
            np.testing.assert_array_equal(value,trace[n['name']]);dense[n['name']]=value
        (out/(label+'_32_model.json')).write_text(json.dumps(m,indent=2));np.savez_compressed(out/(label+'_32_trace.npz'),**trace)
        proof[label]=dict(full256_frozen_nodes_exact=21,small32_independent_dense_nodes_exact=21)
    p=target/'models/nodule';m=json.loads((p/'model.json').read_text());params=dict(np.load(p/'parameters.npz'))
    with np.load(p/'reference.npz') as z:
        with NativeNodule(lib,m,params,2) as net:
            for case in range(4):
                image=z['case'+str(case)+'_input'];net.forward(image)
                for i,n in enumerate(m['nodes']):np.testing.assert_array_equal(net.node_codes(i),z['case'+str(case)+'_'+n['name']])
            trace={'input':z['case0_input'].copy(),**{n['name']:z['case0_'+n['name']].copy() for n in m['nodes']}}
    np.savez_compressed(out/'nodule_trace.npz',**trace)
    proof['nodule']=dict(frozen_real_cases=4,nodes_per_case=11)
    d=dict(status='frozen_full_models_and_small_dense_integer_oracles_pass',proofs=proof,build=build,
        source_sha256={str(p):sha(p) for p in [ROOT/'pynq/lung_cpu.py',ROOT/'pynq/nodule_cpu.py',ROOT/'python/lung_fpga_contract.py',ROOT/'python/lung_tensor_vectors.py',Path(__file__)]},
        outputs_sha256={str(p):sha(p) for p in out.glob('*') if p.is_file()})
    (out/'verification.json').write_text(json.dumps(d,indent=2));checked(root);print(d['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--oracle',action='store_true');p.add_argument('--compiler',default='E:/vivado2/Vivado/2022.2/tps/mingw/6.2.0/win64.o/nt/bin/g++.exe');a=p.parse_args()
    if a.oracle:oracle(a.root,a.compiler)
    else:prepare(a.root)
