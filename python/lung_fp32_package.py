"""Export original folded FP32 weights + Torch traces for native ARM benchmark."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from lung_unet import make_model, mapped_npy, read_manifest, torch_module, file_sha256
from lung_segmentation import fresh_output, write_json
from verify_lung_axil import check_hashes
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_cpu import encode_model
from lung_continuous_test import load_package
from lung_fp32 import NativeFP32, compile_fp32

WORKSPACE=Path(__file__).resolve().parents[1]
ATOL=0.002
RTOL=0.0002


def float_parameters(model,checkpoint):
    torch=torch_module();original=make_model(checkpoint['base_channels'],1).eval();original.load_state_dict(checkpoint['model'],strict=True)
    rows=[];weights=[];bias=[];nw=nb=0
    # Only geometry/topology is shared with the integer graph, never weights.
    for n in model['nodes']:
        if n['kind']!='conv': continue
        name=n['name'];module=original.get_submodule(name)
        if name!='head':
            prefix,suffix=name.split('.');module=torch.nn.utils.fusion.fuse_conv_bn_eval(module,original.get_submodule(prefix+'.'+str(int(suffix)+1)))
        a=module.weight.detach().cpu().numpy().astype(np.float32);b=module.bias.detach().cpu().numpy().astype(np.float32)
        weights.append(a.reshape(-1));bias.append(b);rows.append((name,nw,nb));nw+=a.size;nb+=b.size
    return original,rows,np.concatenate(weights),np.concatenate(bias)


def torch_trace(original,image,model):
    torch=torch_module();trace={};x=torch.from_numpy(image[None])
    with torch.inference_mode():
        for node in model['nodes']:
            a=x if node['sources'][0]=='input' else trace[node['sources'][0]]
            if node['kind']=='conv':
                name=node['name'];out=original.get_submodule(name)(a)
                if name!='head':
                    prefix,suffix=name.split('.');j=int(suffix)
                    out=original.get_submodule(prefix+'.'+str(j+2))(original.get_submodule(prefix+'.'+str(j+1))(out))
            elif node['kind']=='pool': out=original.pool(a)
            else:
                b=trace[node['sources'][1]];out=torch.cat((torch.nn.functional.interpolate(a,size=b.shape[-2:],mode='nearest'),b),dim=1)
            trace[node['name']]=out
        direct=original(x)
        torch.testing.assert_close(direct,trace['head'],rtol=0,atol=0)
    return {name:a.numpy()[0].copy() for name,a in trace.items()}


def generate(root,source,compiler):
    root=Path(root).resolve();source=Path(source).resolve();manifest,model,parameters=load_package(source)
    if root.parent!=WORKSPACE/'data': raise ValueError('Use new workspace/data folder')
    dataset=WORKSPACE/'data/lung_learning_20261009/dataset.json';cache_root,data=read_manifest(dataset)
    ckpt=WORKSPACE/'data/lung_training_20261011/early_stop_run/best.pt'
    if file_sha256(ckpt)!='9f822eb29910325e92b4cd6912d15c6dd687ca535857ee9c27f629172be47f88': raise ValueError('Frozen checkpoint changed')
    torch=torch_module();torch.set_num_threads(4);saved=torch.load(ckpt,map_location='cpu',weights_only=True)
    if saved['dataset_fingerprint']!=data['fingerprint'] or saved['epoch']!=12 or saved['base_channels']!=8: raise ValueError('Wrong original model/data')
    rec=next(r for r in data['cases'] if r['case']==manifest['case'] and r['split']=='validation')
    cache=cache_root/rec['cache_folder']/'images.npy'
    with mapped_npy(cache) as a:image=np.array(a[manifest['slice_index']][None],dtype=np.float32,copy=True)
    q=model['input_quantization'];codes=torch.quantize_per_tensor(torch.from_numpy(image),q['scale'],q['zero_point'],torch.quint8).int_repr().numpy()
    with np.load(source/'input.npz',allow_pickle=False) as a:
        if not np.array_equal(codes,a['codes']): raise ValueError('FP32 slice does not quantize to the identical FPGA input')
    original,offsets,weights,bias=float_parameters(model,saved);d=encode_model(model,parameters)[0]
    by_name={name:(nw,nb) for name,nw,nb in offsets}
    for i,n in enumerate(model['nodes']):
        if n['kind']=='conv': d[i,19:21]=by_name[n['name']]
    refs=torch_trace(original,image,model)
    root=fresh_output(root).resolve();target=root/'lung_fp32_compare';target.mkdir()
    for p in source.iterdir():
        if not p.is_file(): raise ValueError('Source upload package contains a folder')
        shutil.copy2(p,target/p.name)
    additions=['lung_fp32.py','lung_fp32_native.cpp','lung_fp32_kernel.h','lung_threeway_benchmark.py']
    for n in additions: shutil.copy2(WORKSPACE/'pynq'/n,target/n)
    np.savez_compressed(target/'fp32_model.npz',descriptors=d,weights=weights,bias=bias)
    np.savez_compressed(target/'fp32_input.npz',image=image)
    np.savez_compressed(target/'fp32_reference.npz',**refs)
    native=root/'host_native';native.mkdir()
    for n in ['lung_fp32_native.cpp','lung_fp32_kernel.h']:shutil.copy2(target/n,native/n)
    library,build=compile_fp32(native,compiler);checks=[]
    for threads in [1,2]:
        with NativeFP32(library,d,weights,bias,threads) as network:
            network.forward(image);rows={}
            for i,n in enumerate(model['nodes']):
                got=network.trace(i);want=refs[n['name']]
                np.testing.assert_allclose(got,want,atol=ATOL,rtol=RTOL)
                rows[n['name']]=float(np.max(np.abs(got-want)))
            checks.append(dict(threads=threads,node_max_abs_error=rows))
    assembly=native/'armv7_fp32_check.s'
    command=['E:/vivado2/Vivado/2022.2/tps/llvm/7.0/win64/bin/clang++.exe','--target=armv7-none-eabi','-mcpu=cortex-a9',
        '-mfpu=neon','-O3','-std=c++11','-ffreestanding','-DLUNG_FP32_NEON_ONLY','-S',str(native/'lung_fp32_native.cpp'),'-o',str(assembly)]
    subprocess.run(command,check=True)
    if 'vmla.f32' not in assembly.read_text(encoding='utf-8'): raise ValueError('Missing ARM NEON FP32 multiply-accumulate')
    tracked=[ckpt,dataset,cache,Path(__file__).resolve()]+[WORKSPACE/'pynq'/n for n in additions]
    protected=json.loads((WORKSPACE/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256'];check_hashes(protected)
    payload=dict(schema='lung_original_fp32_threeway_v1',checkpoint_sha256=file_sha256(ckpt),epoch=12,case=manifest['case'],slice_index=manifest['slice_index'],
        model_source='Original FP32 checkpoint; eval BN fused into FP32 conv, never dequantized INT8 weights',
        input_source='Same normalized cached CT before quantization; quantizes exactly to frozen FPGA codes',
        atol=ATOL,rtol=RTOL,node_names=[n['name'] for n in model['nodes']],
        host_checks=checks,arm_neon_assembly_sha256=file_sha256(assembly),build=build,board_tested=False,
        protected_sources_unchanged=len(protected),source_sha256={str(p):file_sha256(p) for p in tracked},
        files_sha256={p.name:file_sha256(p) for p in target.iterdir()})
    write_json(target/'fp32_manifest.json',payload);write_json(root/'verification.json',payload)
    print('FP32 native and 21-node Torch reference checks PASS:',target,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',default='data/lung_fp32_compare_20261016')
    ap.add_argument('--source',default='data/lung_cache_return_20261016_v2/lung_cache_return')
    ap.add_argument('--compiler',default='E:/vivado2/Vivado/2022.2/tps/mingw/6.2.0/win64.o/nt/bin/g++.exe')
    a=ap.parse_args();generate(a.root,a.source,a.compiler)
