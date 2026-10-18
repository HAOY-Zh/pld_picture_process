"""Package all accepted hardware/runtime acceleration with reviewed pruning.

Original frozen inputs and old deployment packages are never modified.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

from lung_prune import restore_candidate, ORIGINAL, WORKSPACE
from lung_fp32_package import torch_trace, ATOL, RTOL
from lung_unet import file_sha256,torch_module
from lung_segmentation import fresh_output,write_json
from verify_lung_axil import check_hashes
sys.path.insert(0,str(WORKSPACE/'pynq'))
from lung_cpu import encode_model
from lung_fp32 import NativeFP32,compile_fp32
from lung_continuous_test import load_package
from lung_fast_benchmark import validate_package
from lung_clock_benchmark import validate_addon
from lung_selected_benchmark import validate_selection
from lung_threeway_combo import validate as validate_fp32
from lung_accel_suite import validate as validate_suite

LUT=WORKSPACE/'data/lung_lut_20261017'
REFINE=WORKSPACE/'data/lung_prune_refine_sequential_confirm_20261017'


def fused_parameters(network,graph):
    torch=torch_module();weights=[];bias=[];offsets={};nw=nb=0
    shapes={'input':graph['input_shape'],**{n['name']:n['shape'] for n in graph['nodes']}}
    for node in graph['nodes']:
        if node['kind']!='conv':continue
        name=node['name'];module=network.get_submodule(name)
        if name!='head':
            block,index=name.split('.')
            module=torch.nn.utils.fusion.fuse_conv_bn_eval(module,network.get_submodule(block+'.'+str(int(index)+1)))
        a=module.weight.detach().cpu().numpy().astype(np.float32)
        b=module.bias.detach().cpu().numpy().astype(np.float32)
        expected=[node['shape'][0],shapes[node['sources'][0]][0],node['layer']['kernel'],node['layer']['kernel']]
        if list(a.shape)!=expected:raise ValueError('FP32 parameter geometry differs: '+name)
        offsets[name]=(nw,nb);weights.append(a.reshape(-1));bias.append(b);nw+=a.size;nb+=b.size
    return offsets,np.concatenate(weights),np.concatenate(bias)


def generate(root):
    torch=torch_module();torch.set_num_threads(4)
    original_proof=json.loads((LUT/'verification.json').read_text(encoding='utf-8'))
    check_hashes({str(LUT/'lung_lut'/n):h for n,h in original_proof['package_files_sha256'].items()})
    accepted=json.loads((LUT/'lung_lut_result_threeway_verification.json').read_text(encoding='utf-8'))
    if not accepted['int8_head_bit_exact']:raise ValueError('Original LUT baseline not accepted')
    board=REFINE/'fisher_moderate_board'
    proof=json.loads((board/'verification.json').read_text(encoding='utf-8'))
    source=board/'lung_prune_refine_fisher_moderate'
    check_hashes({str(source/n):h for n,h in proof['package_files_sha256'].items()})
    audit=json.loads((REFINE/'fisher_moderate_fulltest/returned_mask_audit.json').read_text(encoding='utf-8'))
    if audit['cases']!=27 or any(x['cases_dropped_more_than_one_percentage_point'] for x in audit['test_limits'].values()):
        raise ValueError('Need all-case Dice confirmation')
    snapshot=REFINE/'fisher_moderate_snapshot';frozen=json.loads((snapshot/'freeze.json').read_text(encoding='utf-8'))
    check_hashes(frozen['files_sha256']);check_hashes(frozen['source_sha256'])
    output=fresh_output(root).resolve();target=output/'lung_all_accel';target.mkdir()
    shutil.copytree(LUT/'lung_lut',target/'original');shutil.copytree(source,target/'pruned')
    oldfp=json.loads((target/'original/fp32_manifest.json').read_text(encoding='utf-8'))
    pruned_manifest,graph,params=load_package(target/'pruned')
    if (pruned_manifest['case'],pruned_manifest['slice_index'])!=(oldfp['case'],oldfp['slice_index']):
        raise ValueError('Models must use the same CT slice')
    network,saved=restore_candidate(snapshot/'fisher_moderate/best.pt');network.eval()
    with np.load(target/'original/fp32_input.npz',allow_pickle=False) as a:image=np.array(a['image'],copy=True)
    q=graph['input_quantization'];codes=torch.quantize_per_tensor(torch.from_numpy(image),q['scale'],q['zero_point'],torch.quint8).int_repr().numpy()
    with np.load(target/'pruned/input.npz',allow_pickle=False) as a:np.testing.assert_array_equal(codes,a['codes'])
    offsets,weights,bias=fused_parameters(network,graph);descriptors=encode_model(graph,params)[0]
    for i,n in enumerate(graph['nodes']):
        if n['kind']=='conv':descriptors[i,19:21]=offsets[n['name']]
    refs=torch_trace(network,image,graph)
    folder=target/'pruned'
    for name in ['lung_fp32.py','lung_fp32_native.cpp','lung_fp32_kernel.h']:
        shutil.copy2(LUT/'lung_lut'/name,folder/name)
    np.savez_compressed(folder/'fp32_model.npz',descriptors=descriptors,weights=weights,bias=bias)
    np.savez_compressed(folder/'fp32_input.npz',image=image);np.savez_compressed(folder/'fp32_reference.npz',**refs)
    native=output/'host_native_pruned';native.mkdir()
    for name in ['lung_fp32_native.cpp','lung_fp32_kernel.h']:shutil.copy2(folder/name,native/name)
    compiler='E:/vivado2/Vivado/2022.2/tps/mingw/6.2.0/win64.o/nt/bin/g++.exe'
    library,build=compile_fp32(native,compiler);checks=[]
    for threads in [1,2]:
        with NativeFP32(library,descriptors,weights,bias,threads) as engine:
            engine.forward(image);errors={}
            for i,node in enumerate(graph['nodes']):
                actual=engine.trace(i);expected=refs[node['name']]
                np.testing.assert_allclose(actual,expected,atol=ATOL,rtol=RTOL)
                errors[node['name']]=float(np.max(np.abs(actual-expected)))
            checks.append(dict(threads=threads,node_max_abs_error=errors))
    addition=WORKSPACE/'pynq/lung_threeway_combo.py'
    for label in ['original','pruned']:
        folder=target/label;shutil.copy2(addition,folder/addition.name)
        metadata=dict(oldfp) if label=='original' else dict(
            checkpoint_sha256=file_sha256(snapshot/'fisher_moderate/best.pt'),epoch=saved['epoch'],
            case=oldfp['case'],slice_index=oldfp['slice_index'],atol=ATOL,rtol=RTOL,
            node_names=[n['name'] for n in graph['nodes']],host_checks=checks,build=build,
            model_source='Recovered pruned FP32 checkpoint, eval BN fused; never dequantized INT8 weights',
            input_source='Same normalized CT before model-specific input quantization',board_tested=False)
        metadata.update(schema='lung_fp32_threeway_v2',model_label=label,board_tested=False)
        metadata['files_sha256']={p.name:file_sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='fp32_manifest.json'}
        write_json(folder/'fp32_manifest.json',metadata)
        validate_package(folder);validate_addon(folder);validate_selection(folder);validate_fp32(folder)
    shutil.copy2(WORKSPACE/'pynq/lung_accel_suite.py',target/'lung_accel_suite.py')
    sources=[Path(__file__).resolve(),WORKSPACE/'pynq/lung_accel_suite.py',addition,
        WORKSPACE/'python/verify_lung_threeway_combo.py',LUT/'verification.json',
        LUT/'lung_lut_result_threeway_verification.json',board/'verification.json',
        REFINE/'fisher_moderate_fulltest/returned_mask_audit.json']
    suite=dict(schema='lung_all_acceleration_suite_v1',models=['original','pruned'],
        candidate='fisher_moderate',same_normalized_input_sha256=__import__('hashlib').sha256(image.tobytes()).hexdigest(),
        bit_sha256=proof['accepted_bit_sha256'],clock_mhz=100,cpu_mhz=650,
        enabled=['INT8 static quantization','HWC8 eight-channel parallel convolution','prepared weights and persistent buffers',
            'native bulk MMIO loader','convolution cache hit return and response bypass',
            'pooling dual row cache','exact concatenation requantization LUT','routed and readback-checked 100 MHz',
            'structured channel pruning and recovery training (pruned model)'],
        board_tested=False,extra_quality_review_passed=False,
        input_sha256={str(p):file_sha256(p) for p in sources},
        files_sha256={p.relative_to(target).as_posix():file_sha256(p) for p in target.rglob('*') if p.is_file()})
    write_json(target/'suite_manifest.json',suite);validate_suite(target)
    write_json(output/'verification.json',dict(status='prepared_all_acceleration_six_way_suite_not_board_tested',
        board_tested=False,original_inputs_unchanged=True,pruned_fp32_host_checks=checks,
        suite_manifest_sha256=file_sha256(target/'suite_manifest.json'),input_sha256=suite['input_sha256'],
        package_files_sha256={p.relative_to(target).as_posix():file_sha256(p) for p in target.rglob('*') if p.is_file()}))
    print('ALL ACCELERATION SUITE PREPARED:',target,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',default='data/lung_all_accel_20261017_v2')
    generate(p.parse_args().outdir)
