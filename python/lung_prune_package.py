"""Package a reviewed pruned graph on the ACCEPTED flow bit for a clean A/B test.

Only geometry/weights change. Exact integer PC scheduling and native C++ checks
are prerequisites. They are explicitly NOT real board performance evidence.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from lung_continuous_package import make_plan
from lung_fixed_network import FixedUNet
from lung_fpga_contract import conv_accumulator, requantize
from lung_prune import DATASET, WORKSPACE
from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256,mapped_npy,read_manifest,torch_module
from verify_lung_axil import check_hashes

sys.path.insert(0,str(WORKSPACE/'pynq'))
from lung_axil_smoke import feature_words,weight_packets
from lung_continuous import unpack_words,validate_model
from lung_cpu import NativeUNet,compile_native
from lung_fast import FastUNet,prepare_block
from lung_fast_benchmark import validate_package
from lung_clock_benchmark import validate_addon
from lung_selected_benchmark import validate_selection

FLOW=WORKSPACE/'data/lung_flow_20261017'


def build_plan(contract,parameters):
    layers=[];arrays={}
    for row in contract['layers']:
        prefix=row['name'].replace('.','_')
        layers.append(dict(name=row['name'],prefix=prefix,input_shape=row['input_shape'][1:],
            output_shape=row['output_shape'][1:],kernel=row['weight_shape'][-1],padding=row['padding'][0],
            input_zero=row['input_zero_point'],output_zero=row['output_zero_point'],relu=row['fused_relu']))
        for key,suffix in [('weight','weight_int8'),('bias','bias_int32'),('multiplier','multiplier_int32'),('shift','right_shift_int32')]:
            arrays[prefix+'_'+key]=np.array(parameters[prefix+'_'+suffix],copy=True)
    model=make_plan(layers,contract);validate_model(model,arrays)
    return model,arrays


def generate(experiment,outdir,exploratory_candidate=None):
    experiment=Path(experiment).resolve();review=json.loads((experiment/'review.json').read_text())
    selected=exploratory_candidate or review['selected_for_board_experiment']
    if selected is None:raise ValueError('No candidate passed review; an explicitly named exploratory candidate may be benchmarked only')
    experiments=json.loads((experiment/'verification.json').read_text())['results']
    result=next((r for r in experiments if r['candidate']==selected),None)
    if result is None or not result['passes_mean_and_work_screen']:
        raise ValueError('Exploratory candidate must still pass the user mean-Dice/workload limits')
    quality=next(r for r in review['reviews'] if r['candidate']==selected)
    quality_passed=quality['passes_conservative_development_review']
    frozen=json.loads((experiment/'freeze.json').read_text())
    if frozen['status']!='frozen_pruning_candidate_identity_verified':raise ValueError('Need candidate identity freeze')
    check_hashes(frozen['files_sha256']);check_hashes(frozen['source_sha256'])
    check_hashes(review['input_sha256'])
    protocol=json.loads((experiment/'protocol.json').read_text());check_hashes(protocol['input_sha256']);check_hashes(protocol['source_sha256'])
    accepted=json.loads((FLOW/'lung_flow_result_threeway_verification.json').read_text())
    if not accepted['int8_head_bit_exact']:raise ValueError('Need accepted flow board baseline')
    prior=json.loads((FLOW/'verification.json').read_text());source=FLOW/'lung_flow'
    check_hashes({str(source/n):h for n,h in prior['package_files_sha256'].items()})
    candidate=experiment/selected;contract=json.loads((candidate/'contract.json').read_text())
    torch=torch_module();torch.set_num_threads(4)
    with np.load(candidate/'parameters.npz') as parameters:
        model,arrays=build_plan(contract,parameters);fixed=FixedUNet(contract,parameters,'cuda')
    cache_root,manifest=read_manifest(DATASET)
    record=next(r for r in manifest['cases'] if r['split']=='validation')
    index=record['slices']//2
    with mapped_npy(cache_root/record['cache_folder']/'images.npy') as images:
        image=torch.from_numpy(np.array(images[index],dtype=np.float32))[None,None]
    grid=contract['input_quantization'];q=torch.quantize_per_tensor(image,grid['scale'],grid['zero_point'],torch.quint8).int_repr()
    inputs={};original_conv=fixed.conv
    def capture(a,name):
        inputs[name]=a.detach().cpu().numpy()[0].copy();return original_conv(a,name)
    fixed.conv=capture
    with torch.inference_mode():_,trace=fixed.forward_codes(q,capture=True)
    # Verify every conv with a second, dense NumPy INT64 implementation.
    for row in contract['layers']:
        name=row['name'];prefix=name.replace('.','_');p=arrays
        acc=conv_accumulator(inputs[name],p[prefix+'_weight'],p[prefix+'_bias'],row['input_zero_point'],row['padding'][0])
        expected=requantize(acc,p[prefix+'_multiplier'],p[prefix+'_shift'],row['output_zero_point'],row['fused_relu'])
        np.testing.assert_array_equal(expected,trace[name][0])
        print('Pruned dense independent reference:',name,flush=True)
    for pool,following in [('pool0','enc2.0'),('pool1','enc3.0'),('pool2','neck.0')]:trace[pool]=inputs[following][None]
    output=fresh_output(outdir).resolve();target=output/'lung_pruned';target.mkdir()
    # Ship only the generic pinned INT8 runner. Original FP32 package/checkpoint
    # identity is intentionally not repurposed for this newly trained model.
    names=['lung_axil.py','lung_axil_smoke.py','lung_model_layers_test.py','lung_tensor.py',
        'lung_continuous.py','lung_continuous_test.py','lung_cpu.py','lung_cpu_compare.py',
        'lung_cpu_kernel.h','lung_cpu_native.cpp','lung_fast.py','lung_fast_io.cpp','lung_fast_io.h',
        'lung_fast_benchmark.py','lung_clock_benchmark.py','lung_selected_benchmark.py',
        'lung_network.bit','lung_network.hwh']
    for name in names:shutil.copy2(source/name,target/name)
    write_json(target/'model.json',model);np.savez_compressed(target/'parameters.npz',**arrays)
    np.savez_compressed(target/'input.npz',codes=q.numpy()[0])
    refs={n['name']:feature_words(trace[n['name']][0],n['zero']) for n in model['nodes']}
    np.savez_compressed(target/'reference.npz',**refs)
    tracked={str(candidate/n):file_sha256(candidate/n) for n in ['best.pt','int8_model.pt','contract.json','parameters.npz','fixed_evaluation.json']}
    tracked.update({str(experiment/n):file_sha256(experiment/n) for n in ['protocol.json','review.json','verification.json','freeze.json']})
    tracked[str(Path(__file__).resolve())]=file_sha256(__file__)
    def hashes(names):return {name:file_sha256(target/name) for name in names}
    continuous_files=['model.json','parameters.npz','input.npz','reference.npz','lung_network.bit','lung_network.hwh',
        'lung_axil.py','lung_axil_smoke.py','lung_model_layers_test.py','lung_tensor.py','lung_continuous.py']
    write_json(target/'continuous_manifest.json',dict(schema='lung_continuous_board_v1',case=record['case'],slice_index=index,
        selection='First original validation patient, middle CT slice; expert labels not used for timing input selection',
        status='prepared_pruned_graph_on_accepted_flow_bit_not_board_tested',candidate=selected,
        conservative_quality_review_passed=quality_passed,performance_experiment_only=not quality_passed,
        retraining=True,recalibration=True,test_set_used=False,model_changed=True,bit_changed=False,
        files_sha256=hashes(continuous_files),source_sha256=tracked,
        warning='Run the pinned selected benchmark; the older isolated continuous runner assumes 51 commands.'))
    packets=blocks=0
    for node in model['nodes']:
        if node['kind']!='conv':continue
        row=node['layer'];p={k:arrays[row['prefix']+'_'+k] for k in ['weight','bias','multiplier','shift']}
        for block in range((node['shape'][0]+7)//8):
            prepared=prepare_block(p,block,row)
            if list(prepared['packets'])!=weight_packets(p['weight'],block):raise ValueError('Weight packet mismatch')
            packets+=len(prepared['words']);blocks+=1
    fast_files=continuous_files+['continuous_manifest.json','lung_cpu.py','lung_cpu_compare.py','lung_cpu_kernel.h',
        'lung_cpu_native.cpp','lung_fast.py','lung_fast_io.cpp','lung_fast_io.h','lung_fast_benchmark.py','lung_continuous_test.py']
    write_json(target/'fast_manifest.json',dict(schema='lung_prepared_runtime_v1',source_sha256=tracked,
        original_manifest_sha256=file_sha256(target/'continuous_manifest.json'),files_sha256=hashes(fast_files),
        frozen_weight_blocks=blocks,weight_packets_per_frame=packets,weight_mmio_words_per_frame=packets*16,
        model_changed=True,bit_changed=False,board_tested=False,hardware_revision='flow_v1',
        conservative_quality_review_passed=quality_passed,performance_experiment_only=not quality_passed,
        warning='New pruned model, exact accepted 100MHz flow bit. No actual candidate latency yet.'))
    write_json(target/'clock_manifest.json',dict(schema='lung_routed_clock_benchmark_v1',target_clock_mhz=100.,
        original_fast_manifest_sha256=file_sha256(target/'fast_manifest.json'),routed_design=prior['routed_design'],
        files_sha256=hashes(['lung_clock_benchmark.py']),board_tested=False))
    write_json(target/'selection_manifest.json',dict(schema='lung_pinned_entrypoint_v1',
        target_folder='/home/xilinx/ct_test/lung_pruned',files_sha256=hashes(['lung_selected_benchmark.py',
            'fast_manifest.json','clock_manifest.json','lung_network.bit','lung_network.hwh'])))
    validate_package(target);validate_addon(target);validate_selection(target)
    # Real upstream device memory rather than feeding frozen expected activations.
    sys.path.insert(0,str(WORKSPACE/'python/tests'))
    from test_lung_fast import CachedMemory,DeviceConv,DeviceTensor
    memory=CachedMemory();network=FastUNet(DeviceConv(memory),DeviceTensor(memory),memory.allocate,model,arrays);checked=[]
    def observe(name,words):
        np.testing.assert_array_equal(words,refs[name]);checked.append(name)
    try:
        first,stats=network.predict_codes(q.numpy()[0],observe)
        second,_=network.predict_codes(q.numpy()[0]);np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first,trace['head'][0])
    finally:network.close()
    if len(checked)!=21 or stats['command_count']!=model['commands']:raise ValueError('Incomplete pruned schedule')
    native=output/'host_native';native.mkdir()
    for name in ['lung_cpu_native.cpp','lung_cpu_kernel.h']:shutil.copy2(target/name,native/name)
    library,build=compile_native(native,'E:/vivado2/Vivado/2022.2/tps/mingw/6.2.0/win64.o/nt/bin/g++.exe')
    for threads in [1,2]:
        with NativeUNet(library,model,arrays,threads) as cpu:
            actual=cpu.forward(q.numpy()[0]);np.testing.assert_array_equal(actual,trace['head'][0])
            for i,node in enumerate(model['nodes']):np.testing.assert_array_equal(cpu.node_codes(i),trace[node['name']][0])
    check_hashes(tracked);check_hashes(protocol['input_sha256'])
    write_json(output/'verification.json',dict(status='prepared_pruned_package_pc_exact_not_board_tested',
        candidate=selected,commands=model['commands'],nodes_checked=checked,changed_model=True,changed_bit=False,
        conservative_quality_review_passed=quality_passed,performance_experiment_only=not quality_passed,
        quality_review=quality,
        accepted_bit_sha256=file_sha256(target/'lung_network.bit'),independent_dense_conv_checks=15,
        pc_cache_device_memory_separate=True,repeated_pc_frames=2,native_int8_threads_checked=[1,2],native_build=build,
        package_files_sha256={f.name:file_sha256(f) for f in target.iterdir() if f.is_file()},
        source_sha256=tracked,protected_original_inputs_unchanged=True,
        board_tested=False,performance_known=False,
        warning='The hardware core supports this graph, but PC arithmetic/trace checks are not actual RTL or board acceptance.'))
    np.savez_compressed(output/'conv_probe_inputs.npz',**inputs)
    print('Independent pruned upload package:',target,'commands:',model['commands'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',default='data/lung_prune_20261017')
    p.add_argument('--outdir',default='data/lung_prune_board_20261017')
    p.add_argument('--exploratory-candidate',help='Explicit performance-only candidate, not accepted as a replacement model')
    a=p.parse_args();generate(a.experiment,a.outdir,a.exploratory_candidate)
