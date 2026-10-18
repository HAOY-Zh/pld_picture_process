"""Same bit/model/input: ARM dual-core, original guarded PL, prepared PL runtime.

No package downloads. CPU/PL return exact CHW codes; comparisons are outside
timers. Setup is reported separately. Speedup is unknown until this runs on Z2.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from lung_axil import LungAxil
from lung_continuous import ContinuousUNet
from lung_continuous_test import load_package
from lung_cpu import compile_native,NativeUNet
from lung_cpu_compare import environment,statistics
from lung_fast import compile_io,NativeLoader,PreparedConv,FastUNet
from lung_model_layers_test import sha
from lung_tensor import TensorAxil


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2)


def validate_package(folder):
    folder=Path(folder).resolve()
    manifest,model,parameters=load_package(folder)
    path=folder/'fast_manifest.json'; fast=json.loads(path.read_text(encoding='utf-8'))
    if fast.get('schema')!='lung_prepared_runtime_v1': raise ValueError('Wrong optimized runtime package')
    for name,digest in fast['files_sha256'].items():
        p=(folder/name).resolve()
        if p.parent!=folder or sha(p)!=digest: raise ValueError('Changed optimized file: '+name)
    if fast['original_manifest_sha256']!=sha(folder/'continuous_manifest.json'):
        raise ValueError('Original package identity mismatch')
    return manifest,model,parameters,fast


def run(args):
    folder=Path(__file__).resolve().parent; path=Path(args.report).resolve()
    cpu_path=path.with_name(path.stem+'_cpu.json')
    diag_path=path.with_name(path.stem+'_diagnostic.json')
    head_path=path.with_name(path.stem+'_head.npz')
    if any(p.exists() for p in [path,cpu_path,diag_path,head_path]):
        raise FileExistsError('Use a fresh report name; preserve partial results too')
    if platform.machine().lower() not in ('armv7l','armv7'):
        raise RuntimeError('Run on the PYNQ-Z2 ARM, not a desktop')
    if not 1<=args.runs<=20 or not 1<=args.warmup<=5 or not 0<=args.old_runs<=10:
        raise ValueError('Invalid benchmark repeats')
    manifest,model,parameters,fast=validate_package(folder)
    with np.load(folder/'input.npz',allow_pickle=False) as archive: image=archive['codes']
    with np.load(folder/'reference.npz',allow_pickle=False) as archive:
        references={node['name']:archive[node['name']] for node in model['nodes']}
    from lung_continuous import unpack_words
    expected=unpack_words(references['head'],model['nodes'][-1]['shape'])
    print('Compiling local native libraries (no downloads, same frozen model)...',flush=True)
    io_library,io_build=compile_io(folder)
    cpu_library,cpu_build=compile_native(folder)
    env_before=environment()
    print('ARM dual-core baseline: resident complete network...',flush=True)
    with NativeUNet(cpu_library,model,parameters,2) as cpu_net:
        if cpu_net.features&3!=3: raise RuntimeError('Need actual NEON + OpenMP ARM build')
        for _ in range(args.warmup):
            if not np.array_equal(cpu_net.forward(image),expected): raise AssertionError('CPU warmup differs')
        for i,node in enumerate(model['nodes']):
            if not np.array_equal(cpu_net.node_codes(i),unpack_words(references[node['name']],node['shape'])):
                raise AssertionError('CPU node mismatch: '+node['name'])
        samples=[]
        for i in range(args.runs):
            start=time.perf_counter(); actual=cpu_net.forward(image); elapsed=(time.perf_counter()-start)*1000
            if not np.array_equal(actual,expected): raise AssertionError('CPU result differs')
            samples.append(elapsed);print('CPU T2 {}/{}: {:.3f} ms, SAME'.format(i+1,args.runs,elapsed),flush=True)
    common=dict(case=manifest['case'],slice_index=manifest['slice_index'],input_shape=model['input_shape'],
                package_sha256=sha(folder/'fast_manifest.json'),original_manifest_sha256=sha(folder/'continuous_manifest.json'),
                cpu_build=cpu_build,io_build=io_build,environment_before=env_before,
                runs=args.runs,warmup=args.warmup,cpu_two_threads=statistics(samples))
    write_new(cpu_path,dict(status='passed_optimized_comparison_cpu_phase',**common))
    print('Loading UNCHANGED dual-engine bit ONCE...',flush=True)
    from pynq import Overlay,allocate
    bit=folder/'lung_network.bit'; overlay=Overlay(str(bit))
    conv_mmio=overlay.lung_conv8_axil_0.mmio; tensor_mmio=overlay.lung_tensor_axil_0.mmio
    tensor=TensorAxil(tensor_mmio,timeout=args.timeout)
    original=None
    if args.old_runs:
        old=ContinuousUNet(LungAxil(conv_mmio,timeout=args.timeout),tensor,allocate,model,parameters,log=lambda *a,**k:None)
        print('Original guarded PL: warmup then {} measured frame(s)...'.format(args.old_runs),flush=True)
        for _ in range(args.warmup):
            actual,_=old.predict_codes(image)
            if not np.array_equal(actual,expected): raise AssertionError('Original PL warmup differs')
        samples=[];details=[]
        for i in range(args.old_runs):
            start=time.perf_counter(); actual,row=old.predict_codes(image); elapsed=(time.perf_counter()-start)*1000
            if not np.array_equal(actual,expected): raise AssertionError('Original PL differs')
            samples.append(elapsed);details.append(row)
            print('Old PL {}/{}: {:.3f} ms, SAME'.format(i+1,args.old_runs,elapsed),flush=True)
        original=dict(**statistics(samples),details=details)
    conv=PreparedConv(conv_mmio,native_loader=NativeLoader(io_library,conv_mmio),timeout=args.timeout)
    network=FastUNet(conv,tensor,allocate,model,parameters)
    print('Prepared once: {:.1f} ms weights, {:.1f} ms allocation; {:.3f} MiB persistent CMA'.format(
        network.preparation_ms,network.allocation_ms,network.cma_bytes/2**20),flush=True)
    try:
        checked=[]
        def observe(name,words):
            if not np.array_equal(words,references[name]): raise AssertionError('Optimized PL node differs: '+name)
            checked.append(name);print('  optimized PL {}: SAME'.format(name),flush=True)
        print('Untimed validation: all 21 actual optimized PL node outputs...',flush=True)
        actual,diagnostic=network.predict_codes(image,observer=observe)
        if len(checked)!=21 or not np.array_equal(actual,expected): raise AssertionError('Incomplete optimized diagnostic')
        write_new(diag_path,dict(status='passed_real_board_prepared_runtime_all_nodes',node_checks=checked,
                  package_sha256=common['package_sha256'],stats=diagnostic))
        for _ in range(args.warmup):
            actual,_=network.predict_codes(image)
            if not np.array_equal(actual,expected): raise AssertionError('Optimized warmup differs')
        samples=[];details=[]
        for i in range(args.runs):
            start=time.perf_counter(); actual,row=network.predict_codes(image); elapsed=(time.perf_counter()-start)*1000
            if not np.array_equal(actual,expected): raise AssertionError('Optimized output differs; no speed claim')
            samples.append(elapsed);details.append(row)
            t=row['timing_ms']
            print('Fast PL {}/{}: {:.3f} ms; engine {:.3f}, weight load {:.3f}, input sync {:.3f}, final guards {:.3f} ms; SAME'.format(
                i+1,args.runs,elapsed,row['pl_command_cycles']/100000,t['weight_load_ms'],t['input_sync_ms'],t['final_sync_guard_ms']),flush=True)
        with head_path.open('xb') as stream: np.savez_compressed(stream,head_codes=actual)
        optimized=dict(**statistics(samples),details=details,all_21_nodes_verified=True,all_heads_match=True,
                       preparation_ms=network.preparation_ms,allocation_ms=network.allocation_ms,cma_resident_bytes=network.cma_bytes)
        ratio=common['cpu_two_threads']['mean_ms']/optimized['mean_ms']
        result=dict(status='passed_real_board_prepared_runtime_comparison',**common,
            original_pl=original,optimized_pl=optimized,speedup_cpu2_over_fast_pl=ratio,
            original_over_fast_pl=(original['mean_ms']/optimized['mean_ms'] if original else None),
            bit_sha256=sha(bit),hwh_sha256=sha(bit.with_suffix('.hwh')),head_codes_sha256=sha(head_path),
            diagnostic_sha256=sha(diag_path),environment_after=environment(),
            scope='Batch1 resident input codes -> head codes. Same model, exact codes and bit. Setup excluded for CPU and optimized PL; all per-frame physical weight transfers remain inside PL timing. Final-output comparisons and files excluded.',
            warning='Frame-end guards replace old per-command all-buffer/future-block diagnostics. Native volatile AXI-Lite is not AXI burst. Ratio <1 means PL still slower. Single validation slice repeated, not clinical or sustained case-wide throughput proof.')
        write_new(path,result)
        print('Outputs: PASS. CPU T2 {:.3f} ms; prepared PL {:.3f} ms; CPU/PL {:.3f}x{}'.format(
            common['cpu_two_threads']['mean_ms'],optimized['mean_ms'],ratio,' (PL still slower)' if ratio<1 else ''),flush=True)
        print('Results saved: '+str(path),flush=True)
        return result
    finally:
        if network._outstanding():
            print('Uncertain hardware completion: CMA retained. Do NOT rerun/reload/free; shared board recovery required.',flush=True)
        else: network.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',default='/home/xilinx/ct_test/lung_fast_result.json')
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--old-runs',type=int,default=1)
    parser.add_argument('--timeout',type=float,default=30.)
    run(parser.parse_args())
