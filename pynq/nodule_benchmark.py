"""PYNQ-Z2 exact-head check and same-model ARM INT8 versus PL benchmark.

Run as a sole hardware owner. All four real training patches are verified;
repeated timings rotate through them. No Torch/Internet/CPU fallback for PL.
"""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
from nodule_plan import load_package,quantize_hu,lower_stem,sha
from nodule_cpu import compile_native,NativeNodule
from nodule_pl import NodulePL
from lung_fast import compile_io,NativeLoader,PreparedConv
from lung_tensor import TensorAxil
from lung_continuous import unpack_words


def statistics(a):
    a=np.asarray(a,float)
    return dict(mean_ms=float(a.mean()),median_ms=float(np.median(a)),
        min_ms=float(a.min()),max_ms=float(a.max()),p95_ms=float(np.percentile(a,95)),samples_ms=a.tolist())


def hardware(folder,model,params,timeout):
    from pynq import Overlay,allocate,Clocks
    overlay=Overlay(str(folder/'lung_network.bit'))
    conv=PreparedConv(overlay.lung_conv8_axil_0.mmio,timeout=timeout)
    tensor=TensorAxil(overlay.lung_tensor_axil_0.mmio,timeout=timeout)
    conv._idle();tensor._idle()
    # Set only the previously routed FCLK0, before issuing any command.
    Clocks.fclk0_mhz=100.0
    if abs(float(Clocks.fclk0_mhz)-100.)>.1: raise RuntimeError('Cannot set routed 100 MHz FCLK0')
    library,build=compile_io(folder)
    conv.native_loader=NativeLoader(library,conv.mmio)
    net=NodulePL(conv,tensor,allocate,model,params)
    return net,overlay,Clocks,build


def save(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)


def run(args):
    if platform.machine().lower() not in ('armv7l','armv7'):raise RuntimeError('Run on PYNQ-Z2 ARMv7')
    if not 1<=args.runs<=40 or not 1<=args.warmup<=5:raise ValueError('Invalid repeats')
    folder,manifest,model,params=load_package(Path(__file__).resolve().parent)
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError('Preserve existing results; choose a new output directory')
    out.mkdir();start=time.perf_counter()
    with np.load(folder/'nodule_reference.npz',allow_pickle=False) as z:
        refs={n:z[n].copy() for n in z.files}
    inputs=[lower_stem(quantize_hu(refs[f'case{i}_hu'])) for i in range(4)]
    print('Compile local ARM NEON/OpenMP reference...',flush=True)
    library,build=compile_native(folder);cpu_times=[];cpu_outputs={}
    with NativeNodule(library,model,params,threads=2) as cpu:
        if cpu.features&3!=3:raise RuntimeError('Need actual ARM NEON + OpenMP')
        for i,image in enumerate(inputs):
            actual=cpu.forward(image)
            for j,node in enumerate(model['nodes']):
                if not np.array_equal(cpu.node_codes(j),refs[f'case{i}_'+node['name']]):
                    raise AssertionError('CPU node mismatch '+node['name'])
            cpu_outputs[f'case{i}_head']=actual.copy()
        for i in range(args.warmup):cpu.forward(inputs[i%4])
        for i in range(args.runs):
            tick=time.perf_counter();a=cpu.forward(inputs[i%4]);elapsed=(time.perf_counter()-tick)*1000
            if not np.array_equal(a,refs[f'case{i%4}_head']):raise AssertionError('CPU head changed')
            cpu_times.append(elapsed)
    save(out/'cpu_phase.json',dict(status='exact_cpu_all_nodes',build=build,timing=statistics(cpu_times)))
    print('Load reused flow-v1 bit once; seven conv + four tensor ops per patch...',flush=True)
    net,overlay,clocks,io_build=hardware(folder,model,params,args.timeout)
    pl_times=[];patch_times=[];pack_times=[];cycle_times=[];outputs={};details=[]
    try:
        for i in range(4):
            def observe(name,words,i=i):
                if not np.array_equal(unpack_words(words,net.shapes[name][0]),refs[f'case{i}_'+name]):
                    raise AssertionError('PL node mismatch '+name)
            actual,stats=net.predict_codes(inputs[i],observer=observe)
            if not np.array_equal(actual,refs[f'case{i}_head']):raise AssertionError('PL head mismatch')
            outputs[f'case{i}_head']=actual.copy()
            print('Physical all-node check {}/4 PASS'.format(i+1),flush=True)
        for i in range(args.warmup):net.predict_hu(refs[f'case{i%4}_hu'])
        for i in range(args.runs):
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed before run')
            _,stats=net.predict_hu(refs[f'case{i%4}_hu'])
            head=unpack_words(net.buffers['head'],[5,64,64])
            if not np.array_equal(head,refs[f'case{i%4}_head']):raise AssertionError('PL timed result differs')
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed during run')
            pl_times.append(stats['host_total_ms']);patch_times.append(stats['patch_total_ms'])
            pack_times.append(stats['arm_quantize_stride2_pack_ms'])
            cycle_times.append(stats['pl_command_cycles']/100000.)
            details.append(stats)
            print('PL patch {}/{}: {:.3f} ms, SAME'.format(i+1,args.runs,stats['patch_total_ms']),flush=True)
        result=dict(status='physical_board_exact_all_nodes',board_tested=True,clock_mhz=float(clocks.fclk0_mhz),
            cpu_mhz=float(clocks.cpu_mhz),machine=platform.machine(),
            manifest_sha256=sha(folder/'nodule_manifest.json'),model_sha256=manifest['model_sha256'],
            bit_sha256=sha(folder/'lung_network.bit'),runs=args.runs,warmup=args.warmup,
            cpu_int8_two_threads=statistics(cpu_times),pl_resident_network=statistics(pl_times),
            arm_input_quantize_pack=statistics(pack_times),pl_patch_with_input_pack=statistics(patch_times),
            pl_engine_active_cycles_ms=statistics(cycle_times),
            speedup_same_lowered_integer_graph=float(np.mean(cpu_times)/np.mean(pl_times)),
            cma_resident_bytes=net.cma_bytes,io_build=io_build,
            preparation_ms=net.preparation_ms,allocation_ms=net.allocation_ms,details=details,
            scope='128x128 five-plane patch benchmark, NOT a CT volume or whole candidate extraction. CPU and PL use identical lowered INT8 graph. Input packing shown separately. No claim of FP32 clinical accuracy.',
            elapsed_sec=time.perf_counter()-start)
        np.savez(out/'actual_heads.npz',**outputs,**{'cpu_'+n:v for n,v in cpu_outputs.items()})
        result['actual_heads_sha256']=sha(out/'actual_heads.npz');save(out/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k in ['status','cpu_int8_two_threads','pl_resident_network','pl_patch_with_input_pack','speedup_same_lowered_integer_graph','cma_resident_bytes']},indent=2),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(status='failed_do_not_use_timing',error=repr(exc)))
        raise
    finally:
        net.close()  # Refuses to free CMA on uncertain/outstanding AXI.


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=int,default=20)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--timeout',type=float,default=30.)
    p.add_argument('--out',default='nodule_benchmark_result');run(p.parse_args())
