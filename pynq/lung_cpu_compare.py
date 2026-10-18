"""Same-board native CPU versus current guarded PL core-forward benchmark.

One fixed validation slice, batch one. No medical preprocessing/postprocessing
or per-node reference comparisons in timed regions. PL safety guards retained.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from lung_axil import LungAxil
from lung_continuous import ContinuousUNet,unpack_words
from lung_continuous_test import load_package
from lung_cpu import NativeUNet,compile_native
from lung_model_layers_test import sha
from lung_tensor import TensorAxil


def statistics(samples):
    if not samples or any(not np.isfinite(x) or x<=0 for x in samples): raise ValueError('Invalid benchmark samples')
    return dict(samples_ms=samples,mean_ms=float(np.mean(samples)),median_ms=float(np.median(samples)),
                min_ms=float(np.min(samples)),max_ms=float(np.max(samples)))


def environment():
    result=dict(machine=platform.machine(),system=platform.system(),python=platform.python_version(),numpy=np.__version__)
    for name,path in [('cpuinfo','/proc/cpuinfo'),('loadavg','/proc/loadavg'),
                      ('governor','/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
                      ('frequency_khz','/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'),
                      ('temperature_millideg','/sys/class/thermal/thermal_zone0/temp')]:
        p=Path(path)
        if p.exists(): result[name]=p.read_text().strip()
    return result


def run(args):
    folder=Path(__file__).resolve().parent; report_path=Path(args.report).resolve()
    cpu_path=report_path.with_name(report_path.stem+'_cpu.json')
    if report_path.exists() or cpu_path.exists(): raise FileExistsError('Use a fresh report name; preserve CPU partial evidence too')
    env_before=environment()
    if env_before['machine'] not in ('armv7l','armv7'):
        raise RuntimeError('Run this timing comparison on the PYNQ-Z2, not a desktop test double')
    if not 1<=args.runs<=20 or not 1<=args.warmup<=5: raise ValueError('Use runs 1..20, warmup 1..5')
    manifest,model,parameters=load_package(folder)
    compare_path=folder/'cpu_compare_manifest.json'; compare=json.loads(compare_path.read_text(encoding='utf-8'))
    if compare['schema']!='lung_same_board_cpu_pl_v1': raise ValueError('Wrong comparison package')
    for name,digest in compare['files_sha256'].items():
        path=(folder/name).resolve()
        if path.parent!=folder or sha(path)!=digest: raise ValueError('Changed comparison file: '+name)
    with np.load(folder/'input.npz',allow_pickle=False) as a: image=a['codes']
    with np.load(folder/'reference.npz',allow_pickle=False) as a:
        references={n['name']:unpack_words(a[n['name']],n['shape']) for n in model['nodes']}
    expected=references['head']
    print('Compiling optimized native CPU baseline (NEON + OpenMP); no downloads...',flush=True)
    library,build=compile_native(folder)
    cpu=[]
    for threads in [1,2]:
        print('CPU {} thread(s): loading model / preparing constant weights and LUTs'.format(threads),flush=True)
        with NativeUNet(library,model,parameters,threads) as network:
            if network.features&3!=3: raise RuntimeError('Need an actual NEON + OpenMP ARM build; no scalar fallback for board timing')
            for warmup in range(args.warmup):
                print('CPU T{} warmup {}/{}: running complete network...'.format(threads,warmup+1,args.warmup),flush=True)
                actual=network.forward(image)
                if not np.array_equal(actual,expected): raise AssertionError('CPU warmup head differs')
            # Untimed full graph check, independently of timed final-output checks.
            for index,node in enumerate(model['nodes']):
                if not np.array_equal(network.node_codes(index),references[node['name']]):
                    raise AssertionError('Native CPU node differs: '+node['name'])
            samples=[]
            for repeat in range(args.runs):
                print('CPU T{} measured {}/{}: running...'.format(threads,repeat+1,args.runs),flush=True)
                start=time.perf_counter(); actual=network.forward(image); elapsed=(time.perf_counter()-start)*1000
                if not np.array_equal(actual,expected): raise AssertionError('CPU output mismatch; do not report speedup')
                samples.append(elapsed); print('  {:.3f} ms, final codes SAME'.format(elapsed),flush=True)
            cpu.append(dict(threads=threads,neon=True,openmp=True,all_21_nodes_verified=True,
                            all_measured_outputs_match=True,**statistics(samples)))
    partial=dict(status='passed_same_board_native_cpu_only',case=manifest['case'],slice_index=manifest['slice_index'],
                 input_shape=model['input_shape'],cpu=cpu,build=build,environment_before=env_before,
                 library_sha256=sha(library),package_sha256=sha(compare_path),warmup=args.warmup,runs=args.runs,
                 warning='Self-written optimized C++/NEON exact-Q31 baseline, not a claim of best achievable ARM library performance')
    cpu_path.write_text(json.dumps(partial,indent=2),encoding='utf-8')
    print('CPU-only results saved. Loading matching overlay ONCE for PL measurements...',flush=True)
    from pynq import Overlay,allocate
    bit=folder/'lung_network.bit'; overlay=Overlay(str(bit))
    conv=LungAxil(overlay.lung_conv8_axil_0.mmio,timeout=args.timeout)
    tensor=TensorAxil(overlay.lung_tensor_axil_0.mmio,timeout=args.timeout)
    network=ContinuousUNet(conv,tensor,allocate,model,parameters,log=lambda *a,**k:None)
    for warmup in range(args.warmup):
        print('PL warmup {}/{}: complete 51-command inference...'.format(warmup+1,args.warmup),flush=True)
        actual,_=network.predict_codes(image,observer=None)
        if not np.array_equal(actual,expected): raise AssertionError('PL warmup output differs')
    samples=[]; active_ms=[]; stats=[]
    for repeat in range(args.runs):
        print('PL measured {}/{}: complete inference, no per-node reference comparison...'.format(repeat+1,args.runs),flush=True)
        start=time.perf_counter(); actual,row=network.predict_codes(image,observer=None); elapsed=(time.perf_counter()-start)*1000
        if not np.array_equal(actual,expected): raise AssertionError('PL head differs; do not report speedup')
        samples.append(elapsed); active_ms.append(row['pl_command_cycles']/100000); stats.append(row)
        print('  {:.3f} ms total, {:.3f} ms engine-active, final codes SAME'.format(elapsed,active_ms[-1]),flush=True)
    pl=dict(all_measured_outputs_match=True,**statistics(samples),engine_active_ms=active_ms,details=stats)
    speedup={str(r['threads'])+'_cpu_threads':r['mean_ms']/pl['mean_ms'] for r in cpu}
    result=dict(status='passed_same_board_cpu_vs_current_pl',**{k:v for k,v in partial.items() if k!='status'},
                pl=pl,speedup_cpu_over_pl=speedup,environment_after=environment(),
                bit_sha256=sha(bit),hwh_sha256=sha(bit.with_suffix('.hwh')),
                scope='Resident UINT8 CHW input -> exact UINT8 CHW head; batch=1; CPU model/workspace setup and bit download excluded; reference checks outside timing. PL buffer allocation, per-frame weights/control, sync, polling and safety guards included.',
                comparison_warning='Current deployed PL implementation versus custom C++ CPU baseline, not best possible performance on either side. Same slice repeated; not case-wide throughput/accuracy. Ratio <1 means PL is slower.')
    report_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Same-board CPU/PL benchmark: PASS (all outputs match)',flush=True)
    for row in cpu:
        print('CPU {} thread(s): {:.3f} ms; CPU/PL ratio {:.3f}x'.format(row['threads'],row['mean_ms'],speedup[str(row['threads'])+'_cpu_threads']),flush=True)
    print('Current guarded PL: {:.3f} ms'.format(pl['mean_ms']),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',default='/home/xilinx/ct_test/lung_cpu_compare_result.json')
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--timeout',type=float,default=30.)
    run(parser.parse_args())
