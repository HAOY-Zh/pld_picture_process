"""Model CPU FP32, exact CPU INT8 and selected FPGA INT8; no board downloads."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from lung_fp32 import NativeFP32, compile_fp32, sha


def write_new(path,payload):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)


def validate(folder):
    folder=Path(folder).resolve();m=json.loads((folder/'fp32_manifest.json').read_text(encoding='utf-8'))
    if m['schema']!='lung_fp32_threeway_v2':raise ValueError('Wrong FP32 package')
    for n,h in m['files_sha256'].items():
        p=(folder/n).resolve()
        if p.parent!=folder or sha(p)!=h:raise ValueError('Changed threeway package: '+n)
    for name in ['lung_fp32']:
        mod=sys.modules.get(name)
        if mod is not None and sha(mod.__file__)!=m['files_sha256'][name+'.py']:raise RuntimeError('Different cached FP32 helper; stop')
    return m


def run(args):
    folder=Path(__file__).resolve().parent;m=validate(folder);report=Path(args.report).resolve()
    fp_path=report.with_name(report.stem+'_fp32.json');summary_path=report.with_name(report.stem+'_threeway.json')
    fp_head_path=report.with_name(report.stem+'_fp32_head.npz')
    outputs=[report,fp_path,summary_path,fp_head_path]+[report.with_name(report.stem+s) for s in ['_cpu.json','_clock.json','_diagnostic.json','_selection.json','_head.npz']]
    if any(p.exists() for p in outputs):raise FileExistsError('Use fresh report name, preserve partial results')
    if platform.machine().lower() not in ('armv7','armv7l'):raise RuntimeError('Board timing requires PYNQ-Z2 ARM')
    if not 1<=args.runs<=20 or not 1<=args.warmup<=5:raise ValueError('Invalid repeat counts')
    from pynq import Clocks
    cpu_before=float(Clocks.cpu_mhz)
    if abs(cpu_before-650)>0.05:raise RuntimeError('Expected unchanged 650 MHz CPU')
    with np.load(folder/'fp32_model.npz',allow_pickle=False) as a:d,w,b=(a[n] for n in ['descriptors','weights','bias'])
    with np.load(folder/'fp32_input.npz',allow_pickle=False) as a:image=a['image']
    with np.load(folder/'fp32_reference.npz',allow_pickle=False) as a:refs={n:a[n] for n in m['node_names']}
    if image.dtype!=np.float32 or not np.isfinite(image).all():raise ValueError('Invalid original floating input')
    print('Compiling model FP32 CPU baseline (NEON + OpenMP, no downloads)...',flush=True)
    library,build=compile_fp32(folder);start=time.perf_counter()
    with NativeFP32(library,d,w,b,2) as network:
        setup_ms=(time.perf_counter()-start)*1000
        if network.features&3!=3:raise RuntimeError('Need actual NEON and OpenMP FP32, no scalar board fallback')
        network.forward(image);diagnostic={}
        for i,n in enumerate(m['node_names']):
            got=network.trace(i);np.testing.assert_allclose(got,refs[n],atol=m['atol'],rtol=m['rtol'])
            diagnostic[n]=float(np.max(np.abs(got-refs[n])))
        for _ in range(args.warmup):np.testing.assert_allclose(network.forward(image),refs['head'],atol=m['atol'],rtol=m['rtol'])
        samples=[];clock_reads=[]
        for i in range(args.runs):
            before=float(Clocks.cpu_mhz)
            start=time.perf_counter();out=network.forward(image);elapsed=(time.perf_counter()-start)*1000
            after=float(Clocks.cpu_mhz)
            if abs(before-cpu_before)>0.05 or abs(after-cpu_before)>0.05:raise RuntimeError('CPU clock changed')
            clock_reads.append(dict(before_mhz=before,after_mhz=after))
            np.testing.assert_allclose(out,refs['head'],atol=m['atol'],rtol=m['rtol'])
            samples.append(elapsed);print('CPU FP32 T2 {}/{}: {:.3f} ms; Torch reference within tolerance'.format(i+1,args.runs,elapsed),flush=True)
        head=out.copy()
    with fp_head_path.open('xb') as f:np.savez_compressed(f,head_logits=head)
    fp=dict(status='passed_fp32_cpu_phase',model_label=m['model_label'],checkpoint_sha256=m['checkpoint_sha256'],case=m['case'],slice_index=m['slice_index'],runs=args.runs,warmup=args.warmup,
        head_sha256=sha(fp_head_path),clock_reads=clock_reads,
        mean_ms=float(np.mean(samples)),median_ms=float(np.median(samples)),samples_ms=samples,
        min_ms=float(np.min(samples)),max_ms=float(np.max(samples)),setup_ms=setup_ms,build=build,threads=2,cpu_mhz=cpu_before,
        package_sha256=sha(folder/'fp32_manifest.json'),atol=m['atol'],rtol=m['rtol'],node_max_abs_error=diagnostic,
        final_head_max_abs_error=float(np.max(np.abs(head-refs['head']))),fp32_logits_sign_disagreement_pixels=int(np.count_nonzero((head>=0)!=(refs['head']>=0))),
        scope='Resident normalized FP32 CHW input to FP32 CHW logits; BN fused; setup, sigmoid, resize, masks and display excluded; not bit-exact to INT8')
    write_new(fp_path,fp)
    # Fresh explicit selected entry, not a possibly cached old package module.
    spec=importlib.util.spec_from_file_location('_lung_threeway_selected',folder/'lung_selected_benchmark.py')
    selected=importlib.util.module_from_spec(spec);spec.loader.exec_module(selected);selected.run(args)
    result=json.loads(report.read_text(encoding='utf-8'));int8=result['cpu_two_threads'];pl=result['optimized_pl']
    with open(folder/'model.json',encoding='utf-8') as f:model=json.load(f)
    # Export verified with Torch quantize_per_tensor, then source hashes checked.
    # Do not substitute NumPy division here: half-tie implementation may differ.
    with np.load(report.with_name(report.stem+'_head.npz'),allow_pickle=False) as a:pl_head=a['head_codes']
    summary=dict(status='passed_same_board_fp32_int8_fpga',model_label=m['model_label'],checkpoint_sha256=m['checkpoint_sha256'],fp32_ms=fp['mean_ms'],cpu_int8_ms=int8['mean_ms'],fpga_int8_ms=pl['mean_ms'],
        cpu_fp32_over_cpu_int8=fp['mean_ms']/int8['mean_ms'],cpu_int8_over_fpga_int8=int8['mean_ms']/pl['mean_ms'],
        cpu_fp32_over_fpga_int8=fp['mean_ms']/pl['mean_ms'],fp32_report_sha256=sha(fp_path),int8_report_sha256=sha(report),
        selection_sha256=sha(report.with_name(report.stem+'_selection.json')),clock_sha256=sha(report.with_name(report.stem+'_clock.json')),
        fp32_int8_model_grid_mask_disagreement_pixels=int(np.count_nonzero((head>=0)!=(pl_head>=model['output_zero']))),
        warning='One validation slice repeated; FP32 vs quantized models differ numerically. Grid mask disagreement is NOT native Dice or dataset accuracy. Custom optimized C++ baselines, not best possible ARM library claim. CPU and PL phases sequential.')
    write_new(summary_path,summary)
    print('Threeway PASS: CPU FP32 {:.3f} ms; CPU INT8 {:.3f} ms; FPGA INT8 {:.3f} ms'.format(fp['mean_ms'],int8['mean_ms'],pl['mean_ms']),flush=True)
    print('Combined FP32/FPGA {:.3f}x; same-INT8 CPU/FPGA {:.3f}x'.format(summary['cpu_fp32_over_fpga_int8'],summary['cpu_int8_over_fpga_int8']),flush=True)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',default='/home/xilinx/ct_test/lung_threeway_result.json')
    ap.add_argument('--runs',type=int,default=20);ap.add_argument('--warmup',type=int,default=3)
    ap.add_argument('--old-runs',type=int,choices=[0],default=0);ap.add_argument('--timeout',type=float,default=30.)
    run(ap.parse_args())
