"""Same native input: ARM INT8, prior exact-LUT PL, window-prefetch PL."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np


def run(args):
    if platform.machine().lower() not in ('armv7l','armv7'):raise RuntimeError('Run on PYNQ-Z2')
    if not 1<=args.runs<=20 or not 1<=args.warmup<=3:raise ValueError('Invalid repetitions')
    from pynq import allocate
    here=Path(__file__).resolve().parent;parent=Path(args.package).resolve();sys.path.insert(0,str(parent))
    from nodule_plan import load_package,sha
    from nodule_benchmark import hardware,save
    from nodule_canvas_cpu import CanvasCPU
    from nodule_canvas_direct import DirectCanvasPL
    from nodule_canvas_input import compile_input,NativeCanvasInput
    from nodule_cpu import compile_native
    from nodule_canvas_benchmark import canvas_plane_once
    from nodule_fast_input import SharedPlaneSampler
    from lung_nodule_cnn import config_from_dict,decode,consolidate
    from lung_continuous import unpack_words
    manifest=json.loads((here/'prefetch_manifest.json').read_text())
    for n,h in manifest['files_sha256'].items():
        p=(here/n).resolve()
        if here not in p.parents or not p.is_file() or sha(p)!=h:raise ValueError('Changed prefetch package '+n)
    _,_,model,parameters=load_package(parent)
    if sha(parent/'nodule_manifest.json')!=manifest['parent_manifest_sha256']:raise ValueError('Wrong frozen model parent')
    cfg=config_from_dict(model['detector_config']);meta=json.loads((here/'slice_case.json').read_text());g=meta['local_geometry']
    if sha(here/'slice_case.npz')!=meta['payload_sha256']:raise ValueError('Changed slice')
    with np.load(here/'slice_case.npz') as z:hu=z['hu'];search=z['search'];jobs=[tuple(j) for j in z['jobs']]
    sampler=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg);h,w=map(int,sampler.context.shape[1:]);del sampler
    out=Path(args.out).resolve();out.mkdir()
    lib,input_build=compile_input(here);native=NativeCanvasInput(lib)
    cpu_lib,cpu_build=compile_native(parent)
    heads={};all_nodes={};results={};reference=None;reference_candidates=None;stash={};net=None;io_builds={}

    def phase(name,predict):
        nonlocal reference,reference_candidates
        rows=[]
        def capture(a):
            pred,stats=predict(a);stash['full']=pred;return pred,stats
        for i in range(args.runs+args.warmup):
            row,tileheads=canvas_plane_once(capture,hu,search,jobs,g,cfg)
            candidates=consolidate(*decode(tileheads,jobs,g,cfg,search),g,cfg)
            full=stash['full']
            if reference is None:reference=full.copy();reference_candidates=candidates
            if not np.array_equal(full,reference) or candidates!=reference_candidates:
                raise AssertionError(name+' full-head/candidate mismatch; discard timings')
            if i>=args.warmup:
                rows.append(row)
                print('{} {}/{}: {:.3f} ms, SAME full head + candidates'.format(name,i-args.warmup+1,args.runs,row['slice_sampling_to_candidates_ms']),flush=True)
        means={k:float(np.mean([r[k] for r in rows])) for k in ['slice_sampling_to_candidates_ms','sample_ms','network_ms','input_pack_ms','patch_ms','decode_ms']}
        if name!='native_ARM':
            means['hardware_active_ms']=float(np.mean([r['details']['pl_command_cycles']/100000. for r in rows]))
            means['execute_wait_ms']=float(np.mean([r['details']['timing_ms']['execute_wait_ms'] for r in rows]))
        data=dict(status='same_full_canvas_and_candidates',samples=rows,mean_ms=means)
        save(out/(name+'.json'),data);results[name]=data;heads[name]=full.copy()

    try:
        with CanvasCPU(cpu_lib,model,parameters,h,w,2) as cpu:
            if cpu.features&3!=3:raise RuntimeError('Require actual ARM NEON/OpenMP')
            def cpu_predict(image):
                tick=time.perf_counter();q=native.lower(image);packed=time.perf_counter()
                codes=cpu.forward(q);end=time.perf_counter()
                pred=(codes.astype(np.float32)-128)*np.asarray(model['head_scales'],np.float32)[:,None,None]
                return pred,dict(host_total_ms=(end-packed)*1000,arm_quantize_stride2_pack_ms=(packed-tick)*1000,patch_total_ms=(time.perf_counter()-tick)*1000)
            phase('native_ARM',cpu_predict)
        for name,folder in [('baseline_LUT_PL',here/'baseline'),('prefetch_PL',here)]:
            net,overlay,clocks,io_builds[name]=hardware(folder,model,parameters,args.timeout)
            with np.load(parent/'nodule_reference.npz') as z:
                for i in range(4):
                    def observe(node,words,i=i,name=name):
                        actual=unpack_words(words,net.shapes[node][0])
                        np.testing.assert_array_equal(actual,z['case'+str(i)+'_'+node])
                        all_nodes[name+'_case'+str(i)+'_'+node]=actual.copy()
                    net.predict_codes(z['case'+str(i)+'_input'],observe)
                    print('{} physical all-node check {}/4 PASS'.format(name,i+1),flush=True)
            conv,tensor=net.conv,net.tensor;net.close();net=None
            net=DirectCanvasPL(conv,tensor,allocate,model,parameters,h,w,native)
            def checked(image):
                if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
                result=net.predict_hu(image)
                if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
                return result
            phase(name,checked);cma_bytes=net.cma_bytes;net.close();net=None
            conv._idle();tensor._idle();conv=tensor=None
        np.savez_compressed(out/'full_heads.npz',**heads)
        np.savez_compressed(out/'physical_all_nodes.npz',**all_nodes)
        means={k:v['mean_ms']['slice_sampling_to_candidates_ms'] for k,v in results.items()}
        active={k:results[k]['mean_ms']['hardware_active_ms'] for k in ['baseline_LUT_PL','prefetch_PL']}
        gains=dict(complete_plane=means['baseline_LUT_PL']/means['prefetch_PL'],
                   hardware_active=active['baseline_LUT_PL']/active['prefetch_PL'],
                   same_native_arm_vs_prefetch_pl=means['native_ARM']/means['prefetch_PL'])
        save(out/'result.json',dict(status='physical_prefetch_all_nodes_and_full_canvas_exact',means_ms=means,
            hardware_active_ms=active,gains=gains,results=results,runs=args.runs,warmup=args.warmup,
            manifest_sha256=sha(here/'prefetch_manifest.json'),parent_manifest_sha256=sha(parent/'nodule_manifest.json'),
            baseline_bit_sha256=sha(here/'baseline/lung_network.bit'),prefetch_bit_sha256=sha(here/'lung_network.bit'),
            full_heads_sha256=sha(out/'full_heads.npz'),all_nodes_sha256=sha(out/'physical_all_nodes.npz'),
            all_node_physical_checks_per_bit=4,clock_mhz=float(clocks.fclk0_mhz),cpu_mhz=float(clocks.cpu_mhz),
            machine=platform.machine(),cma_resident_bytes=cma_bytes,
            input_build=input_build,cpu_build=cpu_build,io_builds=io_builds,
            candidate_sha256=hashlib.sha256(json.dumps(reference_candidates,sort_keys=True).encode()).hexdigest(),
            model_sampling_quantization_threshold_unchanged=True,
            scope='One complete detector plane with online sampling/input/CNN/decode/NMS. Excludes prior lung segmentation, bit loading/setup, case I/O, cross-layer volume scoring and validation-only checks. Hardware active time is a subset of complete-plane time.'))
        print(json.dumps(dict(means_ms=means,hardware_active_ms=active,gains=gains),indent=2),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(status='failed_discard_timings',error=repr(exc)));raise
    finally:
        if net is not None:net.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',default='/home/xilinx/ct_test/nodule_board')
    p.add_argument('--runs',type=int,default=5);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--timeout',type=float,default=30.);p.add_argument('--out',required=True);run(p.parse_args())
