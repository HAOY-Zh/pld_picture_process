"""Separate native input benefit from the migrated exact-LUT hardware benefit."""
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
    if not 1<=args.runs<=10 or not 1<=args.warmup<=3:raise ValueError('Invalid repetitions')
    here=Path(__file__).resolve().parent;parent=Path(args.package).resolve();sys.path.insert(0,str(parent))
    from nodule_plan import load_package,sha
    from nodule_benchmark import hardware,save
    from nodule_canvas import lower_canvas
    from nodule_canvas_cpu import CanvasCPU
    from nodule_canvas_pl import CanvasPL
    from nodule_canvas_direct import DirectCanvasPL
    from nodule_canvas_input import compile_input,NativeCanvasInput
    from nodule_cpu import compile_native
    from nodule_canvas_benchmark import canvas_plane_once
    from nodule_fast_input import SharedPlaneSampler
    from lung_nodule_cnn import config_from_dict,decode,consolidate
    from lung_continuous import unpack_words
    m=json.loads((here/'direct_manifest.json').read_text())
    for n,h in m['files_sha256'].items():
        p=(here/n).resolve()
        if here not in p.parents or sha(p)!=h:raise ValueError('Changed direct/LUT addon '+n)
    _,_,model,parameters=load_package(parent)
    if sha(parent/'nodule_manifest.json')!=m['parent_manifest_sha256']:raise ValueError('Wrong canonical model/bit parent')
    cfg=config_from_dict(model['detector_config']);meta=json.loads((here/'slice_case.json').read_text());g=meta['local_geometry']
    if sha(here/'slice_case.npz')!=meta['payload_sha256']:raise ValueError('Wrong slice')
    with np.load(here/'slice_case.npz') as z:hu=z['hu'];search=z['search'];jobs=[tuple(j) for j in z['jobs']]
    sampler=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg);h,w=map(int,sampler.context.shape[1:]);del sampler
    out=Path(args.out).resolve();out.mkdir();lib,input_build=compile_input(here);native=NativeCanvasInput(lib)
    cpu_lib,cpu_build=compile_native(parent);heads={};results={};reference=None;reference_candidates=None;stash={};net=None
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
                rows.append(row);print('{} {}/{}: {:.3f} ms, SAME full head + candidates'.format(name,i-args.warmup+1,args.runs,row['slice_sampling_to_candidates_ms']),flush=True)
        means={k:float(np.mean([r[k] for r in rows])) for k in ['slice_sampling_to_candidates_ms','sample_ms','network_ms','input_pack_ms','patch_ms','decode_ms']}
        phase=dict(status='same_full_canvas_and_candidates',samples=rows,mean_ms=means)
        save(out/(name+'.json'),phase);results[name]=phase;heads[name]=full.copy()
    try:
        with CanvasCPU(cpu_lib,model,parameters,h,w,2) as cpu:
            if cpu.features&3!=3:raise RuntimeError('Require actual ARM NEON/OpenMP')
            def cpu_predict(image,direct):
                tick=time.perf_counter()
                q=native.lower(image) if direct else lower_canvas(native.lut[image.astype(np.int32)+32768])
                packed=time.perf_counter();codes=cpu.forward(q);end=time.perf_counter()
                pred=(codes.astype(np.float32)-128)*np.asarray(model['head_scales'],np.float32)[:,None,None]
                return pred,dict(host_total_ms=(end-packed)*1000,arm_quantize_stride2_pack_ms=(packed-tick)*1000,patch_total_ms=(time.perf_counter()-tick)*1000)
            phase('canvas_ARM',lambda a:cpu_predict(a,False));phase('native_ARM',lambda a:cpu_predict(a,True))
        net,overlay,clocks,flow_io=hardware(parent,model,parameters,args.timeout)
        conv,tensor=net.conv,net.tensor;net.close();net=None
        from pynq import allocate
        net=CanvasPL(conv,tensor,allocate,model,parameters,h,w)
        def checked(a):
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            result=net.predict_hu(a)
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            return result
        phase('canvas_PL',checked);net.close();net=None
        net=DirectCanvasPL(conv,tensor,allocate,model,parameters,h,w,native)
        phase('direct_PL',checked);net.close();net=None
        conv._idle();tensor._idle()
        # Switch only after every previous command has completed and buffers
        # have been freed while idle. The new overlay is outside timed scopes.
        conv=tensor=None
        net,overlay,clocks,lut_io=hardware(here,model,parameters,args.timeout)
        with np.load(parent/'nodule_reference.npz') as z:
            for i in range(4):
                def observe(name,words,i=i):
                    np.testing.assert_array_equal(unpack_words(words,net.shapes[name][0]),z['case'+str(i)+'_'+name])
                net.predict_codes(z['case'+str(i)+'_input'],observe)
                print('LUT physical all-node check {}/4 PASS'.format(i+1),flush=True)
        conv,tensor=net.conv,net.tensor;net.close();net=None
        net=DirectCanvasPL(conv,tensor,allocate,model,parameters,h,w,native)
        phase('direct_LUT_PL',checked)
        np.savez_compressed(out/'full_heads.npz',**heads)
        means={k:v['mean_ms']['slice_sampling_to_candidates_ms'] for k,v in results.items()}
        gains=dict(direct_input=means['canvas_PL']/means['direct_PL'],lut_only=means['direct_PL']/means['direct_LUT_PL'],
            total_pl=means['canvas_PL']/means['direct_LUT_PL'],same_native_arm_vs_lut_pl=means['native_ARM']/means['direct_LUT_PL'])
        save(out/'result.json',dict(status='physical_direct_input_and_lut_checked',means_ms=means,gains=gains,results=results,
            addon_manifest_sha256=sha(here/'direct_manifest.json'),parent_manifest_sha256=sha(parent/'nodule_manifest.json'),
            full_heads_sha256=sha(out/'full_heads.npz'),clock_mhz=float(clocks.fclk0_mhz),cma_resident_bytes=net.cma_bytes,
            input_build=input_build,cpu_build=cpu_build,flow_io_build=flow_io,lut_io_build=lut_io,
            flow_bit_sha256=sha(parent/'lung_network.bit'),lut_bit_sha256=sha(here/'lung_network.bit'),
            lut_all_node_physical_checks=4,candidate_sha256=hashlib.sha256(json.dumps(reference_candidates,sort_keys=True).encode()).hexdigest(),
            same_model_weights_quantization_threshold=True,sampling_algorithm_changed=False,
            scope='Same complete detector plane, online sampling/input/whole CNN/decode/NMS each repetition. Model setup, bit switching, validation-only comparisons and previous lung segmentation excluded.'))
        print(json.dumps(dict(means_ms=means,gains=gains),indent=2),flush=True)
    except BaseException as exc:save(out/'failure.json',dict(status='failed_discard_timings',error=repr(exc)));raise
    finally:
        if net is not None:net.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',default='/home/xilinx/ct_test/nodule_board')
    p.add_argument('--runs',type=int,default=3);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--timeout',type=float,default=30.);p.add_argument('--out',required=True);run(p.parse_args())
