"""Matched native ARM / prefetch PL / pixel-overlap PL complete-plane test."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np


def run(args):
    if platform.machine().lower() not in ('armv7','armv7l'):raise RuntimeError('Run on PYNQ-Z2')
    if not 1<=args.runs<=20 or not 1<=args.warmup<=3:raise ValueError('Invalid repeat counts')
    here=Path(__file__).resolve().parent;parent=Path(args.package).resolve()
    sys.path[:0]=[str(here),str(parent)]
    from pynq import allocate
    from nodule_plan import load_package,sha
    from nodule_benchmark import hardware,save
    from nodule_cpu import compile_native
    from nodule_canvas_cpu import CanvasCPU
    from nodule_canvas_direct import DirectCanvasPL
    from nodule_canvas_input import NativeCanvasInput,compile_input
    from nodule_native_sampling import NativeXY,NativePlaneSampler,compile_sampling
    from nodule_fast_input import SharedPlaneSampler
    from nodule_host_pipeline import plane_once
    from nodule_fast_decode import ExactDiameterTable
    from lung_nodule_cnn import config_from_dict
    from lung_continuous import unpack_words
    m=json.loads((here/'overlap_manifest.json').read_text())
    for n,h in m['files_sha256'].items():
        p=(here/n).resolve()
        if here not in p.parents or not p.is_file() or sha(p)!=h:raise ValueError('Changed overlap package '+n)
    _,_,model,parameters=load_package(parent)
    if sha(parent/'nodule_manifest.json')!=m['parent_manifest_sha256']:raise ValueError('Different frozen model')
    cfg=config_from_dict(model['detector_config']);meta=json.loads((here/'slice_case.json').read_text());g=meta['local_geometry']
    if sha(here/'slice_case.npz')!=meta['payload_sha256']:raise ValueError('Changed input case')
    with np.load(here/'slice_case.npz') as z:hu=z['hu'];search=z['search'];jobs=[tuple(v) for v in z['jobs']]
    out=Path(args.out).resolve();out.mkdir();net=None
    try:
        lib,input_build=compile_input(here);packer=NativeCanvasInput(lib)
        lib,sampling_build=compile_sampling(here);native=NativeXY(lib,2)
        old=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg)
        new=NativePlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg,native)
        np.testing.assert_array_equal(old.context,new.context)
        h,w=new.context.shape[1:];context_hash=hashlib.sha256(new.context.tobytes()).hexdigest()
        np.savez_compressed(out/'contexts.npz',previous=old.context,optimized=new.context)
        del old,new
        print('Real five-plane online native sampling: EXACT PASS',flush=True)
        table=ExactDiameterTable(model['head_scales'][4])
        cpu_library,cpu_build=compile_native(parent)
        results={};heads={};raw={};all_nodes={};references=[None,None,None];clocks_seen={};bits={};io_builds={}
        def phase(name,predict):
            rows=[]
            for i in range(args.runs+args.warmup):
                row,head,candidates,decoded=plane_once(predict,hu,search,jobs,g,cfg,native,True,table)
                if references[0] is None:
                    references[:]=[head.copy(),candidates,tuple(a.copy() for a in decoded)]
                np.testing.assert_array_equal(head,references[0])
                for a,b in zip(decoded,references[2]):np.testing.assert_array_equal(a,b)
                if candidates!=references[1]:raise AssertionError(name+' candidates changed; discard timings')
                if i>=args.warmup:
                    rows.append(row)
                    print('{} {}/{}: {:.3f} ms, sample {:.3f}, network {:.3f}, decode {:.3f}; SAME full head + raw decode + candidates'.format(
                        name,i-args.warmup+1,args.runs,row['slice_sampling_to_candidates_ms'],row['sample_ms'],row['network_ms'],row['decode_ms']),flush=True)
            keys=['slice_sampling_to_candidates_ms','sample_ms','network_ms','input_pack_ms','patch_ms','decode_ms','extract_ms','decode_regression_ms','plane_merge_ms']
            means={k:float(np.mean([r[k] for r in rows])) for k in keys}
            if name.endswith('_PL'):means['hardware_active_ms']=float(np.mean([r['details']['pl_command_cycles']/100000. for r in rows]))
            result=dict(status='exact_context_head_raw_decode_candidates',samples=rows,mean_ms=means)
            save(out/(name+'.json'),result);results[name]=result;heads[name]=head.copy()
            for k,v in zip(['points','scores','diameters'],decoded):raw[name+'_'+k]=v.copy()
        with CanvasCPU(cpu_library,model,parameters,h,w,2) as cpu:
            if cpu.features&3!=3:raise RuntimeError('Require native ARM NEON/OpenMP two-thread reference')
            def cpu_predict(image):
                tick=time.perf_counter();q=packer.lower(image);packed=time.perf_counter();codes=cpu.forward(q);end=time.perf_counter()
                pred=(codes.astype(np.float32)-128)*np.asarray(model['head_scales'],np.float32)[:,None,None]
                return pred,dict(host_total_ms=(end-packed)*1000,arm_quantize_stride2_pack_ms=(packed-tick)*1000,patch_total_ms=(time.perf_counter()-tick)*1000)
            phase('native_ARM',cpu_predict)
        for name,folder in [('prefetch_PL',here/'baseline'),('overlap_PL',here)]:
            net,overlay,clocks,io_build=hardware(folder,model,parameters,args.timeout)
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('Unexpected FCLK')
            clocks_seen[name]=dict(cpu_mhz=float(clocks.cpu_mhz),fclk_mhz=float(clocks.fclk0_mhz))
            bits[name]=sha(folder/'lung_network.bit');io_builds[name]=io_build
            with np.load(parent/'nodule_reference.npz') as z:
                for i in range(4):
                    def observe(node,words,i=i):
                        a=unpack_words(words,net.shapes[node][0]);np.testing.assert_array_equal(a,z['case'+str(i)+'_'+node])
                        all_nodes[name+'_case'+str(i)+'_'+node]=a.copy()
                    net.predict_codes(z['case'+str(i)+'_input'],observe)
                    print('{} physical all-node check {}/4 PASS'.format(name,i+1),flush=True)
            conv,tensor=net.conv,net.tensor;net.close();net=None
            net=DirectCanvasPL(conv,tensor,allocate,model,parameters,h,w,packer)
            def checked(image):
                if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
                value=net.predict_hu(image)
                if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
                return value
            phase(name,checked)
            cma_bytes=net.cma_bytes;net.close();net=None
        np.savez_compressed(out/'full_heads.npz',**heads);np.savez_compressed(out/'raw_decode.npz',**raw)
        np.savez_compressed(out/'physical_all_nodes.npz',**all_nodes);save(out/'candidates.json',references[1])
        means={k:v['mean_ms']['slice_sampling_to_candidates_ms'] for k,v in results.items()}
        network={k:v['mean_ms']['network_ms'] for k,v in results.items()}
        active={k:results[k]['mean_ms']['hardware_active_ms'] for k in ['prefetch_PL','overlap_PL']}
        gains=dict(complete_plane_pl_before_after=means['prefetch_PL']/means['overlap_PL'],
            hardware_active_pl_before_after=active['prefetch_PL']/active['overlap_PL'],
            same_native_arm_vs_overlap_pl=means['native_ARM']/means['overlap_PL'],
            resident_call_arm_vs_overlap_pl=network['native_ARM']/network['overlap_PL'])
        files={n:sha(out/n) for n in ['contexts.npz','full_heads.npz','raw_decode.npz','physical_all_nodes.npz','candidates.json']}
        result=dict(status='physical_exact_pixel_overlap_three_way_pass',means_ms=means,resident_network_call_ms=network,
            hardware_active_ms=active,gains=gains,results=results,runs=args.runs,warmup=args.warmup,
            manifest_sha256=sha(here/'overlap_manifest.json'),parent_manifest_sha256=sha(parent/'nodule_manifest.json'),
            bit_sha256=bits,clocks=clocks_seen,machine=platform.machine(),context_sha256=context_hash,
            all_node_physical_checks_per_bit=4,online_sampling_each_run=True,
            cma_resident_bytes=cma_bytes,input_build=input_build,sampling_build=sampling_build,cpu_build=cpu_build,io_builds=io_builds,
            evidence_sha256=files,model_sampling_quantization_decode_threshold_unchanged=True,
            scope='One complete existing detector Z plane, online sampling/packing/CNN/decoding/NMS in every timed run. Excludes bit/model/build setup, validation comparisons, earlier lung segmentation and cross-Z volume scoring. Hardware active is a subset of network call, not an additive stage.')
        save(out/'result.json',result)
        print(json.dumps(dict(means_ms=means,resident_network_call_ms=network,hardware_active_ms=active,gains=gains),indent=2),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(status='failed_discard_timings',error=repr(exc)));raise
    finally:
        if net is not None:net.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',default='/home/xilinx/ct_test/nodule_board')
    p.add_argument('--runs',type=int,default=5);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--timeout',type=float,default=30.);p.add_argument('--out',required=True);run(p.parse_args())
