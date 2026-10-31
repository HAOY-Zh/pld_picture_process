"""Same frozen prefetch bit/model; isolate exact host processing improvements."""
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
    m=json.loads((here/'host_manifest.json').read_text())
    for n,h in m['files_sha256'].items():
        p=(here/n).resolve()
        if here not in p.parents or not p.is_file() or sha(p)!=h:raise ValueError('Changed host addon '+n)
    _,_,model,parameters=load_package(parent)
    if sha(parent/'nodule_manifest.json')!=m['parent_manifest_sha256']:raise ValueError('Wrong frozen model/parameters')
    cfg=config_from_dict(model['detector_config']);meta=json.loads((here/'slice_case.json').read_text());g=meta['local_geometry']
    if sha(here/'slice_case.npz')!=meta['payload_sha256']:raise ValueError('Changed plane payload')
    with np.load(here/'slice_case.npz') as z:hu=z['hu'];search=z['search'];jobs=[tuple(v) for v in z['jobs']]
    out=Path(args.out).resolve();out.mkdir();net=None
    try:
        lib,input_build=compile_input(here);packer=NativeCanvasInput(lib)
        lib,sampling_build=compile_sampling(here);native=NativeXY(lib,2)
        # First validate the installed ARM compiler/SciPy/NumPy combination.
        # These checks and extra comparisons are outside formal timed scopes.
        rng=np.random.RandomState(871)
        synthetic=rng.randint(-32768,32768,(3,29,31)).astype(np.int16)
        y=np.r_[-1.,0.,.5,28.,28.000001,rng.uniform(-1,30,71)]
        x=np.r_[-1.,0.,.5,30.,30.000001,rng.uniform(-1,32,79)]
        actual=native.resample(synthetic,[0,1,2],y,x)
        from scipy import ndimage as ndi
        grid=np.meshgrid(y,x,indexing='ij')
        for i in range(3):
            expected=ndi.map_coordinates(synthetic[i].astype(np.float32),grid,order=1,mode='constant',cval=-1000,prefilter=False)
            np.testing.assert_array_equal(actual[i],expected)
        old=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg);new=NativePlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg,native)
        np.testing.assert_array_equal(new.context,old.context)
        context_hash=hashlib.sha256(old.context.tobytes()).hexdigest();h,w=old.context.shape[1:]
        np.savez_compressed(out/'contexts.npz',previous=old.context,optimized=new.context)
        del old,new,actual
        print('ARM native XY float32 + real five-plane context: EXACT PASS',flush=True)
        table_begin=time.perf_counter();diameter_table=ExactDiameterTable(model['head_scales'][4]);table_ms=(time.perf_counter()-table_begin)*1000
        cpu_library,cpu_build=compile_native(parent);heads={};decoded_bank={};all_nodes={};results={};reference=None;reference_candidates=None;reference_decode=None
        def phase(name,predict,optimized):
            nonlocal reference,reference_candidates,reference_decode
            rows=[]
            for i in range(args.runs+args.warmup):
                row,head,candidates,decoded=plane_once(predict,hu,search,jobs,g,cfg,native if optimized else None,optimized,diameter_table if optimized else None)
                if reference is None:
                    reference=head.copy();reference_candidates=candidates;reference_decode=tuple(a.copy() for a in decoded)
                np.testing.assert_array_equal(head,reference)
                for a,b in zip(decoded,reference_decode):np.testing.assert_array_equal(a,b)
                if candidates!=reference_candidates:raise AssertionError(name+' candidates changed; discard timings')
                if i>=args.warmup:
                    rows.append(row)
                    print('{} {}/{}: {:.3f} ms, sample {:.3f}, network {:.3f}, decode {:.3f}; SAME full head + raw decode + candidates'.format(
                        name,i-args.warmup+1,args.runs,row['slice_sampling_to_candidates_ms'],row['sample_ms'],row['network_ms'],row['decode_ms']),flush=True)
            keys=['slice_sampling_to_candidates_ms','sample_ms','network_ms','input_pack_ms','patch_ms','decode_ms','extract_ms','decode_regression_ms','plane_merge_ms']
            means={k:float(np.mean([r[k] for r in rows])) for k in keys}
            if name.endswith('_PL'):means['hardware_active_ms']=float(np.mean([r['details']['pl_command_cycles']/100000. for r in rows]))
            result=dict(status='exact_context_head_raw_decode_candidates',samples=rows,mean_ms=means)
            save(out/(name+'.json'),result);results[name]=result;heads[name]=head.copy()
            for key,value in zip(['points','scores','diameters'],decoded):decoded_bank[name+'_'+key]=value.copy()
        with CanvasCPU(cpu_library,model,parameters,h,w,2) as cpu:
            if cpu.features&3!=3:raise RuntimeError('Require ARM NEON/OpenMP dual-core reference')
            def cpu_predict(image):
                tick=time.perf_counter();q=packer.lower(image);packed=time.perf_counter();codes=cpu.forward(q);end=time.perf_counter()
                pred=(codes.astype(np.float32)-128)*np.asarray(model['head_scales'],np.float32)[:,None,None]
                return pred,dict(host_total_ms=(end-packed)*1000,arm_quantize_stride2_pack_ms=(packed-tick)*1000,patch_total_ms=(time.perf_counter()-tick)*1000)
            phase('previous_ARM',cpu_predict,False);phase('optimized_ARM',cpu_predict,True)
        net,overlay,clocks,io_build=hardware(here,model,parameters,args.timeout)
        with np.load(parent/'nodule_reference.npz') as z:
            for i in range(4):
                def observe(node,words,i=i):
                    a=unpack_words(words,net.shapes[node][0]);np.testing.assert_array_equal(a,z['case'+str(i)+'_'+node])
                    all_nodes['case'+str(i)+'_'+node]=a.copy()
                net.predict_codes(z['case'+str(i)+'_input'],observe)
                print('Prefetch bit physical all-node check {}/4 PASS'.format(i+1),flush=True)
        conv,tensor=net.conv,net.tensor;net.close();net=None
        net=DirectCanvasPL(conv,tensor,allocate,model,parameters,h,w,packer)
        def checked(image):
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            result=net.predict_hu(image)
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            return result
        phase('previous_PL',checked,False);phase('optimized_PL',checked,True)
        np.savez_compressed(out/'full_heads.npz',**heads);np.savez_compressed(out/'raw_decode.npz',**decoded_bank)
        np.savez_compressed(out/'physical_all_nodes.npz',**all_nodes);save(out/'candidates.json',reference_candidates)
        means={k:v['mean_ms']['slice_sampling_to_candidates_ms'] for k,v in results.items()}
        network={k:v['mean_ms']['network_ms'] for k,v in results.items()}
        active={k:results[k]['mean_ms']['hardware_active_ms'] for k in ['previous_PL','optimized_PL']}
        gains=dict(arm_host_before_after=means['previous_ARM']/means['optimized_ARM'],pl_host_before_after=means['previous_PL']/means['optimized_PL'],
            same_optimized_arm_vs_pl=means['optimized_ARM']/means['optimized_PL'],resident_call_arm_vs_pl=network['optimized_ARM']/network['optimized_PL'])
        save(out/'result.json',dict(status='physical_exact_native_host_four_way_pass',means_ms=means,resident_network_call_ms=network,hardware_active_ms=active,gains=gains,results=results,
            runs=args.runs,warmup=args.warmup,manifest_sha256=sha(here/'host_manifest.json'),parent_manifest_sha256=sha(parent/'nodule_manifest.json'),
            bit_sha256=sha(here/'lung_network.bit'),clock_mhz=float(clocks.fclk0_mhz),cpu_mhz=float(clocks.cpu_mhz),machine=platform.machine(),
            context_sha256=context_hash,synthetic_float_planes_exact=True,real_context_exact=True,all_node_physical_checks=4,
            diameter_table_setup_ms=table_ms,diameter_table_sha256=hashlib.sha256(diameter_table.outputs.tobytes()).hexdigest(),
            contexts_sha256=sha(out/'contexts.npz'),
            full_heads_sha256=sha(out/'full_heads.npz'),raw_decode_sha256=sha(out/'raw_decode.npz'),all_nodes_sha256=sha(out/'physical_all_nodes.npz'),
            candidates_sha256=sha(out/'candidates.json'),cma_resident_bytes=net.cma_bytes,input_build=input_build,sampling_build=sampling_build,cpu_build=cpu_build,io_build=io_build,
            same_bit_model_sampling_values_quantization_threshold=True,
            scope='Each timed run performs online resampling/packing/CNN/decoding/NMS. No precomputed-plane cache. Excludes bit/model/build setup, validation-only comparisons, previous lung segmentation and cross-layer volume scoring. PL network call includes input synchronization/controls/guards/head readback; hardware active is a subset, not a second additive stage.'))
        print(json.dumps(dict(means_ms=means,resident_network_call_ms=network,hardware_active_ms=active,gains=gains),indent=2),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(status='failed_discard_timings',error=repr(exc)));raise
    finally:
        if net is not None:net.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',default='/home/xilinx/ct_test/nodule_board')
    p.add_argument('--runs',type=int,default=5);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--timeout',type=float,default=30.);p.add_argument('--out',required=True);run(p.parse_args())
