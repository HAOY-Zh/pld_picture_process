"""Same-slice baseline/optimized ARM and PL, with exact output checks.

Only preprocessing and tile scheduling change. The sealed package remains the
authority for model, arithmetic, bit, frozen detection config and thresholds.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np


def fast_plane_once(predict_many,hu,search,jobs,geometry,config,batch_size):
    from nodule_fast_input import SharedPlaneSampler
    from lung_nodule_cnn import decode,consolidate
    row=dict(sample_ms=0.,network_ms=0.,input_pack_ms=0.,patch_ms=0.,decode_ms=0.,weight_loads=0)
    start=time.perf_counter();sampler=SharedPlaneSampler(hu,geometry['spacing_zyx_mm'],jobs,config)
    row['sample_ms']=(time.perf_counter()-start)*1000;parts=[[],[],[]];predictions=[];details=[]
    for begin in range(0,len(jobs),batch_size):
        batch=jobs[begin:begin+batch_size];tick=time.perf_counter()
        patches=[sampler.patch(job) for job in batch];row['sample_ms']+=(time.perf_counter()-tick)*1000
        pred,stats=predict_many(patches);predictions.append(pred)
        row['network_ms']+=stats['host_total_ms'];row['patch_ms']+=stats['patch_total_ms']
        row['input_pack_ms']+=stats['arm_quantize_stride2_pack_ms'];row['weight_loads']+=stats.get('weight_loads',0)
        details.append(stats);tick=time.perf_counter()
        for dest,value in zip(parts,decode(pred,batch,geometry,config,search)):dest.append(value)
        row['decode_ms']+=(time.perf_counter()-tick)*1000
    tick=time.perf_counter();candidates=consolidate(*(np.concatenate(a) for a in parts),geometry,config)
    row['plane_merge_ms']=(time.perf_counter()-tick)*1000
    row['slice_sampling_to_candidates_ms']=(time.perf_counter()-start)*1000
    row.update(tiles=len(jobs),plane_candidate_count=len(candidates),resampled_native_planes=sampler.resampled_native_planes,
        resampled_pixels=sampler.resampled_pixels,context_canvas_bytes=sampler.context.nbytes,batches=details)
    return row,np.concatenate(predictions)


def run(args):
    if platform.machine().lower() not in ('armv7l','armv7'):raise RuntimeError('Run on PYNQ-Z2')
    if not 1<=args.runs<=10 or not 1<=args.warmup<=3 or not 1<=args.batch_size<=32:raise ValueError('Invalid repeat/batch setting')
    package=Path(args.package).resolve();sys.path.insert(0,str(package));here=Path(__file__).resolve().parent
    from nodule_plan import load_package,sha,lower_stem,quantize_hu,dequantize_head
    from nodule_cpu import compile_native,NativeNodule
    from nodule_benchmark import hardware,save
    from nodule_slice_benchmark import plane_once,summary
    from nodule_fast_input import compile_input,NativeInput
    from nodule_fast_batch import BatchNodulePL
    from lung_nodule_cnn import config_from_dict
    # Check the addon independently; do not modify the sealed parent manifest.
    addon=json.loads((here/'fast_manifest.json').read_text(encoding='utf-8'))
    for name,digest in addon['files_sha256'].items():
        p=(here/name).resolve()
        if here not in p.parents or sha(p)!=digest:raise ValueError('Changed addon: '+name)
    package,manifest,model,params=load_package(package)
    meta=json.loads((here/'slice_case.json').read_text(encoding='utf-8'))
    if sha(package/'nodule_manifest.json')!=meta['package_manifest_sha256'] or sha(here/'slice_case.npz')!=meta['payload_sha256']:
        raise ValueError('Wrong parent package or slice')
    with np.load(here/'slice_case.npz',allow_pickle=False) as z:hu=z['hu'].copy();search=z['search'].copy();jobs=[tuple(v) for v in z['jobs']]
    geometry=meta['local_geometry'];cfg=config_from_dict(model['detector_config']);out=Path(args.out).resolve();out.mkdir()
    library,input_build=compile_input(here);native=NativeInput(library)
    cpu_library,cpu_build=compile_native(package);results={};outputs={};reference=None;net=None
    def phase(name,call):
        nonlocal reference
        for _ in range(args.warmup):
            _,a=call()
            if reference is None:reference=a
            elif not np.array_equal(reference,a):raise AssertionError(name+' warmup output differs')
        rows=[]
        for i in range(args.runs):
            row,a=call()
            if not np.array_equal(reference,a):raise AssertionError(name+' output differs')
            rows.append(row);print('{} {}/{}: {:.3f} ms, {} tiles, SAME'.format(name,i+1,args.runs,row['slice_sampling_to_candidates_ms'],len(jobs)),flush=True)
        packet=dict(status='same_complete_plane',timing=summary(rows),samples=rows)
        save(out/(name+'.json'),packet);results[name]=packet;outputs[name]=a
    try:
        with NativeNodule(cpu_library,model,params,2) as cpu:
            if cpu.features&3!=3:raise RuntimeError('Need real ARM NEON/OpenMP')
            def old_cpu(patch):
                start=time.perf_counter();q=lower_stem(quantize_hu(patch));packed=time.perf_counter()
                codes=cpu.forward(q);end=time.perf_counter();pred=dequantize_head(codes,model)
                return pred,dict(host_total_ms=(end-packed)*1000,arm_quantize_stride2_pack_ms=(packed-start)*1000,patch_total_ms=(time.perf_counter()-start)*1000)
            def fast_cpu(patches):
                start=time.perf_counter();outputs=[];packing=network=0.
                for patch in patches:
                    tick=time.perf_counter();q=native.lower(patch);packing+=(time.perf_counter()-tick)*1000
                    tick=time.perf_counter();codes=cpu.forward(q);network+=(time.perf_counter()-tick)*1000
                    outputs.append(dequantize_head(codes,model))
                pred=np.stack(outputs)
                return pred,dict(host_total_ms=network,arm_quantize_stride2_pack_ms=packing,patch_total_ms=(time.perf_counter()-start)*1000)
            phase('baseline_ARM',lambda:plane_once(old_cpu,hu,search,jobs,geometry,cfg))
            phase('optimized_ARM',lambda:fast_plane_once(fast_cpu,hu,search,jobs,geometry,cfg,args.batch_size))
        net,overlay,clocks,io_build=hardware(package,model,params,args.timeout)
        def old_pl():
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            result=plane_once(net.predict_hu,hu,search,jobs,geometry,cfg)
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            return result
        phase('baseline_PL',old_pl)
        conv,tensor=net.conv,net.tensor;net.close();net=None
        from pynq import allocate
        net=BatchNodulePL(conv,tensor,allocate,model,params,native,args.batch_size)
        def fast_pl():
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            result=fast_plane_once(net.predict_batch,hu,search,jobs,geometry,cfg,args.batch_size)
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            return result
        phase('optimized_PL',fast_pl)
        means={k:v['timing']['slice_sampling_to_candidates_ms']['mean_ms'] for k,v in results.items()}
        report=dict(status='physical_same_plane_acceleration_checked',plane=meta['plane'],results=results,
            gains=dict(pl_before_after=means['baseline_PL']/means['optimized_PL'],
                arm_before_after=means['baseline_ARM']/means['optimized_ARM'],
                optimized_arm_vs_pl=means['optimized_ARM']/means['optimized_PL']),
            native_input_build=input_build,cpu_build=cpu_build,io_build=io_build,clock_mhz=float(clocks.fclk0_mhz),
            cma_resident_bytes=net.cma_bytes,batch_size=args.batch_size,
            addon_manifest_sha256=sha(here/'fast_manifest.json'),parent_manifest_sha256=sha(package/'nodule_manifest.json'),
            weights_bit_config_changed=False,all_tile_outputs_equal=True,
            scope='Same complete plane, same 32 tiles, same INT8 outputs. Includes online shared sampling and input preparation each run. Excludes precomputed search, prior lung segmentation and volume cross-layer scoring. Never compare desktop CPU timing to this as board timing.')
        np.savez_compressed(out/'plane_predictions.npz',**outputs)
        report['plane_predictions_sha256']=sha(out/'plane_predictions.npz')
        save(out/'result.json',report)
        print(json.dumps(dict(complete_plane_mean_ms=means,gains=report['gains'],cma_resident_bytes=net.cma_bytes),indent=2),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(status='failed_discard_timings',error=repr(exc)));raise
    finally:
        if net is not None:net.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',default='/home/xilinx/ct_test/nodule_board')
    p.add_argument('--runs',type=int,default=3);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--timeout',type=float,default=30.)
    p.add_argument('--out',required=True);run(p.parse_args())
