"""Physical same-plane previous-batch versus exact valid-core shared CNN.

Runs both paths on both ARM and PL. Only decoder-required cells are compared;
unused tile padding borders have different semantics and are never evaluated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np


def canvas_plane_once(predict,hu,search,jobs,geometry,config):
    from nodule_fast_input import SharedPlaneSampler
    from nodule_canvas import extract_tile_heads
    from lung_nodule_cnn import decode,consolidate
    tick=time.perf_counter();sampler=SharedPlaneSampler(hu,geometry['spacing_zyx_mm'],jobs,config)
    sample_ms=(time.perf_counter()-tick)*1000
    pred,stats=predict(sampler.context)
    begin=time.perf_counter();heads=extract_tile_heads(pred,jobs,sampler.y0,sampler.x0,config)
    decoded=decode(heads,jobs,geometry,config,search)
    candidates=consolidate(*decoded,geometry,config)
    decode_ms=(time.perf_counter()-begin)*1000
    row=dict(slice_sampling_to_candidates_ms=(time.perf_counter()-tick)*1000,sample_ms=sample_ms,
        network_ms=stats['host_total_ms'],input_pack_ms=stats['arm_quantize_stride2_pack_ms'],
        patch_ms=stats['patch_total_ms'],decode_ms=decode_ms,tiles=len(jobs),canvas_hw=list(sampler.context.shape[1:]),
        plane_candidate_count=len(candidates),details=stats)
    return row,heads


def run(args):
    if platform.machine().lower() not in ('armv7l','armv7'):raise RuntimeError('Run on PYNQ-Z2')
    if not 1<=args.runs<=10 or not 1<=args.warmup<=3:raise ValueError('Invalid repeat counts')
    here=Path(__file__).resolve().parent;parent=Path(args.package).resolve()
    sys.path.insert(0,str(parent))
    from nodule_plan import load_package,sha,dequantize_head
    from nodule_fast_input import compile_input,NativeInput,SharedPlaneSampler
    from nodule_fast_batch import BatchNodulePL
    from nodule_fast_benchmark import fast_plane_once
    from nodule_canvas import lower_canvas
    from nodule_canvas_cpu import CanvasCPU
    from nodule_canvas_pl import CanvasPL
    from nodule_cpu import compile_native,NativeNodule
    from nodule_benchmark import hardware,save
    from lung_nodule_cnn import config_from_dict,decode,consolidate
    addon=json.loads((here/'canvas_manifest.json').read_text())
    for n,h in addon['files_sha256'].items():
        p=(here/n).resolve()
        if here not in p.parents or sha(p)!=h:raise ValueError('Changed addon: '+n)
    _,_,model,params=load_package(parent)
    if sha(parent/'nodule_manifest.json')!=addon['parent_manifest_sha256']:raise ValueError('Wrong frozen parent')
    cfg=config_from_dict(model['detector_config']);meta=json.loads((here/'slice_case.json').read_text())
    if sha(here/'slice_case.npz')!=meta['payload_sha256']:raise ValueError('Changed slice payload')
    with np.load(here/'slice_case.npz') as z:hu=z['hu'];search=z['search'];jobs=[tuple(j) for j in z['jobs']]
    g=meta['local_geometry'];sample=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg)
    height,width=map(int,sample.context.shape[1:]);out=Path(args.out).resolve();out.mkdir()
    library,input_build=compile_input(here);native=NativeInput(library)
    cpu_library,cpu_build=compile_native(parent)
    a=np.arange(-32768,32768,dtype=np.float64);lut=np.rint((a.clip(-1000,400)+1000)*(255./1400.)).astype(np.uint8)
    results={};outputs={};reference=None;reference_candidates=None;net=None
    def phase(name,call):
        nonlocal reference,reference_candidates
        rows=[]
        for i in range(args.warmup+args.runs):
            row,heads=call();needed=np.ascontiguousarray(heads[:,:,15:49,15:49])
            candidates=consolidate(*decode(heads,jobs,g,cfg,search),g,cfg)
            if reference is None:reference=needed.copy();reference_candidates=candidates
            if not np.array_equal(reference,needed) or candidates!=reference_candidates:
                raise AssertionError(name+' required head/candidate mismatch; discard speed result')
            if i>=args.warmup:
                rows.append(row)
                print('{} {}/{}: {:.3f} ms, SAME valid cells + candidates'.format(name,i-args.warmup+1,args.runs,row['slice_sampling_to_candidates_ms']),flush=True)
        means={k:float(np.mean([r[k] for r in rows])) for k in ['slice_sampling_to_candidates_ms','sample_ms','network_ms','input_pack_ms','patch_ms','decode_ms']}
        packet=dict(status='same_required_heads_and_candidates',mean_ms=means,samples=rows)
        save(out/(name+'.json'),packet);results[name]=packet;outputs[name]=needed.copy()
    try:
        with NativeNodule(cpu_library,model,params,2) as cpu:
            if cpu.features&3!=3:raise RuntimeError('ARM baseline requires NEON/OpenMP')
            def tilecpu(patches):
                tick=time.perf_counter();packing=network=0.;heads=[]
                for p in patches:
                    t=time.perf_counter();q=native.lower(p);packing+=(time.perf_counter()-t)*1000
                    t=time.perf_counter();codes=cpu.forward(q);network+=(time.perf_counter()-t)*1000
                    heads.append(dequantize_head(codes,model))
                return np.stack(heads),dict(host_total_ms=network,arm_quantize_stride2_pack_ms=packing,patch_total_ms=(time.perf_counter()-tick)*1000)
            phase('previous_ARM',lambda:fast_plane_once(tilecpu,hu,search,jobs,g,cfg,8))
        with CanvasCPU(cpu_library,model,params,height,width,2) as cpu:
            def canvascpu(a):
                tick=time.perf_counter();q=lower_canvas(lut[a.astype(np.int32)+32768]);packed=time.perf_counter()
                codes=cpu.forward(q);end=time.perf_counter()
                head=(codes.astype(np.float32)-128)*np.asarray(model['head_scales'],np.float32)[:,None,None]
                return head,dict(host_total_ms=(end-packed)*1000,arm_quantize_stride2_pack_ms=(packed-tick)*1000,patch_total_ms=(time.perf_counter()-tick)*1000)
            phase('canvas_ARM',lambda:canvas_plane_once(canvascpu,hu,search,jobs,g,cfg))
        net,overlay,clocks,io_build=hardware(parent,model,params,args.timeout)
        conv,tensor=net.conv,net.tensor;net.close();net=None
        from pynq import allocate
        net=BatchNodulePL(conv,tensor,allocate,model,params,native,8)
        def previouspl():
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            row,head=fast_plane_once(net.predict_batch,hu,search,jobs,g,cfg,8)
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            return row,head
        phase('previous_PL',previouspl);net.close();net=None
        net=CanvasPL(conv,tensor,allocate,model,params,height,width)
        def canvaspl():
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            row,head=canvas_plane_once(net.predict_hu,hu,search,jobs,g,cfg)
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            return row,head
        phase('canvas_PL',canvaspl)
        np.savez_compressed(out/'required_heads.npz',**outputs)
        means={k:v['mean_ms']['slice_sampling_to_candidates_ms'] for k,v in results.items()}
        report=dict(status='physical_canvas_same_plane_pass',means_ms=means,results=results,
            gains=dict(pl_before_after=means['previous_PL']/means['canvas_PL'],arm_before_after=means['previous_ARM']/means['canvas_ARM'],
                same_canvas_arm_vs_pl=means['canvas_ARM']/means['canvas_PL']),
            cma_resident_bytes=net.cma_bytes,clock_mhz=float(clocks.fclk0_mhz),input_build=input_build,cpu_build=cpu_build,io_build=io_build,
            required_heads_sha256=sha(out/'required_heads.npz'),parent_manifest_sha256=sha(parent/'nodule_manifest.json'),
            addon_manifest_sha256=sha(here/'canvas_manifest.json'),candidate_sha256=hashlib.sha256(json.dumps(reference_candidates,sort_keys=True).encode()).hexdigest(),
            canvas_hw=[height,width],selected_tiles=len(jobs),old_commands=15*len(jobs),canvas_commands=15,
            weights_bit_threshold_changed=False,unused_padding_borders_claimed_equal=False,
            scope='One complete detector plane. Includes online sampling, input pack, all network ops, decode and plane NMS; excludes prior lung segmentation, bit download, case I/O and cross-layer volume scoring.')
        save(out/'result.json',report);print(json.dumps(dict(means_ms=means,gains=report['gains']),indent=2),flush=True)
    except BaseException as exc:save(out/'failure.json',dict(status='failed_discard_timings',error=repr(exc)));raise
    finally:
        if net is not None:net.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',default='/home/xilinx/ct_test/nodule_board')
    p.add_argument('--runs',type=int,default=3);p.add_argument('--warmup',type=int,default=1);p.add_argument('--timeout',type=float,default=30.)
    p.add_argument('--out',required=True);run(p.parse_args())
