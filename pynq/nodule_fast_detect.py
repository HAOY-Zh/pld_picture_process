"""Optimized full-case candidate extraction after an existing predicted lung mask.

Same frozen search, tile order, decode, volume NMS and cross-layer ranker.
Shared-plane sampling and layer-major PL batches replace only execution.
"""
import argparse
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np


def run(args):
    if platform.machine().lower() not in ('armv7l','armv7'):raise RuntimeError('Run on PYNQ-Z2')
    if not 1<=args.batch_size<=32:raise ValueError('Batch size must be 1..32')
    package=Path(args.package).resolve();sys.path.insert(0,str(package));here=Path(__file__).resolve().parent
    from nodule_plan import load_package,sha
    from nodule_benchmark import hardware,save
    from nodule_detect import THRESHOLD
    from nodule_fast_input import compile_input,NativeInput,SharedPlaneSampler
    from nodule_fast_batch import BatchNodulePL
    from lung_nodule_cnn import config_from_dict,search_mask,tiles,decode,consolidate
    from lung_nodule_continuity_boundary import evidence_for_packet
    from lung_nodule_continuity import rerank_scores
    addon=json.loads((here/'fast_manifest.json').read_text(encoding='utf-8'))
    for n,h in addon['files_sha256'].items():
        p=(here/n).resolve()
        if here not in p.parents or sha(p)!=h:raise ValueError('Changed addon: '+n)
    package,manifest,model,params=load_package(package)
    if sha(package/'nodule_manifest.json')!=addon['parent_manifest_sha256']:raise ValueError('Wrong parent package')
    source=Path(args.input).resolve();receipt=json.loads((source/'case.json').read_text(encoding='utf-8'))
    if set(receipt['files_sha256'])!={'hu.npy','lung_mask.npy','geometry.json'} or not receipt.get('predicted_lung_mask'):
        raise ValueError('Need complete HU and PREDICTED lung-mask receipt')
    for n,h in receipt['files_sha256'].items():
        p=(source/n).resolve()
        if source not in p.parents or sha(p)!=h:raise ValueError('Changed case input: '+n)
    hu=np.load(source/'hu.npy',mmap_mode='r',allow_pickle=False)
    lung=np.load(source/'lung_mask.npy',mmap_mode='r',allow_pickle=False)
    geometry=json.loads((source/'geometry.json').read_text(encoding='utf-8'));spacing=np.asarray(geometry['spacing_zyx_mm'],float)
    if (hu.dtype!=np.int16 or lung.dtype!=np.bool_ or hu.ndim!=3 or len(hu)<2 or hu.shape!=lung.shape
            or spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any()):raise ValueError('Invalid native HU/mask geometry')
    out=Path(args.out).resolve();out.mkdir();library,input_build=compile_input(here);native=NativeInput(library)
    old,overlay,clocks,io_build=hardware(package,model,params,args.timeout)
    conv,tensor=old.conv,old.tensor;old.close()
    from pynq import allocate
    net=BatchNodulePL(conv,tensor,allocate,model,params,native,args.batch_size)
    start=time.perf_counter();cfg=config_from_dict(model['detector_config']);parts=[[],[],[]];plane_rows=[]
    timing=dict(search_ms=0.,sample_ms=0.,pl_patch_ms=0.,decode_ms=0.,post_ms=0.,input_pack_ms=0.,network_ms=0.)
    commands=weight_loads=pl_cycles=0
    try:
        tick=time.perf_counter();search=search_mask(lung,spacing,cfg);jobs,schedule=tiles(search,spacing,cfg)
        timing['search_ms']=(time.perf_counter()-tick)*1000
        for zm,plane_iter in itertools.groupby(jobs,key=lambda j:j[0]):
            plane=list(plane_iter);pt=time.perf_counter();tick=pt
            sampler=SharedPlaneSampler(hu,spacing,plane,cfg);sample_ms=(time.perf_counter()-tick)*1000
            compute_ms=decode_ms=0.
            for begin in range(0,len(plane),args.batch_size):
                batch=plane[begin:begin+args.batch_size];tick=time.perf_counter()
                patches=[sampler.patch(j) for j in batch];sample_ms+=(time.perf_counter()-tick)*1000
                if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
                pred,stats=net.predict_batch(patches);compute_ms+=stats['patch_total_ms']
                timing['input_pack_ms']+=stats['arm_quantize_stride2_pack_ms'];timing['network_ms']+=stats['host_total_ms']
                commands+=stats['command_count'];weight_loads+=stats['weight_loads'];pl_cycles+=stats['pl_command_cycles']
                tick=time.perf_counter()
                # Keep ALL decoded raw centers in original tile order. Never
                # insert plane-level NMS before the unchanged volume-level NMS.
                for dest,a in zip(parts,decode(pred,batch,geometry,cfg,search)):dest.append(a)
                decode_ms+=(time.perf_counter()-tick)*1000
            timing['sample_ms']+=sample_ms;timing['pl_patch_ms']+=compute_ms;timing['decode_ms']+=decode_ms
            plane_rows.append(dict(z_mm=float(zm),nearest_native_index=min(len(hu)-1,int(round(zm/spacing[0]))),
                tiles=len(plane),sample_ms=sample_ms,pl_patch_ms=compute_ms,decode_ms=decode_ms,
                plane_raw_candidates_ms=(time.perf_counter()-pt)*1000))
            if len(plane_rows)%10==0:print('Optimized planes:',len(plane_rows),'tiles:',sum(r['tiles'] for r in plane_rows),flush=True)
        tick=time.perf_counter()
        candidates=consolidate(*(np.concatenate(a) for a in parts),geometry,cfg) if jobs else []
        for c in candidates:
            box=np.asarray(c['bbox_zyx']).reshape(3,2);box[:,0]=np.maximum(0,box[:,0]);box[:,1]=np.minimum(hu.shape,box[:,1]);c['bbox_zyx']=box.ravel().tolist()
        features,_=evidence_for_packet(hu,spacing,candidates)
        cross=rerank_scores(np.asarray([c['score'] for c in candidates]),features,
            json.loads((package/'nodule_ranker.json').read_text(encoding='utf-8')))
        selected=[]
        for i in np.flatnonzero(cross>=THRESHOLD):
            c=dict(candidates[i]);c['raw_score']=c['score'];c['score']=float(cross[i]);selected.append(c)
        selected.sort(key=lambda c:-c['score']);timing['post_ms']=(time.perf_counter()-tick)*1000
        timing['whole_candidate_ms']=(time.perf_counter()-start)*1000;timing['pl_active_cycles_ms']=pl_cycles/100000.
        timing['amortized_ms_per_original_slice']=timing['whole_candidate_ms']/len(hu)
        if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed before completion')
        save(out/'prediction.json',dict(status='complete_optimized_experimental_pl_int8',backend='PYNQ_PL_INT8',
            candidates=selected,candidate_count=len(selected),raw_candidate_count=len(candidates),threshold=THRESHOLD,
            schedule=schedule,timing=timing,detector_plane_times=plane_rows,native_slices=len(hu),batch_size=args.batch_size,
            commands=commands,weight_loads=weight_loads,cma_resident_bytes=net.cma_bytes,
            parent_manifest_sha256=sha(package/'nodule_manifest.json'),addon_manifest_sha256=sha(here/'fast_manifest.json'),
            input_receipt_sha256=sha(source/'case.json'),native_input_build=input_build,io_build=io_build,
            lung_inference_rerun=False,annotations_used=False,
            scope='Whole candidate extraction including search, online shared sampling, PL batches, decode, unchanged volume NMS and cross scoring. Excludes previous lung segmentation, file transfer/hash checks, bit download and display. Per-original-slice time is amortized, not a directly measured individual slice latency.',
            warning='Same INT8 algorithm; FP32 detection quality does not automatically carry over. Whole-case speed/memory still requires physical validation.'))
        print('Saved:',out/'prediction.json',timing,flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(status='failed_discard_output',error=repr(exc)));raise
    finally:net.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',default='/home/xilinx/ct_test/nodule_board')
    p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--timeout',type=float,default=30.);run(p.parse_args())
