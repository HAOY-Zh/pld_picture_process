"""A complete axial detection plane, with five-layer context and all lung tiles.

An additive benchmark: reuses the sealed nodule_board package without changing
its weights, bit, runtime or patch benchmark. Final volume-wide cross-layer
scoring is deliberately outside this single-plane measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np


def plane_once(predict,hu,search,jobs,geometry,config):
    from nodule_detect import PlaneCache
    from lung_nodule_sampling_cache import sample_patch_reuse
    from lung_nodule_cnn import decode,consolidate
    if not jobs or any(abs(j[0]-jobs[0][0])>1e-7 for j in jobs):
        raise ValueError('Need all tiles from one nonempty axial plane')
    cache=PlaneCache(hu);parts=[[],[],[]];predictions=[]
    row=dict(sample_ms=0.,network_ms=0.,input_pack_ms=0.,patch_ms=0.,decode_ms=0.)
    start=time.perf_counter()
    for job in jobs:
        tick=time.perf_counter();patch=sample_patch_reuse(cache,geometry['spacing_zyx_mm'],job,config)
        row['sample_ms']+=(time.perf_counter()-tick)*1000
        prediction,stats=predict(patch)
        row['network_ms']+=stats['host_total_ms'];row['patch_ms']+=stats['patch_total_ms']
        row['input_pack_ms']+=stats['arm_quantize_stride2_pack_ms'];predictions.append(prediction)
        tick=time.perf_counter()
        for dest,v in zip(parts,decode(prediction[None],[job],geometry,config,search)):dest.append(v)
        row['decode_ms']+=(time.perf_counter()-tick)*1000
    tick=time.perf_counter()
    candidates=consolidate(*(np.concatenate(p) for p in parts),geometry,config)
    row['plane_merge_ms']=(time.perf_counter()-tick)*1000
    row['slice_sampling_to_candidates_ms']=(time.perf_counter()-start)*1000
    row.update(tiles=len(jobs),plane_candidate_count=len(candidates))
    return row,np.stack(predictions)


def summary(rows):
    return {k:dict(mean_ms=float(np.mean([r[k] for r in rows])),
        min_ms=float(min(r[k] for r in rows)),max_ms=float(max(r[k] for r in rows)))
        for k in ['network_ms','input_pack_ms','patch_ms','sample_ms','decode_ms','plane_merge_ms','slice_sampling_to_candidates_ms']}


def run(args):
    package=Path(args.package).resolve();sys.path.insert(0,str(package))
    from nodule_plan import load_package,sha,quantize_hu,lower_stem,dequantize_head
    from nodule_cpu import compile_native,NativeNodule
    from nodule_benchmark import hardware,save
    from lung_nodule_cnn import config_from_dict
    if platform.machine().lower() not in ('armv7l','armv7'):raise RuntimeError('Run this benchmark on PYNQ-Z2')
    if not 1<=args.runs<=10 or not 1<=args.warmup<=3:raise ValueError('Invalid repeats')
    package,manifest,model,params=load_package(package)
    here=Path(__file__).resolve().parent
    meta=json.loads((here/'slice_case.json').read_text(encoding='utf-8'))
    if (sha(package/'nodule_manifest.json')!=meta['package_manifest_sha256']
            or sha(here/'slice_case.npz')!=meta['payload_sha256']):raise ValueError('Wrong package or slice input')
    with np.load(here/'slice_case.npz',allow_pickle=False) as z:
        hu=z['hu'].copy();search=z['search'].copy();jobs=[tuple(v) for v in z['jobs']]
    geometry=meta['local_geometry'];cfg=config_from_dict(model['detector_config'])
    out=Path(args.out).resolve();out.mkdir()  # Preserve previous measurements.
    cpu_rows=[];pl_rows=[];reference=None
    library,cpu_build=compile_native(package)
    with NativeNodule(library,model,params,2) as cpu:
        if cpu.features&3!=3:raise RuntimeError('Need native ARM NEON/OpenMP baseline')
        def cpu_predict(patch):
            start=time.perf_counter();q=lower_stem(quantize_hu(patch));packed=time.perf_counter()
            codes=cpu.forward(q);end=time.perf_counter();pred=dequantize_head(codes,model)
            return pred,dict(host_total_ms=(end-packed)*1000,
                arm_quantize_stride2_pack_ms=(packed-start)*1000,patch_total_ms=(time.perf_counter()-start)*1000)
        for _ in range(args.warmup):plane_once(cpu_predict,hu,search,jobs,geometry,cfg)
        for i in range(args.runs):
            row,pred=plane_once(cpu_predict,hu,search,jobs,geometry,cfg)
            if reference is None:reference=pred
            elif not np.array_equal(reference,pred):raise AssertionError('ARM plane changed')
            cpu_rows.append(row)
            print('ARM complete plane {}/{}: {:.3f} ms, {} tiles'.format(i+1,args.runs,row['slice_sampling_to_candidates_ms'],len(jobs)),flush=True)
    net,overlay,clocks,io_build=hardware(package,model,params,args.timeout)
    try:
        for _ in range(args.warmup):
            _,pred=plane_once(net.predict_hu,hu,search,jobs,geometry,cfg)
            if not np.array_equal(reference,pred):raise AssertionError('PL plane differs from ARM')
        for i in range(args.runs):
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            row,pred=plane_once(net.predict_hu,hu,search,jobs,geometry,cfg)
            if not np.array_equal(reference,pred):raise AssertionError('PL plane differs from ARM')
            if abs(float(clocks.fclk0_mhz)-100.)>.1:raise RuntimeError('FCLK changed')
            pl_rows.append(row)
            print('PL complete plane {}/{}: {:.3f} ms, {} tiles, SAME'.format(i+1,args.runs,row['slice_sampling_to_candidates_ms'],len(jobs)),flush=True)
        report=dict(status='physical_complete_axial_plane_same_int8',runs=args.runs,warmup=args.warmup,
            input_plane=meta['plane'],cpu=summary(cpu_rows),pl=summary(pl_rows),cpu_samples=cpu_rows,pl_samples=pl_rows,
            same_integer_graph=True,model_sha256=manifest['model_sha256'],
            package_manifest_sha256=sha(package/'nodule_manifest.json'),slice_payload_sha256=sha(here/'slice_case.npz'),
            clock_mhz=float(clocks.fclk0_mhz),cma_resident_bytes=net.cma_bytes,cpu_build=cpu_build,io_build=io_build,
            script_sha256=sha(__file__),whole_volume_cross_scoring_included=False,
            scope='One complete detector axial plane, all selected lung tiles, original XY field of view and five-layer context. Includes sampling, quantization/packing, network, decode and plane merging. Search mask/schedule precomputed. Excludes prior lung segmentation, whole-volume cross-layer scoring, file I/O, overlay download and display.',
            warning='Single-plane diagnostic, not full-volume latency or detection-quality validation. native_index_nearest is a locator; detector plane may lie between native slices.')
        np.savez(out/'plane_predictions.npz',cpu=reference,pl=pred)
        report['plane_predictions_sha256']=sha(out/'plane_predictions.npz')
        save(out/'result.json',report)
        print(json.dumps(dict(plane=meta['plane'],arm=report['cpu'],pl=report['pl']),indent=2),flush=True)
    finally:net.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',default='/home/xilinx/ct_test/nodule_board')
    p.add_argument('--runs',type=int,default=3);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--timeout',type=float,default=30.);p.add_argument('--out',required=True);run(p.parse_args())
