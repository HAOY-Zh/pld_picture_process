"""Real full-plane CPU integration proof and local host-stage timings.

Desktop measurements are diagnostics; they do not predict PYNQ speed.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from nodule_plan import load_package,sha
from nodule_native_sampling import NativeXY,NativePlaneSampler,compile_sampling
from nodule_fast_input import SharedPlaneSampler
from nodule_host_pipeline import plane_once
from nodule_fast_decode import ExactDiameterTable
from nodule_canvas_cpu import CanvasCPU
from nodule_cpu import compile_native
from nodule_canvas_input import compile_input,NativeCanvasInput
from lung_nodule_cnn import config_from_dict


def run(out,compiler):
    out=Path(out).resolve();out.mkdir()
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';_,_,model,params=load_package(parent)
    source=ROOT/'data/nodule_prefetch_20261024_v2/nodule_prefetch'
    for n in ['nodule_sampling_native.cpp','nodule_canvas_input.cpp']:shutil.copy2(ROOT/'pynq'/n,out/n)
    for n in ['lung_cpu_native.cpp','lung_cpu_kernel.h']:shutil.copy2(parent/n,out/n)
    lib,sample_build=compile_sampling(out,compiler);native=NativeXY(lib)
    lib,pack_build=compile_input(out,compiler);pack=NativeCanvasInput(lib)
    lib,cpu_build=compile_native(out,compiler)
    g=json.loads((source/'slice_case.json').read_text())['local_geometry'];cfg=config_from_dict(model['detector_config'])
    with np.load(source/'slice_case.npz') as z:hu=z['hu'];search=z['search'];jobs=[tuple(v) for v in z['jobs']]
    old=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg);new=NativePlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg,native)
    np.testing.assert_array_equal(old.context,new.context);h,w=old.context.shape[1:]
    rows={};outputs={};candidates={};decoded={};scales=np.asarray(model['head_scales'],np.float32)[:,None,None]
    table=ExactDiameterTable(model['head_scales'][4])
    with CanvasCPU(lib,model,params,h,w,2) as cpu:
        def predict(image):
            tick=time.perf_counter();q=pack.lower(image);packed=time.perf_counter();codes=cpu.forward(q);done=time.perf_counter()
            pred=(codes.astype(np.float32)-128)*scales
            return pred,dict(host_total_ms=(done-packed)*1000,arm_quantize_stride2_pack_ms=(packed-tick)*1000,patch_total_ms=(time.perf_counter()-tick)*1000)
        for name,xy,fast in [('previous',None,False),('optimized',native,True)]:
            rows[name],outputs[name],candidates[name],decoded[name]=plane_once(predict,hu,search,jobs,g,cfg,xy,fast,table if fast else None)
            print(name,{k:round(rows[name][k],3) for k in ['sample_ms','network_ms','decode_regression_ms','plane_merge_ms','slice_sampling_to_candidates_ms']},flush=True)
        np.testing.assert_array_equal(outputs['previous'],outputs['optimized'])
        for a,b in zip(decoded['previous'],decoded['optimized']):np.testing.assert_array_equal(a,b)
        if candidates['previous']!=candidates['optimized']:raise ValueError('Real full-plane candidate mismatch')
        for name in decoded:
            for key,value in zip(['points','scores','diameters'],decoded[name]):outputs[name+'_'+key]=value
        np.savez_compressed(out/'outputs.npz',**outputs)
        # Time host stages repeatedly using the SAME already computed head to
        # isolate the decoder; do not call this a new online plane benchmark.
        def fixed(image):return outputs['previous'],dict(host_total_ms=0.,arm_quantize_stride2_pack_ms=0.,patch_total_ms=0.)
        host={}
        for name,xy,fast in [('previous',None,False),('optimized',native,True)]:
            samples=[plane_once(fixed,hu,search,jobs,g,cfg,xy,fast,table if fast else None)[0] for _ in range(3)]
            host[name]={k:float(np.mean([s[k] for s in samples])) for k in ['sample_ms','decode_ms','decode_regression_ms','plane_merge_ms']}
        proof=dict(status='real_448_native_host_context_head_decode_candidates_exact',board_tested=False,
            slice_payload_sha256=sha(source/'slice_case.npz'),outputs_sha256=sha(out/'outputs.npz'),rows=rows,isolated_host_stage_mean_ms=host,
            candidate_count=len(candidates['previous']),parent_manifest_sha256=sha(parent/'nodule_manifest.json'),
            scope='Windows desktop native CPU actual CNN plus frozen plane; host-only repeated timings explicitly exclude CNN, not PYNQ speed',
            sample_build=sample_build,pack_build=pack_build,cpu_build=cpu_build,
            source_sha256={str(ROOT/'pynq'/n):sha(ROOT/'pynq'/n) for n in ['nodule_sampling_native.cpp','nodule_native_sampling.py','nodule_fast_decode.py','nodule_host_pipeline.py']})
        (out/'verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(dict(status=proof['status'],host=host,candidates=proof['candidate_count']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--compiler',default='E:/vivado2/Vivado/2022.2/tps/mingw/6.2.0/win64.o/nt/bin/g++.exe');a=p.parse_args();run(a.out,a.compiler)
