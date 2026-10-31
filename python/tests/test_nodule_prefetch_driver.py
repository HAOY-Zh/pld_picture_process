"""Exercise the board benchmark/report verifier plumbing with explicit fakes.

These checks do not emulate RTL or establish numerical correctness. Actual
RTL all-node, protocol and physical board checks are separate evidence.
"""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq')]
import nodule_prefetch_benchmark as bench
from nodule_plan import load_package,sha
from verify_nodule_prefetch import verify


class PrefetchDriverTests(unittest.TestCase):
    def test_three_phases_capture_nodes_compute_timings_and_verify_return(self):
        parent=ROOT/'data/nodule_board_20261023_v1/nodule_board'
        _,_,model,parameters=load_package(parent)
        with np.load(parent/'nodule_reference.npz') as z:refs={n:z[n] for n in z.files}
        class CPU:
            features=3
            def __init__(self,*a):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def forward(self,q):return np.full((5,224,224),128,np.uint8)
        class Native:
            def __init__(self,*a):pass
            def lower(self,a):return a
        class Idle:
            def _idle(self):pass
        class Net:
            cma_bytes=6348800
            def __init__(self,*a):self.conv=Idle();self.tensor=Idle();self.i=0;self.shapes={n['name']:n['shape'] for n in model['nodes']}
            def close(self):pass
            def predict_codes(self,a,observer):
                for n in model['nodes']:observer(n['name'],refs['case'+str(self.i)+'_'+n['name']])
                self.i+=1
            def predict_hu(self,a):
                commands=[dict(cycles=1000) for _ in range(15)]
                return np.zeros((5,224,224),np.float32),dict(host_total_ms=3,arm_quantize_stride2_pack_ms=2,patch_total_ms=5,
                    pl_command_cycles=15000,commands=commands,timing_ms=dict(execute_wait_ms=.2))
        def plane(predict,*a):
            _,stats=predict(np.zeros((5,448,448),np.int16))
            return dict(slice_sampling_to_candidates_ms=10,sample_ms=2,network_ms=stats['host_total_ms'],
                input_pack_ms=stats['arm_quantize_stride2_pack_ms'],patch_ms=stats['patch_total_ms'],decode_ms=1,details=stats),None
        modules={n:importlib.import_module(n) for n in ['nodule_benchmark','nodule_canvas_cpu','nodule_canvas_direct','nodule_canvas_input',
            'nodule_cpu','nodule_canvas_benchmark','nodule_fast_input','lung_nodule_cnn','lung_continuous']}
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            folder=Path(tmp);out=folder/'result';(folder/'baseline').mkdir()
            for f in ['lung_network.bit','baseline/lung_network.bit']:(folder/f).write_bytes(f.encode())
            source=ROOT/'data/nodule_direct_lut_20261023_v1/nodule_direct_lut'
            for n in ['slice_case.npz','slice_case.json']:(folder/n).write_bytes((source/n).read_bytes())
            manifest=dict(parent_manifest_sha256=sha(parent/'nodule_manifest.json'),files_sha256={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()})
            (folder/'prefetch_manifest.json').write_text(json.dumps(manifest))
            stack.enter_context(patch.object(bench,'__file__',str(folder/'nodule_prefetch_benchmark.py')))
            stack.enter_context(patch.object(bench.platform,'machine',return_value='armv7l'))
            stack.enter_context(patch.dict(sys.modules,{'pynq':types.SimpleNamespace(allocate=None)}))
            replacements=[('nodule_benchmark','hardware',lambda *a:(Net(),None,types.SimpleNamespace(fclk0_mhz=100,cpu_mhz=667),{})),
                ('nodule_canvas_cpu','CanvasCPU',CPU),('nodule_canvas_direct','DirectCanvasPL',Net),
                ('nodule_canvas_input','compile_input',lambda *a:(None,{})),('nodule_canvas_input','NativeCanvasInput',Native),
                ('nodule_cpu','compile_native',lambda *a:(None,{})),('nodule_canvas_benchmark','canvas_plane_once',plane),
                ('nodule_fast_input','SharedPlaneSampler',lambda *a:types.SimpleNamespace(context=np.zeros((5,448,448),np.int16))),
                ('lung_nodule_cnn','decode',lambda *a:([],[])),('lung_nodule_cnn','consolidate',lambda *a:[]),
                ('lung_continuous','unpack_words',lambda words,*a:words)]
            for mod,name,value in replacements:stack.enter_context(patch.object(modules[mod],name,value))
            with contextlib.redirect_stdout(io.StringIO()):
                bench.run(types.SimpleNamespace(package=str(parent),runs=2,warmup=1,out=str(out),timeout=30))
                result=verify(out,folder)
            self.assertEqual(result['all_node_arrays_checked'],88)
            self.assertEqual(result['gains']['hardware_active'],1)
            with np.load(out/'full_heads.npz') as z:self.assertEqual(set(z.files),{'native_ARM','baseline_LUT_PL','prefetch_PL'})
            # A corrupted newly returned node must not pass independent review.
            with np.load(out/'physical_all_nodes.npz') as z:data={n:z[n] for n in z.files}
            data[next(iter(data))]=data[next(iter(data))].copy();data[next(iter(data))].flat[0]^=1
            np.savez_compressed(out/'physical_all_nodes.npz',**data)
            r=json.loads((out/'result.json').read_text());r['all_nodes_sha256']=sha(out/'physical_all_nodes.npz')
            (out/'result.json').write_text(json.dumps(r))
            with self.assertRaises(AssertionError):verify(out,folder)


if __name__=='__main__':unittest.main()
