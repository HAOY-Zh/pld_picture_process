"""Exercise both bit branches, evidence capture and independent tamper rejection.

Networks are explicitly mocked. This is driver verification, not RTL evidence.
The actual native sampler and the frozen all-node reference bank are used.
"""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq'),str(ROOT/'python/tests')]
import nodule_overlap_benchmark as bench
from nodule_native_sampling import compile_sampling
from nodule_plan import load_package,sha
from verify_nodule_overlap import verify
from test_lung_cpu import compiler


class OverlapDriverTests(unittest.TestCase):
    def test_three_matched_phases_capture_both_bits_and_reject_tampered_node(self):
        parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';_,_,model,_=load_package(parent)
        scales=np.asarray(model['head_scales'],np.float32)[:,None,None]
        with np.load(parent/'nodule_reference.npz') as z:refs={n:z[n] for n in z.files}
        class CPU:
            features=3
            def __init__(self,*a):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def forward(self,q):return np.zeros((5,224,224),np.uint8)
        class Packer:
            def __init__(self,*a):pass
            def lower(self,a):return a
        class Idle:
            def _idle(self):pass
        class Net:
            cma_bytes=6348800
            def __init__(self,*a):
                self.conv=Idle();self.tensor=Idle();self.i=0;self.shapes={n['name']:(n['shape'],0) for n in model['nodes']}
            def close(self):pass
            def predict_codes(self,a,observer):
                for n in model['nodes']:observer(n['name'],refs['case'+str(self.i)+'_'+n['name']])
                self.i+=1
            def predict_hu(self,a):
                return -128*scales*np.ones((5,224,224),np.float32),dict(host_total_ms=3,arm_quantize_stride2_pack_ms=2,patch_total_ms=5,
                    pl_command_cycles=15000,commands=[dict(cycles=1000) for _ in range(15)])
        mods={n:importlib.import_module(n) for n in ['nodule_benchmark','nodule_canvas_cpu','nodule_canvas_direct','nodule_canvas_input',
            'nodule_cpu','nodule_native_sampling','lung_continuous']}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);out=folder/'returned';(folder/'baseline').mkdir()
            shutil.copy2(ROOT/'pynq/nodule_sampling_native.cpp',folder/'nodule_sampling_native.cpp')
            lib,build=compile_sampling(folder,compiler())
            source=ROOT/'data/nodule_prefetch_20261024_v2/nodule_prefetch'
            for n in ['slice_case.npz','slice_case.json']:shutil.copy2(source/n,folder/n)
            (folder/'lung_network.bit').write_bytes(b'new-explicit-test-double-not-hardware')
            (folder/'baseline/lung_network.bit').write_bytes(b'old-explicit-test-double-not-hardware')
            m=dict(parent_manifest_sha256=sha(parent/'nodule_manifest.json'),files_sha256={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()})
            (folder/'overlap_manifest.json').write_text(json.dumps(m))
            native_cls=mods['nodule_native_sampling'].NativeXY;instances=[];loaded=[]
            def native(*a,**kw):
                n=native_cls(*a,**kw);instances.append(n);return n
            def hardware(path,*a):
                loaded.append(Path(path));return Net(),None,types.SimpleNamespace(fclk0_mhz=100,cpu_mhz=667),{}
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(bench,'__file__',str(folder/'nodule_overlap_benchmark.py')))
                stack.enter_context(patch.object(bench.platform,'machine',return_value='armv7l'))
                stack.enter_context(patch.dict(sys.modules,{'pynq':types.SimpleNamespace(allocate=None)}))
                changes=[('nodule_benchmark','hardware',hardware),('nodule_canvas_cpu','CanvasCPU',CPU),('nodule_canvas_direct','DirectCanvasPL',Net),
                    ('nodule_canvas_input','compile_input',lambda *a:(None,{})),('nodule_canvas_input','NativeCanvasInput',Packer),
                    ('nodule_cpu','compile_native',lambda *a:(None,{})),('nodule_native_sampling','compile_sampling',lambda *a:(lib,build)),
                    ('nodule_native_sampling','NativeXY',native),('lung_continuous','unpack_words',lambda words,*a:words)]
                for mod,name,value in changes:stack.enter_context(patch.object(mods[mod],name,value))
                with contextlib.redirect_stdout(io.StringIO()):
                    bench.run(types.SimpleNamespace(package=str(parent),runs=2,warmup=1,out=str(out),timeout=30));checked=verify(out,folder)
                self.assertEqual(loaded,[folder/'baseline',folder]);self.assertEqual(checked['all_node_arrays_checked'],88)
                self.assertEqual(set(checked['means_ms']),{'native_ARM','prefetch_PL','overlap_PL'})
                self.assertEqual(checked['candidates'],0)
                with np.load(out/'physical_all_nodes.npz') as z:data={n:z[n] for n in z.files}
                key=next(n for n in data if n.startswith('overlap_PL_'));data[key]=data[key].copy();data[key].flat[0]^=1
                np.savez_compressed(out/'physical_all_nodes.npz',**data)
                r=json.loads((out/'result.json').read_text());r['evidence_sha256']['physical_all_nodes.npz']=sha(out/'physical_all_nodes.npz')
                (out/'result.json').write_text(json.dumps(r))
                with self.assertRaises(AssertionError):verify(out,folder)
            if os.name=='nt':
                import _ctypes
                for n in instances:_ctypes.FreeLibrary(n.lib._handle)


if __name__=='__main__':unittest.main()
