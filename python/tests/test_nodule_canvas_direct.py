from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq'),str(ROOT/'python/tests')]
from nodule_canvas_input import NativeCanvasInput,compile_input
from nodule_canvas_direct import DirectCanvasPL
from nodule_canvas import lower_canvas,extract_tile_heads
from nodule_plan import load_package
from nodule_int8_cuda_reference import IntegerNodule
from nodule_fast_input import SharedPlaneSampler
from lung_nodule_cnn import config_from_dict,decode,consolidate
from lung_axil_smoke import feature_words
from test_lung_cpu import compiler
from test_lung_fast import CachedMemory,DeviceConv,DeviceTensor


class DirectCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();folder=Path(cls.temp.name)
        shutil.copy2(ROOT/'pynq/nodule_canvas_input.cpp',folder/'nodule_canvas_input.cpp')
        cls.library,cls.build=compile_input(folder,compiler());cls.native=NativeCanvasInput(cls.library)
        _,_,cls.model,cls.parameters=load_package(ROOT/'data/nodule_board_20261023_v1/nodule_board')
        cls.cfg=config_from_dict(cls.model['detector_config']);cls.ref=IntegerNodule(cls.model,cls.parameters)
    @classmethod
    def tearDownClass(cls):
        if os.name=='nt':
            import _ctypes
            _ctypes.FreeLibrary(cls.native.lib._handle)
        cls.temp.cleanup()

    def expected(self,a):return lower_canvas(self.native.lut[a.astype(np.int32)+32768])

    def test_all_65536_hu_values_and_hwc8_tail_and_guards(self):
        a=np.resize(np.arange(-32768,32768,dtype=np.int16),(5,192,256));expected=self.expected(a)
        np.testing.assert_array_equal(self.native.lower(a),expected)
        words=feature_words(expected,0);out=np.full(len(words)+16,np.uint64(0xdeadbeefcafef00d))
        self.native.pack(a,out,True)
        np.testing.assert_array_equal(out[:len(words)],words)
        self.assertTrue(np.all(out[len(words):]==np.uint64(0xdeadbeefcafef00d)))
        self.assertTrue(np.all(out[:len(words)].view(np.uint8).reshape(-1,48)[:,45:]==0))

    def test_rectangles_boundaries_extremes_and_positive_strided_readonly_views(self):
        rng=np.random.default_rng(817)
        for h,w in [(128,128),(192,320),(448,448),(1024,128)]:
            raw=rng.integers(-32768,32767,(5,h*2+3,w*2+3),dtype=np.int16)
            a=raw[:,1:h*2+1:2,2:w*2+2:2];a.flags.writeable=False
            expected=self.expected(a);np.testing.assert_array_equal(self.native.lower(a),expected)
            out=np.empty(h//2*(w//2)*6,np.uint64);self.native.pack(a,out,True)
            np.testing.assert_array_equal(out,feature_words(expected,0))
        for value in [-32768,-1000,-600,0,400,32767]:
            a=np.full((5,128,192),value,np.int16);np.testing.assert_array_equal(self.native.lower(a),self.expected(a))

    def test_reject_wrong_shape_type_stride_alias_and_readonly_or_short_output(self):
        a=np.zeros((5,128,128),np.int16)
        for bad in [a.astype(np.float32),a[:,:,::-1],a[:,:,:127],a[:4]]:
            with self.assertRaises(ValueError):self.native.lower(bad)
        for out in [np.empty(1,np.uint8),np.empty(200000,np.float32),a.view(np.uint8).reshape(-1)]:
            with self.assertRaises(ValueError):self.native.pack(a,out,True)
        out=np.empty(64*64*48,np.uint8);out.flags.writeable=False
        with self.assertRaises(ValueError):self.native.pack(a,out,True)

    def test_real_448_all_nodes_and_candidates_equal_with_separate_cpu_ddr_caches(self):
        src=ROOT/'data/nodule_slice_timing_20261023_v1/nodule_slice_timing';meta=json.loads((src/'slice_case.json').read_text());g=meta['local_geometry']
        with np.load(src/'slice_case.npz') as z:hu=z['hu'];search=z['search'];jobs=[tuple(v) for v in z['jobs']]
        sampler=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,self.cfg);a=sampler.context
        mem=CachedMemory();net=DirectCanvasPL(DeviceConv(mem),DeviceTensor(mem),mem.allocate,self.model,self.parameters,448,448,self.native)
        try:
            for canvas in [a,np.clip(a.astype(np.int32)+127,-32768,32767).astype(np.int16)]:
                q=self.native.lut[canvas.astype(np.int32)+32768]
                codes,trace=self.ref.forward_codes(torch.from_numpy(q[None]),True)
                def observer(name,words):np.testing.assert_array_equal(words,feature_words(trace[name][0],net.shapes[name][1]))
                actual,stats=net.predict_hu(canvas,observer)
                expected=(codes[0].cpu().numpy().astype(np.float32)-128)*np.asarray(self.model['head_scales'],np.float32)[:,None,None]
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(stats['command_count'],15)
                self.assertTrue(all(b.flushes==1 for name,b in net.buffers.items() if name!='input'))
                old=extract_tile_heads(expected,jobs,sampler.y0,sampler.x0,self.cfg)
                new=extract_tile_heads(actual,jobs,sampler.y0,sampler.x0,self.cfg)
                self.assertEqual(consolidate(*decode(old,jobs,g,self.cfg,search),g,self.cfg),consolidate(*decode(new,jobs,g,self.cfg,search),g,self.cfg))
        finally:net.close()
        self.assertTrue(all(b.freed for b in mem.buffers))

    def test_timeout_retains_memory_and_blocks_reuse(self):
        import lung_fast
        mem=CachedMemory();conv=DeviceConv(mem);tensor=DeviceTensor(mem)
        net=DirectCanvasPL(conv,tensor,mem.allocate,self.model,self.parameters,128,128,self.native);mark=len(lung_fast._UNSAFE)
        def timeout():conv._poisoned=True;raise TimeoutError('injected')
        conv.wait=timeout
        try:
            with self.assertRaises(TimeoutError):net.predict_hu(np.zeros((5,128,128),np.int16))
            with self.assertRaises(RuntimeError):net.close()
            with self.assertRaises(RuntimeError):net.predict_hu(np.zeros((5,128,128),np.int16))
            self.assertEqual(len(lung_fast._UNSAFE),mark+1);self.assertFalse(any(b.freed for b in mem.buffers))
        finally:conv._inflight=conv._poisoned=False;net.close();del lung_fast._UNSAFE[mark:]

    def test_bad_dma_guard_or_incomplete_count_rejected(self):
        for failure in ['guard','count']:
            mem=CachedMemory();conv=DeviceConv(mem);tensor=DeviceTensor(mem)
            net=DirectCanvasPL(conv,tensor,mem.allocate,self.model,self.parameters,128,128,self.native);wait=conv.wait
            def corrupt():
                result=wait()
                if failure=='guard':mem.device(conv.configuration[1])[-1]=0
                else:result['written_words']-=1
                return result
            conv.wait=corrupt
            try:
                with self.assertRaises(AssertionError):net.predict_hu(np.zeros((5,128,128),np.int16))
            finally:net.close()

    def test_library_name_cannot_shadow_python_import_on_second_run(self):
        folder=Path(self.temp.name);shutil.copy2(ROOT/'pynq/nodule_canvas_input.py',folder/'nodule_canvas_input.py')
        subprocess.run([sys.executable,'-B','-c',"import sys;sys.path.insert(0,sys.argv[1]);import nodule_canvas_input;assert nodule_canvas_input.__file__.endswith('.py')",str(folder)],check=True,capture_output=True)
        self.assertNotEqual(self.library.stem,'nodule_canvas_input')


if __name__=='__main__':unittest.main()
