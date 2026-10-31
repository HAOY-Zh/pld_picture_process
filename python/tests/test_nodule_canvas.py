from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python'),str(ROOT/'python/tests')]
from nodule_canvas import expanded_plan,extract_tile_heads,valid_dependency_proof,lower_canvas
from nodule_plan import load_package,quantize_hu,lower_stem
from nodule_int8_cuda_reference import IntegerNodule
from nodule_fast_input import SharedPlaneSampler
from lung_nodule_cnn import config_from_dict,decode,consolidate


class CanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,_,cls.model,cls.params=load_package(ROOT/'data/nodule_board_20261023_v1/nodule_board')
        cls.cfg=config_from_dict(cls.model['detector_config'])
        cls.net=IntegerNodule(cls.model,cls.params)

    def compare(self,canvas,jobs,y0=0,x0=0):
        cfg=self.cfg
        old=self.net(np.stack([canvas[:,int(j[1])-y0:int(j[1])-y0+128,int(j[2])-x0:int(j[2])-x0+128] for j in jobs]))
        whole=self.net(canvas[None])[0]
        new=extract_tile_heads(whole,jobs,y0,x0,cfg)
        np.testing.assert_array_equal(old[:,:,15:49,15:49],new[:,:,15:49,15:49])
        return old,new

    def test_dependency_proof_and_shape_plan(self):
        proof=valid_dependency_proof(self.model)
        self.assertEqual(proof['nodes']['head']['valid_bounds_yx'],[11,53,11,53])
        for h,w in [(128,128),(448,448),(192,384),(1024,1024)]:
            plan,shapes=expanded_plan(self.model,self.params,h,w)
            self.assertEqual(shapes['head'][0],[5,h//2,w//2])
            self.assertEqual(plan['commands'],15)
        for h,w in [(127,128),(130,128),(1032,128)]:
            with self.assertRaises(ValueError):expanded_plan(self.model,self.params,h,w)
        self.assertEqual(self.model['input_shape'],[45,64,64])

    def test_random_extreme_and_aligned_pool_phases(self):
        rng=np.random.default_rng(2910)
        for h,w in [(192,256),(256,192)]:
            jobs=[(0,y,x) for y in range(0,h-127,8) for x in range(0,w-127,32)]
            canvas=rng.integers(-32768,32767,(5,h,w),dtype=np.int16)
            for begin in range(0,len(jobs),16):self.compare(canvas,jobs[begin:begin+16])
        for value in [-32768,-1000,0,400,32767]:
            self.compare(np.full((5,192,192),value,np.int16),[(0,0,0),(0,64,64)])
        a=rng.integers(-2000,1000,(5,128,128),dtype=np.int16)
        np.testing.assert_array_equal(lower_canvas(quantize_hu(a)),lower_stem(quantize_hu(a)))

    def test_real_32_tiles_codes_and_complete_candidates_exact(self):
        import json
        src=ROOT/'data/nodule_slice_timing_20261023_v1/nodule_slice_timing'
        meta=json.loads((src/'slice_case.json').read_text());g=meta['local_geometry']
        with np.load(src/'slice_case.npz') as a:hu=a['hu'];search=a['search'];jobs=[tuple(v) for v in a['jobs']]
        sampler=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,self.cfg)
        old,new=self.compare(sampler.context,jobs,sampler.y0,sampler.x0)
        a=decode(old,jobs,g,self.cfg,search);b=decode(new,jobs,g,self.cfg,search)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(consolidate(*a,g,self.cfg),consolidate(*b,g,self.cfg))

    def test_enlarged_runtime_all_nodes_cache_ownership_and_guards(self):
        from nodule_canvas_pl import CanvasPL
        from test_lung_fast import CachedMemory,DeviceConv,DeviceTensor
        from lung_axil_smoke import feature_words
        memory=CachedMemory();conv=DeviceConv(memory);tensor=DeviceTensor(memory)
        net=CanvasPL(conv,tensor,memory.allocate,self.model,self.params,192,256)
        try:
            rng=np.random.default_rng(321)
            for _ in range(2):
                hu=rng.integers(-2000,1200,(5,192,256),dtype=np.int16)
                codes=self.net.lut[hu.astype(np.int32)+32768]
                head,trace=self.net.forward_codes(torch.from_numpy(codes[None]),True)
                def observer(name,words):
                    np.testing.assert_array_equal(words,feature_words(trace[name][0],net.shapes[name][1]))
                pred,stats=net.predict_hu(hu,observer)
                expected=(head[0].cpu().numpy().astype(np.float32)-128)*np.asarray(self.model['head_scales'],np.float32)[:,None,None]
                np.testing.assert_array_equal(pred,expected)
                self.assertEqual(stats['command_count'],15)
                self.assertTrue(all(buf.flushes==1 for name,buf in net.buffers.items() if name!='input'))
        finally:net.close()
        self.assertTrue(all(b.freed for b in memory.buffers))

    def test_actual_448_runtime_all_nodes_and_exact_native_cpu(self):
        import json,os,shutil,tempfile
        from nodule_canvas_pl import CanvasPL
        from nodule_canvas_cpu import CanvasCPU
        from nodule_cpu import compile_native
        from test_lung_fast import CachedMemory,DeviceConv,DeviceTensor
        from test_lung_cpu import compiler
        from lung_axil_smoke import feature_words
        src=ROOT/'data/nodule_slice_timing_20261023_v1/nodule_slice_timing'
        meta=json.loads((src/'slice_case.json').read_text())
        with np.load(src/'slice_case.npz') as z:hu=z['hu'];jobs=[tuple(v) for v in z['jobs']]
        sampler=SharedPlaneSampler(hu,meta['local_geometry']['spacing_zyx_mm'],jobs,self.cfg)
        q=self.net.lut[sampler.context.astype(np.int32)+32768]
        head,trace=self.net.forward_codes(torch.from_numpy(q[None]),True)
        memory=CachedMemory();net=CanvasPL(DeviceConv(memory),DeviceTensor(memory),memory.allocate,self.model,self.params,448,448)
        try:
            def observer(name,words):np.testing.assert_array_equal(words,feature_words(trace[name][0],net.shapes[name][1]))
            pred,stats=net.predict_hu(sampler.context,observer)
            self.assertEqual(stats['command_count'],15)
            self.assertEqual(net.cma_bytes,6348800)
            old=self.net(np.stack([sampler.patch(j) for j in jobs]))
            new=extract_tile_heads(pred,jobs,sampler.y0,sampler.x0,self.cfg)
            np.testing.assert_array_equal(old[:,:,15:49,15:49],new[:,:,15:49,15:49])
        finally:net.close()
        with tempfile.TemporaryDirectory() as folder:
            for name in ['lung_cpu_native.cpp','lung_cpu_kernel.h']:shutil.copy2(ROOT/'pynq'/name,Path(folder)/name)
            lib,_=compile_native(folder,compiler())
            cpu=CanvasCPU(lib,self.model,self.params,448,448,2)
            try:
                actual=cpu.forward(lower_canvas(q))
                np.testing.assert_array_equal(actual,head[0].cpu().numpy())
                for i,node in enumerate(self.model['nodes']):np.testing.assert_array_equal(cpu.node_codes(i),trace[node['name']][0])
            finally:
                cpu.close()
                if os.name=='nt':
                    import _ctypes
                    _ctypes.FreeLibrary(cpu.library._handle)

    def test_canvas_timeout_retains_allocations_and_rejects_reuse(self):
        import lung_fast
        from nodule_canvas_pl import CanvasPL
        from test_lung_fast import CachedMemory,DeviceConv,DeviceTensor
        mem=CachedMemory();conv=DeviceConv(mem);tensor=DeviceTensor(mem)
        net=CanvasPL(conv,tensor,mem.allocate,self.model,self.params,192,192)
        mark=len(lung_fast._UNSAFE)
        def timeout():conv._poisoned=True;raise TimeoutError('injected canvas timeout')
        conv.wait=timeout
        try:
            with self.assertRaises(TimeoutError):net.predict_hu(np.zeros((5,192,192),np.int16))
            with self.assertRaises(RuntimeError):net.close()
            with self.assertRaises(RuntimeError):net.predict_hu(np.zeros((5,192,192),np.int16))
            self.assertEqual(len(lung_fast._UNSAFE),mark+1)
            self.assertFalse(any(b.freed for b in mem.buffers))
        finally:conv._inflight=conv._poisoned=False;net.close();del lung_fast._UNSAFE[mark:]

    def test_canvas_end_guard_catches_bad_dma_write(self):
        from nodule_canvas_pl import CanvasPL
        from test_lung_fast import CachedMemory,DeviceConv,DeviceTensor
        mem=CachedMemory();conv=DeviceConv(mem);tensor=DeviceTensor(mem)
        net=CanvasPL(conv,tensor,mem.allocate,self.model,self.params,128,192)
        wait=conv.wait
        def corrupt():
            result=wait();mem.device(conv.configuration[1])[-1]=0;return result
        conv.wait=corrupt
        try:
            with self.assertRaisesRegex(AssertionError,'canary'):net.predict_hu(np.zeros((5,128,192),np.int16))
            with self.assertRaises(RuntimeError):net.predict_hu(np.zeros((5,128,192),np.int16))
        finally:net.close()


if __name__=='__main__':unittest.main()
