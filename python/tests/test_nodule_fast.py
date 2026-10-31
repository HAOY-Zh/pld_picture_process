import copy
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python'),str(ROOT/'python/tests')]
from nodule_fast_input import SharedPlaneSampler,NativeInput,compile_input
from nodule_fast_batch import BatchNodulePL
from nodule_plan import quantize_hu,lower_stem,load_package,dequantize_head
from lung_axil_smoke import feature_words
from lung_nodule_cnn import DetectorConfig,config_from_dict
from lung_nodule_sampling_cache import sample_patch_reuse
from nodule_detect import PlaneCache
from lung_continuous_emulator import Buffer
from test_lung_fast import CachedMemory,DeviceConv,DeviceTensor
from test_lung_cpu import compiler


class OffsetMemory(CachedMemory):
    def device(self,address):
        for b in self.buffers:
            if b.physical_address<=address<b.physical_address+b.nbytes:
                offset=(address-b.physical_address)//8
                v=np.asarray(b.device)[offset:].view(Buffer);v.physical_address=address;v.freed=False
                return v
        raise AssertionError('Unknown device address')


class CountedConv(DeviceConv):
    def __init__(self,m):super().__init__(m);self.loads=0
    def load_prepared(self,p):self.loads+=1;return super().load_prepared(p)


class FastInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();p=Path(cls.temp.name)
        shutil.copy2(ROOT/'pynq/nodule_fast_input.cpp',p/'nodule_fast_input.cpp')
        lib,cls.build=compile_input(p,compiler());cls.native=NativeInput(lib);cls.library_path=lib
    @classmethod
    def tearDownClass(cls):
        if os.name=='nt':
            import _ctypes
            _ctypes.FreeLibrary(cls.native.lib._handle)
        cls.temp.cleanup()

    def test_all_int16_values_and_native_hwc8_padding_match_frozen_formula(self):
        a=np.resize(np.arange(-32768,32768,dtype=np.int16),(5,128,128))
        expected=lower_stem(quantize_hu(a))
        np.testing.assert_array_equal(self.native.lower(a),expected)
        out=np.full(64*64*6+16,np.uint64(0xdeadbeefcafef00d))
        self.native.pack(a,out[:64*64*6],True)
        np.testing.assert_array_equal(out[:64*64*6],feature_words(expected,0))
        self.assertTrue(np.all(out[64*64*6:]==np.uint64(0xdeadbeefcafef00d)))

    def test_compiled_library_does_not_shadow_python_module_on_subsequent_run(self):
        self.assertNotEqual(Path(self.library_path).stem,'nodule_fast_input')
        folder=Path(self.temp.name)
        shutil.copy2(ROOT/'pynq/nodule_fast_input.py',folder/'nodule_fast_input.py')
        code="import sys;sys.path.insert(0,sys.argv[1]);import nodule_fast_input;assert nodule_fast_input.__file__.endswith('.py')"
        subprocess.run([sys.executable,'-c',code,str(folder)],check=True,capture_output=True)

    def test_strided_native_input_and_readonly_view_reject_bad_buffers(self):
        rng=np.random.default_rng(7);big=rng.integers(-2000,1000,(5,260,270),dtype=np.int16)
        a=big[:,1:257:2,3:259:2];a.flags.writeable=False
        np.testing.assert_array_equal(self.native.lower(a),lower_stem(quantize_hu(a)))
        with self.assertRaises(ValueError):self.native.lower(a[:,:,::-1])
        with self.assertRaises(ValueError):self.native.pack(a,np.empty(1,np.uint8))
        with self.assertRaises(ValueError):self.native.lower(a.astype(np.float32))

    def test_shared_sampling_exact_for_boundaries_fractional_spacing_and_z_clamping(self):
        rng=np.random.default_rng(91);hu=rng.integers(-32768,32767,(8,31,35),dtype=np.int16)
        cfg=DetectorConfig(patch=16,halo=4)
        for spacing in [(1.25,.8,.8),(2.5,.78125,.78125),(.7,.65,.91)]:
            for z in [0.,1.25,3.75,30.]:
                jobs=[(z,-4,-4),(z,0,0),(z,8,12),(z,24,28)]
                fast=SharedPlaneSampler(hu,spacing,jobs,cfg);old=PlaneCache(hu)
                for j in jobs:np.testing.assert_array_equal(fast.patch(j),sample_patch_reuse(old,spacing,j,cfg))
                self.assertFalse(fast.context.flags.writeable)
        with self.assertRaises(ValueError):SharedPlaneSampler(hu,(1,1,1),[(0,0,0),(1,0,0)],cfg)

    def test_batch_actual_graph_outputs_weight_reuse_and_cache_ownership(self):
        package=ROOT/'data/nodule_board_20261023_v1/nodule_board'
        _,_,model,params=load_package(package)
        with np.load(package/'nodule_reference.npz') as a:refs={n:a[n].copy() for n in a.files}
        memory=OffsetMemory();conv=CountedConv(memory);tensor=DeviceTensor(memory)
        net=BatchNodulePL(conv,tensor,memory.allocate,model,params,self.native,batch_size=3)
        try:
            for indices in [[0,2,1],[3],[2,0]]:
                before=conv.loads
                def observer(name,slot,words):
                    expected=feature_words(refs[f'case{indices[slot]}_'+name],net.shapes[name][1])
                    np.testing.assert_array_equal(words,expected)
                pred,stats=net.predict_batch([refs[f'case{i}_hu'] for i in indices],observer)
                for slot,i in enumerate(indices):np.testing.assert_array_equal(pred[slot],dequantize_head(refs[f'case{i}_head'],model))
                self.assertEqual(conv.loads-before,11)
                self.assertEqual(stats['weight_loads'],11);self.assertEqual(stats['command_count'],15*len(indices))
            self.assertEqual(len(memory.buffers),12)
        finally:net.close()
        self.assertTrue(all(b.freed for b in memory.buffers))

    def test_real_slice_all_32_shared_patches_and_fused_inputs_are_identical(self):
        src=ROOT/'data/nodule_slice_timing_20261023_v1/nodule_slice_timing'
        meta=json.loads((src/'slice_case.json').read_text())
        package=ROOT/'data/nodule_board_20261023_v1/nodule_board'
        _,_,model,_=load_package(package);cfg=config_from_dict(model['detector_config'])
        with np.load(src/'slice_case.npz') as a:hu=a['hu'];jobs=[tuple(v) for v in a['jobs']]
        fast=SharedPlaneSampler(hu,meta['local_geometry']['spacing_zyx_mm'],jobs,cfg);old=PlaneCache(hu)
        for j in jobs:
            a=fast.patch(j);expected=sample_patch_reuse(old,meta['local_geometry']['spacing_zyx_mm'],j,cfg)
            np.testing.assert_array_equal(a,expected)
            np.testing.assert_array_equal(self.native.lower(a),lower_stem(quantize_hu(expected)))
        self.assertEqual(fast.resampled_native_planes,3)

    def test_batch_timeout_retains_cma_and_refuses_reuse(self):
        import lung_fast
        _,_,model,params=load_package(ROOT/'data/nodule_board_20261023_v1/nodule_board')
        mem=OffsetMemory();conv=CountedConv(mem);tensor=DeviceTensor(mem)
        net=BatchNodulePL(conv,tensor,mem.allocate,model,params,self.native,2);start=len(lung_fast._UNSAFE)
        def timeout():conv._poisoned=True;raise TimeoutError('injected')
        conv.wait=timeout
        try:
            with self.assertRaises(TimeoutError):net.predict_batch([np.zeros((5,128,128),np.int16)])
            with self.assertRaises(RuntimeError):net.close()
            with self.assertRaises(RuntimeError):net.predict_batch([np.zeros((5,128,128),np.int16)])
            self.assertTrue(all(not b.freed for b in mem.buffers));self.assertEqual(len(lung_fast._UNSAFE),start+1)
        finally:conv._inflight=conv._poisoned=False;net.close();del lung_fast._UNSAFE[start:]

    def test_full_real_plane_old_and_fast_native_cpu_predictions_and_candidates_equal(self):
        from nodule_cpu import compile_native,NativeNodule
        from nodule_slice_benchmark import plane_once
        from nodule_fast_benchmark import fast_plane_once
        import time
        folder=Path(self.temp.name)
        for name in ['lung_cpu_native.cpp','lung_cpu_kernel.h']:shutil.copy2(ROOT/'pynq'/name,folder/name)
        library,_=compile_native(folder,compiler())
        _,_,model,params=load_package(ROOT/'data/nodule_board_20261023_v1/nodule_board')
        src=ROOT/'data/nodule_slice_timing_20261023_v1/nodule_slice_timing'
        meta=json.loads((src/'slice_case.json').read_text());cfg=config_from_dict(model['detector_config']);g=meta['local_geometry']
        with np.load(src/'slice_case.npz') as a:hu=a['hu'];search=a['search'];jobs=[tuple(v) for v in a['jobs']]
        net=NativeNodule(library,model,params,threads=2)
        def old(patch):
            tick=time.perf_counter();codes=net.forward(lower_stem(quantize_hu(patch)))
            pred=dequantize_head(codes,model);ms=(time.perf_counter()-tick)*1000
            return pred,dict(host_total_ms=ms,arm_quantize_stride2_pack_ms=0.,patch_total_ms=ms)
        def fast(patches):
            tick=time.perf_counter();pred=np.stack([dequantize_head(net.forward(self.native.lower(p)),model) for p in patches]);ms=(time.perf_counter()-tick)*1000
            return pred,dict(host_total_ms=ms,arm_quantize_stride2_pack_ms=0.,patch_total_ms=ms)
        try:
            before,p=plane_once(old,hu,search,jobs,g,cfg)
            after,q=fast_plane_once(fast,hu,search,jobs,g,cfg,8)
            np.testing.assert_array_equal(p,q)
            self.assertEqual(before['plane_candidate_count'],after['plane_candidate_count'])
            self.assertEqual(after['tiles'],32)
        finally:
            net.close()
            if os.name=='nt':
                import _ctypes
                _ctypes.FreeLibrary(net.library._handle)


if __name__=='__main__':unittest.main()
