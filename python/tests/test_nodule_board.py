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
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python'),str(ROOT/'python/tests')]
from nodule_plan import quantize_hu,lower_stem,requant,load_package,validate,dequantize_head
from nodule_cpu import NativeNodule,compile_native
from nodule_pl import NodulePL
from test_lung_cpu import compiler
from test_lung_fast import CachedMemory,DeviceConv,DeviceTensor

PACKAGE=ROOT/'data/nodule_board_20261023_v1/nodule_board'


class NoduleContractTests(unittest.TestCase):
    def test_stride_two_lowering_exact_for_signed_weights_and_zero_padding(self):
        torch.set_num_threads(4);rng=np.random.default_rng(921)
        q=rng.integers(0,256,(5,128,128),dtype=np.uint8)
        w=rng.integers(-127,128,(8,5,3,3),dtype=np.int8)
        a=torch.nn.functional.conv2d(torch.tensor(q[None],dtype=torch.float64),torch.tensor(w,dtype=torch.float64),stride=2,padding=1)
        b=torch.nn.functional.conv2d(torch.tensor(lower_stem(q)[None],dtype=torch.float64),torch.tensor(w.reshape(8,45,1,1),dtype=torch.float64))
        np.testing.assert_array_equal(a.numpy(),b.numpy())
        self.assertTrue(np.all(lower_stem(q)[0,0,:]==0))

    def test_signed_ties_saturation_and_input_contract(self):
        a=np.array([-7,-5,-3,-1,1,3,5,7],np.int64)
        np.testing.assert_array_equal(requant(a,1,1,128),np.rint(a/2).astype(int)+128)
        with self.assertRaises(OverflowError):requant(np.array([2**31]),1,0,0)
        with self.assertRaises(ValueError):quantize_hu(np.zeros((1,128,128)))
        hu=np.full((5,128,128),-1000.);hu[:,0,1]=400
        self.assertEqual(quantize_hu(hu)[0,0,0],0);self.assertEqual(quantize_hu(hu)[0,0,1],255)

    def test_board_entrypoint_imports_without_torch_or_sklearn(self):
        code="import sys;sys.path[:0]=['pynq','python'];import nodule_detect;assert 'torch' not in sys.modules;assert 'sklearn' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,capture_output=True)


@unittest.skipUnless(PACKAGE.exists(),'Run nodule_board_prepare first')
class NoduleBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.manifest,cls.model,cls.params=load_package(PACKAGE)
        with np.load(PACKAGE/'nodule_reference.npz',allow_pickle=False) as z:cls.refs={n:z[n].copy() for n in z.files}

    def test_reject_changed_head_skip_and_channels_before_allocation(self):
        for defect in ['head','skip','input','scale','shift']:
            m=copy.deepcopy(self.model);p={k:v.copy() for k,v in self.params.items()}
            if defect=='head':m['nodes'][-1]['layer']['relu']=True
            if defect=='skip':m['nodes'][6]['sources'].reverse()
            if defect=='input':m['input_shape'][0]=5
            if defect=='scale':m['head_scales'][0]=float('nan')
            if defect=='shift':p['head_shift'][0]=63
            memory=CachedMemory()
            with self.assertRaises((ValueError,OverflowError)):
                NodulePL(DeviceConv(memory),DeviceTensor(memory),memory.allocate,m,p)
            self.assertEqual(len(memory.buffers),0)

    def test_native_cpp_all_four_patches_every_node_matches_independent_integer_oracle(self):
        cxx=compiler()
        self.assertIsNotNone(cxx,'Actual native compilation required for this verification')
        with tempfile.TemporaryDirectory() as td:
            for n in ['lung_cpu_native.cpp','lung_cpu_kernel.h']:shutil.copy2(ROOT/'pynq'/n,td)
            lib,_=compile_native(td,cxx)
            for threads in [1,2]:
                with NativeNodule(lib,self.model,self.params,threads) as net:
                    for i in range(4):
                        image=lower_stem(quantize_hu(self.refs[f'case{i}_hu']))
                        np.testing.assert_array_equal(net.forward(image),self.refs[f'case{i}_head'])
                        for j,node in enumerate(self.model['nodes']):
                            np.testing.assert_array_equal(net.node_codes(j),self.refs[f'case{i}_'+node['name']])
                if os.name=='nt':
                    # ctypes does not unload this test DLL; release only AFTER
                    # its native network was destroyed and all calls finished.
                    import _ctypes
                    _ctypes.FreeLibrary(net.library._handle)

    def test_persistent_runtime_producer_consumer_and_repeated_changed_inputs(self):
        memory=CachedMemory();conv=DeviceConv(memory);tensor=DeviceTensor(memory)
        net=NodulePL(conv,tensor,memory.allocate,self.model,self.params)
        try:
            for i in [0,2,0]:
                pred,stats=net.predict_hu(self.refs[f'case{i}_hu'])
                np.testing.assert_array_equal(pred,dequantize_head(self.refs[f'case{i}_head'],self.model))
                self.assertEqual(stats['command_count'],15)
                self.assertEqual(stats['buffer_allocations_per_frame'],0)
            self.assertEqual(len(memory.buffers),12)
            self.assertTrue(all(b.flushes==1 for n,b in net.buffers.items() if n!='input'))
        finally:net.close()
        self.assertTrue(all(b.freed for b in memory.buffers))

    def test_partial_allocation_failure_frees_owned_buffers(self):
        memory=CachedMemory()
        def alloc(**kw):
            if len(memory.buffers)==3:raise MemoryError('injected')
            return memory.allocate(**kw)
        with self.assertRaises(MemoryError):NodulePL(DeviceConv(memory),DeviceTensor(memory),alloc,self.model,self.params)
        self.assertTrue(all(b.freed for b in memory.buffers))

    def test_uncertain_hardware_completion_retains_cma_and_refuses_reuse(self):
        import lung_fast
        start=len(lung_fast._UNSAFE)
        memory=CachedMemory();conv=DeviceConv(memory);tensor=DeviceTensor(memory)
        net=NodulePL(conv,tensor,memory.allocate,self.model,self.params)
        def fail():
            conv._poisoned=True
            raise TimeoutError('injected outstanding AXI')
        conv.wait=fail
        try:
            with self.assertRaises(TimeoutError):net.predict_hu(self.refs['case0_hu'])
            with self.assertRaises(RuntimeError):net.close()
            with self.assertRaises(RuntimeError):net.predict_hu(self.refs['case0_hu'])
            self.assertTrue(all(not b.freed for b in memory.buffers))
            self.assertEqual(len(lung_fast._UNSAFE),start+1)
        finally:
            # Fake driver only: simulate completed shared-system recovery.
            conv._inflight=conv._poisoned=False;net.close();del lung_fast._UNSAFE[start:]


if __name__=='__main__':unittest.main()
