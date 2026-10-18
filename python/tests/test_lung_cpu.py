import ctypes
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pynq'))
from lung_cpu import compile_native,NativeUNet,encode_model
from lung_cpu_compare import statistics
from lung_continuous_emulator import Memory,Conv,Tensor
from lung_continuous import ContinuousUNet,unpack_words
from lung_fpga_contract import requantize
from test_lung_continuous import fixture


def compiler():
    candidates=[os.environ.get('LUNG_CXX'),shutil.which('g++'),
                'E:/vivado2/Vivado/2022.2/tps/mingw/6.2.0/win64.o/nt/bin/g++.exe']
    return next((str(p) for p in candidates if p and Path(p).is_file()),None)


class NativeCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cxx=compiler()
        if cxx is None: raise unittest.SkipTest('Native compiler absent; native CPU implementation has NOT been exercised')
        cls.temp=tempfile.TemporaryDirectory(); cls.folder=Path(cls.temp.name)
        for name in ['lung_cpu_native.cpp','lung_cpu_kernel.h']:
            shutil.copy2(Path(__file__).resolve().parents[2]/'pynq'/name,cls.folder/name)
        cls.library,cls.build=compile_native(cls.folder,cxx)

    @classmethod
    def tearDownClass(cls):
        # Windows keeps loaded DLLs locked until process exit; TemporaryDirectory
        # cannot remove one while ctypes still owns a live handle. Avoid unloading
        # an unknown shared handle; tests may keep this tiny OS temp artifact.
        if os.name=='nt': cls.temp._finalizer.detach()
        else: cls.temp.cleanup()

    def test_native_complete_graph_matches_independent_actual_input_emulator(self):
        model,params,image=fixture();memory=Memory();snapshots={}
        reference=ContinuousUNet(Conv(memory),Tensor(memory),memory.allocate,model,params,log=lambda *a,**k:None)
        expected,_=reference.predict_codes(image,lambda name,words:snapshots.update({name:words.copy()}))
        with NativeUNet(self.library,model,params,1) as net:
            np.testing.assert_array_equal(net.forward(image),expected)
            for i,node in enumerate(model['nodes']):
                np.testing.assert_array_equal(net.node_codes(i),unpack_words(snapshots[node['name']],node['shape']))

    def test_single_and_dual_thread_outputs_match_and_input_preserved(self):
        model,params,image=fixture();saved=image.copy()
        with NativeUNet(self.library,model,params,1) as one,NativeUNet(self.library,model,params,2) as two:
            expected=one.forward(image).copy()
            np.testing.assert_array_equal(two.forward(image),expected)
            np.testing.assert_array_equal(two.forward(image),expected)
            self.assertTrue(two.features&2)
        np.testing.assert_array_equal(image,saved)

    def test_rounding_signed_ties_shift_zero_and_62_exact(self):
        model,params,_=fixture()
        with NativeUNet(self.library,model,params,1) as net:
            for value in [-2147483648,-255,-5,-3,-1,0,1,3,5,255,2147483647]:
                for m,s in [(1,0),(1073741824,31),(2147483647,62),(2147483647,31)]:
                    for relu in [False,True]:
                        expected=requantize(np.array([value],np.int32),np.array([m],np.int32),np.array([s],np.int32),128,relu)[0]
                        self.assertEqual(net.library.lung_cpu_requant(value,m,s,128,relu),expected)

    def test_native_rejects_bad_dimensions_and_accumulator_overflow_bound(self):
        model,params,image=fixture()
        with NativeUNet(self.library,model,params,1) as net:
            d,w,b,m,s=encode_model(model,params);d[0,7]+=1
            handle=net.library.lung_cpu_create(d,len(d),*model['input_shape'],w,len(w),b,m,s,len(b),1)
            self.assertFalse(handle)
        bad=dict(params);bad['enc1_0_bias']=np.full(8,2147483647,np.int32)
        with self.assertRaisesRegex(RuntimeError,'ACC32'): NativeUNet(self.library,model,bad,1)

    def test_close_and_invalid_input_or_thread_count_rejected(self):
        model,params,image=fixture(); net=NativeUNet(self.library,model,params,1)
        with self.assertRaises(ValueError): net.forward(image.astype(np.int16))
        net.close();net.close()
        with self.assertRaises(RuntimeError): net.forward(image)
        with self.assertRaises(RuntimeError): NativeUNet(self.library,model,params,3)

    def test_verified_binary_reuse_checks_identity(self):
        target,record=compile_native(self.folder,compiler())
        self.assertEqual(target,self.library)
        self.assertTrue(record['reused_verified_binary'])

    def test_benchmark_statistics_are_latency_not_fps(self):
        result=statistics([10.,20.,30.])
        self.assertEqual(result['mean_ms'],20.)
        self.assertEqual(result['median_ms'],20.)
        for values in [[],[0.],[float('nan')]]:
            with self.assertRaises(ValueError): statistics(values)


if __name__=='__main__': unittest.main()
