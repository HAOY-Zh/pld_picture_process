import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pynq'))
from lung_fp32 import NativeFP32, compile_fp32
from lung_cpu import encode_model
from test_lung_cpu import compiler
from test_lung_continuous import fixture
from lung_unet import torch_module


class FP32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c=compiler()
        if not c:raise unittest.SkipTest('Native compiler absent, no FP32 execution proof')
        cls.temp=tempfile.TemporaryDirectory();cls.folder=Path(cls.temp.name)
        for n in ['lung_fp32_native.cpp','lung_fp32_kernel.h']:shutil.copy2(Path(__file__).resolve().parents[2]/'pynq'/n,cls.folder/n)
        cls.lib,cls.build=compile_fp32(cls.folder,c)
    @classmethod
    def tearDownClass(cls):
        if os.name=='nt':cls.temp._finalizer.detach()
        else:cls.temp.cleanup()
    def arrays(self):
        model,p,image=fixture();d,w,b,_,_=encode_model(model,p)
        rng=np.random.default_rng(16016)
        return model,d,(rng.standard_normal(len(w))*.03).astype(np.float32),(rng.standard_normal(len(b))*.02).astype(np.float32),(image.astype(np.float32)/255)
    def test_complete_graph_matches_independent_torch_and_thread_counts(self):
        torch=torch_module();torch.set_num_threads(2);model,d,w,b,image=self.arrays();refs={};x=torch.from_numpy(image[None])
        for i,node in enumerate(model['nodes']):
            r=d[i];a=x if r[1]<0 else refs[int(r[1])]
            if r[0]==1:
                k=int(r[9]);total=int(r[6]*r[3]*k*k)
                ww=torch.from_numpy(w[r[19]:r[19]+total].reshape(r[6],r[3],k,k));bb=torch.from_numpy(b[r[20]:r[20]+r[6]])
                out=torch.nn.functional.conv2d(a,ww,bb,padding=int(r[10]))
                if r[13]:out=out.relu()
            elif r[0]==2:out=torch.nn.functional.max_pool2d(a,2)
            else:out=torch.cat((torch.nn.functional.interpolate(a,scale_factor=2,mode='nearest'),refs[int(r[2])]),dim=1)
            refs[i]=out
        saved=image.copy()
        for threads in [1,2]:
            with NativeFP32(self.lib,d,w,b,threads) as net:
                for _ in range(2):
                    np.testing.assert_allclose(net.forward(image),refs[len(d)-1].numpy()[0],atol=2e-6,rtol=2e-5)
                for i in range(len(d)):np.testing.assert_allclose(net.trace(i),refs[i].numpy()[0],atol=2e-6,rtol=2e-5)
        np.testing.assert_array_equal(image,saved)
    def test_bad_shapes_parameters_and_threads_rejected(self):
        _,d,w,b,image=self.arrays()
        for thread in [0,3]:
            with self.assertRaises(RuntimeError):NativeFP32(self.lib,d,w,b,thread)
        for index in [3,7,19,20]:
            bad=d.copy();bad[0,index]=-1
            with self.assertRaises(RuntimeError):NativeFP32(self.lib,bad,w,b,1)
        bad=w.copy();bad[0]=np.nan
        with self.assertRaisesRegex(RuntimeError,'nonfinite'):NativeFP32(self.lib,d,bad,b,1)
    def test_close_input_dtype_and_trace_bounds(self):
        _,d,w,b,image=self.arrays();net=NativeFP32(self.lib,d,w,b,1)
        with self.assertRaises(ValueError):net.forward(image.astype(np.float64))
        with self.assertRaises(ValueError):net.trace(-1)
        net.close();net.close()
        with self.assertRaises(RuntimeError):net.forward(image)
    def test_binary_reuse_identity_and_no_fast_math(self):
        _,info=compile_fp32(self.folder,compiler());self.assertTrue(info['reused'])
        self.assertNotIn('-ffast-math',info['command'])
        self.assertIn('-ffp-contract=off',info['command'])


if __name__=='__main__':unittest.main()
