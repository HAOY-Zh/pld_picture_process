from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
import unittest
import numpy as np
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq'),str(ROOT/'python/tests')]
from nodule_native_sampling import NativeXY,NativePlaneSampler,compile_sampling
from nodule_fast_input import SharedPlaneSampler
from nodule_fast_decode import decode_fast,ExactDiameterTable
from lung_nodule_cnn import DetectorConfig,decode,consolidate,config_from_dict
from nodule_plan import load_package
from test_lung_cpu import compiler


class NativeHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();folder=Path(cls.tmp.name)
        shutil.copy2(ROOT/'pynq/nodule_sampling_native.cpp',folder/'nodule_sampling_native.cpp')
        lib,cls.build=compile_sampling(folder,compiler());cls.native=NativeXY(lib)
    @classmethod
    def tearDownClass(cls):
        if os.name=='nt':
            import _ctypes
            _ctypes.FreeLibrary(cls.native.lib._handle)
        cls.tmp.cleanup()

    def test_float_planes_exact_random_full_hu_range_coordinates_and_borders(self):
        rng=np.random.default_rng(285)
        hu=rng.integers(-32768,32767,(5,71,83),dtype=np.int16)
        y=np.r_[-1.,-.0000001,0.,1.,70.,70.0000001,71.,rng.uniform(-2,74,317)]
        x=np.r_[-1.,-.0000001,0.,1.,82.,82.0000001,83.,rng.uniform(-2,86,309)]
        grid=np.meshgrid(y,x,indexing='ij')
        for threads in [1,2]:
            self.native.threads=threads;out=self.native.resample(hu,[4,0,2],y,x)
            for j,index in enumerate([4,0,2]):
                expected=ndi.map_coordinates(hu[index].astype(np.float32),grid,order=1,mode='constant',cval=-1000,prefilter=False)
                np.testing.assert_array_equal(out[j],expected)
        self.native.threads=2

    def test_signed_ties_single_axes_and_readonly_strided_hu(self):
        rng=np.random.default_rng(921)
        for shape in [(3,1,1),(3,1,17),(3,19,1),(5,27,31)]:
            raw=rng.integers(-32768,32767,(shape[0]*2,shape[1]*2,shape[2]*2),dtype=np.int16)
            hu=raw[::2,::2,::2];hu.flags.writeable=False
            y=np.arange(-2,shape[1]+2,.5);x=np.arange(-2,shape[2]+2,.5)
            actual=self.native.resample(hu,[0,shape[0]-1],y,x)
            for j,index in enumerate([0,shape[0]-1]):
                expected=ndi.map_coordinates(hu[index].astype(np.float32),np.meshgrid(y,x,indexing='ij'),order=1,mode='constant',cval=-1000,prefilter=False)
                np.testing.assert_array_equal(actual[j],expected)

    def test_context_z_mix_rounding_overlap_and_clamp_match(self):
        hu=np.random.default_rng(983).integers(-2000,3000,(17,57,61),dtype=np.int16)
        cfg=DetectorConfig(patch=32,halo=8);spacing=np.array([1.37,.63,.79])
        for z in [-3.,0.,1.25,2.5,3.75,5.,20.,30.]:
            jobs=[(z,-8,-8),(z,8,8),(z,24,24),(z,40,40)]
            old=SharedPlaneSampler(hu,spacing,jobs,cfg);new=NativePlaneSampler(hu,spacing,jobs,cfg,self.native)
            np.testing.assert_array_equal(new.context,old.context)
            self.assertFalse(new.context.flags.writeable)
            for job in jobs:np.testing.assert_array_equal(new.patch(job),old.patch(job))

    def test_invalid_native_inputs_rejected(self):
        hu=np.zeros((3,11,13),np.int16)
        for bad in [hu.astype(np.float32),hu[:,:,::-1],hu[0]]:
            with self.assertRaises(ValueError):self.native.resample(bad,[0],[0],[0])
        for ids,y,x in [([-1],[0],[0]),([3],[0],[0]),([.5],[0],[0]),([0],[np.nan],[0]),([0],[],[0])]:
            with self.assertRaises(ValueError):self.native.resample(hu,ids,y,x)

    def check_decode(self,pred,jobs,g,cfg,search):
        a=decode(pred,jobs,g,cfg,search);b=decode_fast(pred,jobs,g,cfg,search)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(consolidate(*a,g,cfg),consolidate(*b,g,cfg))

    def test_decoder_exact_random_ties_saturation_haloborders_and_search(self):
        rng=np.random.default_rng(671);g=dict(spacing_zyx_mm=[1.37,.63,.79],basis_lps_zyx=[[0,0,-.79],[0,.63,0],[1.37,0,0]],origin_lps_mm=[23,-85,77])
        for dtype in [np.float32,np.float64]:
            for halo in [0,8,16]:
                cfg=DetectorConfig(patch=32,halo=halo if halo<16 else 14)
                jobs=[(0.,-8,-8),(3.75,8,8),(20.,24,24)]
                pred=rng.normal(0,2,(3,5,16,16)).astype(dtype)
                pred[0,0,:3,:]=30;pred[0,0,3:6,:]=20;pred[1,0,7:10,7:10]=-30
                search=rng.random((30,91,93))>.2
                self.check_decode(pred,jobs,g,cfg,None);self.check_decode(pred,jobs,g,cfg,search)
                pred[:,0]=-128;self.check_decode(pred,jobs,g,cfg,search)

    def test_real_448_context_and_quantized_decoder_match(self):
        folder=ROOT/'data/nodule_direct_lut_20261023_v1/nodule_direct_lut'
        meta=json.loads((folder/'slice_case.json').read_text());g=meta['local_geometry']
        _,_,model,_=load_package(ROOT/'data/nodule_board_20261023_v1/nodule_board');cfg=config_from_dict(model['detector_config'])
        with np.load(folder/'slice_case.npz') as z:hu=z['hu'];search=z['search'];jobs=[tuple(v) for v in z['jobs']]
        a=SharedPlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg);b=NativePlaneSampler(hu,g['spacing_zyx_mm'],jobs,cfg,self.native)
        np.testing.assert_array_equal(a.context,b.context)
        rng=np.random.default_rng(801);codes=rng.integers(0,256,(len(jobs),5,64,64),dtype=np.uint8)
        pred=(codes.astype(np.float32)-128)*np.asarray(model['head_scales'],np.float32)[None,:,None,None]
        self.check_decode(pred,jobs,g,cfg,search)
        table=ExactDiameterTable(model['head_scales'][4]);a=decode(pred,jobs,g,cfg,search)
        b=decode_fast(pred,jobs,g,cfg,search,table)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(consolidate(*a,g,cfg),consolidate(*b,g,cfg))
        compact=decode_fast(pred[:,:,15:49,15:49],jobs,g,cfg,search,table,15)
        for x,y in zip(a,compact):np.testing.assert_array_equal(x,y)
        self.assertEqual(consolidate(*a,g,cfg),consolidate(*compact,g,cfg))

    def test_diameter_table_all_codes_and_nonquantized_fallback_exact(self):
        for scale in [.05714818999523253,.001,1.]:
            table=ExactDiameterTable(scale)
            for a in [table.inputs,np.arange(17,dtype=np.float32)*.1234567,np.array([-4.,-.5,0.,.5,4.])]:
                expected=[float(6*np.exp(np.clip(v,-1.8,1.8))) for v in a]
                np.testing.assert_array_equal(table.values(a),expected)


if __name__=='__main__':unittest.main()
