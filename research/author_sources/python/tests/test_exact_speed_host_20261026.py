"""Lossless host optimizations and bounded asynchronous save failure handling."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from scipy import ndimage as ndi

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'data/CT_Studio_ExactSpeed_20261026_03'
sys.path[:0]=[str(APP/'python')]
from nodule_exact_host import quantize_exact,fill_holes_exact,search_mask_exact,ExactSamplerCache
from nodule_canvas_contract import quantize_canvas
from nodule_vendor.lung_nodule_cnn import DetectorConfig,search_mask
from nodule_vendor.nodule_fast_input import SharedPlaneSampler
from ldct_workspace import LDCTWorkspaceMixin


class Client:
    def __init__(self,fail=None):self.calls=0;self.fail=fail;self.shared=np.empty((16,16),np.int16)
    def denoise(self,original,method):
        self.calls+=1
        if self.calls==self.fail:raise RuntimeError('synthetic request failure')
        self.shared[:]=original+1
        return self.shared,dict(stats=dict(total_ms=1.,timing_scope='UNIT_TEST_NOT_FPGA'))


class Harness(LDCTWorkspaceMixin):
    def __init__(self,folder,fail=None):
        self.output=Path(folder);self.series_id='test_case';(self.output/self.series_id).mkdir()
        self.original_hu=np.stack([np.full((16,16),z,np.float32)for z in range(6)])
        self.original_details=dict(source_slice_indices=list(range(6)))
        self.ldct_identity=dict(model_id='test_model');self.ldct_groups={'group':{}}
        self.lock=threading.RLock();self.cancel=threading.Event();self.client=Client(fail)
        self.ldct_running=True;self.registered=False;self.status={}
    def _update(self,**values):self.status.update(values)
    def _register_enhanced_source(self,group,backend,method):
        assert len(self.ldct_groups[group])==6
        for z,row in self.ldct_groups[group].items():
            with np.load(self.output/self.series_id/'ldct'/(row['metadata']['result_id']+'.npz'))as a:
                np.testing.assert_array_equal(a['hu'],self.original_hu[z]+1)
        self.registered=True


class ExactSpeedTests(unittest.TestCase):
    def test_all_65536_int16_values_quantize_exactly(self):
        values=np.resize(np.arange(-32768,32768,dtype=np.int16),5*128*128).reshape(5,128,128)
        np.testing.assert_array_equal(quantize_exact(values),quantize_canvas(values))
        np.testing.assert_array_equal(quantize_exact(values[:,:,::-1]),quantize_canvas(values[:,:,::-1]))
        floats=values.astype(np.float32)+.25
        np.testing.assert_array_equal(quantize_exact(floats),quantize_canvas(floats))

    def test_four_connected_background_holes_and_complete_search_equal(self):
        rng=np.random.default_rng(26)
        for plane in [np.zeros((21,23),bool),np.ones((21,23),bool),np.eye(23,dtype=bool),
                *[rng.random((37,43))<p for p in (.1,.3,.5,.7,.9)for _ in range(4)]]:
            np.testing.assert_array_equal(fill_holes_exact(plane),ndi.binary_fill_holes(plane))
        lung=rng.random((9,71,83))<.7;before=lung.copy();cfg=DetectorConfig().checked()
        np.testing.assert_array_equal(search_mask_exact(lung,[2.5,.78,1.1],cfg),search_mask(lung,[2.5,.78,1.1],cfg))
        np.testing.assert_array_equal(lung,before)

    def test_cached_float_interpolation_rounding_and_case_isolation(self):
        hu=np.random.default_rng(30).integers(-1200,2000,(13,47,59),dtype=np.int16).astype(np.float32)+.25
        before=hu.copy();spacing=[2.5,.78125,.9];cfg=DetectorConfig().checked()
        cached=ExactSamplerCache(hu,spacing,max_bytes=1024*1024,max_planes=4)
        for z in [0.,1.25,2.5,3.75,5.,6.25,30.,31.25]:
            for jobs in [[(z,-32,-32)],[(z,-32,-32),(z,32,32)]]:
                a=SharedPlaneSampler(hu,spacing,jobs,cfg);b=cached.sample(jobs,cfg)
                np.testing.assert_array_equal(a.context,b.context)
                self.assertEqual((a.y0,a.x0),(b.y0,b.x0));self.assertFalse(b.context.flags.writeable)
                self.assertLessEqual(len(cached.planes),4)
                self.assertLessEqual(sum(v.nbytes for v in cached.planes.values()),1024*1024)
        self.assertGreater(cached.hits,0);np.testing.assert_array_equal(hu,before)
        fresh=ExactSamplerCache(hu+10,spacing)
        self.assertEqual(fresh.hits,0)
        np.testing.assert_array_equal(fresh.sample([(1.25,-32,-32)],cfg).context,
            SharedPlaneSampler(hu+10,spacing,[(1.25,-32,-32)],cfg).context)

    def test_saved_outputs_are_frozen_and_drained_before_registration(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);h._enhance(list(range(6)),'FPGA_LDCT','cnn','group')
            self.assertTrue(h.registered);self.assertEqual(h.status['phase'],'done')
            self.assertFalse(h.ldct_running);self.assertEqual(h.status['completed'],6)

    def test_request_failure_retains_only_valid_partial_outputs(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp,fail=3);h._enhance(list(range(6)),'FPGA_LDCT','cnn','group')
            self.assertFalse(h.registered);self.assertEqual(h.status['phase'],'error')
            self.assertFalse(h.status['board_connected']);self.assertEqual(h.client.calls,3)
            self.assertEqual(len(h.ldct_groups['group']),2)

    def test_save_failure_prevents_full_volume_publication(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp)
            with patch('ldct_workspace.np.savez_compressed',side_effect=OSError('synthetic disk failure')):
                h._enhance(list(range(6)),'FPGA_LDCT','cnn','group')
            self.assertFalse(h.registered);self.assertEqual(h.status['phase'],'error')
            self.assertFalse(h.ldct_running)

    def test_cancel_drains_inflight_save_without_publishing_partial_source(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(tmp);save=np.savez_compressed
            def cancel_after_save(*a,**kw):
                save(*a,**kw);h.cancel.set()
            with patch('ldct_workspace.np.savez_compressed',side_effect=cancel_after_save):
                h._enhance(list(range(6)),'FPGA_LDCT','cnn','group')
            self.assertFalse(h.registered);self.assertEqual(h.status['phase'],'cancelled')
            self.assertFalse(h.ldct_running)


if __name__=='__main__':unittest.main()
