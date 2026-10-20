"""ROI workflow on native HU, source isolation and real local HTTP; no board."""
import base64
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib import request, error

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct_workspace_web import WorkspaceController, workspace_handler
from roi_workspace import CpuRoiProcessor, physical_spacing
from test_lung_live import RunningServer


def wait(controller):
    deadline = time.monotonic() + 10
    while controller.snapshot()['phase'] in ('loading', 'processing', 'connecting', 'computing', 'reconstructing'):
        if time.monotonic() > deadline: raise AssertionError('ROI job timeout')
        time.sleep(.005)
    return controller.snapshot()


class ROIWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.hu = np.arange(3*16*20, dtype=np.float32).reshape(3,16,20)-100
        self.path = self.root/'ct.npz'
        np.savez(self.path, hu=self.hu, spacing_zyx_mm=[3.,.5,.8],origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
        self.c = WorkspaceController(None, {}, self.root/'results')
        self.load()

    def load(self):
        self.c.submit('load',dict(path=str(self.path),confirm_hu=True,load_expert=False))
        state=wait(self.c);self.assertEqual(state['phase'],'idle',state['error'])

    def tearDown(self):
        wait(self.c);self.tmp.cleanup()

    def values(self, **extra):
        return dict(case_id=self.c.status['case_id'],analysis_source_id=self.c.analysis_source_id,
            index=1, mode='slice', roi_yx=[1,4,2,7],use_hu=False, backend='CPU_REFERENCE', **extra)

    def run_roi(self, **extra):
        values=self.values(); values.update(extra)
        self.c.submit('roi',values);state=wait(self.c)
        self.assertEqual(state['phase'],'done',state['error'])
        return self.c.roi_result

    def payload(self, r, index=1):
        return self.c.roi_payload(r['case_id'],r['analysis_source_id'],r['result_id'],index)

    def test_current_native_hu_and_exact_area_without_mutating_any_input(self):
        original=self.c.hu.copy();r=self.run_roi()
        self.assertEqual(r['valid_count'],15)
        self.assertAlmostEqual(r['slice_results'][0]['area_mm2'],6.)
        self.assertIsNone(r['region_volume_ml'])
        self.assertEqual(r['hu_statistics']['mean'],float(original[1,1:4,2:7].mean()))
        np.testing.assert_array_equal(self.c.hu,original)
        self.assertEqual(int(self.c.known.sum()),0)
        self.assertIsNone(self.c.basic_result)
        p=self.payload(r);mask=np.frombuffer(base64.b64decode(p['mask']['pixels']),np.uint8).reshape(16,20)
        expected=np.zeros((16,20),np.uint8);expected[1:4,2:7]=1
        np.testing.assert_array_equal(mask,expected)

    def test_hu_thresholds_are_inclusive_and_use_hu_not_window_grayscale(self):
        lo=int(self.hu[1,2,3]);hi=lo+3
        r=self.run_roi(roi_yx=[0,8,0,10],use_hu=True,hu_range=[lo,hi])
        self.assertEqual(r['valid_count'],4)
        self.assertEqual(r['hu_statistics']['min'],lo);self.assertEqual(r['hu_statistics']['max'],hi)

    def test_zero_selection_is_known_with_zero_count_and_null_mean(self):
        r=self.run_roi(roi_yx=[3,3,2,7])
        self.assertEqual(r['valid_count'],0);self.assertIsNone(r['hu_statistics']['mean'])
        self.assertTrue(self.payload(r)['known'])
        self.assertEqual(self.payload(r)['slice_result']['area_mm2'],0.)
        self.assertFalse(self.payload(r,0)['known'])
        self.assertIsNone(self.payload(r,0)['mask'])

    def test_slab_volume_has_exact_physical_units_and_scope(self):
        r=self.run_roi(mode='range',z_bounds=[0,3])
        self.assertEqual(r['valid_count'],45)
        self.assertAlmostEqual(r['region_volume_mm3'],54.)
        self.assertAlmostEqual(r['region_volume_ml'],.054)
        self.assertEqual(r['processed_indices'],[0,1,2])
        self.assertEqual(r['backend'],'CPU_REFERENCE')

    def test_bad_geometry_does_not_invent_physical_units(self):
        self.c.geom['basis_lps_zyx'][0][1]=.7
        self.assertIsNone(physical_spacing(self.c.geom))
        r=self.run_roi(mode='range',z_bounds=[0,3])
        self.assertIsNone(r['region_volume_ml']);self.assertIsNone(r['slice_results'][0]['area_mm2'])

    def test_missing_prediction_is_rejected_without_using_expert_or_other_source(self):
        self.c.expert=np.ones(self.hu.shape,bool)
        v=self.values();v['use_lung_mask']=True
        with self.assertRaisesRegex(ValueError,'分割'):self.c.submit('roi',v)
        self.assertIsNone(self.c.roi_result)

    def test_lung_mask_requires_every_requested_layer_and_records_its_hash(self):
        self.c.body[:,1:3,2:5]=True;self.c.known[:]=True;self.c.receipts={0:'a',1:'b',2:'c'}
        r=self.run_roi(use_lung_mask=True,mode='range',z_bounds=[0,3])
        self.assertEqual(r['valid_count'],18)
        self.assertEqual(r['lung_mask']['segmentation_receipts'],{'0':'a','1':'b','2':'c'})
        expected=hashlib.sha256(self.c.body[0].astype(np.uint8).tobytes()).hexdigest()
        self.assertEqual(r['slice_results'][0]['lung_mask_sha256'],expected)
        self.c.known[2]=False
        v=self.values();v.update(use_lung_mask=True,mode='range',z_bounds=[0,3])
        with self.assertRaises(ValueError):self.c.submit('roi',v)

    def add_enhanced(self):
        # Register a distinct frozen source, without invoking or modifying CNNs.
        self.c.ldct_identity={'model_id':'TEST source identity'}
        self.c.ldct_group='test';self.c.ldct_groups['test']={z:{'hu':(self.hu[z]+100).astype(np.int16),'metadata':{}} for z in range(3)}
        self.c._register_enhanced_source('test','CPU_REFERENCE','bypass')
        return self.c.snapshot()['analysis_sources'][-1]['source_id']

    def test_original_enhanced_sources_restore_their_own_results_and_predictions(self):
        original=self.run_roi();source=self.add_enhanced()
        self.c.select_analysis_source(source)
        self.assertIsNone(self.c.roi_result);self.assertEqual(self.c.roi_records,{})
        with self.assertRaises(ValueError):self.payload(original)
        enhanced=self.run_roi()
        self.assertEqual(enhanced['hu_statistics']['mean'],original['hu_statistics']['mean']+100)
        self.c.select_analysis_source('original')
        self.assertEqual(self.c.roi_result['result_id'],original['result_id'])
        self.assertTrue(self.payload(original)['known'])
        self.assertEqual(int(self.c.known.sum()),0)
        self.c.select_analysis_source(source)
        self.assertEqual(self.c.roi_result['result_id'],enhanced['result_id'])

    def test_repeated_runs_and_new_case_never_overwrite_old_saved_masks(self):
        r1=self.run_roi();folder=self.c.case_folder/'roi'/r1['result_id']
        before={p.name:p.read_bytes() for p in folder.iterdir()}
        r2=self.run_roi(use_hu=True,hu_range=[32767,32767])
        self.assertNotEqual(r1['result_id'],r2['result_id']);self.assertEqual(len(self.c.roi_records),2)
        self.load();self.assertIsNone(self.c.roi_result)
        with self.assertRaises(ValueError):self.payload(r1)
        self.assertEqual(before,{p.name:p.read_bytes() for p in folder.iterdir()})

    def test_stale_source_invalid_bounds_and_unverified_fpga_rejected_before_execution(self):
        for extra in ({'analysis_source_id':'other'},{'roi_yx':[0,17,0,10]},
                      {'roi_yx':[True,2,0,1]},{'backend':'FPGA'}, {'mode':'range','z_bounds':[2,1]}):
            v=self.values();v.update(extra)
            with self.assertRaises(ValueError):self.c.submit('roi',v)
        self.assertIsNone(self.c.roi_job)

    def test_cancel_between_slices_preserves_completed_and_uncomputed_stays_unknown(self):
        entered=threading.Event();release=threading.Event();original=CpuRoiProcessor.process_slice
        def held(owner,*args,**kwargs):
            entered.set();release.wait(3);return original(owner,*args,**kwargs)
        with patch.object(CpuRoiProcessor,'process_slice',held):
            v=self.values();v.update(mode='range',z_bounds=[0,3])
            self.c.submit('roi',v);self.assertTrue(entered.wait(2))
            with self.assertRaises(BlockingIOError):self.c.submit('analysis_source',{})
            with self.assertRaises(BlockingIOError):self.c.submit('basic',{})
            self.c.submit('roi_cancel',dict(case_id=v['case_id'],analysis_source_id=v['analysis_source_id'],result_id=self.c.roi_job['result_id']))
            release.set();state=wait(self.c)
        self.assertEqual(state['phase'],'cancelled')
        r=self.c.roi_result;self.assertEqual(r['processed_slices'],1)
        self.assertFalse(r['complete']);self.assertFalse(self.payload(r,1)['known'])
        self.assertIsNone(r['region_volume_ml'])

    def test_processing_failure_is_not_zero_success_or_cpu_fallback(self):
        self.c.status['board_connected']=True
        with patch.object(CpuRoiProcessor,'process_slice',side_effect=RuntimeError('test failure')):
            self.c.submit('roi',self.values());state=wait(self.c)
        self.assertEqual(state['phase'],'error');self.assertTrue(state['board_connected'])
        self.assertEqual(self.c.roi_result['status'],'FAILED')
        self.assertFalse(self.payload(self.c.roi_result)['known'])

    def test_http_roi_routes_auth_export_and_mask_integrity(self):
        with closing(RunningServer(workspace_handler(self.c,'roi-test-token',b''))) as server:
            opener=request.build_opener(request.ProxyHandler({}))
            def call(path,data=None,token='roi-test-token'):
                return json.load(opener.open(request.Request(server.url+path,data=None if data is None else json.dumps(data).encode(),
                    headers={'X-UI-Token':token,'Content-Type':'application/json'}),timeout=5))
            with self.assertRaises(error.HTTPError) as denied:call('/api/roi/run',self.values(),token='wrong')
            self.assertEqual(denied.exception.code,403)
            call('/api/roi/run',self.values());r=wait(self.c)['roi']['result']
            q=f'?case={r["case_id"]}&source=original&result={r["result_id"]}'
            exported=call('/api/roi/export'+q)
            self.assertEqual(exported['valid_count'],15)
            self.assertEqual(call('/api/roi/image'+q+'&index=1')['slice_result']['valid_count'],15)
            self.assertEqual(call('/api/export')['state']['roi']['result']['result_id'],r['result_id'])
            with opener.open(server.url+'/',timeout=5) as response:html=response.read().decode()
            self.assertIn('id="tab-roi"',html);self.assertIn('/roi.js',html)
            for asset in ('/roi.js','/roi.css'):
                with opener.open(server.url+asset,timeout=5) as response:self.assertEqual(response.status,200)


if __name__=='__main__':unittest.main()
