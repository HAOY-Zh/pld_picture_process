"""Synthetic integration: reuse predictions, never impersonate a board benchmark."""
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ct_workspace_web import WorkspaceController, workspace_handler
from ct_nodule_inference import NoduleService, array_sha
from test_lung_live import RunningServer, TOKEN


class RecordingService:
    def __init__(self, block=False):
        self.calls = 0; self.inputs = None
        self.release = threading.Event()
        if not block: self.release.set()

    def availability(self):
        return dict(available=True,reason=None,backend='PC_CNN_FP32',threshold=.64)

    def predict(self, hu, mask, geometry, progress, cancel):
        self.calls += 1; self.inputs = (hu,mask,geometry)
        progress('TEST ONLY synthetic task',0,1)
        while not self.release.wait(.01):
            if cancel.is_set(): raise InterruptedError()
        if cancel.is_set(): raise InterruptedError()
        progress('TEST ONLY synthetic task',1,1)
        return dict(schema='TEST_ONLY_NOT_REAL_INFERENCE',backend='PC_CNN_FP32',device='cpu',
            threshold=.64,candidate_count=1,model_sha256='TEST_ONLY',
            input_hu_sha256=array_sha(hu),lung_mask_sha256=array_sha(mask),
            segmentation_reused=True,lung_inference_rerun=False,annotations_used=False,
            candidates=[dict(candidate_id='TEST_ONLY',score=.8,center_index_zyx=[1.,15.,15.],
                center_lps_xyz_mm=[15.,15.,2.],bbox_zyx=[0,3,12,19,12,19],estimated_scale_diameter_mm=5.)],
            timing=dict(inference_ms=1.,scope='TEST_ONLY'))


class NoduleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.service=RecordingService()
        self.controller=WorkspaceController(None,{},self.root/'results',nodule_service=self.service)
        c=self.controller;c.hu=np.full((4,32,32),-800,np.float32)
        c.body=np.zeros(c.hu.shape,bool);c.body[:,5:25,5:25]=True
        c.raw=c.body.copy();c.known=np.ones(4,bool);c.expert=np.ones(c.hu.shape,bool)
        c.geom=dict(spacing_zyx_mm=[2.,1.,1.],origin_lps_mm=[0.,0.,0.],basis_lps_zyx=[[0,0,1],[0,1,0],[2,0,0]])
        c.details=dict(source_slice_indices=[0,1,2,3],partial_input=False,hu_sha256=array_sha(c.hu))
        c.receipts={z:'TEST_ONLY_RECEIPT_'+str(z) for z in range(4)}
        c.case_folder=self.root/'case';c.case_folder.mkdir()
        c._update(case_id='a'*32,phase='done')

    def tearDown(self):
        self.controller.cancel.set();self.service.release.set()
        if hasattr(self.controller,'nodule_thread'):self.controller.nodule_thread.join(5)
        self.tmp.cleanup()

    def submit(self):
        return self.controller.submit('nodule',dict(case_id='a'*32,backend='PC_CNN_FP32'))

    def finish(self):
        self.controller.nodule_thread.join(5)
        self.assertFalse(self.controller.nodule_thread.is_alive())
        return self.controller.snapshot()

    def test_reuses_exact_current_ct_and_body_without_expert_or_hardware(self):
        c=self.controller;hu=c.hu.copy();mask=c.body.copy();old_receipts=dict(c.receipts)
        self.submit();s=self.finish();r=s['nodule']['result']
        self.assertEqual(s['phase'],'done');self.assertEqual(self.service.calls,1)
        self.assertIs(self.service.inputs[0],c.hu);self.assertIs(self.service.inputs[1],c.body)
        self.assertFalse(np.array_equal(self.service.inputs[1],c.expert))
        np.testing.assert_array_equal(c.hu,hu);np.testing.assert_array_equal(c.body,mask)
        self.assertEqual(c.receipts,old_receipts);self.assertIsNone(c.client)
        self.assertTrue(r['segmentation_reused']);self.assertFalse(r['lung_inference_rerun'])
        self.assertFalse(r['annotations_used']);self.assertNotIn('candidates',r)
        full=c.nodule_payload(s['case_id'],r['result_id']);self.assertEqual(len(full['candidates']),1)
        self.assertEqual(full['source']['slice_receipts'],old_receipts)
        self.assertEqual(len(list(c.case_folder.glob('*_nodule.json'))),1)
        self.submit();self.assertEqual(self.service.calls,1)  # Display cached current result, no rerun.

    def test_unknown_slices_partial_inputs_and_missing_receipts_are_not_background(self):
        c=self.controller;c.known[2]=False
        with self.assertRaisesRegex(ValueError,'完整序列'):self.submit()
        c.known[:]=True;c.details['partial_input']=True
        with self.assertRaisesRegex(ValueError,'完整CT'):self.submit()
        c.details['partial_input']=False;c.receipts.pop(2)
        with self.assertRaisesRegex(ValueError,'来源记录'):self.submit()
        self.assertEqual(self.service.calls,0)

    def test_explicit_pc_backend_stale_case_and_parameter_override_rejected(self):
        for v in (dict(case_id='old',backend='PC_CNN_FP32'),dict(case_id='a'*32),
                  dict(case_id='a'*32,backend='FPGA'),dict(case_id='a'*32,backend='PC_CNN_FP32',threshold=.01)):
            with self.subTest(v=v),self.assertRaises(ValueError):self.controller.submit('nodule',v)
        self.assertEqual(self.service.calls,0)

    def test_active_detection_blocks_load_and_basic_until_safe_cancel(self):
        c=self.controller;self.service.release.clear();self.submit()
        self.assertEqual(c.snapshot()['phase'],'detecting')
        for action in ('load','basic','connect','nodule'):
            with self.subTest(action=action),self.assertRaises(BlockingIOError):
                c.submit(action,dict(case_id='a'*32,backend='PC_CNN_FP32'))
        c.cancel.set();s=self.finish()
        self.assertEqual(s['phase'],'cancelled');self.assertIsNone(s['nodule']['result'])
        self.assertTrue(c.known.all());self.assertEqual(len(c.receipts),4)

    def test_loading_another_case_invalidates_previous_results(self):
        c=self.controller;self.submit();s=self.finish();result_id=s['nodule']['result']['result_id']
        path=self.root/'new.npz'
        np.savez(path,hu=np.zeros((3,16,16),np.int16),spacing_zyx_mm=[2.,1.,1.],
                 origin_lps_mm=[0.,0.,0.],orientation=[1,0,0,0,1,0])
        c.submit('load',dict(path=str(path),confirm_hu=True,load_expert=False))
        deadline=time.monotonic()+5
        while c.snapshot()['phase']=='loading' and time.monotonic()<deadline:time.sleep(.01)
        self.assertIsNone(c.snapshot()['nodule']['result'])
        with self.assertRaises(ValueError):c.nodule_payload('a'*32,result_id)

    def test_new_segmentation_invalidates_detection_even_in_same_case(self):
        c=self.controller;self.submit();s=self.finish();rid=s['nodule']['result']['result_id']
        c.client=object();c._update(board_connected=True)
        with patch('lung_live_web.Controller._run',side_effect=lambda values:c._update(phase='done')):
            c.submit('run',dict(mode='volume',index=0))
            deadline=time.monotonic()+5
            while c.snapshot()['phase']=='computing' and time.monotonic()<deadline:time.sleep(.01)
        self.assertIsNone(c.snapshot()['nodule']['result'])
        with self.assertRaises(ValueError):c.nodule_payload(s['case_id'],rid)

    def test_http_auth_result_identity_and_cancel_require_current_task(self):
        c=self.controller;web=RunningServer(workspace_handler(c,TOKEN,b'// TEST ONLY'))
        try:
            for endpoint in ('/api/nodule/result','/api/nodule/run','/api/nodule/cancel'):
                q=request.Request(web.url+endpoint,data=b'{}' if endpoint.endswith(('run','cancel')) else None,
                                  headers={'Content-Type':'application/json'})
                with self.assertRaises(error.HTTPError) as e:request.urlopen(q)
                self.assertEqual(e.exception.code,403)
            q=request.Request(web.url+'/api/nodule/run',data=json.dumps(dict(case_id='a'*32,backend='PC_CNN_FP32')).encode(),
                              headers={'X-UI-Token':TOKEN,'Content-Type':'application/json'})
            with request.urlopen(q) as response:self.assertIn(response.status,(200,202))
            s=self.finish();rid=s['nodule']['result']['result_id']
            for case_id,status in (('a'*32,200),('old',400)):
                q=request.Request(web.url+f'/api/nodule/result?case={case_id}&result={rid}',headers={'X-UI-Token':TOKEN})
                if status==200:
                    with request.urlopen(q) as response:self.assertEqual(json.load(response)['candidate_count'],1)
                else:
                    with self.assertRaises(error.HTTPError) as e:request.urlopen(q)
                    self.assertEqual(e.exception.code,status)
        finally:web.close()

    def test_missing_model_is_explicit_and_does_not_import_torch(self):
        service=NoduleService(self.root/'missing.pt')
        self.assertFalse(service.availability()['available'])
        self.assertIn('缺少结节模型',service.availability()['reason'])


if __name__=='__main__':unittest.main()
