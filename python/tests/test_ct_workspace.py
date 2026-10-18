"""Data isolation and HTTP integration; synthetic inputs, no hardware claims."""
import base64
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from urllib import error, request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct_basic_processing import parameters, process
from ct_workspace_web import WorkspaceController, workspace_handler
from lung_unet import torch_module
from test_lung_live import (BoardClient, FakeHardware, InferenceService, MODEL,
    RunningServer, TOKEN, board_handler)


class BasicReferenceTests(unittest.TestCase):
    def test_threshold_includes_both_endpoints(self):
        a=np.array([[-1001,-1000,-400,-399]],np.float32)
        b=process(a,dict(operation='threshold',lower=-1000,upper=-400))
        np.testing.assert_array_equal(b,[[0,255,255,0]])

    def test_centered_morphology_zero_pads_each_stage(self):
        a=np.zeros((5,5),np.float32)
        e=process(a,dict(operation='erosion',lower=0,upper=0))
        self.assertEqual(int(e.sum()),9*255)
        c=process(a,dict(operation='closing',lower=0,upper=0))
        np.testing.assert_array_equal(e,c)

    def test_window_matches_existing_integer_reference_including_odd_width(self):
        a=np.array([[-1000,-2,-1,0,1,2,1000]],np.float32)
        b=process(a,dict(operation='window',wl=0,ww=5))
        np.testing.assert_array_equal(b,[[0,0,51,102,153,255,255]])

    def test_processing_does_not_mutate_hu_and_constant_equalization_is_defined(self):
        a=np.full((16,16),40,np.float32);old=a.copy()
        for op in ['window','histogram','pseudocolor','opening']:
            b=process(a,dict(operation=op))
            self.assertEqual(b.dtype,np.uint8);np.testing.assert_array_equal(a,old)
        np.testing.assert_array_equal(process(a,dict(operation='histogram')),
                                      process(a,dict(operation='window')))

    def test_invalid_parameters_and_out_of_int16_hu_rejected(self):
        for values in [dict(ww=0),dict(wl=True),dict(lower=2,upper=1),dict(operation='ldct')]:
            with self.assertRaises(ValueError):parameters(values)
        with self.assertRaises(ValueError):process(np.full((16,16),40000),{})


def wait(controller):
    until=time.monotonic()+20
    while controller.snapshot()['phase'] in ('loading','connecting','computing','reconstructing','processing'):
        if time.monotonic()>until:raise AssertionError('Workspace worker timeout')
        time.sleep(.01)
    return controller.snapshot()


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        # Match both production entrypoints; avoid oversubscribing the board
        # build host during the preserved Torch pre/postprocessing checks.
        torch_module().set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        hu=np.full((3,32,32),-800,np.float32)
        self.path=self.root/'ct.npz'
        np.savez(self.path,hu=hu,spacing_zyx_mm=[3.,1.,1.],origin_lps_mm=[0,0,0],
                 orientation=[1,0,0,0,1,0])
        self.hw=FakeHardware();self.board=RunningServer(board_handler(InferenceService(self.hw,TOKEN)))
        self.controller=WorkspaceController(BoardClient(self.board.url,TOKEN),MODEL,self.root/'results')
        self.controller.submit('load',dict(path=str(self.path),confirm_hu=True,load_expert=False))
        self.assertEqual(wait(self.controller)['phase'],'idle')

    def tearDown(self):
        wait(self.controller);self.board.close();self.temp.cleanup()

    def basic(self):
        s=self.controller.snapshot()
        self.controller.submit('basic',dict(case_id=s['case_id'],index=1,operation='threshold'))
        s=wait(self.controller);self.assertEqual(s['phase'],'done');return s

    def test_basic_has_no_fpga_calls_no_neural_mask_and_preserves_input(self):
        before=self.controller.hu.copy();s=self.basic()
        self.assertEqual(self.hw.calls,0);self.assertEqual(s['known_slices'],0)
        self.assertIsNone(s['latest']);self.assertEqual(s['basic']['backend'],'PC_REFERENCE')
        np.testing.assert_array_equal(before,self.controller.hu)
        p=self.controller.basic_payload(s['case_id'],s['basic']['result_id'])
        self.assertEqual(len(base64.b64decode(p['image']['pixels'])),32*32)
        self.assertEqual(len(list(self.controller.case_folder.glob('*_basic.json'))),1)

    def test_new_case_invalidates_old_basic_result_and_does_not_leak_history(self):
        old=self.basic();self.controller.submit('load',dict(path=str(self.path),confirm_hu=True,load_expert=False))
        s=wait(self.controller);self.assertIsNone(s['basic']);self.assertEqual(s['workspace_history'],[])
        self.assertIsNone(s['threshold_input'])
        with self.assertRaises(ValueError):self.controller.basic_payload(old['case_id'],old['basic']['result_id'])
        with self.assertRaises(ValueError):self.controller.submit('basic',dict(case_id=old['case_id'],index=1))

    def test_segmentation_still_uses_real_transport_and_unchanged_input(self):
        self.basic();self.controller.submit('connect',{});wait(self.controller)
        self.controller.submit('run',dict(mode='slice',index=1));s=wait(self.controller)
        self.assertEqual(s['phase'],'done');self.assertEqual(self.hw.calls,1)
        self.assertEqual(s['known_slices'],1);self.assertIsNotNone(s['basic'])

    def test_processing_blocks_other_jobs_without_resetting_hardware(self):
        self.controller._update(phase='processing')
        with self.assertRaises(BlockingIOError):self.controller.submit('connect',{})
        with self.assertRaises(BlockingIOError):self.controller.submit('load',dict(path=str(self.path)))
        self.controller._update(phase='idle');self.assertEqual(self.hw.calls,0)

    def test_morphology_requires_explicit_same_slice_mask_and_freezes_threshold(self):
        c=self.controller;s=c.snapshot()
        for values in [dict(index=1),dict(index=0,mask_result_id='missing')]:
            with self.assertRaises(ValueError):
                c.submit('basic',dict(case_id=s['case_id'],operation='opening',**values))
        s=self.basic();source=s['threshold_input']
        for values in [dict(index=0,mask_result_id=source['result_id']),dict(index=1,mask_result_id='old')]:
            with self.assertRaises(ValueError):
                c.submit('basic',dict(case_id=s['case_id'],operation='opening',**values))
        c.submit('basic',dict(case_id=s['case_id'],index=1,operation='opening',
            mask_result_id=source['result_id'],lower=100,upper=200))
        s=wait(c);self.assertEqual(s['phase'],'done');b=s['basic']
        self.assertEqual(b['parameters']['lower'],-1000)
        self.assertEqual(b['input_mask']['output_sha256'],source['output_sha256'])
        self.assertEqual(s['threshold_input']['result_id'],source['result_id'])
        self.assertEqual(c.basic_payload(s['case_id'],source['result_id'])['operation'],'threshold')
        self.assertGreaterEqual(b['host_result_ready_ms'],b['processing_ms'])
        self.assertEqual(self.hw.calls,0)

    def test_tampered_selected_mask_stops_processing(self):
        s=self.basic();self.controller.threshold_image[0,0]^=255
        self.controller.submit('basic',dict(case_id=s['case_id'],index=1,operation='closing',
            mask_result_id=s['threshold_input']['result_id']))
        s=wait(self.controller);self.assertEqual(s['phase'],'error')
        self.assertEqual(s['basic']['operation'],'threshold');self.assertEqual(self.hw.calls,0)

    def test_completion_wait_returns_exact_request_without_new_hardware_command(self):
        c=self.controller;case_id=c.snapshot()['case_id'];rid='a'*32
        c.submit('basic',dict(case_id=case_id,index=1,operation='threshold',ui_request_id=rid))
        s=c.wait_basic(case_id,rid,timeout=5.)
        self.assertEqual(s['phase'],'done');self.assertEqual(s['basic']['ui_request_id'],rid)
        self.assertEqual(self.hw.calls,0)
        with self.assertRaises(ValueError):c.wait_basic(case_id,'b'*32,timeout=.01)

    def test_http_wait_returns_ready_result_and_missing_request_is_rejected_first(self):
        web=RunningServer(workspace_handler(self.controller,TOKEN,b'// test only'))
        try:
            values=dict(case_id=self.controller.snapshot()['case_id'],index=1,
                        operation='threshold',wait_for_result=True)
            def send(v):
                return request.urlopen(request.Request(web.url+'/api/basic/run',data=json.dumps(v).encode(),
                    headers={'X-UI-Token':TOKEN,'Content-Type':'application/json'}))
            with self.assertRaises(error.HTTPError) as caught:send(values)
            self.assertEqual(caught.exception.code,400);self.assertIsNone(self.controller.basic_result)
            with send(dict(values,ui_request_id='c'*32)) as response:
                self.assertEqual(response.status,200);s=json.load(response)
            self.assertEqual(s['phase'],'done');self.assertEqual(s['basic']['ui_request_id'],'c'*32)
            self.assertEqual(self.hw.calls,0)
        finally:web.close()

    def test_http_routes_and_auth_inherited_for_new_endpoints(self):
        web=RunningServer(workspace_handler(self.controller,TOKEN,b'// test only'))
        try:
            for endpoint in ['/','/workspace.css','/workspace.js','/segmentation']:
                self.assertEqual(request.urlopen(web.url+endpoint).status,200)
            for endpoint in ['/api/export','/api/benchmarks','/api/basic/image']:
                with self.assertRaises(error.HTTPError) as caught:request.urlopen(web.url+endpoint)
                self.assertEqual(caught.exception.code,403)
            body=json.dumps(dict(case_id=self.controller.snapshot()['case_id'],index=1,
                                 operation='threshold')).encode()
            req=request.Request(web.url+'/api/basic/run',data=body,headers={
                'X-UI-Token':TOKEN,'Content-Type':'application/json'})
            self.assertEqual(request.urlopen(req).status,202);wait(self.controller)
            req=request.Request(web.url+'/api/export',headers={'X-UI-Token':TOKEN})
            value=json.load(request.urlopen(req));self.assertEqual(value['state']['basic']['backend'],'PC_REFERENCE')
        finally:web.close()


if __name__=='__main__':unittest.main()
