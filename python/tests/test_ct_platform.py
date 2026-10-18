"""Explicit hardware doubles for protocol, switching and fault containment tests."""
import base64
import hashlib
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'pynq'))
from ct_basic_processing import process, parameters
from ct_platform_client import PlatformClient
from ct_platform_service import (Hardware, Service, BASIC_OPERATIONS, PIXELS,
    handler_for, sha, startup_cases)
from ct_workspace_web import WorkspaceController
from test_lung_live import (FakeHardware, MODEL, TOKEN, BIT_SHA, MODEL_SHA,
    RunningServer)
from test_ct_workspace import wait

MANIFEST_SHA='b'*64
BASIC={op:dict(bit_sha256=hashlib.sha256(op.encode()).hexdigest(),hwh_sha256='c'*64,
    bit='basic/'+op+'.bit',hwh='basic/'+op+'.hwh',shape=[512,512]) for op in BASIC_OPERATIONS}
HOST=dict(platform_manifest_sha256=MANIFEST_SHA,basic=BASIC)


class FakePlatformHardware(FakeHardware):
    """TEST ONLY. Not selectable by a production entrypoint."""
    def __init__(self):
        super().__init__();self.mode='segmentation';self.active_bit=BIT_SHA;self.fault=None
        self.segmentation_identity=self.identity;self.manifest=dict(basic=BASIC)
        self.basic_calls=0
    def basic(self,hu,params):
        self.basic_calls+=1
        if self.fail:raise TimeoutError('TEST uncertain DMA')
        op=params['operation'];self.mode=op;self.active_bit=BASIC[op]['bit_sha256']
        return process(hu,params),dict(dma_ms=.2,preparation_ms=.1,overlay_switch_ms=10.,
            fclk0_mhz=100.,cpu_mhz=650.,guard_verified=True,input_bytes=PIXELS*4,output_bytes=PIXELS*4)
    def infer(self,codes):
        self.mode='segmentation';self.active_bit=BIT_SHA
        return super().infer(codes)


class PlatformProtocolTests(unittest.TestCase):
    def setUp(self):
        self.hardware=FakePlatformHardware();self.service=Service(self.hardware,TOKEN,MANIFEST_SHA)
        self.server=RunningServer(handler_for(self.service))
        self.client=PlatformClient(self.server.url,TOKEN,HOST)
        self.hu=startup_cases()[1][1].astype(np.float32)
    def tearDown(self):self.server.close()

    def test_actual_transport_six_basic_ops_then_segmentation(self):
        self.client.health()
        for op in BASIC_OPERATIONS:
            image,receipt=self.client.basic(self.hu,dict(operation=op))
            np.testing.assert_array_equal(image,process(self.hu,dict(operation=op)))
            self.assertEqual(receipt['backend'],'FPGA_BASIC')
            self.assertEqual(receipt['bit_sha256'],BASIC[op]['bit_sha256'])
        self.assertEqual(self.hardware.basic_calls,6)
        # Health describes both the actual active bit and the bound segmentation identity.
        identity=self.client.health();self.assertEqual(identity['platform']['active_operation'],'closing')
        head,receipt=self.client.infer(np.zeros((1,256,256),np.uint8))
        self.assertEqual(receipt['bit_sha256'],BIT_SHA);self.assertEqual(self.hardware.calls,1)
        self.assertEqual(self.hardware.mode,'segmentation')

    def test_busy_and_fault_refuse_any_second_hardware_call(self):
        raw=self.hu.astype('<i2').tobytes()
        self.service.gate.acquire()
        try:
            with self.assertRaises(BlockingIOError):self.service.execute('/v1/basic',raw,'a'*32,sha(raw),{})
        finally:self.service.gate.release()
        self.assertEqual(self.hardware.basic_calls,0)
        self.hardware.fail=True
        with self.assertRaises(TimeoutError):self.service.execute('/v1/basic',raw,'a'*32,sha(raw),{})
        self.hardware.fail=False
        with self.assertRaises(RuntimeError):self.service.execute('/v1/basic',raw,'b'*32,sha(raw),{})
        self.assertEqual(self.hardware.basic_calls,1)

    def test_invalid_inputs_do_not_touch_pl_or_fault_service(self):
        raw=self.hu.astype('<i2').tobytes()
        for supplied in [dict(operation='histogram'),dict(lower=1,upper=-1)]:
            with self.assertRaises(ValueError):self.service.execute('/v1/basic',raw,'a'*32,sha(raw),supplied)
        with self.assertRaises(ValueError):self.service.execute('/v1/basic',raw,'a'*32,'d'*64,{})
        with self.assertRaises(ValueError):self.client.basic(self.hu[:16,:16],{})
        self.assertEqual(self.hardware.basic_calls,0);self.assertIsNone(self.service.failed)

    def test_changed_result_parameters_bit_or_clock_evidence_rejected(self):
        original=self.client.call
        for key,replacement in [('request_id','d'*32),('bit_sha256','0'*64),
                ('hwh_sha256','0'*64),('output_sha256','0'*64),('parameters',{})]:
            def tamper(*a,**k):
                value=original(*a,**k);value[key]=replacement;return value
            self.client.call=tamper
            with self.assertRaises(ValueError):self.client.basic(self.hu,{})
        self.client.call=original

    def test_workspace_saves_real_pl_receipt_without_neural_mask_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);source=folder/'input.npz'
            np.savez(source,hu=self.hu[None],spacing_zyx_mm=[3.,1.,1.],origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
            controller=WorkspaceController(self.client,MODEL,folder/'results')
            controller.submit('load',dict(path=str(source),confirm_hu=True,load_expert=False));wait(controller)
            controller.submit('connect',{});s=wait(controller)
            with self.assertRaises(ValueError):
                controller.submit('basic',dict(case_id=s['case_id'],index=0,operation='opening',backend='FPGA_BASIC'))
            self.assertEqual(self.hardware.basic_calls,0)
            controller.submit('basic',dict(case_id=s['case_id'],index=0,operation='threshold',backend='FPGA_BASIC'))
            s=wait(controller)
            original=self.hardware.basic
            def delayed(*args,**kwargs):
                time.sleep(.08)
                return original(*args,**kwargs)
            self.hardware.basic=delayed
            controller.submit('basic',dict(case_id=s['case_id'],index=0,operation='opening',backend='FPGA_BASIC',
                mask_result_id=s['threshold_input']['result_id'],lower=0,upper=1))
            s=wait(controller);self.assertEqual(s['phase'],'done')
            self.assertEqual(s['basic']['backend'],'FPGA_BASIC');self.assertTrue(s['basic']['reference_bit_exact'])
            self.assertEqual(s['known_slices'],0);self.assertEqual(self.hardware.calls,0)
            self.assertEqual(self.hardware.basic_calls,2)
            self.assertGreater(s['basic']['host_result_ready_ms'],80.)
            self.assertEqual(s['basic']['processing_ms'],.2)
            self.assertEqual(s['basic']['parameters']['lower'],-1000)
            self.assertEqual(s['basic']['input_mask']['result_id'],s['threshold_input']['result_id'])

    def test_mismatched_actual_mask_is_saved_and_not_replaced(self):
        old=self.hardware.basic
        def corrupt(*a,**k):
            image,stats=old(*a,**k);image[0,0]^=255;return image,stats
        self.hardware.basic=corrupt
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);source=folder/'input.npz'
            np.savez(source,hu=self.hu[None],spacing_zyx_mm=[3.,1.,1.],origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
            c=WorkspaceController(self.client,MODEL,folder/'results')
            c.submit('load',dict(path=str(source),confirm_hu=True,load_expert=False));wait(c)
            c.submit('connect',{});s=wait(c)
            c.submit('basic',dict(case_id=s['case_id'],index=0,backend='FPGA_BASIC'))
            s=wait(c);self.assertEqual(s['phase'],'error');self.assertIsNone(s['basic'])
            self.assertFalse(s['board_connected']);self.assertEqual(len(list(c.case_folder.glob('*_basic_mismatch.npz'))),1)


class Buffer(np.ndarray):
    def __new__(cls,size):
        value=np.zeros(size,np.uint32).view(cls);value.freed=False;return value
    def flush(self):pass
    def invalidate(self):pass
    def freebuffer(self):self.freed=True


class HardwareSafetyTests(unittest.TestCase):
    def make(self,status):
        h=Hardware.__new__(Hardware);h.fault=None;h.mode='window';h.timeout=.001
        h.cpu_mhz=650.;h.network=None;h.buffers=[Buffer(PIXELS+16),Buffer(PIXELS+16)]
        h.dma=SimpleNamespace(mmio=SimpleNamespace(read=lambda offset:status),
            recvchannel=SimpleNamespace(transfer=lambda *a,**k:None,wait=lambda:None),
            sendchannel=SimpleNamespace(transfer=lambda *a,**k:None,wait=lambda:None))
        h.overlay=SimpleNamespace(axi_gpio_0=SimpleNamespace(channel1=SimpleNamespace(write=lambda *a:None),
            channel2=SimpleNamespace(write=lambda *a:None)))
        h.clocks=SimpleNamespace(cpu_mhz=650.,fclk0_mhz=100.)
        return h

    def test_timeout_keeps_dma_cma_and_prevents_reload(self):
        h=self.make(0);buffers=list(h.buffers);dma=h.dma
        with self.assertRaises(TimeoutError):h.basic(np.zeros((512,512),np.int16),parameters({}))
        self.assertIs(h.dma,dma);self.assertTrue(h.fault)
        self.assertTrue(all(not b.freed for b in buffers))
        with self.assertRaises(RuntimeError):h.switch('threshold')
        with self.assertRaises(RuntimeError):h.release_idle()
        self.assertTrue(all(not b.freed for b in buffers))

    def test_busy_dma_cannot_be_freed_for_switch(self):
        h=self.make(0)
        with self.assertRaises(RuntimeError):h.release_idle()
        self.assertTrue(all(not b.freed for b in h.buffers))

    def test_error_dma_is_not_treated_as_idle(self):
        h=self.make(0x12)
        with self.assertRaises(RuntimeError):h.release_idle()
        self.assertTrue(all(not b.freed for b in h.buffers))


if __name__=='__main__':unittest.main()
