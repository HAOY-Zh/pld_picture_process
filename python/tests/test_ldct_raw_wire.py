"""Raw transport frame, complete parent checks and no replay after failure."""
import base64
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'data/CT_Studio_ExactSpeed_20261027_04'
PACKAGE=APP/'data/ct_studio_integration_20261025'
if not (PACKAGE/'host_platform.json').is_file():
    APP=ROOT
    PACKAGE=ROOT/'data/ct_studio_current_20261027'
sys.path[:0]=[str(ROOT/'python'),str(APP/'python'),str(ROOT/'pynq'),str(PACKAGE/'ct_studio')]
from ct_ldct_raw_client import RawLDCTStudioClient
from ldct_raw_contract import INTERFACE,encode_raw,decode_raw
import ct_ldct_raw_service as wire
from dual_io_contract import digest


class RawWireTests(unittest.TestCase):
    def bundle(self):
        m=json.loads((PACKAGE/'host_platform.json').read_text(encoding='utf8'));m['ldct_raw_interface']=INTERFACE
        return m

    def test_frames_keep_signed_hu_and_reject_corruption(self):
        raw=np.arange(-32768,32768,dtype=np.int16).tobytes()
        encoded=encode_raw(dict(interface=INTERFACE),raw)
        self.assertEqual(decode_raw(encoded,len(raw))[1],raw)
        for bad in [encoded[:-1],encoded+b'\0',b'\xff'*4+encoded[4:],b'']:
            with self.assertRaises(ValueError):decode_raw(bad,len(raw))

    def test_persistent_wire_original_checks_and_failure_never_replayed(self):
        bundle=self.bundle()
        class HW:
            manifest=json.loads((PACKAGE/'ct_studio/platform_manifest.json').read_text());fault=None
        class MockService(wire.Service):
            calls=0;fail=False;wrong_hash=False
            def execute_ldct(self,raw,rid,sha,params):
                if self.failed:raise RuntimeError('Retained failed owner')
                self.calls+=1
                if self.fail:self.failed='Simulated timeout';raise TimeoutError(self.failed)
                assert digest(raw)==sha
                output=np.bitwise_xor(np.frombuffer(raw,'<i2'),np.int16(257)).astype('<i2').tobytes()
                return dict(schema='ct_studio_ldct_result_v1',backend='FPGA_LDCT',request_id=rid,input_sha256=sha,
                    output_sha256=digest(output),shape=params['shape'],parameters=params,
                    platform_manifest_sha256=bundle['platform_manifest_sha256'],hu_base64=base64.b64encode(output).decode(),
                    **{k:bundle['ldct'][k]for k in ('model_id','weights_sha256','bit_sha256','hwh_sha256')},
                    stats=dict(total_ms=1.,overlay_switch_ms=0.,actual_pl_mhz_before=100.,actual_pl_mhz_after=100.,cpu_mhz=650.),
                    state=dict(busy=False,faulted=False,dma_may_access=False,error=None))
        service=MockService(HW(),'b'*32,bundle['service_bundle_sha256'])
        server=ThreadingHTTPServer(('127.0.0.1',0),wire.handler_for(service));worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        client=RawLDCTStudioClient('http://127.0.0.1:'+str(server.server_port),'b'*32,bundle)
        try:
            sock=None
            for shape in [(1,7),(67,69),(512,512)]:
                a=np.random.default_rng(27).integers(-32768,32768,shape,dtype=np.int16)
                actual,receipt=client.denoise(a,'cnn')
                np.testing.assert_array_equal(actual,np.bitwise_xor(a,np.int16(257)))
                self.assertFalse(receipt['raw_wire_timing']['compression'])
                if sock is not None:self.assertIs(sock,client.raw_connection.sock)
                sock=client.raw_connection.sock
            service.fail=True
            with self.assertRaises(RuntimeError):client.denoise(np.zeros((8,8),np.int16))
            self.assertEqual(service.calls,4)
            with self.assertRaises(RuntimeError):client.denoise(np.zeros((8,8),np.int16))
            self.assertEqual(service.calls,4)
        finally:client.close_raw();server.shutdown();server.server_close();worker.join(3)

    def test_unpaired_refused_before_network(self):
        bundle=self.bundle();bundle.pop('ldct_raw_interface')
        client=RawLDCTStudioClient('http://127.0.0.1:1','b'*32,bundle)
        with self.assertRaisesRegex(ValueError,'not configured'):client.denoise(np.zeros((8,8),np.int16))

    def test_all_seven_startup_stages_retained(self):
        calls=[]
        class SimHW:
            manifest=json.loads((PACKAGE/'ct_studio/platform_manifest.json').read_text());fault=None
            def __init__(self,folder):pass
            def __getattr__(self,name):
                if name.startswith('selftest_'):return lambda:calls.append(name)
                raise AttributeError(name)
        with patch.object(wire.parent.parent,'Hardware',SimHW):service=wire.start('127.0.0.1',0,PACKAGE/'ct_studio')
        try:self.assertEqual(calls,['selftest_basic','selftest_ldct','selftest_roi','selftest_shared','selftest_shared_segmentation','selftest_canvas','selftest_compact'])
        finally:service.server.shutdown();service.server.server_close();service.thread.join(3)


if __name__=='__main__':unittest.main()
