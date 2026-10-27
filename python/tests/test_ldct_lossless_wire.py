"""Wire codec, complete frozen validation and no replay; simulated server only."""
import base64
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
import zlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT
PACKAGE=APP/'data/ct_studio_current_20261027'
sys.path[:0]=[str(APP/'python'),str(ROOT/'python'),str(PACKAGE/'ct_studio'),str(ROOT/'pynq')]
from ct_ldct_lossless_client import LosslessStudioClient,INTERFACE,ROUTE
import ct_ldct_lossless_service as wire
from dual_io_contract import digest


class WireTests(unittest.TestCase):
    def test_signed_hu_complete_receipts_persistence_and_failure_no_replay(self):
        bundle=json.loads((PACKAGE/'host_platform.json').read_text())
        bundle['ldct_lossless_interface']=INTERFACE
        class HW:
            manifest=json.loads((PACKAGE/'ct_studio/platform_manifest.json').read_text())
            fault=None
        class MockService(wire.Service):
            calls=0;fail=False;wrong_hash=False
            def execute_ldct(self,raw,rid,sha,params):
                if self.failed:raise RuntimeError('Simulated retained failure')
                self.calls+=1
                if self.fail:self.failed='test';raise TimeoutError('Simulated timeout')
                assert digest(raw)==sha
                a=np.frombuffer(raw,dtype='<i2');output=np.bitwise_xor(a,np.int16(257)).astype('<i2').tobytes()
                return dict(schema='ct_studio_ldct_result_v1',backend='FPGA_LDCT',request_id=rid,
                    input_sha256=sha,output_sha256=('0'*64 if self.wrong_hash else digest(output)),
                    shape=params['shape'],parameters=params,
                    platform_manifest_sha256=bundle['platform_manifest_sha256'],
                    **{k:bundle['ldct'][k]for k in ('model_id','weights_sha256','bit_sha256','hwh_sha256')},
                    hu_base64=base64.b64encode(output).decode('ascii'),
                    stats=dict(total_ms=1.,overlay_switch_ms=0.,actual_pl_mhz_before=100.,actual_pl_mhz_after=100.,cpu_mhz=650.),
                    state=dict(busy=False,faulted=False,dma_may_access=False,error=None))
        service=MockService(HW(),'b'*32,bundle['service_bundle_sha256'])
        server=ThreadingHTTPServer(('127.0.0.1',0),wire.handler_for(service))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        client=LosslessStudioClient('http://127.0.0.1:'+str(server.server_port),'b'*32,bundle)
        try:
            rng=np.random.default_rng(21)
            for shape in [(1,1),(1,7),(67,69),(512,512)]:
                a=rng.integers(-32768,32768,shape,dtype=np.int16)
                result,receipt=client.denoise(a,'cnn')
                np.testing.assert_array_equal(result,np.bitwise_xor(a,np.int16(257)))
                self.assertEqual(receipt['wire_timing']['raw_input_bytes'],a.nbytes)
            service.fail=True
            with self.assertRaises(RuntimeError):client.denoise(np.zeros((8,8),np.int16))
            self.assertEqual(service.calls,5)
            with self.assertRaises(RuntimeError):client.denoise(np.zeros((8,8),np.int16))
            self.assertEqual(service.calls,5)
        finally:
            client.ldct_io.close();client.io.close();server.shutdown();server.server_close();thread.join(3)

    def test_unpaired_wire_refused_before_any_network_or_cpu_fallback(self):
        bundle=json.loads((PACKAGE/'host_platform.json').read_text())
        bundle.pop('ldct_lossless_interface',None)
        client=LosslessStudioClient('http://127.0.0.1:1','b'*32,bundle)
        with self.assertRaisesRegex(ValueError,'not configured'):
            client.denoise(np.zeros((8,8),np.int16))

    def test_start_keeps_all_original_selftest_stages(self):
        calls=[]
        class SimulatedHardware:
            manifest=json.loads((PACKAGE/'ct_studio/platform_manifest.json').read_text())
            fault=None
            def __init__(self,folder):pass
            def __getattr__(self,name):
                if name.startswith('selftest_'):return lambda:calls.append(name)
                raise AttributeError(name)
        with patch.object(wire.parent,'Hardware',SimulatedHardware):
            service=wire.start(bind='127.0.0.1',port=0,folder=PACKAGE/'ct_studio')
        try:
            self.assertEqual(calls,['selftest_basic','selftest_ldct','selftest_roi',
                'selftest_shared','selftest_shared_segmentation','selftest_canvas','selftest_compact'])
        finally:
            service.server.shutdown();service.server.server_close();service.thread.join(3)


if __name__=='__main__':unittest.main()
