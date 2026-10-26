"""Synthetic protocol and fault-containment tests, never board timing proof."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib import request,error
import uuid
import zlib

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'pynq'))
from ct_platform_fast_service import Service,Hardware,handler_for,decompress_exact,TRANSPORT,MODES,upgrade
from ct_platform_client import PlatformClient
from ct_basic_processing import process
from test_ct_platform import FakePlatformHardware,BASIC,MANIFEST_SHA,HardwareSafetyTests
from test_lung_live import RunningServer,TOKEN,MODEL
from ct_workspace_web import WorkspaceController
from test_ct_workspace import wait

SHARED={op:dict(row,bit_sha256='e'*64,hwh_sha256='f'*64) for op,row in BASIC.items()}
HOST=dict(platform_manifest_sha256=MANIFEST_SHA,basic=SHARED,basic_transport=TRANSPORT)


class FakeSharedHardware(FakePlatformHardware):
    def __init__(self):super().__init__();self.manifest=dict(basic=SHARED)
    def basic(self,hu,p):
        previous=self.mode
        image,stats=super().basic(hu,p)
        self.active_bit='e'*64
        stats.update(shared_basic_overlay=True,mode=MODES[p['operation']],
                     overlay_switch_ms=0. if previous in MODES else 10.)
        return image,stats


class FastProtocolTests(unittest.TestCase):
    def setUp(self):
        self.hw=FakeSharedHardware();self.service=Service(self.hw,TOKEN,MANIFEST_SHA)
        self.server=RunningServer(handler_for(self.service));self.client=PlatformClient(self.server.url,TOKEN,HOST)
        y,x=np.mgrid[:512,:512];self.hu=((x*131+y*73)%4000-2000).astype(np.float32)
    def tearDown(self):self.server.close()

    def test_binary_cache_runs_real_command_every_time_and_changes_modes_without_reload(self):
        self.client.health()
        for i,op in enumerate(MODES):
            actual,r=self.client.basic(self.hu,dict(operation=op))
            np.testing.assert_array_equal(actual,process(self.hu,dict(operation=op)))
            self.assertEqual(r['transport']['input_cache_hit'],i>0)
            if i>0:
                self.assertEqual(r['transport']['request_body_bytes'],0)
                self.assertEqual(r['stats']['overlay_switch_ms'],0.)
        self.assertEqual(self.hw.basic_calls,6)  # No output/result cache.
        self.assertEqual(len(self.service.inputs),1)

    def test_cache_miss_retry_is_only_pre_hardware_then_uploads_once(self):
        self.client.basic(self.hu,{})
        self.service.inputs.clear()
        _,r=self.client.basic(self.hu,{})
        self.assertTrue(r['transport']['cache_miss_safe_upload'])
        self.assertEqual(self.hw.basic_calls,2)

    def test_cache_bounded_by_eight_and_does_not_mix_slices(self):
        for i in range(10):self.client.basic(self.hu+i,{})
        self.assertEqual(len(self.service.inputs),8)
        self.assertEqual(self.hw.basic_calls,10)

    def test_invalid_compression_hash_and_shapes_do_not_touch_hardware(self):
        for data,size in [(zlib.compress(b'x'*10000),20),(zlib.compress(b'x')+b'trailing',1),(b'bad',20)]:
            with self.assertRaises((ValueError,zlib.error)):decompress_exact(data,size)
        with self.assertRaises(ValueError):
            self.service.execute_fast(zlib.compress(b'\0'*524288),uuid.uuid4().hex,'a'*64,{},'zlib-int16')
        self.assertEqual(self.hw.basic_calls,0);self.assertIsNone(self.service.failed)

    def test_busy_and_uncertain_fault_never_retry_hardware(self):
        self.client.basic(self.hu,{})
        self.service.gate.acquire()
        try:
            with self.assertRaises(error.HTTPError) as caught:self.client.basic(self.hu,{})
            self.assertEqual(caught.exception.code,409)
        finally:self.service.gate.release()
        self.assertEqual(self.hw.basic_calls,1)
        self.hw.fail=True
        with self.assertRaises(error.HTTPError):self.client.basic(self.hu,{})
        self.assertTrue(self.service.failed);self.assertEqual(self.hw.basic_calls,2)
        with self.assertRaises(error.HTTPError):self.client.basic(self.hu,{})
        self.assertEqual(self.hw.basic_calls,2)

    def test_tampered_output_or_receipt_rejected(self):
        original=self.service.execute_fast
        for field,value in [('output_sha256','0'*64),('request_id','0'*32),('bit_sha256','0'*64),('parameters',{})]:
            def corrupt(*a,**k):
                body,r=original(*a,**k);r[field]=value;return body,r
            self.service.execute_fast=corrupt
            with self.assertRaises(ValueError):self.client.basic(self.hu,{})
        self.service.execute_fast=original

    def test_explicit_host_upgrade_checks_health_without_hardware_command(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            (folder/'model.json').write_text(json.dumps(MODEL),encoding='utf-8')
            (folder/'token.txt').write_text(TOKEN,encoding='utf-8')
            (folder/'manifest.json').write_text(json.dumps(HOST),encoding='utf-8')
            config=folder/'host_config.json'
            config.write_text(json.dumps(dict(board_url=self.server.url,token_file='token.txt',
                model='model.json',platform_manifest='manifest.json')),encoding='utf-8')
            c=WorkspaceController(self.client,MODEL,folder/'results')
            # Bind this synthetic model's file identity; no private installation
            # or production weights are required for a host-only connection test.
            with patch('ct_workspace_web.FAST_CONFIG',config), \
                    patch('ct_workspace_web.MODEL_SHA',hashlib.sha256((folder/'model.json').read_bytes()).hexdigest()):
                c.submit('connect_fast',{});s=wait(c)
            self.assertEqual(s['phase'],'idle');self.assertTrue(s['board_connected'])
            self.assertTrue(s['fast_platform_active']);self.assertEqual(self.hw.basic_calls,0)
            self.assertEqual(self.hw.calls,0)

    def test_package_and_credentials_are_checked_before_old_service_is_stopped(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);token_file=folder/'platform_token.txt'
            token_file.write_text('wrong-token',encoding='utf-8')
            manifest=dict(basic=SHARED,basic_transport=TRANSPORT)
            with patch('ct_platform_fast_service.base.validate_package',return_value=manifest), \
                    patch('ct_platform_fast_service.stop_idle_platform_service') as stop, \
                    patch('ct_platform_fast_service.start',return_value='new-service') as start:
                with self.assertRaises(ValueError):upgrade(self.service,folder)
                stop.assert_not_called();start.assert_not_called()
                token_file.write_text(TOKEN,encoding='utf-8')
                self.assertEqual(upgrade(self.service,folder),'new-service')
                stop.assert_called_once_with(self.service);start.assert_called_once_with(folder=folder)


class ResidentSwitchTests(unittest.TestCase):
    def make(self,status=2):
        old=HardwareSafetyTests().make(status)
        h=Hardware.__new__(Hardware);h.__dict__.update(old.__dict__)
        h.writes=[]
        h.overlay.axi_gpio_1=SimpleNamespace(channel1=SimpleNamespace(write=lambda *a:h.writes.append(a)))
        return h
    def test_mode_switch_reuses_buffers_and_dma(self):
        h=self.make();dma=h.dma;buffers=list(h.buffers)
        self.assertEqual(h.switch('opening'),0.)
        self.assertEqual(h.switch('threshold'),0.)
        self.assertIs(h.dma,dma);self.assertEqual(h.writes,[(4,7),(1,7)])
        self.assertTrue(all(a is b and not a.freed for a,b in zip(buffers,h.buffers)))
    def test_busy_or_error_dma_retains_buffers_and_faults_owner(self):
        for status in [0,0x12]:
            h=self.make(status)
            with self.assertRaises(RuntimeError):h.switch('opening')
            self.assertTrue(h.fault);self.assertEqual(h.writes,[])
            self.assertTrue(all(not b.freed for b in h.buffers))

    def test_first_unarmed_dma_can_change_mode_but_errors_never_pass(self):
        h=self.make(0);h._basic_transfer_started=False
        self.assertEqual(h.switch('opening'),0.)
        self.assertFalse(h._basic_transfer_started)
        self.assertTrue(all(not b.freed for b in h.buffers))
        h=self.make(0x10);h._basic_transfer_started=False
        with self.assertRaises(RuntimeError):h.switch('opening')
        self.assertTrue(h.fault);self.assertEqual(h.writes,[])


if __name__=='__main__':unittest.main()
