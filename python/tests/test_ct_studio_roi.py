"""Synthetic protocol/workflow tests only; not physical FPGA evidence."""
import copy
from contextlib import closing
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq'),str(ROOT/'pynq/roi_vendor')]
from roi_vendor import ct_roi_reference as reference
sys.modules.setdefault('ct_roi_reference',reference)
from roi_contract import encode,decode,TRANSPORT
from ct_studio_roi_client import ROIStudioClient
import ct_studio_roi_service as board
from test_ct_studio_ldct import FakeStudioHardware
from test_ct_roi_workspace import wait
from test_lung_live import RunningServer
from ct_workspace_web import WorkspaceController

ROW=dict(module_version='roi-filter-1.0.0',transport=TRANSPORT,clock_mhz=100.,
         bit_sha256='1'*64,hwh_sha256='2'*64)


class SimulatedRoiHardware(FakeStudioHardware):
    def __init__(self):
        super().__init__();self.manifest['roi']=ROW;self.roi_calls=0
    def roi(self,hu,params,lung):
        self.roi_calls+=1
        if self.fail:raise TimeoutError('TEST uncertain ROI DMA')
        result=reference.CpuRoiProcessor().process_slice(hu,params['roi_yx'],params['hu_range'],params['use_hu'],lung)
        result['backend']='FPGA';result['hardware_status']=0x01000002
        result['timing'].update(dma_compute_seconds=.001,overlay_switch_ms=0.,actual_pl_mhz_before=100.,actual_pl_mhz_after=100.,
                               cpu_mhz=650.,guard_verified=True)
        return result


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.hw=SimulatedRoiHardware();self.service=board.Service(self.hw,'test-roi-token-'*4,'f'*64)
        self.service.startup_verified=self.service.roi_verified=True
        self.server=RunningServer(board.handler_for(self.service))
        self.client=ROIStudioClient(self.server.url,'test-roi-token-'*4,dict(self.hw.manifest,platform_manifest_sha256='f'*64))
        self.hu=np.arange(16*20,dtype=np.int16).reshape(16,20)-120
    def tearDown(self):self.server.close()
    def test_binary_mask_signed_hu_count_and_detached_output(self):
        mask=(np.indices(self.hu.shape).sum(0)%2).astype(np.uint8)
        for roi,limits,use_hu,lung in [(None,None,False,None),((0,16,0,20),(-80,40),True,mask),
            ((16,16,0,20),None,False,None),((0,1,0,1),(-120,-120),True,None)]:
            result=self.client.roi(self.hu,roi,limits,use_hu,lung)
            expected=reference.CpuRoiProcessor().process_slice(self.hu,roi,limits,use_hu,lung)
            np.testing.assert_array_equal(result['filtered_mask'],expected['filtered_mask'])
            self.assertEqual(result['valid_count'],expected['valid_count']);self.assertEqual(result['backend'],'FPGA_ROI')
            self.assertEqual(result['hardware_receipt']['hu_sha256'],__import__('hashlib').sha256(self.hu.tobytes()).hexdigest())
    def test_bad_geometry_modes_and_payload_rejected_before_hardware(self):
        raw,params,_,_=encode(self.hu,None,None,False,None)
        bad=copy.deepcopy(params);bad['shape'][0]=True
        with self.assertRaises(ValueError):decode(raw,bad)
        for invalid in (raw[:-1],raw+b'0'):
            with self.assertRaises(ValueError):decode(invalid,params)
        with self.assertRaises(ValueError):self.client.roi(self.hu,[0,17,0,1],None,False)
        self.assertEqual(self.hw.roi_calls,0)
    def test_busy_has_no_retry_or_fallback(self):
        self.service.gate.acquire()
        try:
            with self.assertRaises(Exception):self.client.roi(self.hu,None,None,False)
        finally:self.service.gate.release()
        self.assertEqual(self.hw.roi_calls,0)
    def test_fault_is_sticky_and_does_not_fall_back(self):
        self.hw.fail=True
        with self.assertRaises(Exception):self.client.roi(self.hu,None,None,False)
        self.assertTrue(self.service.failed)
        with self.assertRaises(Exception):self.client.roi(self.hu,None,None,False)
        self.assertEqual(self.hw.roi_calls,1)
    def test_wrong_returned_clock_or_count_rejected(self):
        original=self.client.call
        def altered(*args,**kwargs):
            value=original(*args,**kwargs);value['valid_count']+=1;return value
        self.client.call=altered
        with self.assertRaisesRegex(ValueError,'mask/count'):self.client.roi(self.hu,None,None,False)
        def wrong_clock(*args,**kwargs):
            value=original(*args,**kwargs);value['stats']['actual_pl_mhz_after']=50.;return value
        self.client.call=wrong_clock
        with self.assertRaisesRegex(ValueError,'clock'):self.client.roi(self.hu,None,None,False)


class RoiDMAStateTests(unittest.TestCase):
    def test_first_transfer_exception_is_only_for_never_armed_roi_dma(self):
        hw=board.Hardware.__new__(board.Hardware);hw.mode='roi';hw.dma=Mock();hw.dma.mmio.read.return_value=0
        hw._basic_transfer_started=False;hw._roi_transfer_started=False;hw._dma_status()
        hw._roi_transfer_started=True
        with self.assertRaisesRegex(RuntimeError,'not confirmed idle'):hw._dma_status()
        hw.dma.mmio.read.return_value=2;hw._dma_status()
        hw.dma.mmio.read.return_value=0x12
        with self.assertRaisesRegex(RuntimeError,'DMA error'):hw._dma_status()


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        hu=np.arange(3*16*20,dtype=np.float32).reshape(3,16,20)-100
        np.savez(self.root/'ct.npz',hu=hu,spacing_zyx_mm=[3.,.5,.8],origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
        self.c=WorkspaceController(None,{},self.root/'results')
        self.c.submit('load',dict(path=str(self.root/'ct.npz'),confirm_hu=True,load_expert=False));wait(self.c)
        class Remote:
            manifest={'roi':ROW}
            def roi(self,*args,**kwargs):
                result=reference.CpuRoiProcessor().process_slice(*args,**kwargs);result['backend']='FPGA_ROI'
                result['timing']['dma_compute_seconds']=.001
                result['hardware_receipt']=dict(roundtrip_ms=2.,stats=dict(overlay_switch_ms=3.),test_only=True)
                return result
        self.c.client=Remote();self.c.status['board_connected']=True
    def tearDown(self):wait(self.c);self.temp.cleanup()
    def values(self):
        return dict(case_id=self.c.status['case_id'],analysis_source_id='original',index=1,mode='range',
                    z_bounds=[0,3],roi_yx=[1,4,2,7],use_hu=False,use_lung_mask=False,backend='FPGA_ROI')
    def test_remote_outputs_timing_receipts_and_saved_masks(self):
        self.c.submit('roi',self.values());s=wait(self.c)
        self.assertEqual(s['phase'],'done',s['error']);r=self.c.roi_result
        self.assertEqual(r['backend'],'FPGA_ROI');self.assertIsNone(r['timing']['cpu_reference_ms'])
        self.assertEqual(r['valid_count'],45);self.assertAlmostEqual(r['timing']['fpga_dma_compute_ms'],3.)
        self.assertTrue(all(row['hardware_receipt']['test_only'] for row in r['slice_results']))
        self.assertTrue(s['roi']['fpga_available']);self.assertEqual(s['known_slices'],0)
    def test_failed_remote_is_not_replaced_with_cpu(self):
        self.c.client.roi=Mock(side_effect=TimeoutError('TEST DMA uncertain'))
        self.c.submit('roi',self.values());s=wait(self.c)
        self.assertEqual(s['phase'],'error');self.assertFalse(s['board_connected'])
        self.assertEqual(self.c.roi_result['processed_slices'],0);self.assertEqual(self.c.roi_result['backend'],'FPGA_ROI')
    def test_unconnected_remote_does_not_submit(self):
        self.c.status['board_connected']=False
        with self.assertRaisesRegex(ValueError,'不会改用电脑'):self.c.submit('roi',self.values())


if __name__=='__main__':unittest.main()
