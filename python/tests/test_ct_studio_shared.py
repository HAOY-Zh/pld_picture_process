"""Simulated transport and package isolation tests; physical proof is separate."""
import copy,importlib,json
from pathlib import Path
import sys,unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'pynq'),str(ROOT/'pynq/roi_vendor')]
import test_ct_studio_roi as old
import ct_studio_shared_service as service
from ct_studio_shared_client import SharedPatchClient
from shared_patch_contract import INTERFACE,INPUT_SHAPE,OUTPUT_SHAPE,input_codes

ROW=dict(interface=INTERFACE,bit_sha256='a'*64,model_sha256='b'*64)


class SimulatedHardware(old.SimulatedRoiHardware):
    def __init__(self):
        super().__init__();self.manifest.update(shared_nodule=ROW,legacy_contract_sha256='f'*64);self.patch_calls=0
    def nodule_patch(self,image):
        self.patch_calls+=1
        if self.fail:raise TimeoutError('TEST uncertain shared core')
        head=np.broadcast_to(image[0],OUTPUT_SHAPE).copy()
        stats=dict(command_count=15,commands=[dict(node=str(i)) for i in range(15)],selected_model='nodule',
            overlay_reloaded=False,buffer_allocations_per_frame=0,host_total_ms=50.,overlay_switch_ms=0.,
            actual_pl_mhz_before=100.,actual_pl_mhz_after=100.,cpu_mhz=650.,pl_command_cycles=1200000)
        return head,stats


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.hw=SimulatedHardware();self.s=service.Service(self.hw,'test-shared-token-'*4,'e'*64)
        self.s.startup_verified=self.s.roi_verified=self.s.shared_verified=True
        self.server=old.RunningServer(service.handler_for(self.s))
        self.client=SharedPatchClient(self.server.url,'test-shared-token-'*4,
            dict(service_bundle_sha256='e'*64,shared_nodule=copy.deepcopy(ROW)))
        self.image=np.arange(np.prod(INPUT_SHAPE),dtype=np.uint8).reshape(INPUT_SHAPE)
    def tearDown(self):self.server.close()
    def test_fixed_model_raw_head_detached_and_health(self):
        self.client.health();head,r=self.client.nodule_patch(self.image)
        np.testing.assert_array_equal(head,np.broadcast_to(self.image[0],OUTPUT_SHAPE))
        self.assertFalse(r['whole_case_detection']);self.assertEqual(r['stats']['command_count'],15)
        head[:]=0;self.assertTrue(self.image.any())
    def test_legacy_health_contract_is_explicit_and_preserved(self):
        h=self.client.call('/v1/platform')
        self.assertEqual(h['platform_manifest_sha256'],'f'*64)
        self.assertEqual(h['service_bundle_sha256'],'e'*64)
        legacy=old.ROIStudioClient(self.server.url,'test-shared-token-'*4,
            dict(self.hw.manifest,platform_manifest_sha256='f'*64))
        result=legacy.roi(np.arange(20,dtype=np.int16).reshape(4,5),use_hu=False)
        self.assertEqual(result['valid_count'],20)
    def test_invalid_input_and_busy_never_reach_hardware(self):
        for image in (self.image.astype(np.float32),self.image[:1]):
            with self.assertRaises(ValueError):input_codes(image)
        self.s.gate.acquire()
        try:
            with self.assertRaises(Exception):self.client.nodule_patch(self.image)
        finally:self.s.gate.release()
        self.assertEqual(self.hw.patch_calls,0)
    def test_bad_hash_rejected_before_hardware(self):
        with self.assertRaises(ValueError):self.s.execute_patch(self.image.tobytes(),'0'*32,'1'*64)
        self.assertEqual(self.hw.patch_calls,0)
    def test_fault_retained_without_retry_or_cpu(self):
        self.hw.fail=True
        with self.assertRaises(Exception):self.client.nodule_patch(self.image)
        self.assertTrue(self.s.failed)
        with self.assertRaises(Exception):self.client.nodule_patch(self.image)
        self.assertEqual(self.hw.patch_calls,1)
    def test_tampered_clock_model_head_and_bundle_rejected(self):
        original=self.client.call
        for mutate in (lambda r:r['stats'].update(actual_pl_mhz_after=50.),
                       lambda r:r.update(service_bundle_sha256='x'*64),
                       lambda r:r.update(output_sha256='0'*64),
                       lambda r:r.update(shared_nodule=dict(ROW,model_sha256='0'*64))):
            def changed(*args,**kwargs):
                r=original(*args,**kwargs);mutate(r);return r
            self.client.call=changed
            with self.assertRaises(ValueError):self.client.nodule_patch(self.image)


class IsolationTests(unittest.TestCase):
    def test_frozen_models_validate_and_new_drivers_cannot_shadow_old(self):
        folder=ROOT/'data/ct_studio_shared_20261024_01/ct_studio'
        if not folder.is_dir():folder=ROOT/'data/ct_studio_integration_20261025/ct_studio'
        if not folder.is_dir():self.skipTest('Install the reviewed integration runtime before asset-isolation tests')
        manifest=service.validate_package(folder)
        self.assertEqual(manifest['shared_nodule']['commands_per_patch'],15)
        sys.path.insert(0,str(folder))
        try:
            private=importlib.import_module('shared_cnn.lung_nodule_shared')
            fast=importlib.import_module('shared_cnn.lung_fast')
            self.assertIs(private.FastUNet,fast.FastUNet)
            self.assertNotEqual(fast.__name__,'lung_fast')
            model,params=private.load_model(folder/'shared_cnn','nodule')
            self.assertEqual(model['input_shape'],list(INPUT_SHAPE));self.assertTrue(params)
        finally:sys.path.remove(str(folder))
    def test_failed_shared_owner_cannot_be_freed(self):
        hw=service.Hardware.__new__(service.Hardware);hw.mode='shared_nodule';hw.fault=None
        hw.shared_owner=SimpleNamespace(failed=True,close=Mock())
        with self.assertRaisesRegex(RuntimeError,'Faulted'):hw.release_idle()
        hw.shared_owner.close.assert_not_called()


if __name__=='__main__':unittest.main()
