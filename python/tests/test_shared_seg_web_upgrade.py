"""Contract and fail-closed tests for the real shared-segmentation route."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT/'data/CT_Studio_SharedSeg_20261025_01'
BOARD = APP/'data/ct_studio_integration_20261025/ct_studio'
sys.path[:0] = [str(APP/'python'), str(BOARD)]
import ct_shared_seg_service as server
from ct_shared_seg_client import SharedSegStudioClient
from shared_segmentation_contract import ROUTE, SHAPE, descriptor, digest


def stats():
    return dict(command_count=51, commands=[{} for _ in range(51)], selected_model='unet_original',
        overlay_reloaded=False, buffer_allocations_per_frame=0, weight_packets_repacked_per_frame=0,
        host_total_ms=400., shared_call_ms=402., overlay_switch_ms=0., pl_command_cycles=30000000,
        clock_before=dict(fclk0_mhz=100., cpu_mhz=650.), clock_after=dict(fclk0_mhz=100., cpu_mhz=650.))


class FakeHardware:
    fault = None
    def __init__(self):
        self.manifest = json.loads((BOARD/'platform_manifest.json').read_text())
        self.calls = 0
        self.fail = False
    def shared_segmentation(self, image):
        self.calls += 1
        if self.fail:
            self.fault = 'simulated hardware timeout'
            raise TimeoutError(self.fault)
        return image.copy(), stats()


class Tests(unittest.TestCase):
    def setUp(self):
        self.hw = FakeHardware()
        self.service = server.Service(self.hw, 'a'*32, digest((BOARD/'platform_manifest.json').read_bytes()))
        manifest = json.loads((APP/'data/ct_studio_integration_20261025/host_platform.json').read_text())
        self.client = SharedSegStudioClient('http://127.0.0.1:8766', 'a'*32, manifest)
        self.calls = []
        self.mutate = lambda value: value
        def call(route, data=None, headers=None, timeout=60):
            self.calls.append(route)
            self.assertEqual(route, ROUTE)
            return self.mutate(self.service.execute_shared_segmentation(data, headers['X-Request-ID'], headers['X-Input-SHA256']))
        self.client.call = call

    def test_real_package_and_original_assets(self):
        manifest = server.validate_package(BOARD)
        self.assertEqual(manifest['shared_segmentation'], descriptor(BOARD))
        self.assertNotEqual(manifest['shared_segmentation']['bit_sha256'], manifest['segmentation_bit_sha256'])

    def test_new_route_and_real_identity(self):
        image = np.arange(65536, dtype=np.uint16).astype(np.uint8).reshape(SHAPE)
        head, receipt = self.client.infer(image)
        np.testing.assert_array_equal(head, image)
        self.assertEqual(self.calls, [ROUTE])
        self.assertEqual(receipt['bit_sha256'], self.client.segmentation_bit_sha256)

    def test_reject_legacy_bit_and_wrong_head(self):
        image = np.zeros(SHAPE, np.uint8)
        for edit in (
            lambda r: dict(r, bit_sha256='0'*64),
            lambda r: dict(r, head_sha256='0'*64),
            lambda r: dict(r, stats=dict(r['stats'], command_count=50)),
            lambda r: dict(r, service_bundle_sha256='0'*64),
            lambda r: dict(r, stats=dict(r['stats'], clock_after=dict(fclk0_mhz=50., cpu_mhz=650.))),
        ):
            self.mutate = edit
            with self.assertRaises(ValueError):
                self.client.infer(image)

    def test_invalid_input_and_busy_do_not_touch_hardware(self):
        with self.assertRaises(ValueError):
            self.service.execute_shared_segmentation(b'bad', 'a'*32, digest(b'bad'))
        self.service.gate.acquire()
        try:
            raw = bytes(65536)
            with self.assertRaises(BlockingIOError):
                self.service.execute_shared_segmentation(raw, 'a'*32, digest(raw))
        finally:
            self.service.gate.release()
        self.assertEqual(self.hw.calls, 0)

    def test_fault_is_retained_and_not_retried(self):
        self.hw.fail = True
        with self.assertRaises(TimeoutError):
            self.client.infer(np.zeros(SHAPE, np.uint8))
        with self.assertRaises(RuntimeError):
            self.client.infer(np.zeros(SHAPE, np.uint8))
        self.assertEqual(self.hw.calls, 1)
        self.assertIsNotNone(self.service.failed)


if __name__ == '__main__':
    unittest.main()
