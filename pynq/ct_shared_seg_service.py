"""Add an explicitly identified original-U-Net route to the existing sole owner.

The legacy route and pinned legacy manifest remain valid. Both new model routes
use the same resident owner. No fault recovery, CPU fallback or timeout retry.
"""
import base64
import json
import re
import threading
import time
import uuid
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import ct_studio_shared_service as previous
from shared_segmentation_contract import INTERFACE, ROUTE, MODEL_LABEL, SHAPE, BYTES, digest, descriptor, validate_stats


def validate_package(folder):
    manifest = previous.validate_package(folder)
    if manifest.get('shared_segmentation') != descriptor(folder):
        raise ValueError('Unbound shared original U-Net assets')
    return manifest


class Hardware(previous.Hardware):
    def shared_segmentation(self, image, observer=None):
        switch_ms = self.switch('shared_nodule')
        try:
            before = dict(fclk0_mhz=float(self.clocks.fclk0_mhz), cpu_mhz=float(self.clocks.cpu_mhz))
            head, stats = self.shared_owner.predict_codes(MODEL_LABEL, image, observer)
            self._dma_status()
            after = dict(fclk0_mhz=float(self.clocks.fclk0_mhz), cpu_mhz=float(self.clocks.cpu_mhz))
            stats.update(overlay_switch_ms=switch_ms, clock_before=before, clock_after=after)
            validate_stats(stats)
            return head, stats
        except BaseException as exc:
            self.fault = type(exc).__name__+': '+str(exc)
            raise

    def selftest_shared_segmentation(self):
        folder = self.folder
        model = json.loads((folder/'segmentation/model.json').read_text())
        from shared_cnn.lung_continuous import unpack_words
        arrays = {}
        checked = []
        with np.load(folder/'segmentation/input.npz', allow_pickle=False) as data:
            image = data['codes']
        with np.load(folder/'segmentation/reference.npz', allow_pickle=False) as refs:
            def observe(name, words):
                expected = unpack_words(refs[name], next(n['shape'] for n in model['nodes'] if n['name'] == name))
                actual = unpack_words(words, expected.shape)
                np.testing.assert_array_equal(actual, expected)
                arrays[name] = actual
                checked.append(name)
            head, stats = self.shared_segmentation(image, observe)
        np.testing.assert_array_equal(head, self.last_startup_head)
        if len(checked) != 21:
            raise AssertionError('Incomplete shared segmentation startup nodes')
        arrays['legacy_head'] = self.last_startup_head.copy()
        arrays['shared_head'] = head.copy()
        stem = folder/('shared_seg_startup_'+uuid.uuid4().hex)
        np.savez_compressed(str(stem)+'.npz', **arrays)
        Path(str(stem)+'.json').write_text(json.dumps(dict(passed=True, checked=checked,
            legacy_head_exact=True, shared_segmentation=descriptor(folder), stats=stats,
            arrays_sha256=digest(Path(str(stem)+'.npz').read_bytes())), indent=2))
        self.shared_segmentation_verified = True


class Service(previous.Service):
    def shared_health(self):
        value = super().shared_health()
        value.update(shared_segmentation=self.hardware.manifest['shared_segmentation'],
            shared_segmentation_startup='all-21-nodes-and-legacy-head-exact'
                if getattr(self.hardware, 'shared_segmentation_verified', False) else 'not-run')
        return value

    def execute_shared_segmentation(self, raw, rid, input_hash):
        if len(raw) != BYTES or not re.fullmatch('[0-9a-f]{32}', rid) or digest(raw) != input_hash:
            raise ValueError('Invalid segmentation input/request/hash')
        if not self.gate.acquire(False):
            raise BlockingIOError('FPGA busy; no concurrent command')
        try:
            if self.failed or self.hardware.fault:
                raise RuntimeError('Faulted owner retained')
            try:
                started = time.perf_counter()
                head, stats = self.hardware.shared_segmentation(np.frombuffer(raw, np.uint8).reshape(SHAPE))
                if head.dtype != np.uint8 or head.shape != SHAPE:
                    raise ValueError('Invalid original U-Net head')
                payload = head.tobytes()
                self.requests += 1
                row = self.hardware.manifest['shared_segmentation']
                return dict(schema='lung_live_result_v1', backend='FPGA_INT8', engine_interface=INTERFACE,
                    bit_sha256=row['bit_sha256'], model_sha256=row['model_sha256'], shape=list(SHAPE),
                    request_id=rid, input_sha256=input_hash, head_base64=base64.b64encode(payload).decode(),
                    head_sha256=digest(payload), stats=stats, sequence=self.requests,
                    service_ms=(time.perf_counter()-started)*1000, shared_segmentation=row,
                    service_bundle_sha256=self.bundle_sha256,
                    state=dict(busy=False, faulted=False, dma_may_access=False))
            except BaseException as exc:
                self.failed = type(exc).__name__+': '+str(exc)
                raise
        finally:
            self.gate.release()


def handler_for(service):
    parent = previous.handler_for(service)
    class Handler(parent):
        def do_POST(self):
            if self.path != ROUTE:
                return super().do_POST()
            if not self.allowed():
                return
            try:
                if (self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length') != str(BYTES)
                        or self.headers.get('Content-Type') != 'application/octet-stream'):
                    raise ValueError('Invalid segmentation framing')
                raw = self.rfile.read(BYTES)
                self.reply(200, service.execute_shared_segmentation(raw, self.headers.get('X-Request-ID', ''),
                    self.headers.get('X-Input-SHA256', '')))
            except BlockingIOError as exc:
                self.reply(409, dict(error=str(exc)))
            except ValueError as exc:
                self.reply(503 if service.failed else 400, dict(error=str(exc)))
            except Exception as exc:
                self.reply(503, dict(error=str(exc)))
    return Handler


def start(bind='0.0.0.0', port=8766, folder=None):
    folder = Path(folder).resolve()
    validate_package(folder)
    server = ThreadingHTTPServer((bind, port), BaseHTTPRequestHandler)
    try:
        hardware = Hardware(folder)
        hardware.selftest_basic()
        hardware.selftest_ldct()
        hardware.selftest_roi()
        hardware.selftest_shared()
        hardware.selftest_shared_segmentation()
        service = Service(hardware, (folder/'platform_token.txt').read_text().strip(),
            digest((folder/'platform_manifest.json').read_bytes()))
        service.startup_verified = service.roi_verified = service.shared_verified = True
        server.RequestHandlerClass = handler_for(service)
    except BaseException:
        server.server_close()
        raise
    service.server = server
    previous.roi.base.OWNERS.append(service)
    service.thread = threading.Thread(target=server.serve_forever, daemon=True)
    service.thread.start()
    print('CT SHARED SEG READY: original U-Net all-node parity and legacy head EXACT PASS.', flush=True)
    return service
