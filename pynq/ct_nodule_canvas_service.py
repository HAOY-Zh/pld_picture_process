"""Canvas inference through the SAME sole shared owner; legacy routes retained."""
import base64
import json
import re
import threading
import time
import uuid
import zlib
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import ct_shared_seg_service as parent
from nodule_canvas_contract import INTERFACE, ROUTE, MAX_RAW, shapes, inflate, digest, validate_stats, quantize_canvas


class Hardware(parent.Hardware):
    def canvas(self, codes, height, width, observer=None):
        source, target = shapes(height, width)
        if codes.dtype != np.uint8 or codes.shape != source:
            raise ValueError('Wrong canvas codes')
        switch = self.switch('shared_nodule')
        tick = time.perf_counter()
        owner = self.shared_owner
        try:
            with owner._exclusive():
                net = owner._nets.get('nodule_canvas')
                if net is None or (net.height, net.width) != (height, width):
                    if net is not None:
                        net.close()
                        del owner._nets['nodule_canvas']
                    from shared_cnn.nodule_canvas_pl import CanvasPL
                    from shared_cnn.lung_nodule_shared import load_model
                    from shared_cnn.lung_axil import LungAxil
                    model, params = load_model(self.folder/'shared_cnn', 'nodule')
                    net = CanvasPL(owner.conv, owner.tensor, owner.allocate, model, params, height, width)
                    # Match SharedCNN registration's physical cross-model checks.
                    for b in net.buffers.values():
                        address = LungAxil._buffer_address(b)
                        for other_net in owner._nets.values():
                            for other in other_net.buffers.values():
                                start = LungAxil._buffer_address(other)
                                if address < start+other.nbytes and start < address+b.nbytes:
                                    raise RuntimeError('Cross-model canvas CMA overlap')
                    owner._nets['nodule_canvas'] = net
            setup = (time.perf_counter()-tick)*1000
            before = float(self.clocks.fclk0_mhz)
            head, stats = owner.predict_codes('nodule_canvas', codes, observer)
            self._dma_status()
            stats.update(overlay_switch_ms=switch, canvas_setup_ms=setup,
                actual_pl_mhz_before=before, actual_pl_mhz_after=float(self.clocks.fclk0_mhz),
                cpu_mhz=float(self.clocks.cpu_mhz), canvas_hw=[height, width])
            validate_stats(stats)
            if head.shape != target:
                raise ValueError('Wrong canvas head')
            return head, stats
        except BaseException as exc:
            owner.failed = True
            self.fault = type(exc).__name__+': '+str(exc)
            raise

    def selftest_canvas(self):
        from shared_cnn.nodule_canvas import lower_canvas
        from shared_cnn.nodule_plan import quantize_hu, lower_stem
        folder = self.folder/'shared_cnn'
        arrays = {}
        self.switch('shared_nodule')
        with np.load(folder/'models/nodule/reference.npz', allow_pickle=False) as refs:
            a, _ = self.canvas(refs['case0_input'], 128, 128)
            np.testing.assert_array_equal(a, refs['case0_head'])
            arrays['canonical_head'] = a
        image = np.random.default_rng(20261025).integers(-1200, 501, (5, 256, 256), dtype=np.int16)
        quantized = quantize_canvas(image)
        canvas, _ = self.canvas(lower_canvas(quantized), 256, 256)
        arrays['canvas_head'] = canvas
        for y, x in ((0, 0), (64, 64), (128, 128)):
            patch = image[:, y:y+128, x:x+128]
            expected, _ = self.nodule_patch(lower_stem(quantize_hu(patch)))
            actual = canvas[:, y//2+15:y//2+49, x//2+15:x//2+49]
            np.testing.assert_array_equal(actual, expected[:, 15:49, 15:49])
            arrays[f'patch_{y}_{x}'] = expected
        stem = self.folder/('canvas_startup_'+uuid.uuid4().hex)
        np.savez_compressed(str(stem)+'.npz', **arrays)
        Path(str(stem)+'.json').write_text(json.dumps(dict(passed=True,
            scope='Actual canonical full head and three translated decoder-required neighbourhoods',
            arrays_sha256=digest(Path(str(stem)+'.npz').read_bytes())), indent=2))
        self.canvas_verified = True


class Service(parent.Service):
    def shared_health(self):
        value = super().shared_health()
        value.update(canvas_interface=INTERFACE, canvas_max_hw=1024,
            canvas_startup='canonical-head-and-translated-valid-cells-exact'
            if getattr(self.hardware, 'canvas_verified', False) else 'not-run')
        return value

    def execute_canvas(self, raw, rid, sha, height, width):
        source, target = shapes(height, width)
        if len(raw) != int(np.prod(source)) or not re.fullmatch('[0-9a-f]{32}', rid) or digest(raw) != sha:
            raise ValueError('Wrong canvas input/request/hash')
        if not self.gate.acquire(False):
            raise BlockingIOError('FPGA busy; no concurrent operation')
        try:
            if self.failed or self.hardware.fault:
                raise RuntimeError('Faulted owner retained')
            try:
                started = time.perf_counter()
                head, stats = self.hardware.canvas(np.frombuffer(raw, np.uint8).reshape(source), height, width)
                if head.dtype != np.uint8 or head.shape != target:
                    raise ValueError('Invalid actual canvas output')
                packed = head.tobytes()
                self.requests += 1
                return dict(schema=INTERFACE, backend='FPGA_NODULE_SHARED', request_id=rid, input_sha256=sha,
                    input_shape=list(source), output_shape=list(target), output_sha256=digest(packed),
                    head_zlib_base64=base64.b64encode(zlib.compress(packed, 1)).decode(), stats=stats,
                    sequence=self.requests, service_ms=(time.perf_counter()-started)*1000,
                    shared_nodule=self.hardware.manifest['shared_nodule'], service_bundle_sha256=self.bundle_sha256,
                    state=dict(busy=False, faulted=False, dma_may_access=False))
            except BaseException as exc:
                self.failed = type(exc).__name__+': '+str(exc)
                raise
        finally:
            self.gate.release()


def handler_for(service):
    previous = parent.handler_for(service)
    class Handler(previous):
        def do_POST(self):
            if self.path != ROUTE: return super().do_POST()
            if not self.allowed(): return
            try:
                size = int(self.headers.get('Content-Length', '-1'))
                if (self.headers.get('Transfer-Encoding') or not 0 < size <= MAX_RAW+8192
                        or self.headers.get('Content-Type') != 'application/octet-stream'):
                    raise ValueError('Wrong canvas framing')
                hw = json.loads(self.headers.get('X-Canvas-Shape', '[]'))
                if not isinstance(hw, list) or len(hw) != 2: raise ValueError('Need canvas H/W')
                source, _ = shapes(*hw)
                packed = self.rfile.read(size)
                if len(packed) != size: raise ValueError('Truncated canvas input')
                raw = inflate(packed, int(np.prod(source)))
                self.reply(200, service.execute_canvas(raw, self.headers.get('X-Request-ID', ''),
                    self.headers.get('X-Input-SHA256', ''), *hw))
            except BlockingIOError as exc: self.reply(409, dict(error=str(exc)))
            except (ValueError, TypeError, zlib.error) as exc: self.reply(503 if service.failed else 400, dict(error=str(exc)))
            except Exception as exc: self.reply(503, dict(error=str(exc)))
    return Handler


def start(bind='0.0.0.0', port=8766, folder=None):
    folder = Path(folder).resolve()
    parent.validate_package(folder)
    server = ThreadingHTTPServer((bind, port), BaseHTTPRequestHandler)
    try:
        hardware = Hardware(folder)
        hardware.selftest_basic(); hardware.selftest_ldct(); hardware.selftest_roi()
        hardware.selftest_shared(); hardware.selftest_shared_segmentation(); hardware.selftest_canvas()
        service = Service(hardware, (folder/'platform_token.txt').read_text().strip(),
            digest((folder/'platform_manifest.json').read_bytes()))
        service.startup_verified = service.roi_verified = service.shared_verified = True
        server.RequestHandlerClass = handler_for(service)
    except BaseException:
        server.server_close()
        raise
    service.server = server
    parent.previous.roi.base.OWNERS.append(service)
    service.thread = threading.Thread(target=server.serve_forever, daemon=True)
    service.thread.start()
    print('CT CANVAS READY: segmentation, original patch and canvas exact startup PASS.', flush=True)
    return service
