"""On-demand FPGA transport, using the accepted runtime unchanged.

Start in a dedicated PYNQ notebook with start(); keep its returned owner alive.
No reset/reload/free is attempted after any inference fault. No CPU fallback.
Only quantized inputs cross this LAN-only, authenticated (not TLS) transport.
"""
import base64
import hashlib
import hmac
import json
from pathlib import Path
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

BIT_SHA = '567ddc0e47e3556e21b248a2d0bab85f26381c8056295f88ddd14335e695c89b'
MODEL_SHA = '70ad14405228ac08421c5d9a732a2825f46c1e364e4cf927488a90cdd0650dbd'
_SERVICES = []  # Retain owners even if notebook drops its own binding.


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_live(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'live_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'lung_live_fpga_v1':
        raise ValueError('Not a live FPGA package')
    for name, expected in manifest['files_sha256'].items():
        path = (folder / name).resolve()
        if path.parent != folder or digest(path.read_bytes()) != expected:
            raise ValueError('Changed live package file: ' + name)
    if (digest((folder / 'lung_network.bit').read_bytes()) != BIT_SHA or
            digest((folder / 'model.json').read_bytes()) != MODEL_SHA):
        raise ValueError('Not the retained accepted bit/model')
    return manifest


class Hardware:
    def __init__(self, folder, timeout=30.):
        # Acquire before importing PYNQ or downloading anything. This prevents
        # duplicate live servers; other notebooks MUST also relinquish the PL.
        import fcntl
        self.lock_file = open('/tmp/lung_live_z2.lock', 'a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.folder = Path(folder).resolve()
        validate_live(self.folder)
        from lung_selected_benchmark import validate_selection, selected_entrypoints, overlay_guard
        selected, fast = validate_selection(self.folder)
        with selected_entrypoints(self.folder, fast) as (bench, clocks, _):
            _, model, parameters, _ = bench.validate_package(self.folder)
            self.model = model
            library, _ = bench.compile_io(self.folder)
            import pynq
            self.clocks = pynq.Clocks
            self.clock_api = clocks
            self.cpu_mhz = clocks.state(self.clocks)['cpu_mhz']
            trace = []
            guarded = overlay_guard(pynq.Overlay, self.folder, selected['files_sha256'], [])
            self.overlay = clocks.clock_overlay(guarded(str(self.folder / 'lung_network.bit')),
                self.clocks, self.cpu_mhz, trace, timeout)
            conv_mmio = self.overlay.lung_conv8_axil_0.mmio
            self.network = bench.FastUNet(
                bench.PreparedConv(conv_mmio, native_loader=bench.NativeLoader(library, conv_mmio), timeout=timeout),
                bench.TensorAxil(self.overlay.lung_tensor_axil_0.mmio, timeout=timeout),
                pynq.allocate, model, parameters)
            # Retain before the first START, including if startup self-test fails.
            _SERVICES.append(self)
            with np.load(self.folder / 'input.npz', allow_pickle=False) as a:
                fixture = a['codes']
            checked = []
            with np.load(self.folder / 'reference.npz', allow_pickle=False) as refs:
                def observe(name, words):
                    if not np.array_equal(words, refs[name]):
                        raise AssertionError('Startup FPGA node mismatch: ' + name)
                    checked.append(name)
                codes, stats = self.infer(fixture, observer=observe)
                from lung_continuous import unpack_words
                expected = unpack_words(refs['head'], model['nodes'][-1]['shape'])
                if len(checked) != 21 or not np.array_equal(codes, expected):
                    raise AssertionError('Incomplete startup FPGA self-test')
            self.identity = dict(schema='lung_live_identity_v1', backend='FPGA_INT8', bit_sha256=BIT_SHA,
                model_sha256=MODEL_SHA, input_quantization=model['input_quantization'],
                output_scale=model['output_scale'], output_zero=model['output_zero'],
                startup_nodes_checked=checked, startup_command_count=stats['command_count'],
                cma_bytes=self.network.cma_bytes, startup_clock_trace=trace)

    def infer(self, image, observer=None):
        before = self.clock_api.state(self.clocks)
        self.clock_api.require_target(before, self.cpu_mhz)
        codes, stats = self.network.predict_codes(image, observer=observer)
        after = self.clock_api.state(self.clocks)
        self.clock_api.require_target(after, self.cpu_mhz)
        if stats['command_count'] != 51 or len(stats['commands']) != 51:
            raise AssertionError('Incomplete FPGA forward')
        stats.update(clock_before=before, clock_after=after)
        return np.array(codes, dtype=np.uint8, copy=True), stats


class InferenceService:
    def __init__(self, hardware, token):
        if not isinstance(token, str) or len(token) < 32:
            raise ValueError('Use a random token of at least 32 characters')
        self.hardware, self.token = hardware, token
        self.gate = threading.Lock()
        self.failed = None
        self.requests = 0

    def infer(self, raw, request_id, expected_hash):
        if len(raw) != 65536 or not re.fullmatch('[0-9a-f]{32}', request_id):
            raise ValueError('Need one UINT8 1x256x256 input and UUID hex')
        if digest(raw) != expected_hash:
            raise ValueError('Input transport hash mismatch')
        if not self.gate.acquire(blocking=False):
            raise BlockingIOError('FPGA busy; no automatic retry')
        try:
            if self.failed:
                raise RuntimeError('Service faulted; inspect notebook, no automatic reset: ' + self.failed)
            try:
                start_time = time.perf_counter()
                codes, stats = self.hardware.infer(np.frombuffer(raw, np.uint8).reshape(1, 256, 256))
                output = codes.tobytes()
                self.requests += 1
                return dict(schema='lung_live_result_v1', request_id=request_id, backend='FPGA_INT8',
                    input_sha256=expected_hash, head_sha256=digest(output),
                    bit_sha256=BIT_SHA, model_sha256=MODEL_SHA, shape=[1, 256, 256],
                    head_base64=base64.b64encode(output).decode('ascii'), stats=stats,
                    service_ms=(time.perf_counter() - start_time) * 1000, sequence=self.requests)
            except BaseException as exc:
                self.failed = type(exc).__name__ + ': ' + str(exc)
                # Keep the process, drivers, overlay and CMA alive. Never close,
                # reset, reload or retry a failed/uncertain operation here.
                raise
        finally:
            self.gate.release()


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *_):
            pass  # Never log auth, image bytes or user paths.

        def reply(self, status, value):
            raw = json.dumps(value, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(raw)

        def allowed(self):
            # No browser CORS: only the local PC proxy knows this token.
            if self.headers.get('Origin') or not hmac.compare_digest(
                    self.headers.get('Authorization', ''), 'Bearer ' + service.token):
                self.reply(403, dict(error='Unauthorized'))
                return False
            self.connection.settimeout(10.)
            return True

        def do_GET(self):
            if not self.allowed(): return
            if self.path != '/v1/health':
                return self.reply(404, dict(error='Unknown endpoint'))
            self.reply(200, dict(ready=service.failed is None, busy=service.gate.locked(),
                error=service.failed, completed_requests=service.requests, **service.hardware.identity))

        def do_POST(self):
            if not self.allowed(): return
            if self.path != '/v1/infer': return self.reply(404, dict(error='Unknown endpoint'))
            try:
                if (self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length') != '65536'
                        or self.headers.get('Content-Type') != 'application/octet-stream'):
                    raise ValueError('Expected exactly 65536 raw input bytes')
                raw = self.rfile.read(65536)
                value = service.infer(raw, self.headers.get('X-Request-ID', ''),
                                      self.headers.get('X-Input-SHA256', ''))
            except BlockingIOError as exc:
                return self.reply(409, dict(error=str(exc)))
            except (ValueError, TimeoutError) as exc:
                return self.reply(400 if not service.failed else 503, dict(error=str(exc)))
            except Exception as exc:
                return self.reply(503, dict(error=str(exc)))
            self.reply(200, value)
    return Handler


def start(bind='127.0.0.1', port=8765, folder=None):
    """Nonblocking Jupyter entry. Run only one live service on this board."""
    folder = Path(folder or Path(__file__).resolve().parent).resolve()
    token = (folder / 'live_token.txt').read_text(encoding='utf-8').strip()
    # Bind first so an occupied port never triggers a needless bit download.
    server = ThreadingHTTPServer((bind, port), BaseHTTPRequestHandler)
    server.daemon_threads = False  # Do not let an active HW handler exit silently.
    try:
        hardware = Hardware(folder)
        service = InferenceService(hardware, token)
        server.RequestHandlerClass = handler_for(service)
    except BaseException:
        server.server_close()
        raise
    service.server = server
    _SERVICES.append(service)
    service.thread = threading.Thread(target=server.serve_forever, daemon=True)
    service.thread.start()
    print('FPGA LIVE READY: {}:{}; 21-node startup PASS; keep this kernel alive.'.format(bind, port), flush=True)
    print('LAN demonstration only; do not run other overlays/benchmarks or stop this kernel during inference.', flush=True)
    return service
