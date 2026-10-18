"""Single Z2 owner for existing basic DMA overlays and preserved INT8 U-Net.

No bit is rebuilt. Only a confirmed-idle, fault-free owner may change overlays.
After ANY hardware fault, retain all owners/buffers and refuse further work.
"""
import base64
import hashlib
import hmac
import json
from pathlib import Path
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ct_basic_processing import parameters, process
from lung_live_service import BIT_SHA, MODEL_SHA, Hardware as SegmentationHardware

OWNERS = []
BASIC_OPERATIONS = ('window', 'threshold', 'erosion', 'dilation', 'opening', 'closing')
PIXELS = 512*512


def sha(raw): return hashlib.sha256(raw).hexdigest()


def startup_cases():
    """Fixed synthetic HU: transport boundaries, structures and zero-pad edges."""
    y,x=np.mgrid[:512,:512]
    patterned=((131*x+73*y)%4000-2000).astype(np.int16)
    blocks=np.full((512,512),40,np.int16)
    blocks[0:85,0:110]=-800;blocks[150:350,150:350]=-800
    blocks[250,250]=40;blocks[420,75]=-800
    blocks[470:490,470:490]=-1000;blocks[20:40,430:460]=-400
    return [('patterned',patterned),('blocks_edges',blocks)]


def validate_package(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'platform_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'ct_platform_package_v1': raise ValueError('Wrong platform package')
    for name, digest in manifest['files_sha256'].items():
        path = (folder/name).resolve()
        if folder not in path.parents or sha(path.read_bytes()) != digest:
            raise ValueError('Changed platform file: '+name)
    if set(manifest['basic']) != set(BASIC_OPERATIONS): raise ValueError('Missing basic module')
    for operation, row in manifest['basic'].items():
        for kind in ('bit', 'hwh'):
            if manifest['files_sha256'].get(row[kind]) != row[kind+'_sha256']:
                raise ValueError('Unbound basic module: '+operation)
    if (manifest['files_sha256'].get('segmentation/lung_network.bit') != BIT_SHA or
        manifest['files_sha256'].get('segmentation/model.json') != MODEL_SHA):
        raise ValueError('Not preserved segmentation bit/model')
    return manifest


class Hardware:
    def __init__(self, folder, timeout=30.):
        import fcntl
        self.folder = Path(folder).resolve(); self.manifest = validate_package(self.folder)
        sys.path.insert(0, str(self.folder/'segmentation'))
        self.timeout = timeout; self.mode = None; self.network = None
        self.dma = None; self.buffers = []; self.fault = None; self.switches = 0
        self.lock_file = open('/tmp/lung_live_z2.lock', 'a')
        try: fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock_file.close(); raise RuntimeError('Another FPGA service owns PL; stop it only when confirmed idle')
        # Retain even when download or startup verification fails.
        OWNERS.append(self)
        import pynq
        self.pynq = pynq
        self.clocks = pynq.Clocks
        self.cpu_mhz = float(self.clocks.cpu_mhz)
        self.switch('segmentation')

    def _dma_status(self):
        if self.dma is None: return
        for offset in (0x04, 0x34):
            status = int(self.dma.mmio.read(offset))
            if status & 0x770: raise RuntimeError('DMA error status 0x%x' % status)
            if not status & 2: raise RuntimeError('DMA not confirmed idle')

    def release_idle(self):
        if self.fault: raise RuntimeError('Faulted owner retained; no release or reload')
        if self.network is not None:
            self.network.close()  # Existing owner checks inflight/poisoned before freeing.
            self.network = None
        if self.dma is not None: self._dma_status()
        for buffer in self.buffers: buffer.freebuffer()
        self.buffers = []; self.dma = None

    def switch(self, mode):
        if self.fault: raise RuntimeError('Faulted owner: '+self.fault)
        if self.mode == mode: return 0.
        start = time.perf_counter()
        try:
            self.release_idle()
            if mode == 'segmentation': self._load_segmentation()
            else:
                row = self.manifest['basic'][mode]
                bit = self.folder/row['bit']; hwh = self.folder/row['hwh']
                if sha(bit.read_bytes()) != row['bit_sha256'] or sha(hwh.read_bytes()) != row['hwh_sha256']:
                    raise ValueError('Changed basic bit/HWH before download')
                self.overlay = self.pynq.Overlay(str(bit))
                self.dma = self.overlay.axi_dma_0
                for _ in range(2):
                    self.buffers.append(self.pynq.allocate(shape=(PIXELS+16,), dtype=np.uint32))
                self.active_bit = row['bit_sha256']
            self.mode = mode; self.switches += 1
            if abs(float(self.clocks.cpu_mhz)-self.cpu_mhz) > .05: raise RuntimeError('CPU clock changed')
            return (time.perf_counter()-start)*1000
        except BaseException as exc:
            self.fault = type(exc).__name__+': '+str(exc); raise

    def _load_segmentation(self):
        from lung_selected_benchmark import validate_selection, selected_entrypoints, overlay_guard
        folder = self.folder/'segmentation'
        selection, fast = validate_selection(folder)
        with selected_entrypoints(folder, fast) as (bench, clocks, _):
            _, model, arrays, _ = bench.validate_package(folder)
            library, _ = bench.compile_io(folder)
            self.clock_api = clocks
            trace = []
            guard = overlay_guard(self.pynq.Overlay, folder, selection['files_sha256'], [])
            self.overlay = clocks.clock_overlay(guard(str(folder/'lung_network.bit')),
                self.clocks, self.cpu_mhz, trace, self.timeout)
            mmio = self.overlay.lung_conv8_axil_0.mmio
            self.network = bench.FastUNet(bench.PreparedConv(mmio,
                native_loader=bench.NativeLoader(library, mmio), timeout=self.timeout),
                bench.TensorAxil(self.overlay.lung_tensor_axil_0.mmio, timeout=self.timeout),
                self.pynq.allocate, model, arrays)
            with np.load(folder/'input.npz', allow_pickle=False) as a: fixture = a['codes']
            checked = []
            with np.load(folder/'reference.npz', allow_pickle=False) as refs:
                def observe(name, words):
                    if not np.array_equal(words, refs[name]): raise AssertionError('Startup mismatch: '+name)
                    checked.append(name)
                head, stats = SegmentationHardware.infer(self, fixture, observe)
                from lung_continuous import unpack_words
                if len(checked) != 21 or not np.array_equal(head,
                    unpack_words(refs['head'], model['nodes'][-1]['shape'])):
                    raise AssertionError('Incomplete segmentation self-test')
            self.segmentation_identity = dict(schema='lung_live_identity_v1', backend='FPGA_INT8',
                bit_sha256=BIT_SHA, model_sha256=MODEL_SHA,
                startup_nodes_checked=checked, startup_command_count=stats['command_count'],
                input_quantization=model['input_quantization'], output_scale=model['output_scale'],
                output_zero=model['output_zero'])
            self.last_startup_head = np.array(head, copy=True)
            self.active_bit = BIT_SHA

    def selftest_basic(self):
        rows=[];actual={}
        for op in BASIC_OPERATIONS:
            for tag,hu in startup_cases():
                p=parameters(dict(operation=op))
                if op=='window' and tag=='patterned': p.update(wl=0,ww=5)
                wanted=process(hu,p);image,stats=self.basic(hu,p)
                key=op+'_'+tag;actual[key]=image
                row=dict(key=key,parameters=p,input_sha256=sha(hu.astype('<i2').tobytes()),
                    output_sha256=sha(image.tobytes()),reference_bit_exact=bool(np.array_equal(image,wanted)),
                    bit_sha256=self.manifest['basic'][op]['bit_sha256'],stats=stats)
                rows.append(row)
                if not row['reference_bit_exact']:
                    self.fault='Startup basic PL/reference mismatch: '+key
                    self._save_startup(rows,actual,False)
                    raise AssertionError(self.fault)
                print('BASIC PL STARTUP PASS: '+key,flush=True)
        self.switch('segmentation')  # Re-download and re-check all 21 real nodes.
        actual['segmentation_head']=self.last_startup_head
        self._save_startup(rows,actual,True)

    def _save_startup(self,rows,actual,passed):
        import uuid
        stem=self.folder/('platform_startup_'+uuid.uuid4().hex)
        arrays=stem.with_suffix('.npz')
        with arrays.open('xb') as f:np.savez_compressed(f,**actual)
        evidence=dict(schema='ct_platform_startup_v1',passed=passed,basic=rows,
            platform_manifest_sha256=sha((self.folder/'platform_manifest.json').read_bytes()),
            actual_arrays_sha256=sha(arrays.read_bytes()),actual_arrays=arrays.name,
            segmentation_after_switch=self.segmentation_identity if passed else None)
        with stem.with_suffix('.json').open('x',encoding='utf-8') as f:json.dump(evidence,f,indent=2)
        print('Startup actual output evidence: '+str(stem),flush=True)

    def infer(self, image):
        switch_ms = self.switch('segmentation')
        try:
            codes, stats = SegmentationHardware.infer(self, image)
            stats['overlay_switch_ms'] = switch_ms
            return codes, stats
        except BaseException as exc:
            self.fault = type(exc).__name__+': '+str(exc); raise

    def basic(self, hu, params):
        op = params['operation']; switch_ms = self.switch(op)
        try:
            if op == 'window':
                self.overlay.axi_gpio_0.channel1.write(params['wl'] & 0xffff, 0xffff)
                self.overlay.axi_gpio_0.channel2.write(params['ww'] & 0xffff, 0xffff)
            if op == 'threshold':
                packed = ((params['upper'] & 0xffff) << 16) | (params['lower'] & 0xffff)
                self.overlay.axi_gpio_0.channel1.write(packed, 0xffffffff)
            # Exactly the existing standalone morphology path: PS prepares the
            # range mask; erosion/dilation/both morphology stages execute in PL.
            preparation_start = time.perf_counter()
            values = (np.where((hu >= params['lower']) & (hu <= params['upper']), 255, 0)
                      if op not in ('window', 'threshold') else hu)
            src, dst = self.buffers; src[:PIXELS] = values.astype(np.int32).reshape(-1).view(np.uint32)
            dst[:] = np.uint32(0xA5C3917E); src.flush(); dst.flush()
            preparation_ms = (time.perf_counter()-preparation_start)*1000
            start = time.perf_counter()
            self.dma.recvchannel.transfer(dst, nbytes=PIXELS*4)
            self.dma.sendchannel.transfer(src, nbytes=PIXELS*4)
            deadline = time.monotonic()+self.timeout
            while True:
                statuses = [int(self.dma.mmio.read(a)) for a in (0x04, 0x34)]
                if any(s & 0x770 for s in statuses): raise RuntimeError('DMA error: '+str(statuses))
                if all(s & 2 for s in statuses): break
                if time.monotonic() > deadline: raise TimeoutError('DMA timeout; owner/CMA retained, no retry')
                time.sleep(.0001)
            self.dma.sendchannel.wait(); self.dma.recvchannel.wait(); dst.invalidate()
            dma_ms = (time.perf_counter()-start)*1000
            if np.any(dst[PIXELS:] != np.uint32(0xA5C3917E)):
                raise RuntimeError('DMA guard changed')
            words = np.array(dst[:PIXELS], copy=True)
            if np.any(words & np.uint32(0xffffff00)): raise RuntimeError('Unexpected upper output bits')
            return (words & 255).astype(np.uint8).reshape(512, 512), dict(
                dma_ms=dma_ms, preparation_ms=preparation_ms, overlay_switch_ms=switch_ms,
                input_bytes=PIXELS*4, output_bytes=PIXELS*4, guard_verified=True,
                fclk0_mhz=float(self.clocks.fclk0_mhz), cpu_mhz=float(self.clocks.cpu_mhz),
                preprocessing='PS HU range mask' if op not in ('window', 'threshold') else 'PS int16 HU packing')
        except BaseException as exc:
            self.fault = type(exc).__name__+': '+str(exc); raise


class Service:
    def __init__(self, hardware, token, manifest_sha256):
        if not isinstance(token, str) or len(token) < 32: raise ValueError('Invalid token')
        self.hardware = hardware; self.token = token; self.manifest_sha256 = manifest_sha256
        self.gate = threading.Lock(); self.failed = None; self.requests = 0

    def health(self):
        return dict(schema='ct_platform_identity_v1', ready=self.failed is None and not self.hardware.fault,
            busy=self.gate.locked(), error=self.failed or self.hardware.fault,
            platform_manifest_sha256=self.manifest_sha256,
            active_operation=self.hardware.mode, active_bit_sha256=self.hardware.active_bit,
            segmentation=self.hardware.segmentation_identity, basic=self.hardware.manifest['basic'])

    def execute(self, route, raw, request_id, input_hash, supplied_params=None):
        if not re.fullmatch('[0-9a-f]{32}', request_id) or sha(raw) != input_hash:
            raise ValueError('Request/input binding mismatch')
        if route == '/v1/infer':
            if len(raw) != 65536: raise ValueError('Need UINT8 1x256x256 input')
            params = None
        elif route == '/v1/basic':
            if len(raw) != PIXELS*2: raise ValueError('Basic hardware requires native 512x512 int16 HU')
            params = parameters(supplied_params or {})
            if params['operation'] not in BASIC_OPERATIONS: raise ValueError('Operation has no packaged PL module')
        else: raise ValueError('Unknown operation')
        if not self.gate.acquire(False): raise BlockingIOError('FPGA busy; no concurrent command or retry')
        try:
            if self.failed or self.hardware.fault: raise RuntimeError('Faulted owner retained; shared board recovery required')
            try:
                start = time.perf_counter()
                if params is None:
                    head, stats = self.hardware.infer(np.frombuffer(raw, np.uint8).reshape(1,256,256))
                    payload = head.tobytes()
                    result = dict(schema='lung_live_result_v1', backend='FPGA_INT8',
                        bit_sha256=BIT_SHA, model_sha256=MODEL_SHA, shape=[1,256,256],
                        head_base64=base64.b64encode(payload).decode('ascii'), head_sha256=sha(payload), stats=stats)
                else:
                    image, stats = self.hardware.basic(np.frombuffer(raw, '<i2').reshape(512,512), params)
                    payload = image.tobytes(); row = self.hardware.manifest['basic'][params['operation']]
                    result = dict(schema='ct_basic_pl_result_v1', backend='FPGA_BASIC',
                        parameters=params, bit_sha256=row['bit_sha256'], hwh_sha256=row['hwh_sha256'],
                        platform_manifest_sha256=self.manifest_sha256, shape=[512,512],
                        image_base64=base64.b64encode(payload).decode('ascii'), output_sha256=sha(payload), stats=stats)
                self.requests += 1
                result.update(request_id=request_id, input_sha256=input_hash, sequence=self.requests,
                    service_ms=(time.perf_counter()-start)*1000)
                return result
            except BaseException as exc:
                self.failed = type(exc).__name__+': '+str(exc); raise
        finally: self.gate.release()


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def reply(self, code, value):
            raw=json.dumps(value,allow_nan=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(raw)
        def allowed(self):
            if self.headers.get('Origin') or not hmac.compare_digest(
                    self.headers.get('Authorization',''),'Bearer '+service.token):
                self.reply(403,dict(error='Unauthorized'));return False
            self.connection.settimeout(10);return True
        def do_GET(self):
            if not self.allowed():return
            if self.path!='/v1/platform':return self.reply(404,dict(error='Use platform-aware host'))
            self.reply(200,service.health())
        def do_POST(self):
            if not self.allowed():return
            try:
                expected={'/v1/infer':65536,'/v1/basic':PIXELS*2}.get(self.path)
                if (expected is None or self.headers.get('Transfer-Encoding') or
                    self.headers.get('Content-Length')!=str(expected) or
                    self.headers.get('Content-Type')!='application/octet-stream'):
                    raise ValueError('Wrong route/content type/input length')
                text=self.headers.get('X-Parameters','{}')
                if len(text)>1024:raise ValueError('Parameters too long')
                params=json.loads(text)
                if not isinstance(params,dict):raise ValueError('Need parameter object')
                value=service.execute(self.path,self.rfile.read(expected),self.headers.get('X-Request-ID',''),
                    self.headers.get('X-Input-SHA256',''),params)
                self.reply(200,value)
            except BlockingIOError as exc:self.reply(409,dict(error=str(exc)))
            except (ValueError,KeyError) as exc:self.reply(503 if service.failed else 400,dict(error=str(exc)))
            except Exception as exc:self.reply(503,dict(error=str(exc)))
    return Handler


def stop_idle_live_service(service):
    """Explicit handover of an existing healthy live_service, never fault recovery."""
    if not service.gate.acquire(False): raise RuntimeError('A request is active; wait for completion')
    try:
        if service.failed: raise RuntimeError('Faulted service retained; do not close/reload')
        service.hardware.network.close()  # Confirms no inflight/poisoned engine.
        service.server.shutdown(); service.server.server_close(); service.thread.join(timeout=2)
        service.hardware.lock_file.close()
    finally: service.gate.release()


def start(bind='0.0.0.0', port=8765, folder=None):
    folder=Path(folder or Path(__file__).resolve().parent).resolve()
    manifest=validate_package(folder)
    token=(folder/'platform_token.txt').read_text(encoding='utf-8').strip()
    server=ThreadingHTTPServer((bind,port),BaseHTTPRequestHandler);server.daemon_threads=False
    try:
        hardware=Hardware(folder)
        hardware.selftest_basic()
        service=Service(hardware,token,sha((folder/'platform_manifest.json').read_bytes()))
        server.RequestHandlerClass=handler_for(service)
    except BaseException:server.server_close();raise
    service.server=server;OWNERS.append(service)
    service.thread=threading.Thread(target=server.serve_forever,daemon=True);service.thread.start()
    print('CT PLATFORM READY: 12 basic PL checks + segmentation 21-node after-switch PASS.',flush=True)
    print('Keep this single owner alive; no other overlays/notebooks may access PL.',flush=True)
    return service
