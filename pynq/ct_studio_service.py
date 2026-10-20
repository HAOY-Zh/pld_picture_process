"""One owner for the preserved platform plus LDCT. Integration awaits board acceptance."""
import base64
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import ct_platform_service as base
import ct_platform_fast_service as fast
from ldct_contract import METHODS, shape_checked, validate_release
from ldct_processor import LDCTProcessor


def validate_package(folder):
    folder = Path(folder).resolve()
    manifest = base.validate_package(folder)
    _, descriptor = validate_release(folder/'ldct')
    if manifest.get('ldct') != descriptor:
        raise ValueError('Platform/LDCT manifest relationship differs')
    if set(manifest.get('basic_clocks_mhz',{})) != set(base.BASIC_OPERATIONS):
        raise ValueError('Per-module basic clocks missing')
    return manifest


class LDCTHardwareMixin:
    @staticmethod
    def _check_clock(actual,expected,tolerance):
        if not np.isfinite([actual,expected]).all() or abs(float(actual)-float(expected))>tolerance:
            raise RuntimeError('Actual clock differs from the declared module clock')

    def _dma_status(self):
        processor = getattr(self, 'ldct_processor', None)
        if processor is not None:
            return processor.assert_idle()
        return super()._dma_status()

    def release_idle(self):
        if self.fault:
            raise RuntimeError('Faulted owner retained; no release or reload')
        processor = getattr(self, 'ldct_processor', None)
        if processor is not None:
            processor.assert_idle()
        super().release_idle()
        self.ldct_processor = None

    def switch(self, mode):
        if self.fault:
            raise RuntimeError('Faulted owner retained: '+self.fault)
        if mode != 'ldct':
            try:
                previous = self.mode
                elapsed = super().switch(mode)
                if mode != 'segmentation':
                    expected = float(self.manifest['basic_clocks_mhz'][mode])
                    if previous not in base.BASIC_OPERATIONS or elapsed:
                        self.clocks.fclk0_mhz = expected
                    self._check_clock(float(self.clocks.fclk0_mhz),expected,.5)
                self._check_clock(float(self.clocks.cpu_mhz),self.cpu_mhz,.05)
                return elapsed
            except BaseException as exc:
                self.fault = type(exc).__name__+': '+str(exc); raise
        if self.mode == 'ldct':
            return 0.
        start = time.perf_counter()
        try:
            # Validate ALL files before releasing a healthy previous module.
            validate_package(self.folder)
            self.release_idle()
            row = self.manifest['ldct']
            self.overlay = self.pynq.Overlay(str(self.folder/'ldct/overlay/ct_denoise_tile.bit'))
            self.clocks.fclk0_mhz = row['clock_mhz']
            self._check_clock(float(self.clocks.fclk0_mhz),row['clock_mhz'],.5)
            self.dma = self.overlay.axi_dma_0
            for _ in range(2):
                self.buffers.append(self.pynq.allocate(shape=(4096,),dtype=np.uint32))
            gpio = self.overlay.axi_gpio_0
            gpio.write(0,1<<31); time.sleep(.001); gpio.write(0,0)
            self.ldct_processor = LDCTProcessor(dma=self.dma,gpio=gpio,
                tx_buffer=self.buffers[0],rx_buffer=self.buffers[1],model_dir=self.folder/'ldct/model',
                read_clock_mhz=lambda:float(self.clocks.fclk0_mhz),
                expected_clock_mhz=row['clock_mhz'],timeout_s=self.timeout)
            self.ldct_processor.assert_idle()
            self._check_clock(float(self.clocks.cpu_mhz),self.cpu_mhz,.05)
            self.mode = 'ldct'; self.active_bit = row['bit_sha256']; self.switches += 1
            return (time.perf_counter()-start)*1000
        except BaseException as exc:
            self.fault = type(exc).__name__+': '+str(exc); raise

    def denoise(self, image, method):
        switch_ms = self.switch('ldct')
        try:
            result = self.ldct_processor.process_slice(image, method)
            self.ldct_processor.assert_idle()
            self._check_clock(float(self.clocks.cpu_mhz),self.cpu_mhz,.05)
            result.update(overlay_switch_ms=switch_ms,cpu_mhz=float(self.clocks.cpu_mhz))
            return result
        except BaseException as exc:
            self.fault = type(exc).__name__+': '+str(exc); raise

    def selftest_ldct(self):
        folder = self.folder/'ldct'
        cases = json.loads((folder/'fixtures/cases.json').read_text())
        arrays = {}; checks = []
        stem = self.folder/('studio_ldct_startup_'+uuid.uuid4().hex)
        try:
            for case in cases:
                image = np.load(folder/case['input'], allow_pickle=False)
                expected = np.load(folder/case['expected'], allow_pickle=False)
                result = self.denoise(image,case['method'])
                arrays[case['name']] = result['hu']
                np.testing.assert_array_equal(result['hu'],expected)
                checks.append(dict(case=case['name'],stats={k:v for k,v in result.items() if k!='hu'}))
            self.switch('segmentation')  # Re-download and verify all 21 real nodes.
            arrays['segmentation_after_ldct']=self.last_startup_head
        except BaseException as exc:
            self.fault = type(exc).__name__+': '+str(exc); raise
        finally:
            with stem.with_suffix('.npz').open('xb') as f:
                np.savez_compressed(f,**arrays)
            with stem.with_suffix('.json').open('x') as f:
                json.dump(dict(passed=self.fault is None,checks=checks,ldct=self.manifest['ldct'],
                    error=self.fault,arrays_file=stem.with_suffix('.npz').name,
                    platform_manifest_sha256=base.sha((self.folder/'platform_manifest.json').read_bytes()),
                    segmentation_after_ldct=getattr(self,'segmentation_identity',None),
                    arrays_sha256=base.sha(stem.with_suffix('.npz').read_bytes())),f,indent=2)


class ClassicHardware(LDCTHardwareMixin, base.Hardware): pass
class FastHardware(LDCTHardwareMixin, fast.Hardware): pass


class Service(fast.Service):
    def health(self):
        result = base.Service.health(self)
        if self.hardware.manifest.get('basic_transport'):
            result.update(basic_transport=fast.TRANSPORT,shared_basic_overlay=True,input_cache_capacity=8)
        result['ldct'] = self.hardware.manifest['ldct']
        processor = getattr(self.hardware,'ldct_processor',None)
        result['ldct_state'] = processor.get_state() if processor else None
        result['integration_board_acceptance'] = 'startup-selftests-passed' if getattr(self,'startup_verified',False) else 'not-run'
        return result

    def execute_ldct(self, raw, rid, digest, params):
        if (not re.fullmatch('[0-9a-f]{32}',rid) or hashlib.sha256(raw).hexdigest()!=digest
                or not isinstance(params,dict) or set(params)!={'shape','method'}):
            raise ValueError('LDCT request framing/hash mismatch')
        shape = shape_checked(params['shape']); method = params['method']
        if not isinstance(method,str) or method not in METHODS or len(raw)!=int(np.prod(shape))*2:
            raise ValueError('LDCT method/input length mismatch')
        if not self.gate.acquire(False):
            raise BlockingIOError('FPGA busy; no concurrent command or retry')
        try:
            if self.failed or self.hardware.fault:
                raise RuntimeError('Faulted owner retained')
            try:
                start = time.perf_counter()
                result = self.hardware.denoise(np.frombuffer(raw,'<i2').reshape(shape), method)
                image = result.pop('hu')
                if image.shape!=shape or image.dtype!=np.int16:
                    raise ValueError('Unexpected LDCT output shape/type')
                payload = np.ascontiguousarray(image,dtype='<i2').tobytes()
                processor = self.hardware.ldct_processor
                processor.assert_idle()
                self.requests += 1; row = self.hardware.manifest['ldct']
                return dict(schema='ct_studio_ldct_result_v1',backend='FPGA_LDCT',parameters=params,
                    shape=list(shape),hu_base64=base64.b64encode(payload).decode(),
                    request_id=rid,input_sha256=digest,output_sha256=base.sha(payload),
                    platform_manifest_sha256=self.manifest_sha256,sequence=self.requests,
                    service_ms=(time.perf_counter()-start)*1000,stats=result,state=processor.get_state(),
                    **{k:row[k] for k in ('model_id','weights_sha256','bit_sha256','hwh_sha256')})
            except BaseException as exc:
                self.failed=type(exc).__name__+': '+str(exc); raise
        finally:
            self.gate.release()


def handler_for(service):
    parent = fast.handler_for(service)
    class Handler(parent):
        def do_POST(self):
            if self.path != '/v1/ldct':
                if self.path == '/v2/basic' and not service.hardware.manifest.get('basic_transport'):
                    if not self.allowed(): return
                    return self.reply(404,dict(error='Classic platform uses /v1/basic'))
                return super().do_POST()
            if not self.allowed(): return
            try:
                count = int(self.headers.get('Content-Length','-1'))
                if (self.headers.get('Transfer-Encoding') or not 0<count<=512*512*2
                        or self.headers.get('Content-Type')!='application/octet-stream'):
                    raise ValueError('Invalid LDCT binary framing')
                text=self.headers.get('X-Parameters','{}')
                if len(text)>1024: raise ValueError('Parameters too long')
                raw=self.rfile.read(count)
                if len(raw)!=count: raise ValueError('Truncated LDCT input')
                value=service.execute_ldct(raw,self.headers.get('X-Request-ID',''),
                    self.headers.get('X-Input-SHA256',''),json.loads(text))
                self.reply(200,value)
            except BlockingIOError as exc: self.reply(409,dict(error=str(exc)))
            except (ValueError,KeyError) as exc: self.reply(503 if service.failed else 400,dict(error=str(exc)))
            except Exception as exc: self.reply(503,dict(error=str(exc)))
    return Handler


def start(bind='0.0.0.0',port=8765,folder=None):
    folder=Path(folder or Path(__file__).resolve().parent).resolve()
    manifest=validate_package(folder)
    token=(folder/'platform_token.txt').read_text().strip()
    server=ThreadingHTTPServer((bind,port),BaseHTTPRequestHandler)
    try:
        hardware=(FastHardware if manifest.get('basic_transport') else ClassicHardware)(folder)
        hardware.selftest_basic(); hardware.selftest_ldct()
        service=Service(hardware,token,base.sha((folder/'platform_manifest.json').read_bytes()))
        service.startup_verified=True
        server.RequestHandlerClass=handler_for(service)
    except BaseException: server.server_close(); raise
    service.server=server; base.OWNERS.append(service)
    service.thread=threading.Thread(target=server.serve_forever,daemon=True); service.thread.start()
    print('CT STUDIO READY: basic + segmentation + LDCT actual startup checks PASS.',flush=True)
    return service


def upgrade(service,folder):
    folder=Path(folder).resolve(); validate_package(folder)
    if (folder/'platform_token.txt').read_text().strip()!=service.token:
        raise ValueError('Credentials differ; old owner kept')
    fast.stop_idle_platform_service(service)
    return start(folder=folder)
