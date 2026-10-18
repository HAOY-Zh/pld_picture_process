"""Resident six-operation basic PL, lossless binary wire format and HU cache.

The original service/segmentation contracts remain available. Faults retain the
same single owner and CMA buffers; no automatic retry, reset or CPU fallback.
"""
import base64
from collections import OrderedDict
import json
from pathlib import Path
import re
import socket
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import ct_platform_service as base
from ct_basic_processing import parameters

TRANSPORT='ct_basic_zlib_cache_v1'
MODES={op:i for i,op in enumerate(base.BASIC_OPERATIONS)}
MAX_RAW=base.PIXELS*2


def decompress_exact(raw, size):
    decoder=zlib.decompressobj()
    result=decoder.decompress(raw,size+1)
    if (len(result)!=size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError('Compressed input length/framing mismatch')
    return result


class CacheMiss(ValueError): pass


class Hardware(base.Hardware):
    def _dma_status(self):
        if self.dma is not None and getattr(self,'_basic_transfer_started',True) is False:
            # PYNQ permits a channel's very first transfer before its Idle bit
            # has asserted. This owner has never armed either BTT register.
            # No outstanding DMA can reference the freshly allocated buffers.
            for offset in (0x04,0x34):
                status=int(self.dma.mmio.read(offset))
                if status&0x770:raise RuntimeError('Fresh DMA error: 0x%x'%status)
            return
        return super()._dma_status()

    def switch(self, mode):
        if mode=='segmentation':return super().switch(mode)
        if mode not in MODES:raise ValueError('Unknown basic mode')
        if self.fault:raise RuntimeError('Faulted owner retained: '+self.fault)
        try:
            if self.mode not in MODES or self.dma is None:
                elapsed=super().switch(mode)
                self._basic_transfer_started=False
            else:
                self._dma_status()  # Mode changes ONLY after both DMA paths idle.
                elapsed=0.
            self.overlay.axi_gpio_1.channel1.write(MODES[mode],7)
            self.mode=mode
            return elapsed
        except BaseException as exc:
            self.fault=type(exc).__name__+': '+str(exc);raise

    def basic(self,hu,params):
        op=params['operation'];switch_ms=self.switch(op)
        try:
            start=time.perf_counter()
            self.overlay.axi_gpio_0.channel1.write(
                (params['ww']<<16)|(params['wl']&65535),0xffffffff)
            self.overlay.axi_gpio_0.channel2.write(
                ((params['upper']&65535)<<16)|(params['lower']&65535),0xffffffff)
            values=(np.where((hu>=params['lower'])&(hu<=params['upper']),255,0)
                    if op not in ('window','threshold') else hu)
            src,dst=self.buffers
            src[:base.PIXELS]=values.astype(np.int32).reshape(-1).view(np.uint32)
            dst[:]=np.uint32(0xA5C3917E);src.flush();dst.flush()
            preparation_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter()
            # Set BEFORE either transfer call: even a partially failed call may
            # have armed hardware, so idle checks can never be skipped afterward.
            self._basic_transfer_started=True
            self.dma.recvchannel.transfer(dst,nbytes=base.PIXELS*4)
            self.dma.sendchannel.transfer(src,nbytes=base.PIXELS*4)
            deadline=time.monotonic()+self.timeout
            while True:
                statuses=[int(self.dma.mmio.read(a)) for a in (0x04,0x34)]
                if any(s&0x770 for s in statuses):raise RuntimeError('DMA error: '+str(statuses))
                if all(s&2 for s in statuses):break
                if time.monotonic()>deadline:raise TimeoutError('DMA timeout; owner retained, no retry')
                time.sleep(.0001)
            self.dma.sendchannel.wait();self.dma.recvchannel.wait();dst.invalidate()
            dma_ms=(time.perf_counter()-start)*1000
            if np.any(dst[base.PIXELS:]!=np.uint32(0xA5C3917E)):raise RuntimeError('DMA guard changed')
            words=np.array(dst[:base.PIXELS],copy=True)
            if np.any(words & np.uint32(0xffffff00)):raise RuntimeError('Unexpected upper output bits')
            return (words&255).astype(np.uint8).reshape(512,512),dict(
                dma_ms=dma_ms,preparation_ms=preparation_ms,overlay_switch_ms=switch_ms,
                input_bytes=base.PIXELS*4,output_bytes=base.PIXELS*4,guard_verified=True,
                fclk0_mhz=float(self.clocks.fclk0_mhz),cpu_mhz=float(self.clocks.cpu_mhz),
                shared_basic_overlay=True,mode=MODES[op],
                preprocessing='PS HU range mask' if op not in ('window','threshold') else 'PS int16 HU packing')
        except BaseException as exc:
            self.fault=type(exc).__name__+': '+str(exc);raise


class Service(base.Service):
    def __init__(self,*args):
        super().__init__(*args);self.inputs=OrderedDict();self.cache_lock=threading.Lock()

    def health(self):
        return dict(super().health(),basic_transport=TRANSPORT,shared_basic_overlay=True,
                    input_cache_capacity=8,startup_ms=getattr(self,'startup_ms',None))

    def execute_fast(self,wire,rid,input_hash,supplied,encoding):
        if not re.fullmatch('[0-9a-f]{32}',rid) or not re.fullmatch('[0-9a-f]{64}',input_hash):
            raise ValueError('Request/hash format mismatch')
        p=parameters(supplied)
        if p['operation'] not in MODES:raise ValueError('Unsupported operation')
        if encoding=='cache':
            if wire:raise ValueError('Cache reference must have an empty body')
            with self.cache_lock:raw=self.inputs.get(input_hash)
            if raw is None:raise CacheMiss('Input cache miss; no hardware command issued')
        elif encoding=='zlib-int16':
            if not 0<len(wire)<=MAX_RAW+1024:raise ValueError('Oversized wire input')
            raw=decompress_exact(wire,MAX_RAW)
            if base.sha(raw)!=input_hash:raise ValueError('Input hash mismatch')
        else:raise ValueError('Unsupported input encoding')
        if not self.gate.acquire(False):raise BlockingIOError('FPGA busy; no concurrent command')
        try:
            if self.failed or self.hardware.fault:raise RuntimeError('Faulted owner retained')
            # Cache insertion after input validation and ownership; a cache hit
            # saves transmission only. It NEVER reuses a previously computed output.
            with self.cache_lock:
                self.inputs[input_hash]=raw;self.inputs.move_to_end(input_hash)
                while len(self.inputs)>8:self.inputs.popitem(last=False)
            try:
                start=time.perf_counter()
                image,stats=self.hardware.basic(np.frombuffer(raw,'<i2').reshape(512,512),p)
                payload=image.tobytes();encode_start=time.perf_counter()
                body=zlib.compress(payload,1)
                encode_ms=(time.perf_counter()-encode_start)*1000
                row=self.hardware.manifest['basic'][p['operation']];self.requests+=1
                receipt=dict(schema='ct_basic_pl_result_v1',backend='FPGA_BASIC',
                    parameters=p,request_id=rid,input_sha256=input_hash,output_sha256=base.sha(payload),
                    bit_sha256=row['bit_sha256'],hwh_sha256=row['hwh_sha256'],
                    platform_manifest_sha256=self.manifest_sha256,shape=[512,512],stats=stats,
                    sequence=self.requests,service_ms=(time.perf_counter()-start)*1000,
                    transport=dict(schema=TRANSPORT,input_cache_hit=encoding=='cache',
                        request_body_bytes=len(wire),response_body_bytes=len(body),
                        raw_input_bytes=MAX_RAW,raw_output_bytes=base.PIXELS,encode_ms=encode_ms))
                return body,receipt
            except BaseException as exc:self.failed=type(exc).__name__+': '+str(exc);raise
        finally:self.gate.release()


def handler_for(service):
    parent=base.handler_for(service)
    class Handler(parent):
        def do_POST(self):
            if self.path!='/v2/basic':return super().do_POST()
            if not self.allowed():return
            try:
                count=int(self.headers.get('Content-Length','-1'))
                if (self.headers.get('Transfer-Encoding') or not 0<=count<=MAX_RAW+1024 or
                        self.headers.get('Content-Type')!='application/octet-stream'):
                    raise ValueError('Invalid binary request framing')
                text=self.headers.get('X-Parameters','{}')
                if len(text)>1024:raise ValueError('Parameters too long')
                p=json.loads(text)
                if not isinstance(p,dict):raise ValueError('Need parameters object')
                wire=self.rfile.read(count)
                if len(wire)!=count:raise ValueError('Truncated request')
                body,receipt=service.execute_fast(wire,self.headers.get('X-Request-ID',''),
                    self.headers.get('X-Input-SHA256',''),p,self.headers.get('X-Input-Encoding',''))
                header=base64.b64encode(json.dumps(receipt,separators=(',',':'),allow_nan=False).encode()).decode()
                self.send_response(200);self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(len(body)));self.send_header('X-Receipt',header)
                self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
            except CacheMiss as exc:self.reply(409,dict(error=str(exc),error_code='INPUT_CACHE_MISS'))
            except BlockingIOError as exc:self.reply(409,dict(error=str(exc),error_code='BUSY'))
            except (ValueError,KeyError,zlib.error) as exc:self.reply(503 if service.failed else 400,dict(error=str(exc)))
            except Exception as exc:self.reply(503,dict(error=str(exc)))
    return Handler


def stop_idle_platform_service(service):
    """Healthy idle handover only; does not recover from a hardware fault."""
    if not service.gate.acquire(False):raise RuntimeError('FPGA task active; wait')
    try:
        if service.failed or service.hardware.fault:raise RuntimeError('Faulted service retained; do not reload')
        service.hardware.release_idle()
        service.server.shutdown();service.server.server_close();service.thread.join(timeout=2)
        service.hardware.lock_file.close()
    finally:service.gate.release()


def upgrade(service,folder=None):
    """Validate the complete new release before stopping a healthy old owner."""
    folder=Path(folder or Path(__file__).resolve().parent).resolve()
    manifest=base.validate_package(folder)
    if (manifest.get('basic_transport')!=TRANSPORT or
            len({r['bit_sha256'] for r in manifest['basic'].values()})!=1):
        raise ValueError('New package is not the expected shared-basic release')
    if (folder/'platform_token.txt').read_text(encoding='utf-8').strip()!=service.token:
        raise ValueError('New package credentials differ; keeping current service')
    stop_idle_platform_service(service)
    return start(folder=folder)


def start(bind='0.0.0.0',port=8765,folder=None):
    startup_start=time.perf_counter()
    folder=Path(folder or Path(__file__).resolve().parent).resolve()
    manifest=base.validate_package(folder)
    if manifest.get('basic_transport')!=TRANSPORT:raise ValueError('Not a fast platform package')
    rows=list(manifest['basic'].values())
    if len({r['bit_sha256'] for r in rows})!=1:raise ValueError('Need one shared basic bit')
    token=(folder/'platform_token.txt').read_text(encoding='utf-8').strip()
    class Server(ThreadingHTTPServer):
        def get_request(self):
            sock,address=super().get_request();sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            return sock,address
    server=Server((bind,port),BaseHTTPRequestHandler)
    try:
        hardware=Hardware(folder);hardware.selftest_basic()
        # Startup includes both complete self-tests, then preloads the basic
        # overlay so the first interactive basic request does not reconfigure.
        hardware.switch('window')
        service=Service(hardware,token,base.sha((folder/'platform_manifest.json').read_bytes()))
        service.startup_ms=(time.perf_counter()-startup_start)*1000
        server.RequestHandlerClass=handler_for(service)
    except BaseException:server.server_close();raise
    service.server=server;base.OWNERS.append(service)
    service.thread=threading.Thread(target=server.serve_forever,daemon=True);service.thread.start()
    print('CT FAST PLATFORM READY: shared basic bit + 12 actual basic checks + segmentation PASS.',flush=True)
    print('Basic bit preloaded; service startup %.1f ms (not counted as interactive latency).'%service.startup_ms,flush=True)
    print('Same basic bit across six methods; lossless binary wire and bounded input cache.',flush=True)
    return service
