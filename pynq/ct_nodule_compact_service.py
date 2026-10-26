"""Native direct-CMA pack plus compact binary transport, same guarded PL graph."""
import json
import re
import socket
import threading
import time
import uuid
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import numpy as np
import ct_nodule_canvas_service as parent
from nodule_compact_contract import INTERFACE,ROUTE,MAX_INPUT,shapes,inflate,digest,encode_response,validate_stats


class Hardware(parent.Hardware):
    def compact(self,image,height,width,observer=None):
        shapes(height,width)
        if image.dtype!=np.uint8 or image.shape!=(5,height,width)or not image.flags.c_contiguous:
            raise ValueError('Invalid five-plane quantized input')
        switch=self.switch('shared_nodule');owner=self.shared_owner;tick=time.perf_counter()
        try:
            with owner._exclusive():
                from shared_cnn.nodule_canvas_packed import PackedCanvasPL
                from shared_cnn.nodule_quantized_pack import compile_pack,QuantizedPacker
                from shared_cnn.lung_nodule_shared import load_model
                from shared_cnn.lung_axil import LungAxil
                net=owner._nets.get('nodule_canvas')
                if net is None or not isinstance(net,PackedCanvasPL)or(net.height,net.width)!=(height,width):
                    if net is not None:net.close();del owner._nets['nodule_canvas']
                    model,params=load_model(self.folder/'shared_cnn','nodule')
                    if model['input_zero']!=0:raise ValueError('Unexpected input padding zero')
                    net=PackedCanvasPL(owner.conv,owner.tensor,owner.allocate,model,params,height,width)
                    for buf in net.buffers.values():
                        address=LungAxil._buffer_address(buf)
                        for previous in owner._nets.values():
                            for other in previous.buffers.values():
                                start=LungAxil._buffer_address(other)
                                if address<start+other.nbytes and start<address+buf.nbytes:
                                    raise RuntimeError('Cross-model CMA overlap')
                    owner._nets['nodule_canvas']=net
                    net.quantized_packer=QuantizedPacker(compile_pack(self.folder/'shared_cnn'))
                setup=(time.perf_counter()-tick)*1000
                before=float(self.clocks.fclk0_mhz);start=time.perf_counter();previous=owner._current
                head,stats=net.predict_quantized(image,observer)
                owner._idle();owner._clock_check();owner._current='nodule_canvas'
                stats.update(selected_model='nodule_canvas',previous_model=previous,model_changed=previous!='nodule_canvas',
                    overlay_reloaded=False,shared_call_ms=(time.perf_counter()-start)*1000,
                    overlay_switch_ms=switch,canvas_setup_ms=setup,actual_pl_mhz_before=before,
                    actual_pl_mhz_after=float(self.clocks.fclk0_mhz),cpu_mhz=float(self.clocks.cpu_mhz),
                    canvas_hw=[height,width],input_pack='native_quantized_5ch_direct_CMA')
                validate_stats(stats)
                return head,stats
        except BaseException as exc:
            owner.failed=True;self.fault=type(exc).__name__+': '+str(exc);raise

    def selftest_compact(self):
        from shared_cnn.nodule_canvas import lower_canvas
        from shared_cnn.lung_axil_smoke import feature_words
        arrays={}
        for h,w in ((128,128),(256,320)):
            image=np.random.default_rng(h+w).integers(0,256,(5,h,w),dtype=np.uint8)
            expected,oldstats=self.canvas(lower_canvas(image),h,w)
            actual,stats=self.compact(image,h,w)
            np.testing.assert_array_equal(actual,expected)
            buf=self.shared_owner._nets['nodule_canvas'].buffers['input']
            wanted=feature_words(lower_canvas(image),0);buf.invalidate()
            np.testing.assert_array_equal(np.asarray(buf)[:len(wanted)],wanted)
            arrays[f'input_{h}_{w}']=image;arrays[f'old_{h}_{w}']=expected;arrays[f'new_{h}_{w}']=actual
        stem=self.folder/('compact_startup_'+uuid.uuid4().hex)
        np.savez_compressed(str(stem)+'.npz',**arrays)
        Path(str(stem)+'.json').write_text(json.dumps(dict(passed=True,full_head_exact=True,input_CMA_words_exact=True,
            arrays_sha256=digest(Path(str(stem)+'.npz').read_bytes())),indent=2))
        self.compact_verified=True


class Service(parent.Service):
    def shared_health(self):
        value=super().shared_health()
        value.update(compact_interface=INTERFACE,compact_startup='full-head-and-CMA-pack-exact'
            if getattr(self.hardware,'compact_verified',False)else'not-run')
        return value

    def execute_compact(self,raw,rid,sha,height,width):
        _,target=shapes(height,width)
        if len(raw)!=5*height*width or not re.fullmatch('[0-9a-f]{32}',rid)or digest(raw)!=sha:
            raise ValueError('Compact input framing/hash mismatch')
        if not self.gate.acquire(False):raise BlockingIOError('FPGA busy; no concurrent command')
        try:
            if self.failed or self.hardware.fault:raise RuntimeError('Faulted owner retained')
            try:
                started=time.perf_counter()
                head,stats=self.hardware.compact(np.frombuffer(raw,np.uint8).reshape(5,height,width),height,width)
                if head.dtype!=np.uint8 or head.shape!=target:raise ValueError('Wrong real head')
                payload=head.tobytes();self.requests+=1
                meta=dict(schema=INTERFACE,backend='FPGA_NODULE_SHARED',request_id=rid,input_sha256=sha,
                    input_shape=[5,height,width],output_shape=list(target),output_sha256=digest(payload),stats=stats,
                    sequence=self.requests,service_ms=(time.perf_counter()-started)*1000,
                    shared_nodule=self.hardware.manifest['shared_nodule'],service_bundle_sha256=self.bundle_sha256,
                    state=dict(busy=False,faulted=False,dma_may_access=False))
                return encode_response(meta,payload)
            except BaseException as exc:self.failed=type(exc).__name__+': '+str(exc);raise
        finally:self.gate.release()


def handler_for(service):
    previous=parent.handler_for(service)
    class Handler(previous):
        protocol_version='HTTP/1.1'
        def setup(self):
            super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        def do_POST(self):
            if self.path!=ROUTE:return super().do_POST()
            if not self.allowed():self.close_connection=True;return
            try:
                size=int(self.headers.get('Content-Length','-1'))
                if self.headers.get('Transfer-Encoding')or not 0<size<=MAX_INPUT+8192 or self.headers.get('Content-Type')!='application/octet-stream':
                    raise ValueError('Invalid compact framing')
                hw=json.loads(self.headers.get('X-Canvas-Shape','[]'))
                if not isinstance(hw,list)or len(hw)!=2:raise ValueError('Need canvas H/W')
                shapes(*hw);packed=self.rfile.read(size)
                if len(packed)!=size:raise ValueError('Truncated compact input')
                raw=inflate(packed,5*hw[0]*hw[1])
                output=service.execute_compact(raw,self.headers.get('X-Request-ID',''),self.headers.get('X-Input-SHA256',''),*hw)
                self.send_response(200);self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(len(output)));self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(output)
            except BlockingIOError as exc:self.reply(409,dict(error=str(exc)))
            except Exception as exc:
                self.close_connection=True;self.reply(503 if service.failed else 400,dict(error=str(exc)))
    return Handler


def start(bind='0.0.0.0',port=8766,folder=None):
    folder=Path(folder).resolve();parent.parent.validate_package(folder)
    server=ThreadingHTTPServer((bind,port),BaseHTTPRequestHandler)
    try:
        hardware=Hardware(folder);hardware.selftest_basic();hardware.selftest_ldct();hardware.selftest_roi()
        hardware.selftest_shared();hardware.selftest_shared_segmentation();hardware.selftest_canvas();hardware.selftest_compact()
        service=Service(hardware,(folder/'platform_token.txt').read_text().strip(),digest((folder/'platform_manifest.json').read_bytes()))
        service.startup_verified=service.roi_verified=service.shared_verified=True
        server.RequestHandlerClass=handler_for(service)
    except BaseException:server.server_close();raise
    service.server=server;parent.parent.previous.roi.base.OWNERS.append(service)
    service.thread=threading.Thread(target=server.serve_forever,daemon=True);service.thread.start()
    print('COMPACT READY: original full heads and direct-CMA input EXACT PASS.',flush=True)
    return service
