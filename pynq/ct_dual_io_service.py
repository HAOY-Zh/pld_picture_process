"""Two unchanged PL models, exact native input, bounded plane cache and binary I/O."""
import json
import re
import socket
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import numpy as np
import ct_nodule_compact_service as parent
from dual_io_contract import INTERFACE,NODULE_ROUTE,SEG_ROUTE,PlaneCache,inflate_exact,digest,encode_response
from nodule_canvas_contract import shapes,validate_stats
from shared_segmentation_contract import SHAPE,BYTES,INTERFACE as SEG_INTERFACE


class Hardware(parent.Hardware):
    def shared_segmentation(self,image,observer=None):
        self.switch('shared_nodule')
        owner=self.shared_owner
        from shared_cnn.exact_io_runtime import ExactUNetPL
        from shared_cnn.exact_native_io import open_packer
        from shared_cnn.lung_nodule_shared import load_model
        from shared_cnn.lung_axil import LungAxil
        with owner._exclusive():
            previous=owner._nets.get('unet_original')
            if not isinstance(previous,ExactUNetPL):
                if previous is not None:
                    previous.close();del owner._nets['unet_original']
                model,params=load_model(self.folder/'shared_cnn','unet_original')
                if model['input_zero']!=0:raise ValueError('Unexpected segmentation padding zero')
                net=ExactUNetPL(owner.conv,owner.tensor,owner.allocate,model,params)
                # Retain ownership before checking; any conflict blocks reuse.
                owner._nets['unet_original']=net
                for buf in net.buffers.values():
                    address=LungAxil._buffer_address(buf)
                    for label,other_net in owner._nets.items():
                        if label=='unet_original':continue
                        for other in other_net.buffers.values():
                            start=LungAxil._buffer_address(other)
                            if address<start+other.nbytes and start<address+buf.nbytes:
                                owner.failed=True;raise RuntimeError('Cross-model CMA overlap')
                net.exact_packer=open_packer(self.folder/'shared_cnn','segmentation')
        return super().shared_segmentation(image,observer)

    def compact(self,image,height,width,observer=None):
        shapes(height,width)
        if image.dtype!=np.uint8 or image.shape!=(5,height,width)or not image.flags.c_contiguous:
            raise ValueError('Invalid five-plane quantized input')
        switch=self.switch('shared_nodule');owner=self.shared_owner;tick=time.perf_counter()
        try:
            with owner._exclusive():
                from shared_cnn.exact_io_runtime import ExactCanvasPL
                from shared_cnn.exact_native_io import open_packer
                from shared_cnn.lung_nodule_shared import load_model
                from shared_cnn.lung_axil import LungAxil
                net=owner._nets.get('nodule_canvas')
                if net is None or not isinstance(net,ExactCanvasPL)or(net.height,net.width)!=(height,width):
                    if net is not None:net.close();del owner._nets['nodule_canvas']
                    model,params=load_model(self.folder/'shared_cnn','nodule')
                    if model['input_zero']!=0:raise ValueError('Unexpected input padding zero')
                    net=ExactCanvasPL(owner.conv,owner.tensor,owner.allocate,model,params,height,width)
                    for buf in net.buffers.values():
                        address=LungAxil._buffer_address(buf)
                        for previous in owner._nets.values():
                            for other in previous.buffers.values():
                                start=LungAxil._buffer_address(other)
                                if address<start+other.nbytes and start<address+buf.nbytes:
                                    raise RuntimeError('Cross-model CMA overlap')
                    owner._nets['nodule_canvas']=net
                    net.exact_packer=open_packer(self.folder/'shared_cnn','nodule')
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


class Service(parent.Service):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.plane_cache=PlaneCache()

    def shared_health(self):
        value=super().shared_health()
        value.update(dual_io_interface=INTERFACE,dual_io_startup='all-nodes-and-native-layout-exact')
        return value

    def execute_dual(self,raw,rid,sha,hw=None,session=None,hashes=None,indices=None):
        if not re.fullmatch('[0-9a-f]{32}',rid):raise ValueError('Invalid request identity')
        if not self.gate.acquire(False):raise BlockingIOError('FPGA busy')
        try:
            if self.failed or self.hardware.fault:raise RuntimeError('Faulted owner retained')
            if hw is None:
                if len(raw)!=BYTES or digest(raw)!=sha:raise ValueError('Segmentation input hash/size mismatch')
                image=np.frombuffer(raw,np.uint8).reshape(SHAPE)
            else:
                shapes(*hw)
                raw=self.plane_cache.reconstruct(session,hashes,indices,raw,hw[0]*hw[1],sha)
                image=np.frombuffer(raw,np.uint8).reshape(5,*hw)
            # Input/cache errors above cannot have submitted any device command.
            try:
                began=time.perf_counter()
                head,stats=(self.hardware.shared_segmentation(image) if hw is None
                            else self.hardware.compact(image,*hw))
                target=SHAPE if hw is None else shapes(*hw)[1]
                if head.dtype!=np.uint8 or head.shape!=target:raise ValueError('Invalid physical full head')
                payload=head.tobytes();self.requests+=1
                result=dict(schema=INTERFACE,request_id=rid,input_sha256=sha,stats=stats,
                    sequence=self.requests,service_ms=(time.perf_counter()-began)*1000,
                    service_bundle_sha256=self.bundle_sha256,
                    state=dict(busy=False,faulted=False,dma_may_access=False))
                if hw is None:
                    row=self.hardware.manifest['shared_segmentation']
                    result.update(backend='FPGA_INT8',engine_interface=SEG_INTERFACE,bit_sha256=row['bit_sha256'],
                        model_sha256=row['model_sha256'],shape=list(SHAPE),head_sha256=digest(payload),shared_segmentation=row)
                else:
                    result.update(backend='FPGA_NODULE_SHARED',input_shape=[5,*hw],output_shape=list(target),
                        output_sha256=digest(payload),shared_nodule=self.hardware.manifest['shared_nodule'],
                        cache_session=session,cached_plane_hashes=self.plane_cache.keys(),
                        cpu_plane_cache_bytes=sum(map(len,self.plane_cache.planes.values())))
                return encode_response(result,payload)
            except BaseException as exc:
                self.failed=type(exc).__name__+': '+str(exc)
                raise
        finally:self.gate.release()


def handler_for(service):
    previous=parent.handler_for(service)
    class Handler(previous):
        def do_POST(self):
            if self.path not in (NODULE_ROUTE,SEG_ROUTE):return super().do_POST()
            if not self.allowed():self.close_connection=True;return
            try:
                size=int(self.headers.get('Content-Length','-1'))
                if (self.headers.get('Transfer-Encoding') or not 0<size<=5*1024*1024+8192
                        or self.headers.get('Content-Type')!='application/octet-stream'):
                    raise ValueError('Invalid binary framing')
                kwargs={}
                if self.path==SEG_ROUTE:
                    expected=BYTES
                else:
                    hw=json.loads(self.headers.get('X-Canvas-Shape','[]'))
                    if not isinstance(hw,list) or len(hw)!=2:raise ValueError('Need H/W')
                    shapes(*hw)
                    indices=json.loads(self.headers.get('X-Sent-Plane-Indices','null'))
                    if not isinstance(indices,list) or len(indices)>5:raise ValueError('Invalid indices')
                    expected=len(indices)*hw[0]*hw[1]
                    kwargs=dict(hw=hw,session=self.headers.get('X-Cache-Session',''),
                        hashes=json.loads(self.headers.get('X-Plane-Hashes','null')),indices=indices)
                packed=self.rfile.read(size)
                if len(packed)!=size:raise ValueError('Truncated input')
                raw=inflate_exact(packed,expected)
                output=service.execute_dual(raw,self.headers.get('X-Request-ID',''),
                    self.headers.get('X-Input-SHA256',''),**kwargs)
                self.send_response(200);self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(len(output)));self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(output)
            except BlockingIOError as exc:self.reply(409,dict(error=str(exc)))
            except Exception as exc:
                self.close_connection=True;self.reply(503 if service.failed else 400,dict(error=str(exc)))
    return Handler


def start(bind='0.0.0.0',port=8766,folder=None):
    folder=Path(folder).resolve();parent.parent.parent.validate_package(folder)
    server=ThreadingHTTPServer((bind,port),BaseHTTPRequestHandler)
    try:
        hardware=Hardware(folder)
        hardware.selftest_basic();hardware.selftest_ldct();hardware.selftest_roi()
        hardware.selftest_shared();hardware.selftest_shared_segmentation()
        hardware.selftest_canvas();hardware.selftest_compact()
        service=Service(hardware,(folder/'platform_token.txt').read_text().strip(),digest((folder/'platform_manifest.json').read_bytes()))
        service.startup_verified=service.roi_verified=service.shared_verified=True
        server.RequestHandlerClass=handler_for(service)
    except BaseException:server.server_close();raise
    service.server=server;parent.parent.parent.previous.roi.base.OWNERS.append(service)
    service.thread=threading.Thread(target=server.serve_forever,daemon=True);service.thread.start()
    print('DUAL EXACT I/O READY: all U-Net nodes, nodule full heads and native CMA exact.',flush=True)
    return service
