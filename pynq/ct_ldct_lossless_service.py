"""Optional LDCT transport extension. All original PL algorithms are reused."""
import base64
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
import numpy as np
import ct_dual_io_service as parent
from dual_io_contract import inflate_exact,encode_response,decode_response,digest

INTERFACE='ct_ldct_zlib_v1'
ROUTE='/v3/ldct'


class Service(parent.Service):
    def shared_health(self):
        return dict(super().shared_health(),ldct_lossless_interface=INTERFACE)

    def execute_ldct_binary(self,raw,rid,sha,parameters):
        # Original implementation checks shape, request identity and all PL
        # outputs, and owns the exclusive hardware lock. Never submit twice.
        value=self.execute_ldct(raw,rid,sha,parameters)
        head=base64.b64decode(value.pop('hu_base64'),validate=True)
        if len(head)!=len(raw)or digest(head)!=value['output_sha256']:
            self.failed='Unexpected LDCT output framing'
            raise ValueError(self.failed)
        value.update(ldct_lossless_interface=INTERFACE,service_bundle_sha256=self.bundle_sha256)
        return encode_response(value,head)


def handler_for(service):
    previous=parent.handler_for(service)
    class Handler(previous):
        def do_POST(self):
            if self.path!=ROUTE:return super().do_POST()
            if not self.allowed():self.close_connection=True;return
            try:
                size=int(self.headers.get('Content-Length','-1'))
                if (self.headers.get('Transfer-Encoding')or not 0<size<=512*512*2+8192 or
                        self.headers.get('Content-Type')!='application/octet-stream'):
                    raise ValueError('Invalid lossless LDCT frame')
                text=self.headers.get('X-Parameters','')
                if len(text)>1024:raise ValueError('Oversized parameters')
                parameters=json.loads(text);shape=parameters.get('shape')
                if (not isinstance(shape,list)or len(shape)!=2 or
                        any(type(x)is not int or not 1<=x<=512 for x in shape)):
                    raise ValueError('Invalid LDCT shape')
                packed=self.rfile.read(size)
                if len(packed)!=size:raise ValueError('Truncated input')
                raw=inflate_exact(packed,shape[0]*shape[1]*2)
                output=service.execute_ldct_binary(raw,self.headers.get('X-Request-ID',''),
                    self.headers.get('X-Input-SHA256',''),parameters)
                self.send_response(200);self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(len(output)));self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(output)
            except BlockingIOError as exc:self.reply(409,dict(error=str(exc)))
            except Exception as exc:
                self.close_connection=True;self.reply(503 if service.failed else 400,dict(error=str(exc)))
    return Handler


def start(bind='0.0.0.0',port=8766,folder=None):
    folder=Path(folder).resolve();parent.parent.parent.parent.validate_package(folder)
    server=ThreadingHTTPServer((bind,port),BaseHTTPRequestHandler)
    try:
        hardware=parent.Hardware(folder)
        hardware.selftest_basic();hardware.selftest_ldct();hardware.selftest_roi()
        hardware.selftest_shared();hardware.selftest_shared_segmentation()
        hardware.selftest_canvas();hardware.selftest_compact()
        # Codec check does not replace the actual original LDCT selftests above.
        sample=np.array([-32768,-1000,-1,0,1,400,32767],dtype='<i2').tobytes()
        meta,decoded=decode_response(encode_response(dict(test='integer_endpoints'),sample),len(sample))
        if decoded!=sample:raise ValueError('Lossless LDCT codec check failed')
        service=Service(hardware,(folder/'platform_token.txt').read_text().strip(),digest((folder/'platform_manifest.json').read_bytes()))
        service.startup_verified=service.roi_verified=service.shared_verified=True
        server.RequestHandlerClass=handler_for(service)
    except BaseException:server.server_close();raise
    service.server=server
    parent.parent.parent.parent.previous.roi.base.OWNERS.append(service)
    service.thread=threading.Thread(target=server.serve_forever,daemon=True);service.thread.start()
    print('LOSSLESS LDCT WIRE READY: original algorithms/selftests retained.',flush=True)
    return service
