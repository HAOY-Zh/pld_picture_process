"""Candidate no-codec binary transport; inherited LDCT PL execution unchanged."""
import base64
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import ct_ldct_lossless_service as parent
from ldct_raw_contract import INTERFACE,ROUTE,MAX_RAW,encode_raw,decode_raw
from dual_io_contract import digest


class Service(parent.Service):
    def shared_health(self):return dict(super().shared_health(),ldct_raw_interface=INTERFACE)

    def execute_ldct_raw(self,raw,rid,sha,parameters):
        value=self.execute_ldct(raw,rid,sha,parameters)
        output=base64.b64decode(value.pop('hu_base64'),validate=True)
        if len(output)!=len(raw)or digest(output)!=value['output_sha256']:
            self.failed='Unexpected raw LDCT output framing';raise ValueError(self.failed)
        value.update(ldct_raw_interface=INTERFACE,service_bundle_sha256=self.bundle_sha256)
        return encode_raw(value,output)


def handler_for(service):
    previous=parent.handler_for(service)
    class Handler(previous):
        protocol_version='HTTP/1.1'
        def do_POST(self):
            if self.path!=ROUTE:return super().do_POST()
            if not self.allowed():self.close_connection=True;return
            try:
                size=int(self.headers.get('Content-Length','-1'))
                if self.headers.get('Transfer-Encoding')or not 0<size<=MAX_RAW or self.headers.get('Content-Type')!='application/octet-stream':
                    raise ValueError('Invalid raw LDCT frame')
                text=self.headers.get('X-Parameters','')
                if len(text)>1024:raise ValueError('Oversized raw LDCT parameters')
                raw=self.rfile.read(size)
                if len(raw)!=size:raise ValueError('Truncated raw LDCT input')
                output=service.execute_ldct_raw(raw,self.headers.get('X-Request-ID',''),self.headers.get('X-Input-SHA256',''),json.loads(text))
                self.send_response(200);self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(len(output)));self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(output)
            except BlockingIOError as exc:self.reply(409,dict(error=str(exc)))
            except Exception as exc:
                self.close_connection=True;self.reply(503 if service.failed else 400,dict(error=str(exc)))
    return Handler


def start(bind='0.0.0.0',port=8766,folder=None):
    folder=Path(folder).resolve();parent.parent.parent.parent.parent.validate_package(folder)
    server=ThreadingHTTPServer((bind,port),BaseHTTPRequestHandler)
    try:
        hardware=parent.parent.Hardware(folder)
        hardware.selftest_basic();hardware.selftest_ldct();hardware.selftest_roi()
        hardware.selftest_shared();hardware.selftest_shared_segmentation();hardware.selftest_canvas();hardware.selftest_compact()
        sample=b'\x00\x80\xff\xff\x00\x00\xff\x7f'
        if decode_raw(encode_raw(dict(test='endpoints'),sample),len(sample))[1]!=sample:raise ValueError('Raw codec check failed')
        service=Service(hardware,(folder/'platform_token.txt').read_text().strip(),digest((folder/'platform_manifest.json').read_bytes()))
        service.startup_verified=service.roi_verified=service.shared_verified=True
        server.RequestHandlerClass=handler_for(service)
    except BaseException:server.server_close();raise
    service.server=server
    parent.parent.parent.parent.parent.previous.roi.base.OWNERS.append(service)
    service.thread=threading.Thread(target=server.serve_forever,daemon=True);service.thread.start()
    print('RAW LDCT BINARY READY: original algorithms and all actual startup tests retained.',flush=True)
    return service
