"""Candidate raw binary persistent LDCT transport, no compression or retry."""
import base64
import http.client
import threading
import time
from urllib.parse import urlsplit
from ct_dual_io_client import DualSegStudioClient
from ldct_raw_contract import INTERFACE,ROUTE,MAX_RAW,MAX_META,decode_raw


class RawLDCTStudioClient(DualSegStudioClient):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.raw_connection=None;self.raw_last_success=0.;self.raw_lock=threading.Lock()

    def close_raw(self):
        if self.raw_connection is not None:self.raw_connection.close();self.raw_connection=None

    def health(self):
        self.close_raw();value=super().health()
        actual=super().call('/v1/shared/platform',timeout=10)
        if (self.manifest.get('ldct_raw_interface')!=INTERFACE or actual.get('ldct_raw_interface')!=INTERFACE):
            raise ValueError('Raw binary LDCT service has not been paired')
        return value

    def call(self,path,data=None,headers=None,timeout=60):
        if path!='/v1/ldct':return super().call(path,data,headers,timeout)
        if self.manifest.get('ldct_raw_interface')!=INTERFACE:raise ValueError('Raw binary LDCT not configured; no fallback')
        if not isinstance(data,bytes)or not 0<len(data)<=MAX_RAW:raise ValueError('Need bounded original HU bytes')
        if not self.raw_lock.acquire(False):raise RuntimeError('Concurrent LDCT request refused')
        began=time.perf_counter()
        try:
            if time.monotonic()-self.raw_last_success>3.:self.close_raw()
            if self.raw_connection is None:
                url=urlsplit(self.url);self.raw_connection=http.client.HTTPConnection(url.hostname,url.port or 80,timeout=timeout)
            self.raw_connection.request('POST',ROUTE,body=data,headers=dict(headers or {},Authorization='Bearer '+self.token))
            response=self.raw_connection.getresponse();length=response.getheader('Content-Length','')
            if not length.isdigit()or not 0<int(length)<=4+MAX_META+MAX_RAW:raise ValueError('Invalid raw LDCT response length')
            encoded=response.read(int(length))
            if len(encoded)!=int(length):raise ValueError('Incomplete raw LDCT response')
            if response.status!=200:raise RuntimeError('Board refused raw LDCT: '+str(response.status)+' '+encoded[:512].decode(errors='replace'))
            if response.getheader('Content-Type')!='application/octet-stream':raise ValueError('Invalid LDCT response type')
            meta,raw=decode_raw(encoded,len(data))
            if (meta.get('ldct_raw_interface')!=INTERFACE or meta.get('service_bundle_sha256')!=self.manifest['service_bundle_sha256']):
                raise ValueError('Raw LDCT interface/bundle mismatch')
            # Parent denoise retains all original request/model/hash/clock checks.
            meta['hu_base64']=base64.b64encode(raw).decode('ascii')
            meta['raw_wire_timing']=dict(input_bytes=len(data),response_bytes=len(encoded),
                wire_ms=(time.perf_counter()-began)*1000,compression=False)
            self.raw_last_success=time.monotonic()
            return meta
        except BaseException:
            self.close_raw();raise
        finally:self.raw_lock.release()
