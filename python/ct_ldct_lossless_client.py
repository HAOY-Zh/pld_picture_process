"""Optional lossless compressed LDCT wire path; same original output checks."""
import base64
import time
import zlib
from ct_dual_io_client import DualSegStudioClient,Transport

INTERFACE='ct_ldct_zlib_v1'
ROUTE='/v3/ldct'


class LosslessStudioClient(DualSegStudioClient):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.ldct_io=Transport(self.url,self.token)

    def health(self):
        self.ldct_io.close()
        value=super().health()
        actual=super().call('/v1/shared/platform',timeout=10)
        if self.manifest.get('ldct_lossless_interface')!=INTERFACE or actual.get('ldct_lossless_interface')!=INTERFACE:
            raise ValueError('Lossless LDCT endpoint has not been paired')
        return value

    def call(self,path,data=None,headers=None,timeout=60):
        if path!='/v1/ldct':return super().call(path,data,headers,timeout)
        if self.manifest.get('ldct_lossless_interface')!=INTERFACE:
            raise ValueError('Lossless LDCT contract is not configured; no fallback')
        if not isinstance(data,bytes)or not 0<len(data)<=512*512*2:
            raise ValueError('Need bounded original HU bytes')
        began=time.perf_counter();packed=zlib.compress(data,1)
        meta,raw,received=self.ldct_io.request(ROUTE,packed,headers or {},len(data))
        if (meta.get('ldct_lossless_interface')!=INTERFACE or
                meta.get('service_bundle_sha256')!=self.manifest['service_bundle_sha256']):
            raise ValueError('Lossless LDCT bundle mismatch')
        # Preserve StudioClient.denoise's entire frozen validation routine.
        # This local conversion is not network traffic or a CPU CNN fallback.
        meta['hu_base64']=base64.b64encode(raw).decode('ascii')
        meta['wire_timing']=dict(raw_input_bytes=len(data),sent_input_bytes=len(packed),
            received_bytes=received,wire_ms=(time.perf_counter()-began)*1000)
        return meta
