"""Persistent connection, compact exact input and binary head; no retries."""
import http.client
import json
import threading
import time
import uuid
from urllib.parse import urlsplit
import zlib
import numpy as np
from ct_nodule_canvas_client import CanvasClient
from nodule_compact_contract import INTERFACE,ROUTE,MAX_HEAD,MAX_META,shapes,digest,decode_response,validate_stats


class CompactClient(CanvasClient):
    def __init__(self,url,token,bundle):
        super().__init__(url,token,bundle)
        self.connection=None
        self.transport_lock=threading.Lock()

    def health(self):
        result=super().health()
        if (result.get('compact_interface')!=INTERFACE
                or result.get('compact_startup')!='full-head-and-CMA-pack-exact'):
            raise ValueError('Compact transport not accepted')
        return result

    def predict(self,image,height,width):
        _,target=shapes(height,width)
        if not isinstance(image,np.ndarray)or image.dtype!=np.uint8 or image.shape!=(5,height,width):
            raise ValueError('Need five original quantized planes')
        if not self.transport_lock.acquire(False):raise RuntimeError('Concurrent transport refused')
        try:
            tick=time.perf_counter();raw=np.ascontiguousarray(image).tobytes();rid=uuid.uuid4().hex
            began=time.perf_counter();payload=zlib.compress(raw,1);compress_ms=(time.perf_counter()-began)*1000
            if self.connection is None:
                url=urlsplit(self.url)
                self.connection=http.client.HTTPConnection(url.hostname,url.port or 80,timeout=120)
            self.connection.request('POST',ROUTE,body=payload,headers={'Authorization':'Bearer '+self.token,
                'Content-Type':'application/octet-stream','X-Request-ID':rid,'X-Input-SHA256':digest(raw),
                'X-Canvas-Shape':json.dumps([height,width])})
            response=self.connection.getresponse()
            length=response.getheader('Content-Length','')
            if not length.isdigit()or not 0<int(length)<=4+MAX_META+MAX_HEAD+8192:
                raise ValueError('Invalid transport response length')
            encoded=response.read(int(length))
            if len(encoded)!=int(length):raise ValueError('Truncated transport response')
            if response.status!=200:raise RuntimeError('Board refused compact command: '+str(response.status)+' '+encoded[:512].decode(errors='replace'))
            if response.getheader('Content-Type')!='application/octet-stream':raise ValueError('Wrong binary response type')
            result,head=decode_response(encoded,int(np.prod(target)))
            if (result.get('schema')!=INTERFACE or result.get('backend')!='FPGA_NODULE_SHARED'
                    or result.get('request_id')!=rid or result.get('input_sha256')!=digest(raw)
                    or result.get('input_shape')!=[5,height,width]or result.get('output_shape')!=list(target)
                    or result.get('service_bundle_sha256')!=self.bundle['service_bundle_sha256']
                    or result.get('shared_nodule')!=self.bundle['shared_nodule']
                    or result.get('output_sha256')!=digest(head)
                    or result.get('state')!={'busy':False,'faulted':False,'dma_may_access':False}):
                raise ValueError('Compact actual result identity/hash mismatch')
            validate_stats(result['stats'])
            result.update(roundtrip_ms=(time.perf_counter()-tick)*1000,compressed_input_bytes=len(payload),
                raw_input_bytes=len(raw),response_bytes=len(encoded),client_compress_ms=compress_ms)
            return np.frombuffer(head,np.uint8).reshape(target).copy(),result
        except BaseException:
            if self.connection is not None:self.connection.close();self.connection=None
            raise
        finally:self.transport_lock.release()
