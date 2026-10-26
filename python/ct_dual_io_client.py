"""Persistent, idle-aware transport. Failed inference POSTs are never retried."""
import http.client
import json
import threading
import time
import uuid
from urllib.parse import urlsplit
import zlib
import numpy as np
from ct_nodule_canvas_client import CanvasClient
from ct_shared_seg_client import SharedSegStudioClient
from shared_segmentation_contract import SHAPE, validate_stats as validate_seg
from nodule_canvas_contract import shapes, validate_stats
from dual_io_contract import INTERFACE, NODULE_ROUTE, SEG_ROUTE, MAX_HEAD, MAX_META, digest, decode_response


class Transport:
    def __init__(self, url, token):
        self.url, self.token = url, token
        self.connection = None
        self.last_success = 0.
        self.lock = threading.Lock()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def request(self, route, payload, headers, expected):
        if not self.lock.acquire(False):
            raise RuntimeError('Concurrent transport refused')
        try:
            # Server closes after 10 s idle. Reconnect before sending; never replay.
            if time.monotonic()-self.last_success > 3.:
                self.close()
            if self.connection is None:
                url = urlsplit(self.url)
                self.connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=120)
            self.connection.request('POST', route, body=payload, headers=dict(headers,
                Authorization='Bearer '+self.token, **{'Content-Type':'application/octet-stream'}))
            response = self.connection.getresponse()
            length = response.getheader('Content-Length', '')
            if not length.isdigit() or not 0 < int(length) <= 4+MAX_META+MAX_HEAD+8192:
                raise ValueError('Invalid response length')
            encoded = response.read(int(length))
            if len(encoded) != int(length):
                raise ValueError('Truncated response')
            if response.status != 200:
                raise RuntimeError('Board refused command: '+str(response.status)+' '+encoded[:512].decode(errors='replace'))
            if response.getheader('Content-Type') != 'application/octet-stream':
                raise ValueError('Invalid binary response type')
            result, head = decode_response(encoded, expected)
            self.last_success = time.monotonic()
            return result, head, len(encoded)
        except BaseException:
            self.close()
            raise
        finally:
            self.lock.release()


def check(result, rid, sha, bundle):
    if (result.get('schema') != INTERFACE or result.get('request_id') != rid
            or result.get('input_sha256') != sha
            or result.get('service_bundle_sha256') != bundle['service_bundle_sha256']
            or result.get('state') != dict(busy=False, faulted=False, dma_may_access=False)):
        raise ValueError('Request, bundle or owner identity mismatch')


class CachedCanvasClient(CanvasClient):
    def __init__(self, url, token, bundle):
        super().__init__(url, token, bundle)
        self.io = Transport(url, token)
        self.session = uuid.uuid4().hex
        self.keys = set()

    def health(self):
        self.io.close()
        self.session = uuid.uuid4().hex
        self.keys.clear()
        value = super().health()
        if value.get('dual_io_interface') != INTERFACE or value.get('dual_io_startup') != 'all-nodes-and-native-layout-exact':
            raise ValueError('Dual I/O startup not accepted')
        return value

    def predict(self, image, height, width):
        _, target = shapes(height, width)
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (5,height,width):
            raise ValueError('Need original five quantized planes')
        started = time.perf_counter()
        raw = np.ascontiguousarray(image).tobytes()
        size = height*width
        hashes = [digest(raw[i*size:(i+1)*size]) for i in range(5)]
        indices = [i for i,h in enumerate(hashes) if h not in self.keys]
        sent = b''.join(raw[i*size:(i+1)*size] for i in indices)
        t = time.perf_counter(); payload = zlib.compress(sent,1); compress = (time.perf_counter()-t)*1000
        rid = uuid.uuid4().hex; sha = digest(raw)
        result, head, received = self.io.request(NODULE_ROUTE, payload, {
            'X-Request-ID':rid, 'X-Input-SHA256':sha, 'X-Canvas-Shape':json.dumps([height,width]),
            'X-Cache-Session':self.session, 'X-Plane-Hashes':json.dumps(hashes),
            'X-Sent-Plane-Indices':json.dumps(indices)}, int(np.prod(target)))
        check(result,rid,sha,self.bundle)
        if (result.get('backend') != 'FPGA_NODULE_SHARED' or result.get('shared_nodule') != self.bundle['shared_nodule']
                or result.get('input_shape') != [5,height,width] or result.get('output_shape') != list(target)
                or result.get('output_sha256') != digest(head) or result.get('cache_session') != self.session):
            raise ValueError('Actual nodule full-head identity mismatch')
        keys = result.get('cached_plane_hashes')
        if not isinstance(keys,list) or len(keys)>10 or any(not isinstance(h,str) or len(h)!=64 for h in keys):
            raise ValueError('Invalid cache inventory')
        validate_stats(result['stats']); self.keys = set(keys)
        result.update(roundtrip_ms=(time.perf_counter()-started)*1000, raw_input_bytes=len(raw),
            transmitted_raw_input_bytes=len(sent), compressed_input_bytes=len(payload),
            response_bytes=received, client_compress_ms=compress, reused_planes=5-len(indices))
        return np.frombuffer(head,np.uint8).reshape(target).copy(), result


class DualSegStudioClient(SharedSegStudioClient):
    def __init__(self,url,token,manifest):
        super().__init__(url,token,manifest)
        self.io = Transport(url,token)

    def health(self):
        self.io.close()
        value = super().health()
        shared = self.call('/v1/shared/platform',timeout=10)
        if shared.get('dual_io_interface') != INTERFACE or shared.get('dual_io_startup') != 'all-nodes-and-native-layout-exact':
            raise ValueError('Dual segmentation I/O not accepted')
        return value

    def infer(self,codes):
        if not isinstance(codes,np.ndarray) or codes.dtype != np.uint8 or codes.shape != SHAPE:
            raise ValueError('Need original U-Net input')
        started=time.perf_counter();raw=np.ascontiguousarray(codes).tobytes();sha=digest(raw);rid=uuid.uuid4().hex
        result,head,received=self.io.request(SEG_ROUTE,zlib.compress(raw,1),{
            'X-Request-ID':rid,'X-Input-SHA256':sha},int(np.prod(SHAPE)))
        check(result,rid,sha,self.manifest);expected=self.manifest['shared_segmentation']
        if (result.get('backend')!='FPGA_INT8' or result.get('shape')!=list(SHAPE)
                or result.get('shared_segmentation')!=expected or result.get('head_sha256')!=digest(head)
                or result.get('bit_sha256')!=expected['bit_sha256'] or result.get('model_sha256')!=expected['model_sha256']):
            raise ValueError('Actual segmentation full-head identity mismatch')
        validate_seg(result['stats']);result.update(roundtrip_ms=(time.perf_counter()-started)*1000,response_bytes=received)
        return np.frombuffer(head,np.uint8).reshape(SHAPE).copy(),result
