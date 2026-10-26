"""Use the new segmentation route while retaining LDCT/basic/ROI contracts."""
import base64
import time
import uuid
import numpy as np
from ct_studio_roi_client import ROIStudioClient
from shared_segmentation_contract import INTERFACE, ROUTE, SHAPE, BYTES, digest, validate_stats


class SharedSegStudioClient(ROIStudioClient):
    @property
    def segmentation_bit_sha256(self):
        return self.manifest['shared_segmentation']['bit_sha256']

    def health(self):
        legacy = super().health()
        shared = self.call('/v1/shared/platform', timeout=10)
        expected = self.manifest['shared_segmentation']
        if (shared.get('service_bundle_sha256') != self.manifest['service_bundle_sha256']
                or shared.get('shared_segmentation') != expected
                or shared.get('shared_segmentation_startup') != 'all-21-nodes-and-legacy-head-exact'
                or not shared.get('ready') or shared.get('busy') or shared.get('error')):
            raise ValueError('Shared segmentation service is not matched, accepted and idle')
        return dict(legacy, bit_sha256=expected['bit_sha256'],
            engine_interface=INTERFACE, shared_segmentation=expected,
            service_bundle_sha256=shared['service_bundle_sha256'])

    def infer(self, codes):
        if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.shape != SHAPE:
            raise ValueError('Need original U-Net uint8 [1,256,256] input')
        raw = np.ascontiguousarray(codes).tobytes()
        rid = uuid.uuid4().hex
        started = time.perf_counter()
        result = self.call(ROUTE, raw, {'Content-Type': 'application/octet-stream',
            'X-Request-ID': rid, 'X-Input-SHA256': digest(raw)}, timeout=60)
        expected = self.manifest['shared_segmentation']
        if (result.get('schema') != 'lung_live_result_v1' or result.get('backend') != 'FPGA_INT8'
                or result.get('engine_interface') != INTERFACE or result.get('request_id') != rid
                or result.get('input_sha256') != digest(raw) or result.get('shape') != list(SHAPE)
                or result.get('bit_sha256') != expected['bit_sha256']
                or result.get('model_sha256') != expected['model_sha256']
                or result.get('shared_segmentation') != expected
                or result.get('service_bundle_sha256') != self.manifest['service_bundle_sha256']
                or result.get('state') != {'busy': False, 'faulted': False, 'dma_may_access': False}):
            raise ValueError('Actual shared segmentation receipt identity/request mismatch')
        head = base64.b64decode(result.pop('head_base64'), validate=True)
        if len(head) != BYTES or digest(head) != result.get('head_sha256'):
            raise ValueError('Shared segmentation output hash/size mismatch')
        validate_stats(result['stats'])
        result['roundtrip_ms'] = (time.perf_counter()-started)*1000
        return np.frombuffer(head, np.uint8).reshape(SHAPE).copy(), result
