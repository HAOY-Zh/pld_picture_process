"""Authenticated canvas call; every response binds real model/bit/input/head."""
import base64
import json
import time
import uuid
import zlib
import numpy as np
from lung_live_web import BoardClient
from nodule_canvas_contract import INTERFACE, ROUTE, shapes, inflate, digest, validate_stats


class CanvasClient(BoardClient):
    def __init__(self, url, token, bundle):
        super().__init__(url, token)
        self.bundle = bundle

    def health(self):
        value = self.call('/v1/shared/platform', timeout=10)
        if (value.get('service_bundle_sha256') != self.bundle['service_bundle_sha256']
                or value.get('shared_nodule') != self.bundle['shared_nodule']
                or value.get('canvas_interface') != INTERFACE
                or value.get('canvas_startup') != 'canonical-head-and-translated-valid-cells-exact'
                or not value.get('ready') or value.get('busy') or value.get('error')):
            raise ValueError('Canvas service not matched, accepted and idle')
        return value

    def predict(self, codes, height, width):
        source, target = shapes(height, width)
        if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.shape != source:
            raise ValueError('Wrong lowered canvas input')
        raw = np.ascontiguousarray(codes).tobytes()
        rid = uuid.uuid4().hex
        started = time.perf_counter()
        compressed = zlib.compress(raw, 1)
        result = self.call(ROUTE, compressed, {'Content-Type': 'application/octet-stream',
            'X-Request-ID': rid, 'X-Input-SHA256': digest(raw),
            'X-Canvas-Shape': json.dumps([height, width])}, timeout=120)
        if (result.get('schema') != INTERFACE or result.get('backend') != 'FPGA_NODULE_SHARED'
                or result.get('request_id') != rid or result.get('input_sha256') != digest(raw)
                or result.get('input_shape') != list(source) or result.get('output_shape') != list(target)
                or result.get('service_bundle_sha256') != self.bundle['service_bundle_sha256']
                or result.get('shared_nodule') != self.bundle['shared_nodule']
                or result.get('state') != {'busy': False, 'faulted': False, 'dma_may_access': False}):
            raise ValueError('Canvas response identity/request/state mismatch')
        head = inflate(base64.b64decode(result.pop('head_zlib_base64'), validate=True), int(np.prod(target)))
        if digest(head) != result.get('output_sha256'):
            raise ValueError('Canvas output hash mismatch')
        validate_stats(result['stats'])
        result.update(roundtrip_ms=(time.perf_counter()-started)*1000,
            compressed_input_bytes=len(compressed), raw_input_bytes=len(raw))
        return np.frombuffer(head, np.uint8).reshape(target).copy(), result
