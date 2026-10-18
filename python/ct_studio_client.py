"""The existing basic/segmentation client with an explicit raw-HU LDCT route."""
import base64
import hashlib
import json
import time
import uuid
import numpy as np
from ct_platform_client import PlatformClient
from ldct_contract import METHODS, quantize_slice


class StudioClient(PlatformClient):
    def health(self):
        identity = super().health()
        if identity['platform'].get('ldct') != self.manifest.get('ldct') or not self.manifest.get('ldct'):
            raise ValueError('Unified LDCT service identity differs')
        if identity['platform'].get('integration_board_acceptance') != 'startup-selftests-passed':
            raise RuntimeError('Unified service actual startup checks have not passed')
        return identity

    def denoise(self, hu, method='cnn'):
        if method not in METHODS:
            raise ValueError('Unknown LDCT method')
        image = quantize_slice(hu)
        raw = image.tobytes(); rid = uuid.uuid4().hex
        digest = hashlib.sha256(raw).hexdigest()
        params = dict(method=method, shape=list(image.shape))
        start = time.perf_counter()
        result = self.call('/v1/ldct', raw, {'Content-Type':'application/octet-stream',
            'X-Request-ID':rid, 'X-Input-SHA256':digest, 'X-Parameters':json.dumps(params)}, timeout=120)
        expected = self.manifest['ldct']
        if (result.get('schema') != 'ct_studio_ldct_result_v1' or result.get('backend') != 'FPGA_LDCT'
                or result.get('request_id') != rid or result.get('input_sha256') != digest
                or result.get('parameters') != params or result.get('shape') != list(image.shape)
                or result.get('platform_manifest_sha256') != self.manifest['platform_manifest_sha256']
                or any(result.get(k) != expected[k] for k in ('model_id','weights_sha256','bit_sha256','hwh_sha256'))):
            raise ValueError('LDCT request/output/model identity mismatch')
        output = base64.b64decode(result.pop('hu_base64'), validate=True)
        if len(output) != len(raw) or hashlib.sha256(output).hexdigest() != result.get('output_sha256'):
            raise ValueError('LDCT output hash/size mismatch')
        stats = result['stats']
        for key in ('total_ms','overlay_switch_ms','actual_pl_mhz_before','actual_pl_mhz_after','cpu_mhz'):
            if not isinstance(stats.get(key),(float,int)) or not np.isfinite(stats[key]) or stats[key] < 0:
                raise ValueError('Incomplete LDCT timing/clock receipt')
        for key in ('actual_pl_mhz_before','actual_pl_mhz_after'):
            if abs(stats[key]-expected['clock_mhz']) > expected['clock_tolerance_mhz']:
                raise ValueError('Wrong actual LDCT frequency')
        if result.get('state') != dict(busy=False,faulted=False,dma_may_access=False,error=None):
            raise ValueError('LDCT resources not confirmed idle')
        result['roundtrip_ms'] = (time.perf_counter()-start)*1000
        return np.frombuffer(output, '<i2').reshape(image.shape).copy(), result
