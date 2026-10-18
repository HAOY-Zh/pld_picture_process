"""Host client for the single-owner basic/segmentation FPGA platform."""
import base64
import hashlib
import json
import time
import uuid

import numpy as np

from ct_basic_processing import parameters
from lung_live_web import BoardClient, BIT_SHA, MODEL_SHA


def sha(raw): return hashlib.sha256(raw).hexdigest()


class PlatformClient(BoardClient):
    def __init__(self, url, token, manifest):
        super().__init__(url, token)
        self.manifest = manifest

    def health(self):
        value = self.call('/v1/platform', timeout=5)
        if (value.get('schema') != 'ct_platform_identity_v1' or
                value.get('platform_manifest_sha256') != self.manifest['platform_manifest_sha256']):
            raise ValueError('Unified FPGA package identity differs')
        if not value.get('ready') or value.get('busy'):
            raise RuntimeError('FPGA busy/faulted; no retry or reload')
        identity = value.get('segmentation', {})
        if (identity.get('schema') != 'lung_live_identity_v1' or identity.get('backend') != 'FPGA_INT8' or
            identity.get('bit_sha256') != BIT_SHA or identity.get('model_sha256') != MODEL_SHA or
            len(identity.get('startup_nodes_checked', [])) != 21 or identity.get('startup_command_count') != 51):
            raise ValueError('Segmentation identity/startup differs')
        if value.get('basic') != self.manifest['basic']: raise ValueError('Basic module identities differ')
        if self.manifest.get('basic_transport'):
            if (value.get('basic_transport') != self.manifest['basic_transport'] or
                    value.get('shared_basic_overlay') is not True):
                raise ValueError('Fast transport/shared overlay not available')
        return dict(identity, platform=value)

    def basic(self, hu, supplied):
        if self.manifest.get('basic_transport') == 'ct_basic_zlib_cache_v1':
            from ct_platform_fast_client import basic
            return basic(self, hu, supplied)
        params = parameters(supplied); op = params['operation']
        if op not in self.manifest['basic']: raise ValueError('This operation has no connected PL module')
        if np.shape(hu) != (512,512):
            raise ValueError('现有基础PL模块要求原生512×512切片；不自动缩放，可选择电脑参考预览')
        rounded = np.rint(hu)
        if not np.isfinite(rounded).all() or rounded.min() < -32768 or rounded.max() > 32767:
            raise ValueError('HU exceeds signed int16 hardware range')
        raw = rounded.astype('<i2').tobytes(); rid = uuid.uuid4().hex
        start = time.perf_counter()
        value = self.call('/v1/basic', raw, {'Content-Type':'application/octet-stream',
            'X-Request-ID':rid, 'X-Input-SHA256':sha(raw), 'X-Parameters':json.dumps(params)}, timeout=120)
        expected = self.manifest['basic'][op]
        if (value.get('schema') != 'ct_basic_pl_result_v1' or value.get('backend') != 'FPGA_BASIC' or
            value.get('request_id') != rid or value.get('input_sha256') != sha(raw) or
            value.get('shape') != [512,512] or value.get('parameters') != params or
            value.get('platform_manifest_sha256') != self.manifest['platform_manifest_sha256'] or
            value.get('bit_sha256') != expected['bit_sha256'] or
            value.get('hwh_sha256') != expected['hwh_sha256']):
            raise ValueError('Actual basic PL result identity/request/parameters mismatch')
        output = base64.b64decode(value.pop('image_base64'), validate=True)
        if len(output) != 512*512 or sha(output) != value['output_sha256']:
            raise ValueError('Actual basic PL output hash/length mismatch')
        stats = value['stats']
        required = ('dma_ms', 'preparation_ms', 'overlay_switch_ms', 'fclk0_mhz', 'cpu_mhz')
        if (not all(isinstance(stats.get(k),(int,float)) and np.isfinite(stats[k]) for k in required) or
                min(stats[k] for k in required[:3]) < 0 or stats['fclk0_mhz'] <= 0 or
                stats['cpu_mhz'] <= 0 or stats.get('guard_verified') is not True or
                stats.get('input_bytes') != 512*512*4 or stats.get('output_bytes') != 512*512*4):
            raise ValueError('Incomplete basic DMA/time/clock receipt')
        value['roundtrip_ms'] = (time.perf_counter()-start)*1000
        return np.frombuffer(output, np.uint8).reshape(512,512).copy(), value
