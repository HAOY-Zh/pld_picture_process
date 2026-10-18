"""Shared host/service validation; importing this module never accesses hardware."""
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS = {'bypass': 0, 'gaussian5': 1, 'bilateral5': 2, 'joint7_guide5': 3,
           'joint7_guide3': 4, 'guided3': 5, 'cnn': 6, 'bilateral7': 7}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape_checked(shape):
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2 or
            any(type(v) is not int or not 1 <= v <= 512 for v in shape)):
        raise ValueError('LDCT shape must have two integer dimensions in 1..512')
    return tuple(shape)


def quantize_slice(hu):
    if not isinstance(hu, np.ndarray) or hu.dtype not in (np.dtype('int16'), np.dtype('float32')):
        raise ValueError('LDCT requires a numpy int16/float32 raw-HU slice')
    shape_checked(list(hu.shape))
    rounded = np.rint(hu) if hu.dtype != np.int16 else hu
    if not np.isfinite(rounded).all() or rounded.min() < -32768 or rounded.max() > 32767:
        raise ValueError('Nonfinite/out-of-int16 HU rejected before hardware access')
    return np.ascontiguousarray(rounded, dtype='<i2')


def validate_release(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files_sha256'].items():
        path = (folder/name).resolve()
        if folder not in path.parents or sha(path) != digest:
            raise ValueError('Changed LDCT release file: '+name)
    if (manifest.get('interface_version') != 'ldct_slice_v1' or manifest.get('methods') != METHODS
            or manifest['clock_mhz'] not in (50., 100.) or not manifest.get('cnn_compact_protocol')):
        raise ValueError('Unsupported LDCT release contract')
    spec = json.loads((folder/'model/quantization.json').read_text(encoding='utf-8'))
    if spec != manifest['model'] or spec['weights_sha256'] != sha(folder/'model/weights_integer.npz'):
        raise ValueError('LDCT model/ROM release identity mismatch')
    descriptor = dict(interface_version='ldct_slice_v1', release_id=manifest['release_id'],
        release_manifest_sha256=sha(folder/'manifest.json'), model_id=spec['checkpoint_sha256'],
        weights_sha256=spec['weights_sha256'], bit_sha256=sha(folder/'overlay/ct_denoise_tile.bit'),
        hwh_sha256=sha(folder/'overlay/ct_denoise_tile.hwh'), clock_mhz=float(manifest['clock_mhz']),
        clock_tolerance_mhz=.5, methods=METHODS, input_units='HU', output_dtype='int16',
        dimensions=[1,512], standalone_board_verified=True, integrated_board_verified=False)
    return manifest, descriptor
