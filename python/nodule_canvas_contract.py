"""Bounded, explicit variable-canvas transport for frozen INT8 nodule graph."""
import hashlib
import zlib
import numpy as np

INTERFACE = 'ct_shared_nodule_canvas_v1'
ROUTE = '/v1/shared/nodule-canvas'
MAX_RAW = 45*512*512


def digest(raw): return hashlib.sha256(raw).hexdigest()


def quantize_canvas(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[0] != 5 or not np.isfinite(image).all():
        raise ValueError('Need finite five-plane canvas')
    shapes(*image.shape[1:])
    return np.rint((image.astype(np.float64).clip(-1000, 400)+1000)*(255./1400.)).astype(np.uint8)


def shapes(height, width):
    if any(type(v) is not int or v < 128 or v > 1024 or v % 8 for v in (height, width)):
        raise ValueError('Canvas H/W must be multiples of 8 in 128..1024')
    return (45, height//2, width//2), (5, height//2, width//2)


def inflate(raw, expected):
    if not 0 < expected <= MAX_RAW or len(raw) > MAX_RAW+8192:
        raise ValueError('Canvas payload exceeds bounds')
    stream = zlib.decompressobj()
    value = stream.decompress(raw, expected+1)
    if len(value) != expected or not stream.eof or stream.unused_data or stream.unconsumed_tail:
        raise ValueError('Wrong compressed canvas length/framing')
    return value


def validate_stats(stats):
    if (stats.get('command_count') != 15 or len(stats.get('commands', [])) != 15
            or stats.get('selected_model') != 'nodule_canvas' or stats.get('overlay_reloaded') is not False
            or stats.get('buffer_allocations_per_frame') != 0):
        raise ValueError('Incomplete canvas PL receipt')
    for key in ('host_total_ms', 'shared_call_ms', 'overlay_switch_ms', 'canvas_setup_ms', 'pl_command_cycles',
                'actual_pl_mhz_before', 'actual_pl_mhz_after', 'cpu_mhz'):
        if type(stats.get(key)) not in (int, float) or not np.isfinite(stats[key]) or stats[key] < 0:
            raise ValueError('Invalid canvas receipt: '+key)
    if any(abs(stats[k]-100) > .5 for k in ('actual_pl_mhz_before', 'actual_pl_mhz_after')):
        raise ValueError('Wrong canvas PL clock')
