"""Explicit identity for original U-Net on the accepted shared CNN bit."""
import hashlib
import json
from pathlib import Path
import numpy as np

INTERFACE = 'ct_shared_segmentation_v1'
ROUTE = '/v1/shared/segmentation'
MODEL_LABEL = 'unet_original'
SHAPE = (1, 256, 256)
BYTES = 65536


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def descriptor(folder):
    root = Path(folder)
    shared = root/'shared_cnn'
    model_path = shared/'models/unet_original/model.json'
    legacy = root/'segmentation/model.json'
    if model_path.read_bytes() != legacy.read_bytes():
        raise ValueError('Original U-Net contract changed')
    params = shared/'models/unet_original/parameters.npz'
    if params.read_bytes() != (root/'segmentation/parameters.npz').read_bytes():
        raise ValueError('Original U-Net parameters changed')
    model = json.loads(model_path.read_text())
    if model['commands'] != 51 or model['input_shape'] != list(SHAPE) or len(model['nodes']) != 21:
        raise ValueError('Wrong original U-Net graph')
    return dict(interface=INTERFACE, route=ROUTE, selected_model=MODEL_LABEL,
        bit_sha256=digest((shared/'lung_network.bit').read_bytes()),
        hwh_sha256=digest((shared/'lung_network.hwh').read_bytes()),
        model_sha256=digest(model_path.read_bytes()), parameters_sha256=digest(params.read_bytes()),
        input_shape=list(SHAPE), output_shape=list(SHAPE), command_count=51, node_count=21,
        input_quantization=model['input_quantization'], output_scale=model['output_scale'],
        output_zero=model['output_zero'], clock_mhz=100)


def validate_stats(stats):
    if (not isinstance(stats, dict) or stats.get('command_count') != 51
            or len(stats.get('commands', [])) != 51 or stats.get('selected_model') != MODEL_LABEL
            or stats.get('overlay_reloaded') is not False or stats.get('buffer_allocations_per_frame') != 0
            or stats.get('weight_packets_repacked_per_frame') != 0):
        raise ValueError('Incomplete shared original U-Net execution')
    for key in ('host_total_ms', 'shared_call_ms', 'overlay_switch_ms', 'pl_command_cycles'):
        if type(stats.get(key)) not in (int, float) or not np.isfinite(stats[key]) or stats[key] < 0:
            raise ValueError('Invalid shared segmentation timing: '+key)
    clocks = [stats.get('clock_before', {}), stats.get('clock_after', {})]
    for value in clocks:
        if (not np.isfinite([value.get('fclk0_mhz', np.nan), value.get('cpu_mhz', np.nan)]).all()
                or abs(value['fclk0_mhz']-100) > .5 or value['cpu_mhz'] <= 0):
            raise ValueError('Wrong shared segmentation clocks')
    if abs(clocks[0]['cpu_mhz']-clocks[1]['cpu_mhz']) > .05:
        raise ValueError('CPU clock changed')
