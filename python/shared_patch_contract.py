"""Frozen nodule core transport; a patch head is not a whole-case detection."""
import hashlib,json
from pathlib import Path
import numpy as np

INTERFACE='ct_shared_nodule_patch_v1'
BACKEND='FPGA_NODULE_SHARED'
INPUT_SHAPE=(45,64,64)
OUTPUT_SHAPE=(5,64,64)
INPUT_BYTES=int(np.prod(INPUT_SHAPE))
OUTPUT_BYTES=int(np.prod(OUTPUT_SHAPE))
BATCH_INTERFACE='ct_shared_nodule_batch_v1'
MAX_BATCH=8


def batch_patch_id(batch_id,index):
    return digest((batch_id+':'+str(index)).encode('ascii'))[:32]


def digest(data):return hashlib.sha256(data).hexdigest()


def input_codes(value):
    if not isinstance(value,np.ndarray) or value.dtype!=np.uint8 or value.shape!=INPUT_SHAPE:
        raise ValueError('Need frozen lowered uint8 [45,64,64], not raw HU or lung mask')
    return np.ascontiguousarray(value)


def validate_assets(folder,row):
    root=Path(folder).resolve()
    if (row.get('interface')!=INTERFACE or row.get('clock_mhz')!=100 or
            row.get('input_shape')!=list(INPUT_SHAPE) or row.get('output_shape')!=list(OUTPUT_SHAPE)
            or row.get('commands_per_patch')!=15 or row.get('model_kind')!='nodule'):
        raise ValueError('Wrong shared nodule ABI')
    for key in ('bit','hwh','model','parameters','source_manifest'):
        p=(root/row[key]).resolve()
        if root not in p.parents or digest(p.read_bytes())!=row[key+'_sha256']:
            raise ValueError('Changed shared core asset: '+key)
    model=json.loads((root/row['model']).read_text())
    if model['input_shape']!=list(INPUT_SHAPE) or model['nodes'][-1]['shape']!=list(OUTPUT_SHAPE) or model['commands']!=15:
        raise ValueError('Wrong frozen nodule graph')
    return model


def validate_receipt_stats(stats):
    if (not isinstance(stats,dict) or stats.get('command_count')!=15 or len(stats.get('commands',[]))!=15
            or stats.get('selected_model')!='nodule' or stats.get('overlay_reloaded')is not False
            or stats.get('buffer_allocations_per_frame')!=0):raise ValueError('Incomplete real nodule execution receipt')
    for key in ('host_total_ms','overlay_switch_ms','actual_pl_mhz_before','actual_pl_mhz_after','cpu_mhz','pl_command_cycles'):
        if type(stats.get(key))not in (int,float) or not np.isfinite(stats[key]) or stats[key]<0:
            raise ValueError('Invalid shared-core timing: '+key)
    if any(abs(stats[k]-100)>.5 for k in ('actual_pl_mhz_before','actual_pl_mhz_after')):
        raise ValueError('Wrong shared-core clock')
