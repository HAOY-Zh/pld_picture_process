"""ROI1 transport contract, shared by PC and the single board owner."""
import hashlib
from pathlib import Path
import numpy as np
try:
    from roi_vendor.ct_roi_reference import validate_slice, hu_statistics, make_result
except ImportError:
    from ct_roi_reference import validate_slice, hu_statistics, make_result

TRANSPORT='ct_roi_i16_mask_v1'
BACKEND='FPGA_ROI'
PARAMETERS={'shape','roi_yx','hu_range','use_hu','use_lung_mask'}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def encode(hu,roi_yx=None,hu_range=None,use_hu=True,lung_mask=None):
    hu,roi,limits,use_hu,mask=validate_slice(hu,roi_yx,hu_range,use_hu,lung_mask)
    raw=hu.astype('<i2',copy=False).tobytes()+(mask.tobytes() if mask is not None else b'')
    params=dict(shape=list(hu.shape),roi_yx=list(roi),hu_range=list(limits) if use_hu else None,
                use_hu=use_hu,use_lung_mask=mask is not None)
    return raw,params,hu,mask


def decode(raw,params):
    if not isinstance(params,dict) or set(params)!=PARAMETERS:raise ValueError('ROI parameter fields differ')
    shape=params['shape']
    if not isinstance(shape,list) or len(shape)!=2 or any(type(v)is not int or not 1<=v<=512 for v in shape):
        raise ValueError('ROI requires native 1..512 H/W')
    if type(params['use_lung_mask']) is not bool:raise ValueError('ROI lung-mask flag must be boolean')
    n=int(np.prod(shape));expected=n*(3 if params['use_lung_mask'] else 2)
    if len(raw)!=expected:raise ValueError('ROI payload length differs')
    hu=np.frombuffer(raw[:n*2],'<i2').reshape(shape)
    mask=np.frombuffer(raw[n*2:],np.uint8).reshape(shape) if params['use_lung_mask'] else None
    # Canonicalize once and require the exact bytes/parameters on both sides.
    encoded,canonical,hu,mask=encode(hu,params['roi_yx'],params['hu_range'],params['use_hu'],mask)
    if canonical!=params or encoded!=raw:raise ValueError('Noncanonical ROI payload')
    return hu,mask


def validate_assets(folder,row):
    folder=Path(folder).resolve()
    if row.get('transport')!=TRANSPORT or row.get('abi_magic')!='0x524f4931' or row.get('clock_mhz')!=100:
        raise ValueError('Unsupported ROI ABI/clock')
    for key in ('bit','hwh','driver','reference'):
        path=(folder/row[key]).resolve()
        if folder not in path.parents or digest(path.read_bytes())!=row[key+'_sha256']:
            raise ValueError('Changed ROI asset: '+key)
    if row.get('statistics_abi'):
        from roi_statistics_contract import STATISTICS_ABI
        if row['statistics_abi']!=STATISTICS_ABI:raise ValueError('Unsupported statistics ABI')
        for key in ('statistics_contract',):
            path=(folder/row[key]).resolve()
            if folder not in path.parents or digest(path.read_bytes())!=row[key+'_sha256']:
                raise ValueError('Changed ROI statistics contract')
    return row
