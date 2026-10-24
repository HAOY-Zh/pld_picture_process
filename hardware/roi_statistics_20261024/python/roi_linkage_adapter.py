"""Borrow the existing owner's ROI processor. No Overlay, DMA allocation or HTTP routes."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from math import sqrt
import numpy as np

def _key(view):
    return (view['case_id'], view['result_id'], view['candidate_id'], view['context_epoch'], view['z_index'])

def process_selected_slice(processor, request: Mapping, current_view: Callable[[], Mapping]):
    """Call only from an explicit user ROI action, inside the owner's task queue/lock.

    current_view() must return current case/result/candidate/epoch/z identity.
    The transport must additionally bind requests to the owner's actual loaded CT,
    not trust client pixels as authenticated case data. No transport is invented here.
    """
    original_key = _key(request)
    if original_key != _key(current_view()):
        raise ValueError('Stale ROI identity before processing')
    hu = np.asarray(request['hu_slice'])
    shape = tuple(request['shape_yx'])
    if len(shape) != 2 or any(isinstance(n,bool) or not isinstance(n,int) or n < 1 or n > 512 for n in shape):
        raise ValueError('Expected native CT shape <=512x512')
    if hu.dtype != np.int16 or hu.size != shape[0]*shape[1]:
        raise ValueError('HU transport must explicitly supply signed int16, no gray image/resize')
    hu = hu.reshape(shape)
    lung = request.get('lung_mask')
    if lung is not None:
        lung = np.asarray(lung)
        if lung.size != hu.size:
            raise ValueError('Lung mask geometry mismatch')
        lung = lung.reshape(shape)
    result = processor.process_slice(hu, roi_yx=request['roi_yx'], hu_range=request.get('hu_range'),
                                     use_hu=request['use_hu'], lung_mask=lung)
    if original_key != _key(current_view()):
        # Work already issued is never retried or cancelled behind the hardware owner's back.
        raise ValueError('Stale ROI completion; discard result without retry')
    return result

def read_roi_statistics(read_u32: Callable[[str], int], count: int, status: int):
    """Read OPTIONAL new RTL ports mapped by the owner, under the same job lock.

    read_u32 names: identity, valid, sum_lo, sum_hi, sumsq_lo, sumsq_hi, minmax.
    Ports are stable until next input packet. Existing bit/HWH has NONE of these
    new registers. Only call after the integrated new design is really deployed.
    """
    if read_u32('identity') != 0x52535431:
        raise ValueError('Optional statistics ABI unavailable: expected RST1')
    if status != 0x01000002 or read_u32('valid') != 1:
        raise ValueError('ROI packet incomplete/invalid; statistics not usable')
    if isinstance(count,bool) or not isinstance(count,int) or not 0 <= count <= 512*512:
        raise ValueError('Invalid selected count')
    def word(name):
        value = read_u32(name)
        if isinstance(value,bool) or not isinstance(value,int) or not 0 <= value < 2**32:
            raise ValueError('Invalid GPIO word: '+name)
        return value
    total = word('sum_lo') | (word('sum_hi') << 32)
    if total & (1<<63):
        total -= 1<<64
    squares = word('sumsq_lo') | (word('sumsq_hi') << 32)
    minmax = word('minmax')
    def signed16(n):
        return n-65536 if n & 32768 else n
    low,high = signed16(minmax&65535),signed16(minmax>>16)
    variance_numerator = count*squares - total*total
    if (not count and (total or squares or minmax)) or not -32768*count <= total <= 32767*count or squares > (1<<30)*count or variance_numerator < 0:
        raise ValueError('Inconsistent statistics words/count')
    if count and (low > high or not count*low <= total <= count*high):
        raise ValueError('Inconsistent min/max')
    return {'computed_on':'FPGA_OPTIONAL_STATISTICS_PORTS', 'count':count,'sum':total,'sumsq':squares,
            'min':low if count else None,'max':high if count else None,
            'mean':total/count if count else None,
            'std_population':sqrt(variance_numerator)/count if count else None,
            'scope':'current_slice_roi_selected_voxels_not_lesion_segmentation'}
