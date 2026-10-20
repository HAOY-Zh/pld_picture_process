"""ROI1 independent CPU reference and PC-only volume aggregation.

Manual ROI/HU/mask screening, NOT a nodule segmentation/detection algorithm.
Index convention: ZYX, half-open bounds, inclusive signed HU endpoints.
"""
from __future__ import annotations
import numbers
from time import perf_counter
import numpy as np

MODULE_VERSION='roi-filter-1.0.0'
ALGORITHM_ID='manual_roi_hu_mask'
PARAMS_VERSION='roi-v1'
MAX_SHAPE=(512,512)


def integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def prepare_hu(value):
    image=np.asarray(value)
    if image.ndim!=2 or any(n<1 or n>512 for n in image.shape):
        raise ValueError('HU must be a nonempty native 2D slice, each dimension <=512')
    if image.dtype.kind not in 'iuf' or not np.isfinite(image).all():
        raise ValueError('HU must be finite numeric CT values, not display gray or bool')
    rounded=np.rint(image)
    if (rounded < -32768).any() or (rounded > 32767).any():
        raise ValueError('Rounded HU is outside int16; refusing wraparound')
    return np.ascontiguousarray(rounded,dtype=np.int16)


def normalize_mask(value,shape):
    mask=np.asarray(value)
    if mask.shape!=shape or mask.dtype not in (np.dtype('uint8'),np.dtype('bool')):
        raise ValueError('Mask must have matching geometry and uint8/bool dtype')
    unique=set(np.unique(mask).tolist())
    if not (unique <= {0,1} or unique <= {0,255}):
        raise ValueError('Mask must be 0/1 or explicitly normalized 0/255, not arbitrary nonzero')
    return np.ascontiguousarray(mask!=0,dtype=np.uint8)


def validate_bounds(bounds,shape,name='roi'):
    if bounds is None:
        return tuple(v for n in shape for v in (0,n))
    if len(bounds)!=2*len(shape):
        raise ValueError(f'{name} requires {2*len(shape)} boundaries')
    values=tuple(integer(v,name) for v in bounds)
    for i,n in enumerate(shape):
        start,end=values[2*i:2*i+2]
        if not 0<=start<=end<=n:
            raise ValueError(f'{name}: reversed or out-of-range bounds')
    return values


def validate_slice(hu_slice,roi_yx=None,hu_range=None,use_hu=True,lung_mask=None):
    if not isinstance(use_hu,(bool,np.bool_)):
        raise ValueError('use_hu must be boolean')
    hu=prepare_hu(hu_slice)
    roi=validate_bounds(roi_yx,hu.shape)
    if hu_range is None:
        if use_hu:
            raise ValueError('hu_range is required when use_hu=True')
        hu_range=(-32768,32767)
    if len(hu_range)!=2:
        raise ValueError('hu_range requires (low, high)')
    low,high=(integer(v,'HU threshold') for v in hu_range)
    if not -32768<=low<=high<=32767:
        raise ValueError('Require -32768 <= low <= high <= 32767')
    mask=None if lung_mask is None else normalize_mask(lung_mask,hu.shape)
    return hu,roi,(low,high),bool(use_hu),mask


def configuration(hu,roi,hu_range,use_hu,mask):
    h,w=hu.shape
    y0,y1,x0,x1=roi
    low,high=hu_range
    return {'geometry':w|(h<<16),'bounds_x':x0|(x1<<16),'bounds_y':y0|(y1<<16),
            'thresholds':(low&65535)|((high&65535)<<16),
            'modes':int(use_hu)|(int(mask is not None)<<1)}


def pack_input(hu,mask=None):
    words=hu.view(np.uint16).astype(np.uint32)
    if mask is not None:
        words|=mask.astype(np.uint32)<<16
    return np.ascontiguousarray(words.ravel())


def hu_statistics(hu,mask):
    values=hu[mask!=0].astype(np.int64)
    count=int(values.size)
    return {'count':count,'sum':int(values.sum()),
            'min':int(values.min()) if count else None,
            'max':int(values.max()) if count else None,
            'mean':float(values.mean()) if count else None,
            'computed_on':'CPU_FROM_RETURNED_MASK'}


def make_result(hu,roi,hu_range,use_hu,lung,mask,backend,timing,clock_hz=None):
    # Always detach returned arrays from DMA buffers or injected callback views.
    mask=np.array(mask,dtype=np.uint8,copy=True)
    return {'filtered_mask':mask,'valid_count':int(mask.sum(dtype=np.int64)),
            'backend':backend,'status':'SUCCESS','module_version':MODULE_VERSION,
            'algorithm_id':ALGORITHM_ID,'params_version':PARAMS_VERSION,
            'shape_yx':list(hu.shape),'roi_yx':list(roi),
            'use_hu':use_hu,'hu_range':list(hu_range) if use_hu else None,
            'use_lung_mask':lung is not None,'clock_hz':clock_hz,
            'hu_statistics':hu_statistics(hu,mask),'timing':timing}


class CpuRoiProcessor:
    def process_slice(self,hu_slice_i16,roi_yx=None,hu_range=None,use_hu=True,lung_mask=None):
        start=perf_counter()
        hu,roi,thresholds,use_hu,lung=validate_slice(hu_slice_i16,roi_yx,hu_range,use_hu,lung_mask)
        prepared=perf_counter()
        selected=np.zeros(hu.shape,dtype=np.uint8)
        y0,y1,x0,x1=roi
        keep=np.ones((y1-y0,x1-x0),dtype=bool)
        if use_hu:
            keep&=(hu[y0:y1,x0:x1]>=thresholds[0])&(hu[y0:y1,x0:x1]<=thresholds[1])
        if lung is not None:
            keep&=lung[y0:y1,x0:x1]!=0
        selected[y0:y1,x0:x1]=keep
        finished=perf_counter()
        result=make_result(hu,roi,thresholds,use_hu,lung,selected,'CPU_REFERENCE',
            {'prepare_seconds':prepared-start,'cpu_reference_seconds':finished-prepared,
             'dma_compute_seconds':None,'postprocess_seconds':None,'total_seconds':None})
        end=perf_counter()
        result['timing'].update(postprocess_seconds=end-finished,total_seconds=end-start)
        return result


def process_slice_reference(*args,**kwargs):
    return CpuRoiProcessor().process_slice(*args,**kwargs)


def roi_analyze(volume_hu,roi_zyx=None,hu_range=None,use_hu=True,lung_mask=None,
                process_slice=None,spacing_zyx_mm=None,regular_orthogonal=False):
    """PC only. Inject B's real per-slice service call; no HTTP/Overlay ownership.

    Full native output volume, zeros outside requested range. Physical volume
    only if caller explicitly confirms regular orthogonal spacing.
    """
    start=perf_counter()
    volume=np.asarray(volume_hu)
    if volume.ndim!=3 or volume.shape[0]<1 or any(n<1 or n>512 for n in volume.shape[1:]):
        raise ValueError('Need nonempty native Z,H,W CT volume, H/W<=512')
    roi=validate_bounds(roi_zyx,volume.shape,'roi_zyx')
    # Validate options even for an empty z range, before any injected call.
    validate_slice(np.zeros(volume.shape[1:],np.int16),roi[2:],hu_range,use_hu,None)
    if spacing_zyx_mm is not None:
        spacing=np.asarray(spacing_zyx_mm,dtype=float)
        if not regular_orthogonal or spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any():
            raise ValueError('Physical volume requires confirmed regular orthogonal positive spacing')
    else:
        spacing=None
    if lung_mask is not None:
        mask_array=np.asarray(lung_mask)
        if mask_array.shape!=volume.shape:
            raise ValueError('Volume lung mask geometry mismatch')
        if mask_array.dtype not in (np.dtype('uint8'),np.dtype('bool')):
            raise ValueError('Volume lung mask must use uint8/bool encoding')
    callback=process_slice or CpuRoiProcessor().process_slice
    z0,z1,y0,y1,x0,x1=roi
    output=np.zeros(volume.shape,dtype=np.uint8)
    rows=[]; backend=None; total_sum=0; total_min=None; total_max=None
    for z in range(z0,z1):
        hu=prepare_hu(volume[z])
        result=callback(hu,roi_yx=(y0,y1,x0,x1),hu_range=hu_range,use_hu=use_hu,
                        lung_mask=None if lung_mask is None else lung_mask[z])
        if result.get('status')!='SUCCESS' or result.get('backend') not in ('CPU_REFERENCE','FPGA'):
            raise ValueError('Injected processor returned failed/unknown backend result')
        if backend is not None and backend!=result['backend']:
            raise ValueError('Mixed backends within a volume are not accepted')
        backend=result['backend']
        mask=normalize_mask(result['filtered_mask'],hu.shape)
        count=int(mask.sum(dtype=np.int64))
        if count!=result['valid_count']:
            raise ValueError('Returned mask/count mismatch')
        if mask[:y0].any() or mask[y1:].any() or mask[:,:x0].any() or mask[:,x1:].any():
            raise ValueError('Returned mask exceeds ROI')
        output[z]=mask
        stats=hu_statistics(hu,mask)
        total_sum+=stats['sum']
        if count:
            total_min=stats['min'] if total_min is None else min(total_min,stats['min'])
            total_max=stats['max'] if total_max is None else max(total_max,stats['max'])
        rows.append({'z':z,'valid_count':count,'area_mm2':count*float(spacing[1]*spacing[2]) if spacing is not None else None,
                     'timing':result.get('timing')})
    count=int(output.sum(dtype=np.int64))
    mm3=count*float(np.prod(spacing)) if spacing is not None else None
    return {'filtered_mask':output,'valid_count':count,'slice_results':rows,'roi_zyx':list(roi),
            'backend':backend or 'NOT_EXECUTED','status':'SUCCESS','scope':'FULL_VOLUME' if roi==validate_bounds(None,volume.shape) else 'LOCAL',
            'region_volume_mm3':mm3,'region_volume_ml':None if mm3 is None else mm3/1000,
            'hu_statistics':{'count':count,'sum':total_sum,'min':total_min,'max':total_max,
                             'mean':total_sum/count if count else None,'computed_on':'PC_FROM_RETURNED_MASKS'},
            'module_version':MODULE_VERSION,'algorithm_id':ALGORITHM_ID,
            'total_seconds':perf_counter()-start,
            'warning':'Conditional screening region, not precise nodule segmentation/true lesion volume.'}
