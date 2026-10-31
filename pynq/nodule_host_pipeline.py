"""Online one-plane processing with independently replaceable exact host ops."""
import time
import numpy as np
from nodule_fast_input import SharedPlaneSampler
from nodule_native_sampling import NativePlaneSampler
from nodule_fast_decode import decode_fast
from nodule_canvas import extract_tile_heads
from lung_nodule_cnn import decode,consolidate


def extract_required_heads(head,jobs,y0,x0,config):
    """Copy the SAME decoder core+one-neighbor footprint, without zero halos."""
    a=np.asarray(head);p=config.patch//2;halo=config.halo//2
    if config.patch!=128 or config.halo!=32 or a.ndim!=3 or a.shape[0]!=5:raise ValueError('Frozen 128/32 decoder required')
    lo=halo-1;hi=p-halo+1;out=np.empty((len(jobs),5,hi-lo,hi-lo),a.dtype)
    for i,j in enumerate(jobs):
        dy=int(j[1])-y0;dx=int(j[2])-x0
        if dy%8 or dx%8 or min(dy,dx)<0 or dy//2+p>a.shape[1] or dx//2+p>a.shape[2]:raise ValueError('Tile phase/outside canvas')
        out[i]=a[:,dy//2+lo:dy//2+hi,dx//2+lo:dx//2+hi]
    return out,lo


def plane_once(predict,hu,search,jobs,geometry,config,native_xy=None,fast_decode=False,diameter_table=None):
    tick=time.perf_counter()
    if native_xy is None:sampler=SharedPlaneSampler(hu,geometry['spacing_zyx_mm'],jobs,config)
    else:sampler=NativePlaneSampler(hu,geometry['spacing_zyx_mm'],jobs,config,native_xy)
    sample_ms=(time.perf_counter()-tick)*1000
    full,stats=predict(sampler.context)
    begin=time.perf_counter()
    if fast_decode:heads,origin=extract_required_heads(full,jobs,sampler.y0,sampler.x0,config)
    else:heads=extract_tile_heads(full,jobs,sampler.y0,sampler.x0,config);origin=0
    extracted=time.perf_counter()
    decoded=decode_fast(heads,jobs,geometry,config,search,diameter_table,origin) if fast_decode else decode(heads,jobs,geometry,config,search)
    decoded_at=time.perf_counter();candidates=consolidate(*decoded,geometry,config);end=time.perf_counter()
    row=dict(slice_sampling_to_candidates_ms=(end-tick)*1000,sample_ms=sample_ms,
        network_ms=stats['host_total_ms'],input_pack_ms=stats['arm_quantize_stride2_pack_ms'],
        patch_ms=stats['patch_total_ms'],decode_ms=(end-begin)*1000,
        extract_ms=(extracted-begin)*1000,decode_regression_ms=(decoded_at-extracted)*1000,plane_merge_ms=(end-decoded_at)*1000,
        tiles=len(jobs),canvas_hw=list(sampler.context.shape[1:]),plane_candidate_count=len(candidates),details=stats,
        sampler='native_xy_same_z_numpy' if native_xy is not None else 'original_scipy',
        decoder='exact_core_vectorized' if fast_decode else 'original_scalar',
        resampled_native_planes=sampler.resampled_native_planes,resampled_pixels=sampler.resampled_pixels)
    return row,full,candidates,decoded
