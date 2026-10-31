"""Reuse identical native-plane XY interpolation across overlapping Z contexts.

The reference sample_patch arithmetic, coordinates, padding and int16 rounding
are retained. Cache lifetime belongs to one FloatPlaneCache/CT. No new resampling
grid and no change to neural-network input values or detection semantics.
"""
from collections import OrderedDict
import numpy as np
from scipy import ndimage as ndi


def sample_patch_reuse(hu,spacing,job,config):
    if not hasattr(hu,'planes'):
        raise TypeError('Use one FloatPlaneCache per native CT')
    signature=(tuple(spacing),config.patch,config.xy_mm,tuple(config.context_mm))
    if not hasattr(hu,'bilinear_tiles'):
        hu.bilinear_tiles=OrderedDict()
        hu.coordinate_grids={}
        hu.sampling_signature=signature
    if hu.sampling_signature!=signature:
        raise ValueError('Do not reuse sampling cache across geometry/config changes')
    zm,y0,x0=job
    xy=(y0,x0)
    if xy not in hu.coordinate_grids:
        y=(y0+np.arange(config.patch))*config.xy_mm/spacing[1]
        x=(x0+np.arange(config.patch))*config.xy_mm/spacing[2]
        hu.coordinate_grids[xy]=np.meshgrid(y,x,indexing='ij')
    grid=hu.coordinate_grids[xy]
    output=np.empty((5,config.patch,config.patch),np.float32)

    def plane(z):
        key=(z,y0,x0)
        if key not in hu.bilinear_tiles:
            hu.bilinear_tiles[key]=ndi.map_coordinates(np.asarray(hu[z],np.float32),grid,
                order=1,mode='constant',cval=-1000,prefilter=False)
            if len(hu.bilinear_tiles)>1024:
                hu.bilinear_tiles.popitem(last=False)
        hu.bilinear_tiles.move_to_end(key)
        return hu.bilinear_tiles[key]

    for i,offset in enumerate(config.context_mm):
        z=np.clip((zm+offset)/spacing[0],0,len(hu)-1)
        lo,hi=int(np.floor(z)),int(np.ceil(z));alpha=z-lo
        a=plane(lo)
        if hi!=lo:
            b=plane(hi)
            a=a*(1-alpha)+b*alpha
        output[i]=a
    return np.rint(output).clip(-32768,32767).astype(np.int16)
