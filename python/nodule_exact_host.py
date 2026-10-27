"""Equivalent host operations for the frozen five-plane nodule detector.

No changes to native coordinates, interpolation, quantization, search envelope,
tile ordering, CNN, decoder, cross-layer scores or thresholds. Caches belong to
one immutable case/run, are bounded, and never contain reference annotations.
"""
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
from scipy import ndimage as ndi
from nodule_canvas_contract import quantize_canvas, shapes

_INTEGER_CODES = np.arange(65536, dtype=np.uint16).view(np.int16).astype(np.float64)
_INTEGER_LUT = np.rint((_INTEGER_CODES.clip(-1000, 400)+1000)*(255./1400.)).astype(np.uint8)
_INTEGER_LUT.flags.writeable = False
del _INTEGER_CODES


def quantize_exact(image):
    a=np.asarray(image)
    if a.dtype != np.dtype(np.int16):
        return quantize_canvas(a)
    if a.ndim != 3 or a.shape[0] != 5:
        raise ValueError('Need five integer context planes')
    shapes(*a.shape[1:])
    return _INTEGER_LUT[a.view(np.uint16)]


def fill_holes_exact(plane):
    """Same default four-connected outside-background flood as fill_holes."""
    a=np.asarray(plane,bool)
    labels,count=ndi.label(~a)  # Default structure is four-connected in 2-D.
    outside=np.zeros(count+1,bool)
    for edge in (labels[0],labels[-1],labels[:,0],labels[:,-1]):outside[edge]=True
    outside[0]=False  # Label zero is the original foreground, never exterior.
    return ~outside[labels]


def search_mask_exact(lung, spacing, config):
    lung=np.asarray(lung,bool)
    filled=np.empty_like(lung)
    for z in range(len(lung)):filled[z]=fill_holes_exact(lung[z])
    if config.margin_mm:
        size=tuple(2*int(np.ceil(config.margin_mm/s))+1 for s in spacing)
        return ndi.maximum_filter(filled,size=size,mode='constant',cval=0)
    return filled


class ExactSamplerCache:
    """Reuse float32 XY interpolation of identical native planes/grid only."""
    def __init__(self, hu, spacing, max_bytes=32*1024*1024, max_planes=12):
        self.hu=hu;self.spacing=np.asarray(spacing,float)
        if self.spacing.shape!=(3,) or not np.isfinite(self.spacing).all() or (self.spacing<=0).any():
            raise ValueError('Need finite positive ZYX spacing')
        if max_bytes<0 or max_planes<0:raise ValueError('Invalid cache bound')
        self.max_bytes=max_bytes;self.max_planes=max_planes
        self.planes=OrderedDict();self.grid_key=None;self.grid=None
        self.hits=0;self.misses=0;self.peak_bytes=0

    def sample(self,jobs,config):
        if not jobs:raise ValueError('Need one nonempty plane')
        spacing=self.spacing;zm=float(jobs[0][0])
        if any(float(j[0])!=zm or int(j[1])!=j[1] or int(j[2])!=j[2] for j in jobs):
            raise ValueError('Mixed planes or fractional tile grid')
        y0=int(min(j[1] for j in jobs));x0=int(min(j[2] for j in jobs))
        y1=int(max(j[1] for j in jobs))+config.patch
        x1=int(max(j[2] for j in jobs))+config.patch
        key=(y0,y1,x0,x1,float(config.xy_mm))
        if key!=self.grid_key:
            y=(y0+np.arange(y1-y0))*config.xy_mm/spacing[1]
            x=(x0+np.arange(x1-x0))*config.xy_mm/spacing[2]
            self.grid=np.meshgrid(y,x,indexing='ij');self.grid_key=key
        planes={};context=[]
        for offset in config.context_mm:
            z=np.clip((zm+offset)/spacing[0],0,len(self.hu)-1)
            lo,hi=int(np.floor(z)),int(np.ceil(z));alpha=z-lo
            for index in {lo,hi}:
                if index in planes:continue
                cache_key=(key,index)
                if cache_key in self.planes:
                    plane=self.planes.pop(cache_key);self.planes[cache_key]=plane;self.hits+=1
                else:
                    plane=ndi.map_coordinates(np.asarray(self.hu[index],np.float32),self.grid,
                        order=1,mode='constant',cval=-1000,prefilter=False)
                    plane.flags.writeable=False;self.misses+=1
                    if self.max_planes and plane.nbytes<=self.max_bytes:
                        self.planes[cache_key]=plane
                        while len(self.planes)>self.max_planes or sum(p.nbytes for p in self.planes.values())>self.max_bytes:
                            self.planes.popitem(last=False)
                        self.peak_bytes=max(self.peak_bytes,sum(p.nbytes for p in self.planes.values()))
                planes[index]=plane
            a=planes[lo]
            if hi!=lo:a=a*(1-alpha)+planes[hi]*alpha
            context.append(np.rint(a).clip(-32768,32767).astype(np.int16))
        value=np.stack(context);value.flags.writeable=False
        return SimpleNamespace(context=value,y0=y0,x0=x0)
