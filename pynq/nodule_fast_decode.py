"""Same peaks, order and scalar diameter formula; vectorized center/search work."""
import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit


class ExactDiameterTable:
    """All 256 values of the frozen INT8 diameter channel, scalar-exact."""
    def __init__(self,scale):
        self.scale=np.float32(scale)
        if not np.isfinite(self.scale) or self.scale<=0:raise ValueError('Invalid diameter scale')
        self.inputs=(np.arange(256,dtype=np.float32)-128)*self.scale
        self.outputs=np.asarray([float(6*np.exp(np.clip(v,-1.8,1.8))) for v in self.inputs],float)

    def values(self,a):
        a=np.asarray(a)
        if a.dtype==np.float32 and np.isfinite(a).all():
            ids=np.rint(a/self.scale+128).astype(np.int32)
            if ((ids>=0)&(ids<256)).all() and np.array_equal((ids.astype(np.float32)-128)*self.scale,a):
                return self.outputs[ids]
        # Generic/nonquantized heads retain original scalar behavior.
        return [float(6*np.exp(np.clip(v,-1.8,1.8))) for v in a]


def decode_fast(prediction,jobs,geometry,config,search=None,diameter_table=None,head_origin=0):
    spacing=np.asarray(geometry['spacing_zyx_mm'])
    pred=np.asarray(prediction)[:len(jobs)]
    if not len(pred):return np.empty((0,3)),np.empty(0),np.empty(0)
    if not np.isfinite(pred).all():raise ValueError('Nonfinite detection head')
    h=config.halo//2;stop=(config.patch-config.halo)//2
    # Filter all tiles together, with extent ONE along the tile axis. Tiles
    # never interact. Row-major nonzero retains the original tile/y/x order.
    if head_origin<0 or head_origin>max(0,h-1):raise ValueError('Invalid compact head origin')
    lo=max(0,h-1-head_origin);hi=min(pred.shape[-1],stop+1-head_origin)
    probability=expit(pred[:,0,lo:hi,lo:hi])
    peaks=(probability>=ndi.maximum_filter(probability,size=(1,3,3),mode='constant'))&(probability>=config.decode_floor)
    peaks[:,:h-head_origin-lo]=False;peaks[:,stop-head_origin-lo:]=False
    peaks[:,:,:h-head_origin-lo]=False;peaks[:,:,stop-head_origin-lo:]=False
    tile,iy,ix=np.nonzero(peaks);iy=iy+lo+head_origin;ix=ix+lo+head_origin
    if not len(tile):return np.empty((0,3)),np.empty(0),np.empty(0)
    regression=np.clip(np.stack([pred[tile,c,iy-head_origin,ix-head_origin] for c in [1,2,3]]),-.5,.5)
    dx,dy,dz=regression
    # Preserve the installed NumPy version's SCALAR promotions, even when a
    # job list mixes Python floats and strong numpy.float64 Z coordinates.
    scalar=np.zeros((),dtype=regression.dtype)[()]
    mul_dtype=np.asarray(scalar*config.z_step_mm).dtype
    product=np.multiply(dz.astype(mul_dtype),config.z_step_mm,dtype=mul_dtype)
    add_types=[np.asarray(job[0]+scalar*config.z_step_mm).dtype.str for job in jobs[:len(pred)]]
    job_values=np.asarray(jobs[:len(pred)],float);z=np.empty(len(tile),float)
    for dtype in set(add_types):
        selected=np.asarray(add_types)[tile]==dtype
        z[selected]=np.add(job_values[tile[selected],0].astype(dtype),product[selected].astype(dtype),dtype=np.dtype(dtype))
    y=(job_values[tile,1]+(iy+dy)*2)*config.xy_mm
    x=(job_values[tile,2]+(ix+dx)*2)*config.xy_mm
    points=np.column_stack((z,y,x))/spacing
    if search is not None:
        q=np.rint(points).astype(int);valid=((q>=0)&(q<np.asarray(search.shape))).all(axis=1)
        take=np.nonzero(valid)[0];valid[take]&=search[tuple(q[take].T)].astype(bool)
        points=points[valid];tile=tile[valid];iy=iy[valid];ix=ix[valid]
    scores=probability[tile,iy-lo-head_origin,ix-lo-head_origin].astype(float)
    values=pred[tile,4,iy-head_origin,ix-head_origin]
    diameters=diameter_table.values(values) if diameter_table is not None else [float(6*np.exp(np.clip(v,-1.8,1.8))) for v in values]
    return points,scores,np.asarray(diameters)
