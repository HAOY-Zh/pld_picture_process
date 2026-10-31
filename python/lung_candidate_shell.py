"""Physical 3-D shell gradient/context descriptors on small native-HU patches.

Inspired by shell-gradient CAD literature, not an exact paper reproduction.
Fixed antipodal directions and radii use candidate scale, never expert labels.
No full-volume resampling or new network is needed.
"""
import numpy as np
from scipy import ndimage as ndi

SHELL_RADII = (.8, 1.25, 1.8)
FEATURE_NAMES = tuple(f'shell_{r}_{name}' for r in ('inner','middle','outer') for name in (
    'mean_hu_scaled','std_hu_scaled','gradient_magnitude_scaled','inward_gradient_alignment',
    'antipodal_difference_scaled','bright_fraction'))+(
    'shell_core_mean_scaled','shell_background_scaled','shell_contrast_scaled')


def directions():
    index = np.arange(21); z = (index+.5)/21
    phi = index*np.pi*(3-np.sqrt(5)); radial = np.sqrt(1-z*z)
    half = np.column_stack((z, radial*np.sin(phi), radial*np.cos(phi)))
    return np.concatenate((half,-half))


def measure(hu, center_zyx, spacing, sigma_mm):
    spacing = np.asarray(spacing,float); center = np.asarray(center_zyx,float)
    if (np.ndim(hu)!=3 or spacing.shape!=(3,) or center.shape!=(3,) or
            not np.isfinite(spacing).all() or (spacing<=0).any() or
            not np.isfinite(center).all() or not np.isfinite(sigma_mm) or sigma_mm<=0):
        raise ValueError('Invalid physical shell geometry')
    radius=float(np.clip(np.sqrt(2)*sigma_mm,1.5,12.))
    half=np.ceil((1.8*radius+2.5)/spacing).astype(int)+1
    begin=np.maximum(0,np.floor(center).astype(int)-half)
    end=np.minimum(hu.shape,np.ceil(center).astype(int)+half+1)
    if (end-begin<2).any(): raise ValueError('Shell patch outside image')
    image=ndi.gaussian_filter(np.clip(hu[tuple(slice(a,b) for a,b in zip(begin,end))].astype(np.float32),
        -1000,600),sigma=.6/spacing,mode='reflect',truncate=2.5)
    gradient=np.gradient(image,*spacing)
    unit=directions(); p=center-begin
    def sample(array,factor):
        coords=p[:,None]+(unit*factor*radius/spacing).T
        return ndi.map_coordinates(array,coords,order=1,mode='nearest')
    values=[sample(image,r) for r in SHELL_RADII]
    core=float(sample(image,.25).mean());background=float(np.percentile(values[-1],25))
    contrast=max(0.,core-background);threshold=background+.5*max(contrast,30.)
    result=[]
    for factor,gray in zip(SHELL_RADII,values):
        g=np.column_stack([sample(a,factor) for a in gradient])
        magnitude=np.linalg.norm(g,axis=1)
        # Outward unit vector and inward gradient yield positive alignment.
        alignment=float(np.sum(-np.sum(g*unit,axis=1))/max(float(magnitude.sum()),1e-6))
        result.extend((float(gray.mean())/1000,float(gray.std())/1000,
            float(magnitude.mean())/1000,alignment,
            float(np.mean(np.abs(gray[:21]-gray[21:])))/1000,float((gray>=threshold).mean())))
    return np.asarray([*result,core/1000,background/1000,contrast/1000],np.float32)
