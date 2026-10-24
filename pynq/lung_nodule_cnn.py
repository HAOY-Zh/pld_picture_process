"""Experimental lung-guided 2.5D center detector. No annotations at inference.

Uses native HU, a supplied lung prediction and millimetre geometry. This is a
new CPU/CUDA reference, not a reproduction or an FPGA deployment.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DetectorConfig:
    patch: int = 128
    halo: int = 32
    xy_mm: float = .8
    z_step_mm: float = 1.25
    context_mm: tuple = (-2.5, -1.25, 0., 1.25, 2.5)
    margin_mm: float = 5.
    output_stride: int = 2
    base: int = 8
    decode_floor: float = .02
    merge_mm: float = 2.5

    def checked(self):
        if (self.patch % 8 or self.output_stride != 2 or self.halo % 2 or
                not 0 <= 2*self.halo < self.patch or self.xy_mm <= 0 or self.z_step_mm <= 0 or
                self.margin_mm < 0 or len(self.context_mm) != 5 or self.base < 4 or
                not 0 < self.decode_floor < 1 or self.merge_mm < 0):
            raise ValueError('Invalid detector configuration')
        return self


def config_from_dict(value):
    return DetectorConfig(**dict(value, context_mm=tuple(value['context_mm']))).checked()


def model(config):
    import torch
    from torch import nn
    c = config.checked().base

    def conv(a, b, stride=1):
        return nn.Sequential(nn.Conv2d(a, b, 3, stride=stride, padding=1), nn.ReLU())

    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(conv(5, c, 2), conv(c, c))
            self.enc1 = nn.Sequential(nn.MaxPool2d(2), conv(c, 2*c))
            self.enc2 = nn.Sequential(nn.MaxPool2d(2), conv(2*c, 3*c))
            self.dec1 = conv(5*c, 2*c)
            self.dec0 = conv(3*c, c)
            self.head = nn.Conv2d(c, 5, 1)  # center, dx, dy, dz, log diameter
            nn.init.normal_(self.head.weight, std=.001)
            nn.init.zeros_(self.head.bias)
            self.head.bias.data[0] = -4.6

        def forward(self, x):
            a = self.stem(x); b = self.enc1(a); d = self.enc2(b)
            d = self.dec1(torch.cat((nn.functional.interpolate(d, size=b.shape[-2:], mode='nearest'), b), 1))
            d = self.dec0(torch.cat((nn.functional.interpolate(d, size=a.shape[-2:], mode='nearest'), a), 1))
            return self.head(d)

    return Detector()


def z_planes(shape, spacing, config):
    return np.arange(int(np.ceil((shape[0]-1)*spacing[0]/config.z_step_mm))+1)*config.z_step_mm


def search_mask(lung, spacing, config):
    """Three-dimensional conservative envelope; supplied lung mask is unchanged."""
    lung = np.asarray(lung, bool)
    filled = np.empty_like(lung)
    for z in range(len(lung)):
        filled[z] = ndi.binary_fill_holes(lung[z])
    if config.margin_mm:
        # Rectangular expansion is conservative, inexpensive and documented.
        size = tuple(2*int(np.ceil(config.margin_mm/s))+1 for s in spacing)
        return ndi.maximum_filter(filled, size=size, mode='constant', cval=0)
    return filled


def tiles(search, spacing, config):
    """Full regular grid, skip only lung-empty cores; no reference coordinates."""
    p, h = config.patch, config.halo
    core = p-2*h
    ny = int(np.ceil(search.shape[1]*spacing[1]/config.xy_mm))
    nx = int(np.ceil(search.shape[2]*spacing[2]/config.xy_mm))
    yy = np.minimum(np.rint(np.arange(ny)*config.xy_mm/spacing[1]).astype(int), search.shape[1]-1)
    xx = np.minimum(np.rint(np.arange(nx)*config.xy_mm/spacing[2]).astype(int), search.shape[2]-1)
    jobs = []; full = 0
    for zm in z_planes(search.shape, spacing, config):
        z = min(int(round(zm/spacing[0])), len(search)-1)
        plane = search[z][np.ix_(yy, xx)]
        for y in range(0, ny, core):
            for x in range(0, nx, core):
                full += 1
                if plane[y:min(y+core,ny), x:min(x+core,nx)].any():
                    jobs.append((float(zm), int(y-h), int(x-h)))
    return jobs, dict(dense_grid_tiles=full, lung_tiles=len(jobs),
                     skipped_tiles=full-len(jobs), actual_input_pixels=len(jobs)*p*p,
                     note='Dense CNN computes all pixels in each selected tile, including halo/padding')


def sample_patch(hu, spacing, job, config):
    zm, y0, x0 = job
    y = (y0+np.arange(config.patch))*config.xy_mm/spacing[1]
    x = (x0+np.arange(config.patch))*config.xy_mm/spacing[2]
    yy, xx = np.meshgrid(y, x, indexing='ij')
    output = np.empty((5, config.patch, config.patch), np.float32)
    for i, offset in enumerate(config.context_mm):
        # Bilinear XY and linear Z. Only the required slices are converted to float.
        z = np.clip((zm+offset)/spacing[0], 0, len(hu)-1)
        lo, hi = int(np.floor(z)), int(np.ceil(z)); alpha = z-lo
        a = ndi.map_coordinates(np.asarray(hu[lo], np.float32), [yy, xx], order=1,
                                mode='constant', cval=-1000, prefilter=False)
        if hi != lo:
            b = ndi.map_coordinates(np.asarray(hu[hi], np.float32), [yy, xx], order=1,
                                    mode='constant', cval=-1000, prefilter=False)
            a = a*(1-alpha)+b*alpha
        output[i] = a
    return np.rint(output).clip(-32768,32767).astype(np.int16)


def relative_labels(nodules, excluded, geometry, job, config):
    """Return local centers in input pixels and z offset in z-step units."""
    basis = np.asarray(geometry['basis_lps_zyx']); origin = np.asarray(geometry['origin_lps_mm'])
    spacing = np.asarray(geometry['spacing_zyx_mm'])
    positive = []; ignored = []
    for annotation, is_excluded in [(n, False) for n in nodules]+[(n, True) for n in excluded]:
        z, y, x = np.linalg.solve(basis, np.asarray(annotation['center_lps_xyz_mm'])-origin)*spacing
        dz = z-job[0]; diameter = annotation['diameter_mm']
        diameter = diameter if diameter > 0 else 10.
        px, py = x/config.xy_mm-job[2], y/config.xy_mm-job[1]
        if abs(dz) > diameter/2+config.z_step_mm or not (-diameter/config.xy_mm < px < config.patch+diameter/config.xy_mm and
                                                       -diameter/config.xy_mm < py < config.patch+diameter/config.xy_mm):
            continue
        if not is_excluded and abs(dz) <= config.z_step_mm/2+1e-7:
            positive.append([px, py, dz/config.z_step_mm, diameter])
        else:
            # Findings without reference status and nodules in neighbouring z bins
            # are ignored here, never silently relabeled as healthy background.
            ignored.append([px, py, diameter])
    return positive, ignored


def targets(positive, ignored, config):
    s = config.output_stride; n = config.patch//s
    heat = np.zeros((n,n),np.float32); valid = np.ones((n,n),np.float32)
    reg = np.zeros((4,n,n),np.float32); centers = np.zeros((n,n),np.float32)
    yy, xx = np.ogrid[:n,:n]
    for x,y,d in ignored:
        radius = max(2.,d/(2*config.xy_mm*s))
        valid[(xx-x/s)**2+(yy-y/s)**2 <= radius**2] = 0
    for x,y,z,d in positive:
        px,py = x/s,y/s; ix,iy = int(round(px)),int(round(py))
        if not 0 <= ix < n or not 0 <= iy < n:
            continue
        sigma = np.clip(d/(6*config.xy_mm*s),.65,3.)
        gaussian = np.exp(-((xx-ix)**2+(yy-iy)**2)/(2*sigma*sigma))
        np.maximum(heat,gaussian,out=heat)
        heat[iy,ix] = 1; valid[iy,ix] = 1; centers[iy,ix] = 1
        reg[:,iy,ix] = (px-ix,py-iy,z,np.log(d/6.))
    return heat,valid,reg,centers


def loss(prediction, target):
    import torch
    import torch.nn.functional as f
    heat, valid, reg, center = target
    p = prediction[:,0].sigmoid().clamp(1e-5,1-1e-5)
    pos = center
    neg = (heat < 1).float()*valid
    norm = pos.sum().clamp(min=1)
    classification = (-(1-p).pow(2)*p.log()*pos-p.pow(2)*(1-p).log()*(1-heat).pow(4)*neg).sum()/norm
    offsets = f.smooth_l1_loss(prediction[:,1:4],reg[:,:3],reduction='none')
    diameter = f.smooth_l1_loss(prediction[:,4],reg[:,3],reduction='none')
    regression = (offsets*center[:,None]).sum()/norm+.2*(diameter*center).sum()/norm
    return classification+regression, classification.detach(), regression.detach()


def decode(prediction, jobs, geometry, config, search=None):
    """Heatmap local maxima, regression and physical coordinates. No annotations."""
    from scipy.special import expit
    spacing = np.asarray(geometry['spacing_zyx_mm'])
    basis = np.asarray(geometry['basis_lps_zyx']); origin = np.asarray(geometry['origin_lps_mm'])
    points = []; scores = []; diameters = []
    h = config.halo//2; stop = (config.patch-config.halo)//2
    for pred, job in zip(prediction, jobs):
        probability = expit(pred[0])
        peaks = (probability >= ndi.maximum_filter(probability, size=3, mode='constant')) & (probability >= config.decode_floor)
        peaks[:h] = False; peaks[stop:] = False; peaks[:,:h] = False; peaks[:,stop:] = False
        for iy, ix in np.argwhere(peaks):
            dx,dy,dz = np.clip(pred[1:4,iy,ix],-.5,.5)
            point = np.array([job[0]+dz*config.z_step_mm,
                              (job[1]+(iy+dy)*2)*config.xy_mm,
                              (job[2]+(ix+dx)*2)*config.xy_mm])/spacing
            if search is not None:
                q = np.rint(point).astype(int)
                if (q < 0).any() or (q >= search.shape).any() or not search[tuple(q)]:
                    continue
            points.append(point);scores.append(float(probability[iy,ix]));diameters.append(float(6*np.exp(np.clip(pred[4,iy,ix],-1.8,1.8))))
    return np.asarray(points,float).reshape(-1,3),np.asarray(scores),np.asarray(diameters)


def consolidate(points, scores, diameters, geometry, config):
    points=np.asarray(points).reshape(-1,3)
    if not len(points):return []
    spacing=np.asarray(geometry['spacing_zyx_mm']);physical=points*spacing
    order=np.lexsort((points[:,2],points[:,1],points[:,0],-scores));tree=cKDTree(physical)
    suppressed=np.zeros(len(points),bool);result=[]
    basis=np.asarray(geometry['basis_lps_zyx']);origin=np.asarray(geometry['origin_lps_mm'])
    for i in order:
        if suppressed[i]:continue
        radius=max(config.merge_mm,min(5.,diameters[i]*.3))
        suppressed[tree.query_ball_point(physical[i],radius)]=True
        lo=np.floor(points[i]-diameters[i]/(2*spacing)).astype(int)
        hi=np.ceil(points[i]+diameters[i]/(2*spacing)).astype(int)+1
        result.append(dict(candidate_id=f'cnn_{len(result)+1:05d}',center_index_zyx=points[i].tolist(),
            center_lps_xyz_mm=(origin+basis@points[i]).tolist(),bbox_zyx=np.column_stack((lo,hi)).ravel().tolist(),
            score=float(scores[i]),estimated_scale_diameter_mm=float(diameters[i]),
            score_kind='CNN_center_score_not_calibrated_probability',region_kind='predicted_center_and_size_not_voxel_segmentation'))
    return result
