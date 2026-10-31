"""Experimental CPU candidate refinement without another neural network.

Uses native HU and existing candidate/Hessian measurements only. References are
not inputs. Original v3 code and defaults remain unchanged. All distances are mm.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

FEATURE_NAMES = (
    'log_response_2d', 'log_response_3d', 'log_blob_score', 'blobness',
    'log_region_diameter', 'log_scale_mm', 'log_elongation', 'mean_hu_scaled',
    'max_hu_scaled', 'local_contrast_scaled', 'local_core_std_scaled',
    'local_occupancy', 'local_elongation', 'axis_continuation',
    'center_shift_mm', 'z_extent_mm', 'xy_extent_mm', 'local_background_scaled',
)


@dataclass(frozen=True)
class RulesConfig:
    minimum_pool_blobness: float = .04
    minimum_pool_blob_score: float = 40.
    minimum_region_diameter_mm: float = 2.5
    maximum_pool_elongation: float = 6.
    center_blend: float = .5
    maximum_center_shift_mm: float = 2.
    merge_distance_mm: float = 2.5
    overlap_fraction: float = .65

    def checked(self):
        if (not np.isfinite(list(asdict(self).values())).all() or
                not 0 <= self.minimum_pool_blobness <= 1 or self.minimum_pool_blob_score < 0 or
                self.minimum_region_diameter_mm <= 0 or self.maximum_pool_elongation < 1 or
                not 0 <= self.center_blend <= 1 or self.maximum_center_shift_mm < 0 or
                self.merge_distance_mm < 0 or not 0 <= self.overlap_fraction <= 1):
            raise ValueError('Invalid refinement configuration')
        return self


def broad_pool(candidates, features, params=None):
    cfg = (params or RulesConfig()).checked()
    if len(candidates) != len(features):
        raise ValueError('Candidate/Hessian identity count mismatch')
    return np.array([i for i, (c,f) in enumerate(zip(candidates,features)) if
        c['equivalent_region_diameter_mm'] >= cfg.minimum_region_diameter_mm and
        c['elongation'] <= cfg.maximum_pool_elongation and
        f['blobness'] >= cfg.minimum_pool_blobness and f['blob_score'] >= cfg.minimum_pool_blob_score], int)


def local_measurement(hu, candidate, spacing, params=None):
    cfg = (params or RulesConfig()).checked()
    spacing = np.asarray(spacing, float)
    center = np.asarray(candidate['center_index_zyx'], float)
    sigma = candidate['estimated_scale_diameter_mm']/(2*np.sqrt(2))
    radius = float(np.clip(1.5*sigma, 2., 7.))
    half = np.ceil(2*radius/spacing).astype(int)+1
    begin = np.maximum(0, np.floor(center).astype(int)-half)
    end = np.minimum(hu.shape, np.ceil(center).astype(int)+half+1)
    sl = tuple(slice(int(a),int(b)) for a,b in zip(begin,end))
    image = ndi.gaussian_filter(np.clip(hu[sl].astype(np.float32),-1000,600),
                                sigma=.5/spacing, mode='reflect', truncate=2.5)
    p = center-begin
    axes = np.ogrid[tuple(slice(0,int(v)) for v in image.shape)]
    distance2 = sum(((g-c)*s)**2 for g,c,s in zip(axes,p,spacing))
    core = distance2 <= radius**2
    ring = (distance2 >= (1.35*radius)**2) & (distance2 <= (1.9*radius)**2)
    background = float(np.median(image[ring])) if ring.any() else float(image.min())
    core_values = image[core]
    top = float(np.percentile(core_values,90))
    contrast = max(0.,top-background)
    support = core & (image >= background+.45*contrast)
    labels,_ = ndi.label(support, np.ones((3,3,3),bool))
    nearest = tuple(np.clip(np.rint(p).astype(int),0,np.array(image.shape)-1))
    number = labels[nearest]
    component = labels==number if number else np.zeros_like(support)
    if not component.any():
        component[nearest]=True
    points = np.argwhere(component)
    weights = np.maximum(image[component]-background,1.)
    centroid = np.average(points,axis=0,weights=weights)+begin
    delta_mm = (centroid-center)*spacing*cfg.center_blend
    length = float(np.linalg.norm(delta_mm))
    if length > cfg.maximum_center_shift_mm:
        delta_mm *= cfg.maximum_center_shift_mm/length
    refined = center+delta_mm/spacing
    physical = points*spacing
    residual = physical-np.average(physical,axis=0,weights=weights)
    covariance = (residual.T*weights)@residual/weights.sum()+np.diag(spacing**2/12)
    eigen, vectors = np.linalg.eigh(covariance)
    elongation = float(np.sqrt(eigen[-1]/max(eigen[0],1e-9)))
    direction = vectors[:,-1]
    endpoints = p[:,None]+direction[:,None]*np.array([-1.6*radius,1.6*radius])/spacing[:,None]
    endpoint_hu = ndi.map_coordinates(image,endpoints,order=1,mode='nearest')
    # A vessel can continue through both sides; a single attachment should not
    # carry the same penalty. Keep this a feature rather than a hard rejection.
    continuation = float(np.clip((min(endpoint_hu)-background)/max(contrast,30.),-2.,2.))
    return refined,dict(contrast=contrast,std=float(core_values.std()),
        occupancy=float(component.sum()/max(core.sum(),1)), elongation=min(elongation,20.),
        continuation=continuation,shift=float(np.linalg.norm(delta_mm)),background=background)


def measure_pool(hu,candidates,features,geometry,params=None):
    cfg=(params or RulesConfig()).checked()
    spacing=np.asarray(geometry['spacing_zyx_mm'],float)
    if (np.ndim(hu)!=3 or spacing.shape!=(3,) or (spacing<=0).any() or
            not np.isfinite(spacing).all()):
        raise ValueError('Need 3-D HU and finite positive spacing')
    indices=broad_pool(candidates,features,cfg)
    matrix=[]; centers=[]; boxes=[]
    for i in indices:
        c,f=candidates[i],features[i]
        refined,local=local_measurement(hu,c,spacing,cfg)
        extents=(np.array(c['bbox_zyx'])[1::2]-np.array(c['bbox_zyx'])[::2])*spacing
        matrix.append([np.log1p(c['score']),np.log1p(f['response_3d_hu']),np.log1p(f['blob_score']),
            f['blobness'],np.log1p(c['equivalent_region_diameter_mm']),np.log1p(f['sigma_mm']),
            np.log1p(c['elongation']),c['mean_hu']/1000,c['max_hu']/1000,
            local['contrast']/1000,local['std']/1000,local['occupancy'],local['elongation'],
            local['continuation'],local['shift'],extents[0],float(max(extents[1:])),local['background']/1000])
        centers.append(refined)
        lo=np.minimum(np.array(c['bbox_zyx'])[::2],np.floor(refined).astype(int))
        hi=np.maximum(np.array(c['bbox_zyx'])[1::2],np.floor(refined).astype(int)+1)
        boxes.append(np.column_stack([lo,hi]).ravel())
    return dict(features=np.array(matrix,np.float32).reshape(-1,len(FEATURE_NAMES)),
        refined_centers_zyx=np.array(centers,float).reshape(-1,3),
        original_centers_zyx=np.array([candidates[i]['center_index_zyx'] for i in indices],float).reshape(-1,3),
        boxes_zyx=np.array(boxes,np.int32).reshape(-1,6),source_indices=indices)


def merge_regions(centers_zyx,boxes_zyx,scores,spacing_zyx_mm,params=None):
    """Score-ordered non-transitive suppression; do not union chains of nodules."""
    cfg=(params or RulesConfig()).checked()
    centers=np.asarray(centers_zyx,float).reshape(-1,3)
    boxes=np.asarray(boxes_zyx,int).reshape(-1,6)
    scores=np.asarray(scores,float)
    spacing=np.asarray(spacing_zyx_mm,float)
    if len(centers)!=len(boxes) or len(scores)!=len(centers) or not np.isfinite(scores).all():
        raise ValueError('Candidate identities/scores differ')
    if not len(centers):return np.array([],int)
    physical=centers*spacing
    tree=cKDTree(physical);removed=np.zeros(len(centers),bool);keep=[]
    extent=(boxes[:,1::2]-boxes[:,::2])*spacing
    if (extent<=0).any():raise ValueError('Invalid half-open boxes')
    volumes=np.prod(extent,axis=1)
    for i in np.lexsort((centers[:,2],centers[:,1],centers[:,0],-scores)):
        if removed[i]:continue
        keep.append(int(i))
        for j in tree.query_ball_point(physical[i],max(cfg.merge_distance_mm,.5*float(np.linalg.norm(extent[i])))):
            if j==i or removed[j]:continue
            distance=float(np.linalg.norm(physical[i]-physical[j]))
            overlap=np.maximum(0,np.minimum(boxes[i,1::2],boxes[j,1::2])-
                                np.maximum(boxes[i,::2],boxes[j,::2]))*spacing
            fraction=float(np.prod(overlap)/min(volumes[i],volumes[j]))
            if distance<=cfg.merge_distance_mm or (fraction>=cfg.overlap_fraction and
                    distance<=.5*min(np.linalg.norm(extent[i]),np.linalg.norm(extent[j]))):
                removed[j]=True
    return np.array(keep,int)


def linear_scores(features,model):
    """Exportable PS implementation: one dot product, no sklearn runtime needed."""
    if tuple(model['feature_names'])!=FEATURE_NAMES:raise ValueError('Feature contract differs')
    x=np.asarray(features,float)
    if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid feature table')
    return x@np.asarray(model['weights'])+float(model['bias'])


def rule_scores(features):
    x=np.asarray(features,float)
    # Contrast/continuation are graded penalties, not additional hard vetoes.
    contrast=np.clip(x[:,9]/.3,.15,1.5)
    continuation=np.clip(1.-.65*np.maximum(x[:,13],0),.15,1.)
    shape=1/np.sqrt(np.maximum(x[:,12],1))
    return np.expm1(x[:,2])*contrast*continuation*shape
