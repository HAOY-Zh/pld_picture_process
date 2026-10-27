"""Optional stronger full-volume island filtering; original mask is retained.

Only prediction geometry is used. Reference masks never select components.
This changes display/postprocessing and must be evaluated separately from the
lossless acceleration work. Disconnected true lung can exist, so the largest
two-component shortcut is deliberately not used.
"""
from time import perf_counter
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

POLICY=dict(minimum_main_ml=50.,anchor_fraction=.10,far_distance_mm=20.,
    far_max_fraction=.03,far_floor_ml=5.,short_max_layers=3,short_max_mm=6.,
    short_max_fraction=.005,short_floor_ml=2.,short_min_distance_mm=15.)


def clean_structural(prediction,spacing,*,whole_volume=True,policy=None):
    config=dict(POLICY if policy is None else policy)
    if set(config)!=set(POLICY)or not np.isfinite(list(config.values())).all():
        raise ValueError('Need the declared finite complete policy')
    if any(v<=0 for v in config.values())or config['short_max_layers']!=int(config['short_max_layers']):
        raise ValueError('Need positive limits and integer short layer count')
    if any(not 0<config[k]<1 for k in ('anchor_fraction','far_max_fraction','short_max_fraction')):
        raise ValueError('Need fractions in (0,1)')
    a=np.asarray(prediction)
    spacing=np.asarray(spacing,float)
    if a.ndim!=3 or not a.size or not np.isin(a,[0,1,255]).all():raise ValueError('Need 3-D binary prediction')
    if spacing.shape!=(3,)or not np.isfinite(spacing).all()or(spacing<=0).any():raise ValueError('Need positive physical spacing')
    a=a!=0;start=perf_counter()
    if not whole_volume:
        return a.copy(),dict(applied=False,reason='Partial input is protected',policy=config)
    labels,count=ndi.label(a,structure=ndi.generate_binary_structure(3,3))
    sizes=np.bincount(labels.ravel(),minlength=count+1);sizes[0]=0
    volumes=sizes*float(np.prod(spacing)/1000);largest=float(volumes.max())
    if largest<config['minimum_main_ml']:
        return a.copy(),dict(applied=False,reason='No reliable large lung anchor',policy=config,largest_ml=largest)
    anchors=np.flatnonzero((volumes>=max(config['minimum_main_ml'],largest*config['anchor_fraction']))&(sizes>0))
    anchor_lookup=np.zeros(count+1,bool);anchor_lookup[anchors]=True
    anchor_mask=anchor_lookup[labels]
    surface=anchor_mask&~ndi.binary_erosion(anchor_mask,border_value=0)
    tree=cKDTree(np.argwhere(surface).astype(float)*spacing)
    del anchor_mask,surface
    far_limit=max(config['far_floor_ml'],largest*config['far_max_fraction'])
    short_limit=max(config['short_floor_ml'],largest*config['short_max_fraction'])
    boxes=ndi.find_objects(labels);components=[];removed=[]
    for i,box in enumerate(boxes,1):
        if box is None:continue
        layers=box[0].stop-box[0].start;span_mm=layers*spacing[0]
        far_eligible=not anchor_lookup[i]and volumes[i]<=far_limit
        short_eligible=(not anchor_lookup[i]and volumes[i]<=short_limit and
            layers<=config['short_max_layers']and span_mm<=config['short_max_mm'])
        distance=None;reasons=[]
        if far_eligible or short_eligible:
            points=(np.argwhere(labels[box]==i)+[s.start for s in box])*spacing
            distance=float(tree.query(points,workers=1)[0].min())
            if far_eligible and distance>config['far_distance_mm']:reasons.append('small_relative_to_main_and_far')
            if short_eligible and distance>config['short_min_distance_mm']:reasons.append('few_native_layers_small_and_detached')
        if reasons:removed.append(i)
        components.append(dict(id=i,voxels=int(sizes[i]),volume_ml=float(volumes[i]),
            anchor=bool(anchor_lookup[i]),native_layers=int(layers),z_extent_mm=float(span_mm),
            nearest_anchor_mm=distance,bbox_zyx=[[s.start,s.stop]for s in box],remove_reasons=reasons))
    keep=np.ones(count+1,bool);keep[0]=False;keep[removed]=False
    result=keep[labels]
    return result,dict(applied=True,policy=config,connectivity=26,anchors=anchors.tolist(),
        largest_ml=largest,far_volume_limit_ml=float(far_limit),short_volume_limit_ml=float(short_limit),
        removed_component_ids=removed,removed_voxels=int(sizes[removed].sum()),components=components,
        original_prediction_modified=False,geometry_changed=False,expert_used=False,
        cpu_ms=(perf_counter()-start)*1000,
        warning='Optional structural cleanup; not proof that every removed island was false lung.')


def clean_for_display(prediction,spacing,**legacy_parameters):
    """Explicit preview policy: retain original conservative removals as well."""
    from lung_island_cleanup import clean_prediction
    legacy,legacy_info=clean_prediction(prediction,spacing,**legacy_parameters)
    candidate,info=clean_structural(prediction,spacing)
    candidate &= legacy
    info.update(kind='structural_preview_v2',legacy_policy=legacy_parameters,
        legacy_removed_component_ids=legacy_info['removed_component_ids'],
        original_conservative_mask_sha256=__import__('hashlib').sha256(legacy.tobytes()).hexdigest(),
        additional_removed_voxels=int((legacy&~candidate).sum()))
    return candidate,info
