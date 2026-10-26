"""Boundary-compatible evidence adapter for frozen continuity v1.

Match the original decoder's nearest-native-voxel gate. Centers are not moved.
Interior results are exactly the original implementation. Only half-voxel edge
centers use a coordinate-equivalent local view, with NO interpolated Z evidence.
"""
from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from .lung_nodule_continuity import (ContinuityConfig, feature_names, local_evidence,
                                    component_statistics)


def compatible_evidence(hu, spacing, candidate, config=ContinuityConfig()):
    center = np.asarray(candidate['center_index_zyx'],float)
    shape = np.asarray(hu.shape)
    if ((center >= 0) & (center <= shape-1)).all():
        return local_evidence(hu,spacing,candidate,config)
    q = np.rint(center).astype(int)
    if not np.isfinite(center).all() or (q < 0).any() or (q >= shape).any():
        raise ValueError('Center outside original nearest-voxel decode gate')
    spacing = np.asarray(spacing,float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Invalid native spacing')
    diameter = float(candidate['estimated_scale_diameter_mm'])
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError('Invalid estimated diameter')
    diameter = float(np.clip(diameter,2.,36.))
    radius = float(np.clip(diameter*.85+2.,config.minimum_radius_mm,config.maximum_radius_mm))
    lo = np.maximum(0,np.floor(center-radius/spacing).astype(int))
    hi = np.minimum(hu.shape,np.ceil(center+radius/spacing).astype(int)+1)
    y = np.arange(lo[1],hi[1]-1+1e-6,config.xy_step_mm/spacing[1])
    x = np.arange(lo[2],hi[2]-1+1e-6,config.xy_step_mm/spacing[2])
    grids = np.meshgrid(np.arange(lo[0],hi[0]),y,x,indexing='ij')
    image = ndi.map_coordinates(hu,grids,order=1,mode='nearest',output=np.float32,prefilter=False)
    step = np.array([spacing[0],config.xy_step_mm,config.xy_step_mm])
    local_center = np.array([center[0]-lo[0],(center[1]-y[0])*spacing[1]/step[1],
                             (center[2]-x[0])*spacing[2]/step[2]])
    image = ndi.gaussian_filter(image,sigma=(0,config.smoothing_mm/step[1],
        config.smoothing_mm/step[2]),mode='nearest')
    distance = sum(((g-c)*s)**2 for g,c,s in zip(np.ogrid[tuple(slice(0,n) for n in image.shape)],local_center,step))
    features = [np.log(diameter),spacing[0]/diameter]
    supports = []
    for threshold in config.thresholds_hu:
        mask = image >= threshold
        seeds = np.argwhere(mask & (distance <= config.seed_radius_mm**2))
        if len(seeds):
            seed = seeds[np.argmin(distance[tuple(seeds.T)])]
            labels,_ = ndi.label(mask,structure=np.ones((3,3,3),bool))
            component = labels == labels[tuple(seed)]
        else:
            component = np.zeros_like(mask)
        values = component_statistics(component,local_center,step,diameter)
        features.extend(values)
        supports.append(dict(threshold_hu=threshold,native_planes=int(component.any(axis=(1,2)).sum()),
            adjacent_overlap=float(values[7]),center_drift_over_diameter=float(values[5]),
            evidence_kind='local_HU_component_not_expert_nodule_contour'))
    result = np.asarray(features,float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite edge evidence')
    return result,dict(native_z_range=[int(lo[0]),int(hi[0])],supports=supports,
        estimated_diameter_mm=diameter,window_radius_mm=radius)


def evidence_for_packet(hu,spacing,candidates,config=ContinuityConfig()):
    start = perf_counter()
    selected = [i for i,c in enumerate(candidates) if c['score'] >= config.score_floor]
    matrix = np.zeros((len(candidates),len(feature_names(config))),float)
    details = []
    edge = 0
    for i in selected:
        center = np.asarray(candidates[i]['center_index_zyx'])
        edge += int(((center < 0) | (center > np.asarray(hu.shape)-1)).any())
        matrix[i],detail = compatible_evidence(hu,spacing,candidates[i],config)
        details.append(dict(candidate_index=i,candidate_id=candidates[i]['candidate_id'],**detail))
    return matrix,dict(selected_indices=selected,details=details,
        feature_seconds=perf_counter()-start,candidates_total=len(candidates),
        candidates_examined=len(selected),native_slices_only=True,config=asdict(config),
        half_voxel_boundary_candidates=edge)
