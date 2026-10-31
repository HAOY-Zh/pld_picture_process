"""Bounded, candidate-local native-slice evidence; no annotation inputs.

Threshold components are image evidence, NOT expert nodule segmentations.
Keep native Z planes: interpolated slices must not count as independent support.
This module never moves candidate centers, deletes candidates or smooths boxes.
"""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit, logit


@dataclass(frozen=True)
class ContinuityConfig:
    thresholds_hu: tuple = (-750., -500., -250.)
    xy_step_mm: float = 1.
    smoothing_mm: float = .6
    seed_radius_mm: float = 2.
    minimum_radius_mm: float = 6.
    maximum_radius_mm: float = 14.
    score_floor: float = .1
    correction_limit: float = 2.


COMPONENT_FEATURES = ('present', 'log_relative_volume', 'axial_span_over_diameter',
    'transverse_span_over_diameter', 'log_elongation', 'center_drift_over_diameter',
    'largest_step_over_diameter', 'adjacent_iou', 'central_area_over_maximum',
    'end_area_over_maximum', 'boundary_fraction', 'native_planes_log1p')


def feature_names(config=ContinuityConfig()):
    return ['log_diameter', 'native_z_spacing_over_diameter'] + [
        f'hu_{int(t)}_{n}' for t in config.thresholds_hu for n in COMPONENT_FEATURES]


def component_statistics(component, center, spacing, diameter):
    """Position changes use mm, not raw slice/pixel counts."""
    q = np.argwhere(component)
    if not len(q):
        return np.zeros(len(COMPONENT_FEATURES), float)
    physical = (q-center)*spacing
    extent = (q.max(axis=0)-q.min(axis=0)+1)*spacing
    centered = physical-physical.mean(axis=0)
    eig = np.linalg.eigvalsh(centered.T@centered/max(len(q), 1))
    elongation = np.sqrt((eig[-1]+.25)/(eig[0]+.25))
    area = component.sum(axis=(1, 2)).astype(float)
    supported = np.flatnonzero(area)
    yy = np.arange(component.shape[1])*spacing[1]
    xx = np.arange(component.shape[2])*spacing[2]
    cy = (component.sum(axis=2)@yy)[supported]/area[supported]
    cx = (component.sum(axis=1)@xx)[supported]/area[supported]
    centers = np.column_stack((cy, cx))
    drift = np.sqrt(np.mean(np.sum((centers-centers.mean(axis=0))**2, axis=1)))
    adjacent = np.diff(supported) == 1
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)[adjacent]
    intersection = (component[:-1] & component[1:]).sum(axis=(1, 2))
    union = (component[:-1] | component[1:]).sum(axis=(1, 2))
    both = (area[:-1] > 0) & (area[1:] > 0)
    mean_iou = np.mean(intersection[both]/union[both]) if both.any() else 0.
    boundary = ((q == 0) | (q == np.asarray(component.shape)-1)).any(axis=1)
    z = int(np.clip(round(center[0]), 0, len(area)-1))
    volume = len(q)*np.prod(spacing)
    return np.array([1., np.log1p(volume/diameter**3), extent[0]/diameter,
        max(extent[1:])/diameter, np.log1p(min(elongation, 50.)), drift/diameter,
        steps.max()/diameter if len(steps) else 0., mean_iou, area[z]/area.max(),
        (area[supported[0]]+area[supported[-1]])/(2*area.max()), boundary.mean(),
        np.log1p(len(supported))], float)


def local_evidence(hu, spacing, candidate, config=ContinuityConfig()):
    """Only hu/geometry/prediction are allowed; no labels or reference boxes."""
    spacing = np.asarray(spacing, float)
    center = np.asarray(candidate['center_index_zyx'], float)
    if (hu.ndim != 3 or spacing.shape != (3,) or center.shape != (3,) or
            not np.isfinite(spacing).all() or (spacing <= 0).any() or
            not np.isfinite(center).all() or (center < 0).any() or
            (center > np.asarray(hu.shape)-1).any()):
        raise ValueError('Need native CT, finite in-bounds center and positive ZYX spacing')
    diameter = float(candidate['estimated_scale_diameter_mm'])
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError('Invalid estimated diameter')
    diameter = float(np.clip(diameter, 2., 36.))
    radius = float(np.clip(diameter*.85+2., config.minimum_radius_mm, config.maximum_radius_mm))
    lo = np.maximum(0, np.floor(center-radius/spacing).astype(int))
    hi = np.minimum(hu.shape, np.ceil(center+radius/spacing).astype(int)+1)
    # Native Z, regular physical XY. All new XY samples stay in native bounds.
    y = np.arange(lo[1], hi[1]-1+1e-6, config.xy_step_mm/spacing[1])
    x = np.arange(lo[2], hi[2]-1+1e-6, config.xy_step_mm/spacing[2])
    grids = np.meshgrid(np.arange(lo[0], hi[0]), y, x, indexing='ij')
    image = ndi.map_coordinates(hu, grids, order=1, mode='nearest', output=np.float32,
                                prefilter=False)
    step = np.array([spacing[0], config.xy_step_mm, config.xy_step_mm])
    local_center = np.array([center[0]-lo[0], (center[1]-y[0])*spacing[1]/step[1],
                             (center[2]-x[0])*spacing[2]/step[2]])
    # No blur between native Z planes: slice support comes from actual planes.
    image = ndi.gaussian_filter(image, sigma=(0, config.smoothing_mm/step[1],
                                               config.smoothing_mm/step[2]), mode='nearest')
    distance = sum(((g-c)*s)**2 for g, c, s in zip(np.ogrid[tuple(slice(0,n) for n in image.shape)],
                                                             local_center, step))
    features = [np.log(diameter), spacing[0]/diameter]
    supports = []
    for threshold in config.thresholds_hu:
        mask = image >= threshold
        seeds = np.argwhere(mask & (distance <= config.seed_radius_mm**2))
        if len(seeds):
            seed = seeds[np.argmin(distance[tuple(seeds.T)])]
            labels, _ = ndi.label(mask, structure=np.ones((3, 3, 3), bool))
            component = labels == labels[tuple(seed)]
        else:
            component = np.zeros_like(mask)
        values = component_statistics(component, local_center, step, diameter)
        features.extend(values)
        supports.append(dict(threshold_hu=threshold, native_planes=int((component.any(axis=(1,2))).sum()),
            adjacent_overlap=float(values[7]), center_drift_over_diameter=float(values[5]),
            evidence_kind='local_HU_component_not_expert_nodule_contour'))
    result = np.asarray(features, float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite local evidence')
    return result, dict(native_z_range=[int(lo[0]), int(hi[0])], supports=supports,
                        estimated_diameter_mm=diameter, window_radius_mm=radius)


def evidence_for_packet(hu, spacing, candidates, config=ContinuityConfig()):
    start = perf_counter()
    selected = [i for i, c in enumerate(candidates) if c['score'] >= config.score_floor]
    matrix = np.zeros((len(candidates), len(feature_names(config))), float)
    details = []
    for i in selected:
        matrix[i], detail = local_evidence(hu, spacing, candidates[i], config)
        details.append(dict(candidate_index=i, candidate_id=candidates[i]['candidate_id'], **detail))
    return matrix, dict(selected_indices=selected, details=details,
        feature_seconds=perf_counter()-start, candidates_total=len(candidates),
        candidates_examined=len(selected), native_slices_only=True, config=asdict(config))


def rerank_scores(scores, features, ranker, config=ContinuityConfig()):
    """Bounded soft change; no forced multi-slice rule, no new candidate."""
    scores = np.asarray(scores, float)
    features = np.asarray(features, float)
    if features.shape != (len(scores), len(feature_names(config))):
        raise ValueError('Feature/score alignment mismatch')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid original scores')
    if not np.isfinite(features).all():
        raise ValueError('Invalid evidence')
    output = scores.copy()
    selected = scores >= config.score_floor
    if not ranker['alpha'] or not selected.any():
        return output
    standardized = (features[selected]-np.asarray(ranker['mean']))/np.asarray(ranker['scale'])
    correction = standardized@np.asarray(ranker['coef'])+ranker['intercept']
    correction = np.clip(correction, -config.correction_limit, config.correction_limit)
    output[selected] = expit(logit(np.clip(scores[selected], 1e-7, 1-1e-7))+ranker['alpha']*correction)
    return output
