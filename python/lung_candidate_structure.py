"""3-D physical Hessian features to refine software nodule candidates.

No annotations are accepted. The original high-recall points stay available for
development comparisons. This is a CPU reference, not FPGA implementation.
"""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class StructureConfig:
    minimum_blobness: float = .10
    minimum_3d_response_hu: float = 30.
    minimum_blob_score: float = 15.
    minimum_region_diameter_mm: float = 2.5
    maximum_elongation: float = 4.
    minimum_merge_mm: float = 3.

    def checked(self):
        if (not np.isfinite(list(asdict(self).values())).all() or
                not 0 <= self.minimum_blobness <= 1 or min(self.minimum_3d_response_hu,
                self.minimum_blob_score, self.minimum_region_diameter_mm, self.minimum_merge_mm) < 0 or
                self.maximum_elongation < 1):
            raise ValueError('Invalid structure configuration')
        return self


def hessian_features(hu, candidates, spacing_zyx_mm, truncate=3.5):
    """Gaussian smooth in millimetres, then sample all six physical Hessian entries.

    Discrete central second differences / spacing^2. Eigenvalue ordering is
    ascending, bright spherical blobs have three negative curvatures. A tube
    typically has one weak longitudinal curvature. Ratios are experimental.
    """
    spacing = np.asarray(spacing_zyx_mm, float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Invalid physical spacing')
    if np.ndim(hu) != 3 or not np.isfinite(hu).all():
        raise ValueError('Need finite 3-D native HU')
    if not candidates:
        return []
    points = np.asarray([c['center_index_zyx'] for c in candidates], int)
    if ((points < 0) | (points >= np.asarray(hu.shape))).any():
        raise ValueError('Candidate center outside HU')
    sigma_values = np.asarray([c['estimated_scale_diameter_mm']/(2*np.sqrt(2)) for c in candidates])
    if not np.isfinite(sigma_values).all() or (sigma_values <= 0).any():
        raise ValueError('Invalid candidate scale')
    output = [None] * len(candidates)
    max_sigma = float(sigma_values.max())
    halo = np.ceil(truncate*max_sigma/spacing).astype(int)+2
    begin = np.maximum(0, points.min(axis=0)-halo)
    end = np.minimum(hu.shape, points.max(axis=0)+halo+1)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(begin, end))
    image = np.clip(np.asarray(hu[slices], np.float32), -1000., 600.)+1000.
    local = points-begin
    for sigma in sorted(set(np.round(sigma_values, 8))):
        indices = np.flatnonzero(np.isclose(sigma_values, sigma, atol=1e-7, rtol=0))
        p = local[indices]
        smooth = ndi.gaussian_filter(image, sigma=sigma/spacing, mode='reflect', truncate=truncate)

        def sample(delta):
            q = np.clip(p+np.asarray(delta), 0, np.asarray(smooth.shape)-1)
            return smooth[tuple(q.T)].astype(np.float64)

        center = sample([0, 0, 0])
        hessian = np.empty((len(p), 3, 3), np.float64)
        unit = np.eye(3, dtype=int)
        for a in range(3):
            hessian[:, a, a] = (sample(unit[a])-2*center+sample(-unit[a]))/spacing[a]**2
            for b in range(a+1, 3):
                v = (sample(unit[a]+unit[b])-sample(unit[a]-unit[b])-
                     sample(-unit[a]+unit[b])+sample(-unit[a]-unit[b]))/(4*spacing[a]*spacing[b])
                hessian[:, a, b] = hessian[:, b, a] = v
        eigen = np.linalg.eigvalsh(hessian)*sigma**2
        ratios = np.clip(-eigen[:, 2]/np.maximum(-eigen[:, 0], 1e-9), 0, 1)
        response = np.maximum(0, -eigen.sum(axis=1))
        score = response*np.sqrt(ratios)
        for j, index in enumerate(indices):
            output[index] = dict(hessian_eigenvalues=eigen[j].tolist(), blobness=float(ratios[j]),
                response_3d_hu=float(response[j]), blob_score=float(score[j]), sigma_mm=float(sigma))
        del smooth
    return output


def select_structural_candidates(candidates, features, geometry, params=None):
    cfg = (params or StructureConfig()).checked()
    if len(candidates) != len(features):
        raise ValueError('Candidate/feature count mismatch')
    eligible = [i for i, (c, f) in enumerate(zip(candidates, features)) if
                f['blobness'] >= cfg.minimum_blobness and f['response_3d_hu'] >= cfg.minimum_3d_response_hu and
                f['blob_score'] >= cfg.minimum_blob_score and
                c['equivalent_region_diameter_mm'] >= cfg.minimum_region_diameter_mm and
                c['elongation'] <= cfg.maximum_elongation]
    eligible.sort(key=lambda i: (-features[i]['blob_score'], tuple(candidates[i]['center_index_zyx'])))
    if not eligible:
        return []
    positions = np.asarray([candidates[i]['center_lps_xyz_mm'] for i in eligible])
    tree = cKDTree(positions)
    suppressed = np.zeros(len(eligible), bool)
    result = []
    for j, i in enumerate(eligible):
        if suppressed[j]:
            continue
        radius = max(cfg.minimum_merge_mm, min(6., 1.5*features[i]['sigma_mm']))
        suppressed[tree.query_ball_point(positions[j], radius)] = True
        c = dict(candidates[i])
        c.update(source_candidate_id=c['candidate_id'], candidate_id=f'candidate_{len(result)+1:05d}',
                 score=features[i]['blob_score'], score_kind='3d_curvature_blob_response_not_probability',
                 structure_features=features[i])
        result.append(c)
    return result


def refine_candidates(hu, candidates, geometry, params=None):
    start = perf_counter()
    features = hessian_features(hu, candidates, geometry['spacing_zyx_mm'])
    result = select_structural_candidates(candidates, features, geometry, params)
    return result, features, (perf_counter()-start)*1000


BALANCED_PILOT_CONFIG = StructureConfig(minimum_blobness=.2, minimum_blob_score=180.,
                                      minimum_region_diameter_mm=2.5)


def extract_suspicious_regions(hu, lung_mask, geometry, candidate_params=None, structure_params=None):
    """Callable complete software module; supplied lung mask must be on native CT grid.

    Default structure parameters are selected on the 12 development scans, not
    validated on the reserved scans. Caller supplies predicted lung mask, never
    reference nodule annotations. The routine does not run or claim FPGA work.
    """
    from lung_candidates import detect_candidates
    start = perf_counter()
    result, seed = detect_candidates(hu, lung_mask, geometry, candidate_params)
    structure = (structure_params or BALANCED_PILOT_CONFIG).checked()
    coarse_count = len(result['candidates'])
    candidates, _, elapsed = refine_candidates(hu, result['candidates'], geometry, structure)
    result.update(algorithm_id='lung_suspicious_regions_log_hessian_pilot_v3', candidates=candidates,
                  structure_parameters=asdict(structure), coarse_candidate_count=coarse_count,
                  schema_version='lung_candidate_result_v1', source_shape_zyx=list(np.shape(hu)),
                  geometry={k: np.asarray(geometry[k]).tolist() for k in (
                      'spacing_zyx_mm', 'origin_lps_mm', 'basis_lps_zyx')},
                  validation_scope='parameters selected on 12 development scans; heldout not run')
    result['timings_ms'].update(structure=elapsed, candidate_total=(perf_counter()-start)*1000)
    return result, seed


def roi_candidate_packet(result):
    """Compact JSON-ready consumer packet; omit debug arrays and reference labels."""
    required = ['algorithm_id', 'backend', 'status', 'source_shape_zyx', 'geometry', 'candidates']
    if not all(key in result for key in required):
        raise ValueError('Candidate result lacks ROI consumer metadata')
    packet = {key: result[key] for key in required}
    packet.update(schema_version='lung_candidate_result_v1', coordinates='DICOM LPS XYZ millimetres',
                  bbox_convention='zero-based ZYX half-open [z0,z1,y0,y1,x0,x1]',
                  purpose='suspicious region candidates, not diagnosis or expert voxel segmentation')
    for key in ('series_uid', 'input_raw_sha256', 'lung_mask_sha256', 'parameters', 'structure_parameters',
                'validation_scope', 'lung_backend', 'algorithm_source_sha256', 'structure_source_sha256'):
        if key in result:
            packet[key] = result[key]
    return packet
