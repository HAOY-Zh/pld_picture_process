"""Native-grid pulmonary nodule candidates: reference algorithm, not diagnosis.

This module does not read reference annotations or run a neural network. A frozen
lung prediction is supplied by the caller. All bounding boxes are ZYX half-open;
world coordinates are LPS XYZ millimetres. Scores are not probabilities.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CandidateConfig:
    sigma_mm: tuple = (1., 2., 4., 6.)
    response_threshold_hu: float = 12.
    hu_low: float = -950.
    hu_high: float = 2000.
    response_clip_low: float = -1000.
    response_clip_high: float = 600.
    search_margin_mm: float = 3.
    peak_window_mm: float = 1.5
    merge_distance_mm: float = 2.5
    region_response_fraction: float = .35
    min_equivalent_diameter_mm: float = 1.5
    max_equivalent_diameter_mm: float = 35.
    max_elongation: float = 8.
    truncate: float = 3.5
    workers: int = 4

    def checked(self):
        numbers = [x for k, v in asdict(self).items() for x in (v if isinstance(v, (tuple, list)) else [v])]
        if not np.isfinite(numbers).all():
            raise ValueError('Nonfinite configuration')
        if (not self.sigma_mm or min(self.sigma_mm) <= 0 or len(self.sigma_mm) > 255 or
                self.response_threshold_hu <= 0 or self.hu_low > self.hu_high or
                self.response_clip_low >= self.response_clip_high or self.search_margin_mm < 0 or
                min(self.peak_window_mm, self.merge_distance_mm) <= 0 or
                not 0 < self.region_response_fraction <= 1 or
                not 0 < self.min_equivalent_diameter_mm < self.max_equivalent_diameter_mm or
                self.max_elongation < 1 or not 2 <= self.truncate <= 6 or
                not isinstance(self.workers, int) or not 1 <= self.workers <= 16):
            raise ValueError('Invalid candidate configuration')
        return self


def checked_inputs(hu, lung_mask, geometry):
    hu, mask = np.asarray(hu), np.asarray(lung_mask)
    if (hu.ndim != 3 or not hu.size or min(hu.shape) < 1 or hu.shape != mask.shape or
            not np.isfinite(hu).all() or not np.isin(mask, [0, 1]).all()):
        raise ValueError('Need finite native ZYX HU and matching 0/1 lung mask')
    spacing = np.asarray(geometry['spacing_zyx_mm'], float)
    basis = np.asarray(geometry['basis_lps_zyx'], float)
    origin = np.asarray(geometry['origin_lps_mm'], float)
    if (spacing.shape != (3,) or basis.shape != (3, 3) or origin.shape != (3,) or
            not np.isfinite(basis).all() or not np.isfinite(origin).all() or
            not np.isfinite(spacing).all() or (spacing <= 0).any() or
            not np.allclose(basis.T @ basis, np.diag(spacing ** 2), atol=1e-6, rtol=1e-6)):
        raise ValueError('Need orthogonal millimetre geometry consistent with ZYX spacing')
    return hu, mask.astype(bool, copy=False), spacing


def mask_bounds(mask):
    if not mask.any():
        return None
    ranges = [np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    return tuple(slice(int(r[0]), int(r[-1]) + 1) for r in ranges)


def make_search_mask(lung_mask, spacing, margin_mm):
    """Derived search area; original lung prediction is never changed."""
    mask = np.asarray(lung_mask, bool)
    sy, sx = spacing[1:]
    ry, rx = int(np.ceil(margin_mm / sy)), int(np.ceil(margin_mm / sx))
    y, x = np.ogrid[-ry:ry+1, -rx:rx+1]
    kernel = (y * sy) ** 2 + (x * sx) ** 2 <= margin_mm ** 2 + 1e-9
    result = np.zeros_like(mask)
    for z in range(len(mask)):
        filled = ndi.binary_fill_holes(mask[z])
        result[z] = ndi.binary_dilation(filled, structure=kernel) if margin_mm else filled
    return result


def log_kernels(sigma_mm, spacing_yx, truncate):
    """Separable LoG, physical derivatives, unit Gaussian and zero-DC second derivative."""
    kernels = []
    for spacing in spacing_yx:
        sigma = sigma_mm / spacing
        radius = max(1, int(np.ceil(truncate * sigma)))
        coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
        gaussian = np.exp(-.5 * (coordinate / sigma) ** 2)
        gaussian /= gaussian.sum()
        second = ((coordinate ** 2 - sigma ** 2) / sigma ** 4) * gaussian
        second -= second.sum() * gaussian
        kernels.append((gaussian, second / spacing ** 2))
    return kernels


def response_slice(image, kernels, sigma_mm):
    (gy, dy), (gx, dx) = kernels
    yy = ndi.convolve1d(ndi.convolve1d(image, dy, axis=0, mode='reflect'), gx, axis=1, mode='reflect')
    xx = ndi.convolve1d(ndi.convolve1d(image, gy, axis=0, mode='reflect'), dx, axis=1, mode='reflect')
    return -(sigma_mm ** 2) * (xx + yy)


def generate_seed_reference(hu, lung_mask, geometry, params=None):
    """Return explicitly cropped coarse mask plus response/scale for CPU postprocessing.

    Filter before applying the search mask: zero-padding a lung mask into CT would
    create artificial high-contrast edges. No annotation is an input.
    """
    cfg = (params or CandidateConfig()).checked()
    hu, lung, spacing = checked_inputs(hu, lung_mask, geometry)
    start = perf_counter()
    search = make_search_mask(lung, spacing, cfg.search_margin_mm)
    box = mask_bounds(search)
    if box is None:
        return dict(source_shape=list(hu.shape), bbox_zyx=[0, 0, 0, 0, 0, 0],
                    response=np.empty((0, 0, 0), np.float32), scale_index=np.empty((0, 0, 0), np.uint8),
                    seed_mask=np.empty((0, 0, 0), bool), search_mask=search,
                    elapsed_ms=(perf_counter()-start)*1000, parameters=asdict(cfg))
    zs, ys, xs = box
    shape = (zs.stop-zs.start, ys.stop-ys.start, xs.stop-xs.start)
    response = np.zeros(shape, np.float32)
    scales = np.zeros(shape, np.uint8)
    kernels = [log_kernels(s, spacing[1:], cfg.truncate) for s in cfg.sigma_mm]
    hy = max(len(k[0][0]) // 2 for k in kernels)
    hx = max(len(k[1][0]) // 2 for k in kernels)
    py0, py1 = max(0, ys.start-hy), min(hu.shape[1], ys.stop+hy)
    px0, px1 = max(0, xs.start-hx), min(hu.shape[2], xs.stop+hx)
    crop = (slice(ys.start-py0, ys.stop-py0), slice(xs.start-px0, xs.stop-px0))

    def one_slice(z):
        local = z-zs.start
        if not search[z].any():
            return
        image = np.clip(hu[z, py0:py1, px0:px1].astype(np.float32),
                        cfg.response_clip_low, cfg.response_clip_high)
        # A constant offset leaves a zero-DC LoG unchanged and improves numerical conditioning.
        image -= cfg.response_clip_low
        best, winner = response[local], scales[local]
        for i, (sigma, kernel) in enumerate(zip(cfg.sigma_mm, kernels)):
            current = response_slice(image, kernel, sigma)[crop]
            chosen = current > best
            best[chosen], winner[chosen] = current[chosen], i

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        list(pool.map(one_slice, range(zs.start, zs.stop)))
    seed = ((response >= cfg.response_threshold_hu) & search[box] &
            (hu[box] >= cfg.hu_low) & (hu[box] <= cfg.hu_high))
    return dict(source_shape=list(hu.shape), bbox_zyx=[zs.start, zs.stop, ys.start, ys.stop, xs.start, xs.stop],
                response=response, scale_index=scales, seed_mask=seed, search_mask=search,
                elapsed_ms=(perf_counter()-start)*1000, parameters=asdict(cfg))


def finalize_candidates(hu, seed, geometry, params=None):
    """3-D peak consolidation and local connected-region features on the PC."""
    cfg = (params or CandidateConfig()).checked()
    if tuple(seed['source_shape']) != np.shape(hu) or seed['parameters'] != asdict(cfg):
        raise ValueError('Seed shape/parameters differ from postprocessor')
    response, coarse = seed['response'], seed['seed_mask']
    if not response.size:
        return [], dict(peaks=[], merged=[], size=[], shape=[])
    spacing = np.asarray(geometry['spacing_zyx_mm'])
    basis, origin = np.asarray(geometry['basis_lps_zyx']), np.asarray(geometry['origin_lps_mm'])
    offset = np.asarray(seed['bbox_zyx'])[::2]
    radius = np.maximum(1, np.ceil(cfg.peak_window_mm / spacing).astype(int))
    valid_response = np.where(coarse, response, -np.inf)
    maxima = ndi.maximum_filter(valid_response, size=tuple(2*radius+1), mode='constant', cval=-np.inf)
    points = np.argwhere(coarse & (valid_response == maxima))
    del maxima, valid_response
    if not len(points):
        return [], dict(peaks=[], merged=[], size=[], shape=[])
    scores = response[tuple(points.T)]
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], -scores))
    native = points + offset
    tree = cKDTree(native * spacing)
    suppressed = np.zeros(len(points), bool)
    kept = []
    for i in order:
        if not suppressed[i]:
            kept.append(int(i))
            suppressed[tree.query_ball_point(native[i]*spacing, cfg.merge_distance_mm)] = True
    stage = dict(peaks=native.tolist(), merged=native[kept].tolist(), size=[], shape=[])
    candidates = []
    for i in kept:
        p, p_native = points[i], native[i]
        sigma = cfg.sigma_mm[int(seed['scale_index'][tuple(p)])]
        region_radius = min(18., max(2., 2*sigma))
        half = np.maximum(1, np.ceil(region_radius / spacing).astype(int))
        begin = np.maximum(0, p-half)
        end = np.minimum(np.asarray(coarse.shape), p+half+1)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(begin, end))
        grid = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(begin-p, end-p))]
        sphere = sum((g*s) ** 2 for g, s in zip(grid, spacing)) <= region_radius ** 2
        region = coarse[slices] & sphere & (response[slices] >= scores[i]*cfg.region_response_fraction)
        labels, _ = ndi.label(region, structure=np.ones((3, 3, 3), bool))
        label = labels[tuple(p-begin)]
        if not label:
            continue
        component = labels == label
        local_points = np.argwhere(component)
        count = len(local_points)
        volume_mm3 = count * float(np.prod(spacing))
        diameter = (6*volume_mm3/np.pi) ** (1/3)
        if not cfg.min_equivalent_diameter_mm <= diameter <= cfg.max_equivalent_diameter_mm:
            continue
        stage['size'].append(p_native.tolist())
        physical = local_points * spacing
        centered = physical-physical.mean(axis=0)
        covariance = centered.T @ centered / count + np.diag(spacing**2/12)
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = float(np.sqrt(eigenvalues[-1]/eigenvalues[0]))
        if elongation > cfg.max_elongation:
            continue
        stage['shape'].append(p_native.tolist())
        low = offset+begin+local_points.min(axis=0)
        high = offset+begin+local_points.max(axis=0)+1
        box = [int(v) for pair in zip(low, high) for v in pair]
        hu_slices = tuple(slice(int(a), int(b)) for a, b in zip(offset+begin, offset+end))
        values = np.asarray(hu[hu_slices])[component]
        candidates.append(dict(candidate_id=f'candidate_{len(candidates)+1:05d}',
            center_index_zyx=p_native.tolist(), center_lps_xyz_mm=(origin+basis@p_native).tolist(),
            bbox_zyx=box, score=float(scores[i]), score_kind='scale_normalized_2d_Log_response_HU_not_probability',
            estimated_scale_diameter_mm=float(2*np.sqrt(2)*sigma),
            equivalent_region_diameter_mm=float(diameter), region_volume_ml=volume_mm3/1000,
            voxel_count=count, mean_hu=float(values.mean()), min_hu=float(values.min()), max_hu=float(values.max()),
            elongation=elongation, region_kind='coarse_response_component_not_expert_nodule_segmentation'))
    return candidates, stage


def detect_candidates(hu, lung_mask, geometry, params=None):
    cfg = (params or CandidateConfig()).checked()
    start = perf_counter()
    seed = generate_seed_reference(hu, lung_mask, geometry, cfg)
    post_start = perf_counter()
    candidates, stages = finalize_candidates(hu, seed, geometry, cfg)
    metadata = dict(algorithm_id='lung_candidates_2d_log_3d_regions_v1', backend='CPU_REFERENCE',
        parameters=asdict(cfg), candidates=candidates, stage_centers_index_zyx=stages,
        seed_voxels=int(seed['seed_mask'].sum()), search_voxels=int(seed['search_mask'].sum()),
        timings_ms=dict(seed=seed['elapsed_ms'], postprocess=(perf_counter()-post_start)*1000,
                        total=(perf_counter()-start)*1000), status='success')
    return metadata, seed


def point_metrics(centers_lps_xyz_mm, nodules, excluded=()):
    """Development matching: reference diameter/2, ignore duplicate/excluded hits.

    Not the official FROC/CPM evaluator. Reports every reference individually;
    reference counts are independent of how many candidate points hit a nodule.
    """
    centers = np.asarray(centers_lps_xyz_mm, float).reshape(-1, 3)
    if not np.isfinite(centers).all():
        raise ValueError('Nonfinite candidate coordinates')
    matched_any = np.zeros(len(centers), bool)
    rows = []
    for n in nodules:
        distance = np.linalg.norm(centers-np.asarray(n['center_lps_xyz_mm']), axis=1)
        hits = distance < float(n['diameter_mm'])/2
        matched_any |= hits
        rows.append(dict(reference=n, detected=bool(hits.any()),
                         nearest_distance_mm=float(distance.min()) if len(distance) else None,
                         matching_candidate_indices=np.flatnonzero(hits).tolist()))
    ignored = np.zeros(len(centers), bool)
    for n in excluded:
        distance = np.linalg.norm(centers-np.asarray(n['center_lps_xyz_mm']), axis=1)
        diameter = float(n['diameter_mm'])
        if diameter < 0:
            diameter = 10.  # Explicit default in the official LUNA16 reference evaluator.
        ignored |= distance < diameter/2
    false_positive = ~(matched_any | ignored)
    return dict(reference_nodules=len(nodules), detected=sum(r['detected'] for r in rows),
                sensitivity=sum(r['detected'] for r in rows)/len(rows) if rows else None,
                false_positive_candidates=int(false_positive.sum()), candidates=len(centers),
                ignored_excluded_candidates=int((ignored & ~matched_any).sum()),
                duplicate_reference_hits=sum(max(0, len(r['matching_candidate_indices'])-1) for r in rows),
                reference_details=rows, matching='development_center_within_reference_radius',
                official_froc_cpm=False)
