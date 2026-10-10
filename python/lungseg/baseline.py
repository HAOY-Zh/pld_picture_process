"""Explainable per-slice baseline. NOT a trained model or validated lung AI.

Uses no reference labels. Dense disease, airway, cropped scans and incomplete
chest walls can fail. Evaluate on expert annotations before choosing a route.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class BaselineConfig:
    threshold_hu: float = -400
    min_area_mm2: float = 25
    max_components: int = 2
    closing_iterations: int = 1

    def validate(self):
        if not np.isfinite(self.threshold_hu) or not -32768 <= self.threshold_hu <= 32767:
            raise ValueError('Threshold must be finite and in the signed 16-bit range')
        if not np.isfinite(self.min_area_mm2) or self.min_area_mm2 <= 0:
            raise ValueError('Minimum component area must be positive')
        if self.max_components < 1 or self.closing_iterations < 0:
            raise ValueError('Invalid component count or closing iteration count')


def segment_slice(hu, spacing_yx=(1, 1), config=None):
    config = config or BaselineConfig()
    config.validate()
    hu = np.asarray(hu)
    spacing_yx = np.asarray(spacing_yx, dtype=float)
    if hu.ndim != 2 or min(hu.shape) < 3 or not np.isfinite(hu).all():
        raise ValueError('Need a finite 2-D HU slice of at least 3x3')
    if spacing_yx.shape != (2,) or not np.isfinite(spacing_yx).all() or (spacing_yx <= 0).any():
        raise ValueError('Need positive row/column spacing')
    candidate = hu < config.threshold_hu
    labels, count = ndi.label(candidate, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    eligible = sizes * float(np.prod(spacing_yx)) >= config.min_area_mm2
    eligible[border_labels] = False
    eligible[0] = False
    interior = eligible[labels]
    ids = np.flatnonzero(eligible)
    # Stable tie-breaking; component labels are assigned in raster order.
    ids = ids[np.argsort(-sizes[ids], kind='stable')][:config.max_components]
    chosen = np.isin(labels, ids)
    cleaned = chosen.copy()
    # Each iteration is a complete 3x3 dilation -> erosion, both zero-padded,
    # matching the standalone closing PL. iterations=0 means bypass, NOT convergence.
    for _ in range(config.closing_iterations):
        cleaned = ndi.binary_dilation(cleaned, structure=np.ones((3, 3)), border_value=0)
        cleaned = ndi.binary_erosion(cleaned, structure=np.ones((3, 3)), border_value=0)
    prediction = ndi.binary_fill_holes(cleaned)
    info = {'candidate_components': int(count), 'retained_components': int(len(ids)),
            'prediction_pixels': int(prediction.sum()),
            'warning': 'No lung candidate found' if not ids.size else None}
    stages = {'candidate': candidate, 'interior': interior, 'chosen': chosen,
              'cleaned': cleaned, 'prediction': prediction}
    return prediction, stages, info


def segment_volume(hu, spacing_yx, config=None):
    """Independent 2-D slices, deliberately not a 3-D connected-components method."""
    if np.asarray(hu).ndim != 3 or len(hu) == 0:
        raise ValueError('Expected nonempty (slice, row, column) volume')
    predictions = np.empty(np.shape(hu), dtype=bool)
    diagnostics = []
    start = perf_counter()
    for index, image in enumerate(hu):
        slice_start = perf_counter()
        predictions[index], _, info = segment_slice(image, spacing_yx, config)
        info.update(slice=index, elapsed_ms=(perf_counter() - slice_start) * 1000)
        diagnostics.append(info)
    return predictions, diagnostics, (perf_counter() - start) * 1000
