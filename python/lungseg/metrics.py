"""Segmentation metrics with explicit empty-mask handling."""
import numpy as np


def overlap_metrics(prediction, reference):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or prediction.size == 0:
        raise ValueError('Prediction/reference must have the same nonempty shape')
    if not np.isin(prediction, [0, 1, 255]).all() or not np.isin(reference, [0, 1, 255]).all():
        raise ValueError('Metrics require binary masks, not probabilities or grayscale')
    prediction, reference = prediction != 0, reference != 0
    tp = int(np.count_nonzero(prediction & reference))
    fp = int(np.count_nonzero(prediction & ~reference))
    fn = int(np.count_nonzero(~prediction & reference))
    denom, union = 2 * tp + fp + fn, tp + fp + fn
    return {'dice': 2 * tp / denom if denom else None,
            'iou': tp / union if union else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
            'both_empty': denom == 0}
