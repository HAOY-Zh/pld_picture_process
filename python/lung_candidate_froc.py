"""Local LUNA-style FROC matching, without top-K truncation or CT access.

This implements center-in-reference-radius matching and duplicate/excluded
handling from the LUNA16 evaluator. It is a development diagnostic, not the
official challenge submission, ten-fold protocol or bootstrap evaluation.
"""
import numpy as np

CPM_POINTS = (.125, .25, .5, 1., 2., 4., 8.)


def evaluate(rows, predictions, points=CPM_POINTS):
    """Each row contains uid, world points, record.nodules and excluded findings."""
    positive = []; negative = []; references = 0; ignored = 0
    for row in rows:
        xyz = np.asarray(row['world'], float).reshape(-1, 3)
        scores = np.asarray(predictions[row['uid']], float)
        if scores.shape != (len(xyz),) or not np.isfinite(scores).all() or not np.isfinite(xyz).all():
            raise ValueError('Invalid candidate positions/scores')
        matched = np.zeros(len(xyz), bool)
        for nodule in row['record']['nodules']:
            references += 1
            hits = np.linalg.norm(xyz-nodule['center_lps_xyz_mm'], axis=1) < nodule['diameter_mm']/2
            matched |= hits
            if hits.any(): positive.append(float(scores[hits].max()))
        excluded = np.zeros(len(xyz), bool)
        for nodule in row['excluded']:
            diameter = nodule['diameter_mm'] if nodule['diameter_mm'] >= 0 else 10.
            excluded |= np.linalg.norm(xyz-nodule['center_lps_xyz_mm'], axis=1) < diameter/2
        ignored += int((excluded & ~matched).sum())
        negative.extend(scores[~(matched | excluded)].tolist())
    if not rows or not references: raise ValueError('Need scans and reference nodules')
    values = np.array(positive+negative)
    labels = np.r_[np.ones(len(positive), int), np.zeros(len(negative), int)]
    order = np.argsort(-values, kind='stable'); values = values[order]; labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(values)), len(values)-1] if len(values) else np.array([], int)
    tp = np.r_[0, np.cumsum(labels)[ends]]
    fp = np.r_[0, np.cumsum(1-labels)[ends]]
    thresholds = [None]+values[ends].tolist()  # None means above maximum score.
    fps = fp/len(rows); sensitivity = tp/references
    operating = []
    for budget in points:
        eligible = np.flatnonzero(fps <= budget)
        i = int(eligible[-1])
        operating.append(dict(fp_budget=float(budget), sensitivity=float(sensitivity[i]),
            detected=int(tp[i]), actual_fp_per_scan=float(fps[i]), threshold=thresholds[i],
            interpolated_sensitivity=float(np.interp(budget, fps, sensitivity))))
    return dict(scans=len(rows), reference_nodules=references, detectable=len(positive),
        maximum_sensitivity=len(positive)/references, ignored_excluded_candidates=ignored,
        operating_points=operating,
        cpm_interpolated=float(np.mean([p['interpolated_sensitivity'] for p in operating])),
        mean_empirical_sensitivity=float(np.mean([p['sensitivity'] for p in operating])),
        curve=dict(fp_per_scan=fps.tolist(), sensitivity=sensitivity.tolist(), thresholds=thresholds),
        matching='strict distance < diameter/2; best hit per reference; duplicates/excluded ignored',
        candidate_limit=None, official_challenge_evaluation=False)
