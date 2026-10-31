"""Apply existing frozen cross-layer scores and threshold; no fitting/Top-K."""
import copy
import gzip
import json
from pathlib import Path
import time
import numpy as np
from nodule_vendor.lung_nodule_continuity_boundary import evidence_for_packet
from nodule_vendor.lung_nodule_continuity import rerank_scores, feature_names
from nodule_canvas_contract import digest
from nodule_threshold_policy import DEFAULT_THRESHOLD,checked_threshold,select_candidates


def load_scoring(package):
    path = Path(package)/'nodule_scoring.json'
    setup = json.loads(path.read_text(encoding='utf8'))
    ranker_path = path.parent/setup['ranker']
    if ranker_path.parent != path.parent or digest(ranker_path.read_bytes()) != setup['ranker_sha256']:
        raise ValueError('Frozen cross-layer ranker differs')
    ranker = json.loads(ranker_path.read_text())
    if (ranker['feature_names'] != feature_names() or setup['threshold'] != 0.6418673229668569
            or setup['model_checkpoint_sha256'] != 'c1fefe55a74d1cdc81ede2f2e249190aef7d0b71a8a81378c950427ddcd8dc06'):
        raise ValueError('Wrong frozen score contract')
    return setup, ranker


def score_candidates(hu, spacing, candidates, setup, ranker, folder, cancelled=lambda:False, selection_threshold=None):
    started = time.perf_counter()
    if cancelled(): raise InterruptedError('Scoring cancelled')
    features, evidence = evidence_for_packet(hu, spacing, candidates)
    if cancelled(): raise InterruptedError('Scoring cancelled')
    scores = rerank_scores([c['score'] for c in candidates], features, ranker)
    rows = []
    for candidate, value in zip(candidates, scores):
        row = copy.deepcopy(candidate)
        row.update(raw_score=float(candidate['score']), score=float(value),
            score_kind='frozen_cross_layer_ranked_score_not_malignancy_probability')
        rows.append(row)
    threshold=checked_threshold(DEFAULT_THRESHOLD if selection_threshold is None else selection_threshold)
    selected=select_candidates(rows,threshold)
    # No candidate IDs, coordinates, diameter or bounds are moved/renumbered.
    folder = Path(folder)
    with gzip.open(folder/'all_ranked_candidates.json.gz', 'xt', encoding='utf8') as stream:
        json.dump(rows, stream, allow_nan=False)
    with (folder/'continuity_evidence.json').open('x', encoding='utf8') as stream:
        json.dump(evidence, stream, allow_nan=False)
    with (folder/'continuity_features.npz').open('xb') as stream:
        np.savez_compressed(stream, features=features, scores=scores)
    return selected, dict(scoring_mode='frozen_cross_layer', ranker_applied=True,
        ranker_sha256=setup['ranker_sha256'], frozen_threshold=setup['threshold'], selection_threshold=threshold,
        selection_policy='fixed_low_false_positive',
        threshold_source='75-case development calibration, never selected on current scan',
        raw_candidate_count=len(rows), selected_candidate_count=len(selected),
        removed_candidate_count=len(rows)-len(selected), scoring_ms=(time.perf_counter()-started)*1000,
        geometry_changed=False, candidate_ids_changed=False, top_k_cap=None,
        quality_scope='Existing FP32 heldout and INT8 calibration evidence; new live end-to-end quality not yet accepted')
