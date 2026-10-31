"""Verified development-only descriptors shared by optional ranking experiments."""
import json
from pathlib import Path
import numpy as np
from lung_candidate_data import read_split, sha_file
from lung_candidate_rules_run import SPLIT, EXCLUDED, world, labels_for, reference_weights
from lung_candidate_run import excluded_findings
from lung_candidate_rules import RulesConfig, rule_scores, merge_regions
from lung_candidates import point_metrics
from lung_candidate_svm import from_local_table
from lung_candidate_lowfp import LOCAL_NAMES

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'data/lung_candidate_rules_20261018'
LOCAL_COLUMNS = (9, 10, 11, 12, 13, 14, 17)


def load():
    split = read_split(SPLIT); records = {r['series_uid']: r for r in split['cases']}
    excluded = excluded_findings(EXCLUDED)
    protocol_path = PRIOR/'features_scale2/protocol.json'
    fold_path = PRIOR/'evaluation_scale2/report.json'
    primary_path = PRIOR/'evaluation_scale2/balanced_rank_model.json'
    protocol = json.loads(protocol_path.read_text('utf-8'))
    folds = json.loads(fold_path.read_text('utf-8'))['folds']
    primary = json.loads(primary_path.read_text('utf-8'))
    protected = {str(p): sha_file(p) for p in (SPLIT, EXCLUDED, protocol_path, fold_path, primary_path)}
    rows = []
    for meta in protocol['cases']:
        uid = meta['series_uid']; record = records[uid]
        if record['split'] != 'development': raise ValueError('Heldout refused')
        path = PRIOR/'features_scale2'/(uid+'.npz')
        if sha_file(path) != meta['feature_sha256']: raise ValueError('Feature cache changed')
        protected[str(path)] = sha_file(path)
        with np.load(path, allow_pickle=False) as arrays: table = {k: arrays[k] for k in arrays.files}
        scores = rule_scores(table['features'].astype(float))
        keep = np.flatnonzero(scores >= primary['threshold'])
        keep = keep[merge_regions(table[primary['center_mode']][keep], table['boxes_zyx'][keep],
            scores[keep], meta['geometry']['spacing_zyx_mm'], RulesConfig(**primary['rules_config']))]
        path = PRIOR/'oof_comparisons'/uid/'candidates.json'
        protected[str(path)] = sha_file(path); packet = json.loads(path.read_text('utf-8'))
        np.testing.assert_array_equal(table[primary['center_mode']][keep],
            np.array([c['center_index_zyx'] for c in packet['candidates']]))
        np.testing.assert_array_equal(scores[keep], [c['score'] for c in packet['candidates']])
        xyz = world(table[primary['center_mode']][keep], meta['geometry'])
        x = from_local_table(table, keep, scores[keep], meta['geometry']['spacing_zyx_mm'])
        row = dict(uid=uid, record=record, excluded=excluded.get(uid, []), world=xyz,
            packet=packet, geometry=meta['geometry'], source_indices=table['source_indices'][keep],
            features=x, features_local=np.column_stack((x, table['features'][keep][:, LOCAL_COLUMNS])),
            primary_scores=scores[keep], labels=labels_for(xyz, record, excluded.get(uid, [])),
            positive_weights=reference_weights(xyz, record),
            baseline=point_metrics(xyz, record['nodules'], excluded.get(uid, [])))
        rows.append(row)
    if len(rows) != 24 or len({r['uid'] for r in rows}) != 24: raise ValueError('Need original 24 scans')
    return rows, folds, protected
