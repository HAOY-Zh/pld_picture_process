"""Cache-only low-FP ranking experiment on the same frozen 24 development scans."""
import argparse
import json
from pathlib import Path
import numpy as np
from lung_candidate_cached import load, ROOT
from lung_candidate_data import write_json, sha_file
from lung_candidate_froc import evaluate
from lung_candidate_lowfp import fit, score, FEATURE_NAMES
from lung_candidate_svm import FEATURE_NAMES as STRUCTURAL_NAMES
from lung_candidate_svm_experiment import aggregate, preserve_threshold

GRID = [dict(family='svm', features=n, C=c) for n in (13,20) for c in (.1,1.,10.)]
GRID += [dict(family=f, features=20, depth=d) for f in ('boosted_trees','extra_trees') for d in (2,4)]


def training(rows, n):
    x = np.concatenate([r['features_local'][:, :n] for r in rows])
    y = np.concatenate([r['labels'] for r in rows])
    raw = np.concatenate([r['positive_weights'] for r in rows])
    weights = np.ones(len(y)); positive = y == 1
    weights[positive] = raw[positive]*positive.sum()/raw[positive].sum()
    valid = y >= 0
    return x[valid], y[valid], weights[valid]


def run(out, shell_features=None):
    if out.exists(): raise FileExistsError('Preserve prior experiments; use new directory')
    out.mkdir(parents=True)
    rows, folds, protected = load(); by_uid = {r['uid']: r for r in rows}
    grid = GRID
    names_by_count = {13: STRUCTURAL_NAMES, 20: FEATURE_NAMES}
    if shell_features is not None:
        from lung_candidate_shell import FEATURE_NAMES as SHELL_NAMES
        protocol_path = shell_features/'protocol.json'
        protocol = json.loads(protocol_path.read_text('utf-8'))
        if protocol['status']!='COMPLETE' or tuple(protocol['feature_names'])!=SHELL_NAMES:
            raise ValueError('Incomplete/wrong shell descriptors')
        protected[str(protocol_path)] = sha_file(protocol_path)
        meta = {c['series_uid']: c for c in protocol['cases']}
        if set(meta)!=set(by_uid): raise ValueError('Wrong shell cohort')
        for row in rows:
            path=shell_features/(row['uid']+'.npz')
            if sha_file(path)!=meta[row['uid']]['sha256']: raise ValueError('Shell cache changed')
            protected[str(path)] = sha_file(path)
            with np.load(path,allow_pickle=False) as a:
                np.testing.assert_array_equal(a['candidate_ids'],[c['candidate_id'] for c in row['packet']['candidates']])
                row['features_local']=np.column_stack((row['features_local'],a['features']))
        count=20+len(SHELL_NAMES);names_by_count[count]=FEATURE_NAMES+SHELL_NAMES
        grid=[dict(family='svm',features=count,C=c) for c in (.1,1.,10.)]
        grid += [dict(family='boosted_trees',features=count,depth=d) for d in (2,4)]
    for path in (Path(__file__), ROOT/'python/lung_candidate_lowfp.py', ROOT/'python/lung_candidate_froc.py',
                 ROOT/'python/lung_candidate_cached.py'):
        protected[str(path)] = sha_file(path)
    snapshots=out/'executed_sources';snapshots.mkdir()
    for source in (Path(__file__), ROOT/'python/lung_candidate_lowfp.py', ROOT/'python/lung_candidate_froc.py',
                   ROOT/'python/lung_candidate_cached.py',ROOT/'python/lung_candidate_shell.py'):
        (snapshots/source.name).write_bytes(source.read_bytes())
    write_json(out/'protocol.json', dict(grid=grid, folds=folds,
        objective='maximum interpolated local CPM at .125,.25,.5,1,2,4,8 FP/scan',
        tie_break='fixed grid order', development_selection_not_independent_test=True,
        normalization_and_training='training scans only; equal class and per-reference positive weighting',
        primary_candidate_pool='unchanged rule v4 final candidates', threshold_or_topk_suppression=False,
        protected_files_sha256=protected, heldout_scans_run=0, ct_volumes_read=0,
        network_forwards=0, local_features_extracted=0, six_check_loaded=False))
    best = None; trials = []
    for i, parameters in enumerate(grid):
        n = parameters['features']; names = names_by_count[n]
        folder = out/f'trial_{i:02d}'; folder.mkdir(); predictions = {}
        for fold in folds:
            train = fold['training_scans']; validation = fold['validation_scans']
            if set(train)&set(validation) or set(train)|set(validation)!=set(by_uid): raise ValueError('Fold leakage')
            model = fit(*training([by_uid[uid] for uid in train], n), parameters, names)
            model.update(training_scans=train, fold=fold['fold'], formal_deployment=False)
            write_json(folder/f"fold_{fold['fold']}.json", model)
            for uid in validation: predictions[uid] = score(by_uid[uid]['features_local'][:, :n], model)
        np.savez_compressed(folder/'oof_scores.npz', **predictions)
        froc = evaluate(rows, predictions)
        threshold = preserve_threshold(rows, predictions)
        high_recall = aggregate(rows, predictions, threshold)
        result = dict(trial=i, parameters=parameters, froc=froc,
            preserve_threshold=threshold, high_recall=high_recall['summary'])
        write_json(folder/'report.json', result); trials.append(result)
        print('trial', i, parameters, 'CPM',round(froc['cpm_interpolated'],4),
            'sens@1/4/8',[(p['fp_budget'],round(p['sensitivity'],4)) for p in froc['operating_points'] if p['fp_budget'] in (1,4,8)],
            'preserve FP',round(high_recall['summary']['after']['fp_per_scan'],3), flush=True)
        if best is None or froc['cpm_interpolated'] > best['froc']['cpm_interpolated']: best=result
        write_json(out/'trials.json',dict(status='RUNNING',trials=trials))
    n=best['parameters']['features']; names=names_by_count[n]
    selected=fit(*training(rows,n),best['parameters'],names)
    selected.update(training_scans=[r['uid'] for r in rows], formal_deployment=False,
        selected_trial=best['trial'], high_recall_threshold=best['preserve_threshold'],
        operating_thresholds={str(p['fp_budget']): p['threshold'] for p in best['froc']['operating_points']},
        threshold_scope='development OOF estimates, not a guarantee on other scans')
    write_json(out/'selected_model.json',selected)
    for path,digest in protected.items():
        if sha_file(path)!=digest: raise ValueError('Protected source/cache changed')
    write_json(out/'report.json',dict(status='COMPLETE_CACHED_DEVELOPMENT_SELECTION',selected=best,
        selected_model_sha256=sha_file(out/'selected_model.json'),model_bytes=(out/'selected_model.json').stat().st_size,
        trials=trials, protected_files_sha256=protected,heldout_scans_run=0,ct_volumes_read=0,
        network_forwards=0,local_features_extracted=0,six_check_loaded=False,
        limitations='24 scans selected hyperparameters; CPM is development exploration, not unbiased nested CV or LUNA16 challenge score'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--shell-features',type=Path)
    args=p.parse_args();run(args.outdir,args.shell_features)
