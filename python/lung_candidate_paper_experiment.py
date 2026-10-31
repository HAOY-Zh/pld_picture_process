"""Paper-inspired workflow on existing feature caches; no CT/CNN rerun."""
import argparse
import json
from pathlib import Path
import numpy as np
from lung_candidate_cached import load,ROOT
from lung_candidate_data import sha_file,write_json
from lung_candidate_lowfp import FEATURE_NAMES as BASE_NAMES
from lung_candidate_shell import FEATURE_NAMES as SHELL_NAMES
from lung_candidate_lowfp_experiment import training
from lung_candidate_paper import select_features,fit,score
from lung_candidate_froc import evaluate
from lung_candidate_svm_experiment import aggregate,preserve_threshold

GRID=[dict(C=c,negative_ratio=ratio) for ratio in (None,10,20) for c in (.1,1.,10.)]


def run(out,features):
    if out.exists():raise FileExistsError('Preserve earlier experiment')
    rows,folds,protected=load();by_uid={r['uid']:r for r in rows};names=BASE_NAMES+SHELL_NAMES
    protocol_path=features/'protocol.json';protocol=json.loads(protocol_path.read_text('utf-8'))
    if protocol['status']!='COMPLETE' or tuple(protocol['feature_names'])!=SHELL_NAMES:raise ValueError('Invalid shell cache')
    protected[str(protocol_path)]=sha_file(protocol_path)
    meta={r['series_uid']:r for r in protocol['cases']}
    if set(meta)!=set(by_uid):raise ValueError('Wrong scan cohort')
    for row in rows:
        path=features/(row['uid']+'.npz')
        if sha_file(path)!=meta[row['uid']]['sha256']:raise ValueError('Shell cache changed')
        protected[str(path)]=sha_file(path)
        with np.load(path,allow_pickle=False) as a:
            np.testing.assert_array_equal(a['candidate_ids'],[c['candidate_id'] for c in row['packet']['candidates']])
            row['features_local']=np.column_stack((row['features_local'],a['features']))
    out.mkdir(parents=True);snapshots=out/'executed_sources';snapshots.mkdir()
    for filename in ('lung_candidate_paper.py','lung_candidate_paper_experiment.py','lung_candidate_lowfp.py',
                     'lung_candidate_lowfp_experiment.py','lung_candidate_cached.py','lung_candidate_froc.py','lung_candidate_shell.py'):
        path=ROOT/'python'/filename;protected[str(path)]=sha_file(path)
        (snapshots/filename).write_bytes(path.read_bytes())
    write_json(out/'protocol.json',dict(grid=GRID,folds=folds,protected_files_sha256=protected,
        feature_selection='64 random 12-feature subsets; 3 grouped inner checks; keep 16',
        sampling='all training positives + 75% hard/25% random negatives, ratio fixed per trial',
        selection='maximum pooled development OOF CPM; not unbiased nested parameter-selection estimate',
        source='Gu et al 2019, doi:10.1371/journal.pone.0210551',exact_reproduction=False,
        heldout_scans_run=0,ct_volumes_read=0,network_forwards=0,six_check_loaded=False))
    datasets={}
    for fold in folds:
        subset=[by_uid[u] for u in fold['training_scans']]
        x,y,w=training(subset,len(names))
        groups=np.concatenate([np.repeat(r['uid'],len(r['labels']))[r['labels']>=0] for r in subset])
        selection=select_features(x,y,groups,w,names)
        write_json(out/f"selection_fold_{fold['fold']}.json",selection)
        datasets[fold['fold']]=(x,y,w,selection)
        print('selected features for fold',fold['fold'],selection['selected_names'],flush=True)
    trials=[];best=None
    for i,parameters in enumerate(GRID):
        folder=out/f'trial_{i:02d}';folder.mkdir();predictions={}
        for fold in folds:
            x,y,w,selection=datasets[fold['fold']]
            model=fit(x,y,w,parameters,names,selection)
            model.update(training_scans=fold['training_scans'],fold=fold['fold'])
            write_json(folder/f"fold_{fold['fold']}.json",model)
            for uid in fold['validation_scans']:predictions[uid]=score(by_uid[uid]['features_local'],model)
        np.savez_compressed(folder/'oof_scores.npz',**predictions)
        froc=evaluate(rows,predictions);threshold=preserve_threshold(rows,predictions)
        high=aggregate(rows,predictions,threshold)
        result=dict(trial=i,parameters=parameters,froc=froc,preserve_threshold=threshold,high_recall=high['summary'])
        write_json(folder/'report.json',result);trials.append(result)
        if best is None or froc['cpm_interpolated']>best['froc']['cpm_interpolated']:best=result
        write_json(out/'trials.json',dict(status='RUNNING',trials=trials))
        print('trial',i,parameters,'CPM',round(froc['cpm_interpolated'],4),'sens@1/4/8',
            [(p['fp_budget'],round(p['sensitivity'],4)) for p in froc['operating_points'] if p['fp_budget'] in (1,4,8)],
            'preserve FP',round(high['summary']['after']['fp_per_scan'],3),flush=True)
    x,y,w=training(rows,len(names))
    groups=np.concatenate([np.repeat(r['uid'],len(r['labels']))[r['labels']>=0] for r in rows])
    selection=select_features(x,y,groups,w,names)
    model=fit(x,y,w,best['parameters'],names,selection)
    model.update(training_scans=[r['uid'] for r in rows],selected_trial=best['trial'],
        high_recall_threshold=best['preserve_threshold'],
        operating_thresholds={str(p['fp_budget']):p['threshold'] for p in best['froc']['operating_points']},
        threshold_scope='development OOF estimates; no per-scan cap or guarantee on other scans')
    write_json(out/'selected_model.json',model)
    for path,digest in protected.items():
        if sha_file(path)!=digest:raise ValueError('Protected input/source changed')
    write_json(out/'report.json',dict(status='COMPLETE_PAPER_INSPIRED_DEVELOPMENT_EXPERIMENT',selected=best,
        selected_model_sha256=sha_file(out/'selected_model.json'),trials=trials,
        protected_files_sha256=protected,heldout_scans_run=0,ct_volumes_read=0,network_forwards=0,
        six_check_loaded=False,exact_paper_reproduction=False,
        limitations='24 development scans used for pipeline/hyperparameter exploration; not independent test or official LUNA16 score'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--shell-features',type=Path,required=True);a=p.parse_args();run(a.outdir,a.shell_features)
