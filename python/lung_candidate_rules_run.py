"""Development-only candidate refinement experiment using cached frozen lungs.

Preserves v3, LDCT and U-Net. No heldout mode, no neural-network training or PL
claims. Large intermediate candidate JSONs are not written back to disk.
"""
import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile
import numpy as np

from lung_candidate_data import read_split,development_records,load_development_case,sha_file,write_json
from lung_candidate_run import excluded_findings,preview
from lung_candidates import CandidateConfig,detect_candidates,point_metrics
from lung_candidate_structure import hessian_features
from lung_candidate_rules import (RulesConfig,FEATURE_NAMES,measure_pool,merge_regions,rule_scores,linear_scores)

ROOT=Path(__file__).resolve().parents[1]
SPLIT=ROOT/'data/lung_candidate_split_20261018/split.json'
ARCHIVE=ROOT/'data/project_cleanup_audit_20261018/retired_candidate_payloads.zip'
EXCLUDED=ROOT/'data/lung_candidate_planning_20261017/annotations_excluded.csv'
BASELINES=[ROOT/'data/lung_candidates_pilot_20261018_v3',ROOT/'data/lung_candidates_checkdev_20261018_v3']


def frozen_sources():
    files=[SPLIT,EXCLUDED,Path(__file__),ROOT/'python/lung_candidate_rules.py',
           ROOT/'python/lung_candidates.py',ROOT/'python/lung_candidate_structure.py',
           ROOT/'python/lung_candidate_data.py']
    return {str(p):sha_file(p) for p in files}


def expected_records():
    split=read_split(SPLIT)
    pilot=development_records(split,'pilot');check=development_records(split,'check-development')
    if {r['series_uid'] for r in pilot}&{r['series_uid'] for r in check}:raise ValueError('Cohort overlap')
    return split,pilot+check


def prepare(out):
    if out.exists():raise ValueError('Use fresh feature directory; preserve completed experiment')
    split,records=expected_records();frozen=frozen_sources();cfg=RulesConfig()
    out.mkdir(parents=True)
    protocol=dict(status='RUNNING',scope='24 already viewed development scans only',
        frozen_sources=frozen,selected_uids=[r['series_uid'] for r in records],
        rules=asdict(cfg),feature_names=FEATURE_NAMES,heldout_scans_run=0,
        lung_source='previously computed frozen FP32 original U-Net masks, byte identity checked',
        timings='CPU feature/reference timings; mixed cached/recomputed coarse stages; not end-to-end or PL',cases=[])
    write_json(out/'protocol.json',protocol)
    archive_expected=json.loads((ARCHIVE.parent/'verified_cache_plan.json').read_text(encoding='utf-8'))['archive']['sha256']
    if sha_file(ARCHIVE)!=archive_expected:raise ValueError('Historical archive identity differs')
    with ZipFile(ARCHIVE) as z:
        for i,r in enumerate(records,1):
            uid=r['series_uid'];pilot=uid in split['pilot_development_uids'];base=BASELINES[0 if pilot else 1]
            old=json.loads((base/uid/'candidates.json').read_text(encoding='utf-8'))
            if old['input_raw_sha256']!=r['raw_sha256']:raise ValueError('Cached case input differs')
            print(f'[{i}/24] {uid}: loading verified native HU',flush=True)
            hu,geom=load_development_case(split,r)
            if pilot:
                prefix=f'data/lung_candidates_pilot_20261018_v1/{uid}'
                coarse=json.loads(z.read(prefix+'/candidates.json'))
                packed=np.load(BytesIO(z.read(prefix+'/masks_packed.npz')),allow_pickle=False)
                cached=json.loads(z.read(f'data/lung_candidates_pilot_20261018_v2/{uid}/all_structure_features.json'))
                if cached['baseline_candidate_ids']!=[c['candidate_id'] for c in coarse['candidates']]:
                    raise ValueError('Cached Hessian ordering differs')
                hessian=cached['features'];coarse_mode='archived original LoG candidates and Hessian'
            else:
                packed=np.load(base/uid/'masks_packed.npz',allow_pickle=False)
            if tuple(packed['shape_zyx'])!=hu.shape:raise ValueError('Cached lung grid differs')
            body=np.unpackbits(packed['lung_body'],count=hu.size).reshape(hu.shape).astype(bool)
            packed.close()
            if hashlib.sha256(body.tobytes()).hexdigest()!=old['lung_mask_sha256']:
                raise ValueError('Frozen predicted lung mask differs')
            start=perf_counter()
            if not pilot:
                coarse,seed=detect_candidates(hu,body,geom,CandidateConfig());del seed
                hessian=hessian_features(hu,coarse['candidates'],geom['spacing_zyx_mm'])
                coarse_mode='recomputed unchanged LoG and Hessian from verified cached lung mask'
            print('  coarse',len(coarse['candidates']),'local measurements',flush=True)
            local_start=perf_counter();table=measure_pool(hu,coarse['candidates'],hessian,geom,cfg)
            # Keep a compact numeric feature table, not hundreds of MB of loose JSON.
            np.savez_compressed(out/(uid+'.npz'),**table)
            meta=dict(series_uid=uid,source_shape_zyx=list(hu.shape),geometry=geom,
                input_raw_sha256=r['raw_sha256'],lung_mask_sha256=old['lung_mask_sha256'],
                coarse_mode=coarse_mode,coarse_count=len(coarse['candidates']),pool_count=len(table['features']),
                local_measurement_ms=(perf_counter()-local_start)*1000,elapsed_ms=(perf_counter()-start)*1000,
                feature_sha256=sha_file(out/(uid+'.npz')),baseline_candidate_sha256=sha_file(base/uid/'candidates.json'),
                baseline_path=str(base/uid/'candidates.json'))
            write_json(out/(uid+'.json'),meta);protocol['cases'].append(meta)
            write_json(out/'protocol.json',protocol)
            print('  retained broad pool:',len(table['features']),'feature seconds:',round(meta['local_measurement_ms']/1000,2),flush=True)
            del hu,body,coarse,hessian,table
    for path,expected in frozen.items():
        if sha_file(path)!=expected:raise ValueError('Frozen source changed during feature extraction')
    protocol['status']='COMPLETE';write_json(out/'protocol.json',protocol)


def world(points,geom):
    return np.asarray(points).reshape(-1,3)@np.array(geom['basis_lps_zyx']).T+geom['origin_lps_mm']


def labels_for(points,record,excluded):
    pts=np.asarray(points).reshape(-1,3);labels=np.zeros(len(pts),np.int8)
    for n in record['nodules']:
        hit=np.linalg.norm(pts-np.array(n['center_lps_xyz_mm']),axis=1)<n['diameter_mm']/2
        labels[hit]=1
    for n in excluded:
        diameter=n['diameter_mm'] if n['diameter_mm']>=0 else 10.
        hit=np.linalg.norm(pts-np.array(n['center_lps_xyz_mm']),axis=1)<diameter/2
        labels[hit&(labels==0)]=-1
    return labels


def metric_key(uid,detail):
    return uid+':'+','.join(str(x) for x in detail['reference']['center_lps_xyz_mm'])


def export_linear(scaler,clf):
    weights=clf.coef_[0]/scaler.scale_
    return dict(feature_names=FEATURE_NAMES,weights=weights.tolist(),
        bias=float(clf.intercept_[0]-weights@scaler.mean_),score_kind='linear_rank_logit_not_calibrated_probability',
        inference='features @ weights + bias; no sklearn or neural network needed')


def reference_weights(points,record):
    weights=np.zeros(len(points),float)
    for n in record['nodules']:
        hits=np.linalg.norm(points-np.array(n['center_lps_xyz_mm']),axis=1)<n['diameter_mm']/2
        if hits.any():weights[hits]+=1/hits.sum()
    return weights


def training_weights(y,raw):
    weights=np.ones(len(y));pos=y==1
    if pos.any():weights[pos]=raw[pos]*pos.sum()/raw[pos].sum()
    return weights


def evaluate(features_dir,out):
    # Evaluation is separate from the long-running, frozen feature extraction.
    from lung_candidate_rules_evaluate import evaluate as evaluate_features
    return evaluate_features(features_dir,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--outdir',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('--features',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.outdir)
    else:evaluate(args.features,args.outdir)
