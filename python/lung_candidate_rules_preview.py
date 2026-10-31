"""Export real OOF candidate packets and comparisons at a fixed operating point."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from lung_candidate_data import read_split,load_development_case,sha_file,write_json
from lung_candidate_run import excluded_findings
from lung_candidate_rules_run import SPLIT,EXCLUDED
from lung_candidate_rules import merge_regions,rule_scores
from lung_candidate_rank_runtime import score_feature_table
from lung_candidate_rules_check import compare_preview
from lung_candidates import point_metrics


def export(features_dir,evaluation,out,point):
    if out.exists():raise ValueError('Preserve earlier exports')
    report=json.loads((evaluation/'report.json').read_text(encoding='utf-8'))
    desired=report['balanced_experimental_point'] if point=='balanced' else report['selected_preserving_all_baseline_nodules']
    if desired is None:raise ValueError('No such operating point')
    protocol=json.loads((features_dir/'protocol.json').read_text(encoding='utf-8'))
    if sha_file(features_dir/'protocol.json')!=report['feature_protocol_sha256']:raise ValueError('Wrong feature cohort')
    split=read_split(SPLIT);records={r['series_uid']:r for r in split['cases']};excluded=excluded_findings(EXCLUDED)
    out.mkdir(parents=True);models=out/'fold_models';models.mkdir()
    common=dict(kind={'rule':'rule','linear_oof':'linear','boosted_rules_oof':'boosted_rules'}[desired['method']],
        center_mode=desired['center_mode'],threshold=desired['threshold'],rules_config=protocol['rules'],
        complementary_sigma_mm=protocol.get('complementary_sigma_mm'))
    folds={}
    for fold in report['folds']:
        f=fold['fold'];prefix='boost' if common['kind']=='boosted_rules' else 'linear'
        model=json.loads((evaluation/f'{prefix}_fold_{f}.json').read_text(encoding='utf-8')) if common['kind']!='rule' else {}
        model.update(common,training_scans=fold['training_scans'],scope='OOF model at development-selected threshold')
        write_json(models/f'fold_{f}.json',model)
        for uid in fold['validation_scans']:
            if uid in model['training_scans']:raise ValueError('Scan leakage')
            folds[uid]=(f,model)
    rows=[];preview_uids=set(split['forced_development_uids'])
    # Pre-existing four misses plus the previously shown first scan, regardless
    # of whether the new result improves. Do not choose pictures after scoring.
    for meta in protocol['cases']:
        p=Path(meta['baseline_path']).parent/'metrics.json'
        m=json.loads(p.read_text(encoding='utf-8'))['metrics']
        if m['detected']<m['reference_nodules']:preview_uids.add(meta['series_uid'])
    for meta in protocol['cases']:
        uid=meta['series_uid'];record=records[uid]
        if record['split']!='development':raise ValueError('Heldout refused')
        f,model=folds[uid];p=features_dir/(uid+'.npz')
        if sha_file(p)!=meta['feature_sha256']:raise ValueError('Feature bytes changed')
        with np.load(p) as t:table={k:t[k] for k in t.files}
        x=table['features'].astype(np.float64)
        scores=rule_scores(x) if common['kind']=='rule' else score_feature_table(x,model)
        keep=np.flatnonzero(scores>=desired['threshold'])
        keep=keep[merge_regions(table[desired['center_mode']][keep],table['boxes_zyx'][keep],scores[keep],meta['geometry']['spacing_zyx_mm'])]
        centers=table[desired['center_mode']][keep];geom=meta['geometry']
        world=centers@np.array(geom['basis_lps_zyx']).T+geom['origin_lps_mm']
        candidates=[dict(candidate_id=f'refined_{j+1:05d}',center_index_zyx=p.tolist(),
            center_lps_xyz_mm=w.tolist(),bbox_zyx=table['boxes_zyx'][k].tolist(),
            score=float(scores[k]),score_kind='development_rank_not_calibrated_probability')
            for j,(k,p,w) in enumerate(zip(keep,centers,world))]
        folder=out/uid;folder.mkdir()
        packet=dict(schema_version='lung_candidate_result_v1',algorithm_id='lung_candidate_refined_v4_experimental',
            backend='CPU_REFERENCE',status='success',series_uid=uid,source_shape_zyx=meta['source_shape_zyx'],
            geometry=geom,input_raw_sha256=meta['input_raw_sha256'],lung_mask_sha256=meta['lung_mask_sha256'],
            candidates=candidates,rank_model_kind=common['kind'],rank_threshold=common['threshold'],
            rank_model_sha256=hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
            validation_scope='scan-grouped OOF; threshold selected on this development cohort; not final test',fold=f)
        write_json(folder/'candidates.json',packet)
        metrics=point_metrics(world,record['nodules'],excluded.get(uid,[]))
        baseline=json.loads(Path(meta['baseline_path']).read_text(encoding='utf-8'))
        before=point_metrics([c['center_lps_xyz_mm'] for c in baseline['candidates']],record['nodules'],excluded.get(uid,[]))
        write_json(folder/'metrics.json',dict(original_v3=before,refined=metrics))
        images=[]
        if uid in preview_uids:
            hu,_=load_development_case(split,record)
            images=compare_preview(hu,baseline['candidates'],candidates,record['nodules'],geom,folder);del hu
        rows.append(dict(series_uid=uid,original_v3=before,refined=metrics,previews=images))
    summary=dict(reference_nodules=sum(r['refined']['reference_nodules'] for r in rows),
        detected=sum(r['refined']['detected'] for r in rows),
        false_positive_candidates=sum(r['refined']['false_positive_candidates'] for r in rows))
    if (summary['detected']!=desired['detected'] or summary['false_positive_candidates']!=desired['false_positive_candidates']):
        raise ValueError('Exported OOF packets differ from evaluated operating point')
    write_json(out/'report.json',dict(status='COMPLETE',point=desired,summary=summary,cases=rows,
        evaluation_report_sha256=sha_file(evaluation/'report.json'),heldout_scans_run=0,
        preview_selection='previous first scan and all four previously missed references; not selected by improved outcome'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--point',choices=['balanced','strict'],default='balanced');a=p.parse_args()
    export(a.features,a.evaluation,a.outdir,a.point)
