"""Audit saved second-stage packets, geometry and actual model identities."""
import argparse
import json
from pathlib import Path
import numpy as np

from lung_candidate_data import read_split,sha_file,write_json
from lung_candidate_run import excluded_findings
from lung_candidates import point_metrics
from lung_candidate_svm_apply import filter_result

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'data/lung_candidate_rules_20261018'


def audit(out):
    destination=out/'output_audit.json'
    if destination.exists():raise FileExistsError('Preserve earlier output audit')
    report=json.loads((out/'report.json').read_text('utf-8'))
    for path,digest in report['protected_files_sha256'].items():
        if sha_file(path)!=digest:raise ValueError('Protected source/cache changed: '+path)
    split=read_split(ROOT/'data/lung_candidate_split_20261018/split.json')
    records={r['series_uid']:r for r in split['cases']}
    excluded=excluded_findings(ROOT/'data/lung_candidate_planning_20261017/annotations_excluded.csv')
    checks=0;metric_checks=0;localization={};maximum_margin_difference=0.
    for folder,base in [('development_oof','oof_comparisons'),('six_check','new_development_check')]:
        distances={'before':[],'after':[]};counts=dict(detected=0,fp=0,candidates=0)
        for path in sorted((out/folder).glob('*.json')):
            uid=path.stem
            if records[uid]['split']!='development':raise ValueError('Heldout refused')
            result=json.loads(path.read_text('utf-8'))
            old=json.loads((PRIOR/base/uid/'candidates.json').read_text('utf-8'))
            originals={c['candidate_id']:c for c in old['candidates']}
            model_path=(out/result['second_stage_model_path']).resolve()
            if out.resolve() not in model_path.parents:raise ValueError('Model path outside experiment')
            model=json.loads(model_path.read_text('utf-8'))
            if sha_file(model_path)!=result['second_stage_model_sha256']:raise ValueError('Wrong actual model identity')
            if folder=='development_oof' and uid in model['training_scans']:raise ValueError('OOF scan leakage')
            for c in result['candidates']:
                before=originals[c['candidate_id']]
                if any(c[k]!=before[k] for k in ('center_index_zyx','center_lps_xyz_mm','bbox_zyx')):
                    raise ValueError('Candidate geometry moved')
                if c['primary_rule_score']!=before['score'] or c['score']<model['threshold']:
                    raise ValueError('Score/filter mismatch')
            if folder=='six_check':
                primary_path=PRIOR/'evaluation_scale2/balanced_rank_model.json'
                primary=json.loads(primary_path.read_text('utf-8'))
                actual=filter_result(old,model,primary,sha_file(primary_path))['candidates']
                expected=result['candidates']
                if len(actual)!=len(expected):raise ValueError('Runtime filter selection changed')
                for a,b in zip(actual,expected):
                    if {k:v for k,v in a.items() if k!='score'}!={k:v for k,v in b.items() if k!='score'}:
                        raise ValueError('Runtime packet content changed')
                # BLAS thread count may change final float64 accumulation ulps.
                a=np.array([c['score'] for c in actual]);b=np.array([c['score'] for c in expected])
                np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
                if len(a):maximum_margin_difference=max(maximum_margin_difference,float(np.max(np.abs(a-b))))
            checks+=1;computed={}
            for kind,packet in [('before',old),('after',result)]:
                m=point_metrics([c['center_lps_xyz_mm'] for c in packet['candidates']],records[uid]['nodules'],excluded.get(uid,[]))
                computed[kind]=m;metric_checks+=1
                distances[kind].extend(d['nearest_distance_mm'] for d in m['reference_details'] if d['detected'])
                if kind=='after':
                    counts['detected']+=m['detected'];counts['fp']+=m['false_positive_candidates'];counts['candidates']+=m['candidates']
            if any(a['detected'] and not b['detected'] for a,b in zip(computed['before']['reference_details'],computed['after']['reference_details'])):
                raise ValueError('New missed reference')
        expected=report['development_24_oof' if folder=='development_oof' else 'development_check_6']['after']
        if counts!=dict(detected=expected['detected'],fp=expected['false_positive_candidates'],candidates=expected['candidates']):
            raise ValueError('Aggregate metrics mismatch')
        localization[folder]={k:dict(mean_mm=float(np.mean(v)),median_mm=float(np.median(v)),
            p95_mm=float(np.percentile(v,95)),max_mm=float(max(v))) for k,v in distances.items()}
    old=json.loads((PRIOR/'final_summary.json').read_text('utf-8'))
    for path,digest in old['current_source_sha256'].items():
        if sha_file(path)!=digest:raise ValueError('Original candidate source changed')
    proof=dict(status='PASS_INDEPENDENT_CACHED_OUTPUT_AUDIT',packets=checks,metric_recomputations=metric_checks,
        protected_files_unchanged=len(report['protected_files_sha256']),
        original_candidate_source_files_unchanged=len(old['current_source_sha256']),
        heldout_scans_run=0,ct_volumes_read=0,geometry_unchanged=True,
        per_packet_actual_model_bound=True,runtime_six_selection_exact=True,
        maximum_runtime_margin_difference=maximum_margin_difference,localization=localization)
    write_json(destination,proof);print(json.dumps(proof,indent=2));return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    audit(p.parse_args().experiment)
