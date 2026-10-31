"""Apply the optional second-stage SVM to a full v4 candidate result.

Accepts saved HU/geometry-independent descriptors only. No CT, annotations,
Torch, CNN, hardware download or automatic production/default replacement.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from lung_candidate_svm import from_packet,decision_scores


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def filter_result(result, model, primary_rule, primary_file_sha256):
    if (model.get('source_primary_model_sha256')!=primary_file_sha256
            or result.get('algorithm_id')!='lung_candidate_refined_v4_experimental'
            or result.get('rank_model_sha256')!=canonical_sha(primary_rule)):
        raise ValueError('Wrong primary-rule result/configuration for this SVM')
    threshold=float(model['threshold'])
    if not np.isfinite(threshold):raise ValueError('Need finite SVM threshold')
    start=perf_counter()
    features=from_packet(result['candidates'],result['geometry']['spacing_zyx_mm'])
    packed=perf_counter();scores=decision_scores(features,model);scored=perf_counter()
    candidates=[dict(c,primary_rule_score=c['score'],score=float(s),
        score_kind='uncalibrated_second_stage_svm_margin')
        for c,s in zip(result['candidates'],scores) if s>=threshold]
    return dict(result,algorithm_id='rule_v4_plus_weighted_rbf_svm_experimental',
        candidates=candidates,primary_candidate_count=len(result['candidates']),
        second_stage_model_identity_sha256=canonical_sha(model),second_stage_threshold=threshold,
        second_stage_backend='CPU_REFERENCE',
        second_stage_timings_ms=dict(feature_pack=(packed-start)*1000,
            score=(scored-packed)*1000,total=(perf_counter()-start)*1000),
        second_stage_note='Additional descriptor/SVM/filter time only; primary timings refer to the supplied original result. No PL claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--primary-rule',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing predictions')
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    result=filter_result(load(a.candidates),load(a.model),load(a.primary_rule),
        hashlib.sha256(a.primary_rule.read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('SVM second stage:',result['primary_candidate_count'],'->',len(result['candidates']),
        'candidates; descriptors only, no CT/network/PL execution')


if __name__=='__main__':main()
