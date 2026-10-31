"""Optional local-feature ranker adapter; no change to web/default detector."""
import hashlib
import json
from time import perf_counter
import numpy as np
from lung_candidate_svm import FEATURE_NAMES as BASE_NAMES,from_packet
from lung_candidate_lowfp import FEATURE_NAMES as LOCAL_NAMES,score as local_score
from lung_candidate_paper import score as paper_score


def decision_scores(features,model):
    return paper_score(features,model) if model.get('kind')=='paper_candidate_ranker_v1' else local_score(features,model)


def extract_features(hu,result,feature_names):
    """Measure only existing candidate patches from original HU, no annotations."""
    from lung_candidate_rules import local_measurement,RulesConfig
    from lung_candidate_shell import FEATURE_NAMES as SHELL_NAMES,measure
    names=tuple(feature_names)
    if names not in (BASE_NAMES,LOCAL_NAMES,LOCAL_NAMES+SHELL_NAMES):raise ValueError('Unknown feature contract')
    if list(np.shape(hu))!=result['source_shape_zyx']:raise ValueError('CT/candidate grid mismatch')
    candidates=result['candidates'];spacing=result['geometry']['spacing_zyx_mm']
    x=from_packet(candidates,spacing)
    if len(names)>13:
        cfg=RulesConfig(**result['rules_parameters']).checked();measurements=[]
        for c in candidates:
            original=dict(c,center_index_zyx=c['original_peak_index_zyx'])
            refined,local=local_measurement(hu,original,spacing,cfg)
            np.testing.assert_allclose(refined,c['center_index_zyx'],rtol=0,atol=1e-8)
            measurements.append([local['contrast']/1000,local['std']/1000,local['occupancy'],
                local['elongation'],local['continuation'],local['shift'],local['background']/1000])
        x=np.column_stack((x,np.array(measurements,np.float32).reshape(-1,7)))
    if len(names)>20:
        shell=np.array([measure(hu,c['center_index_zyx'],spacing,np.expm1(a[3]))
            for c,a in zip(candidates,x)],np.float32).reshape(-1,len(SHELL_NAMES))
        x=np.column_stack((x,shell))
    return x


def filter_features(result,features,model,profile,model_sha256):
    """Pure scoring/filtering. Caller binds feature row order to candidate order."""
    if profile=='high_recall':threshold=model['high_recall_threshold']
    else:threshold=model['operating_thresholds'][profile]
    if threshold is None or not np.isfinite(threshold):raise ValueError('No finite development threshold for profile')
    if len(features)!=len(result['candidates']):raise ValueError('Candidate/feature count mismatch')
    start=perf_counter();scores=decision_scores(features,model)
    kept=[dict(c,primary_rule_score=c['score'],score=float(s),
        score_kind='uncalibrated_local_classifier_margin') for c,s in zip(result['candidates'],scores) if s>=threshold]
    return dict(result,algorithm_id='rule_v4_plus_local_classifier_experimental',candidates=kept,
        second_stage_model_sha256=model_sha256,second_stage_profile=profile,
        second_stage_threshold=float(threshold),second_stage_backend='CPU_REFERENCE',
        second_stage_scoring_ms=(perf_counter()-start)*1000,
        validation_scope='development only; profile FP target is not guaranteed on new scans')


def apply(hu,result,model,primary_rule,expected_primary_file_sha256,contract,profile):
    canonical=hashlib.sha256(json.dumps(primary_rule,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    model_sha=hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (result.get('algorithm_id')!='lung_candidate_refined_v4_experimental' or
            result.get('rank_model_sha256')!=canonical or
            contract['primary_file_sha256']!=expected_primary_file_sha256 or
            contract['model_canonical_sha256']!=model_sha):raise ValueError('Wrong primary or second-stage model identity')
    start=perf_counter();features=extract_features(hu,result,model['feature_names'])
    output=filter_features(result,features,model,profile,contract['model_file_sha256'])
    output['second_stage_feature_and_scoring_ms']=(perf_counter()-start)*1000
    return output
