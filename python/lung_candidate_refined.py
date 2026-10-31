"""Standalone experimental refinement entry, reusing the existing lung mask.

No Torch/sklearn runtime dependency. Accept ORIGINAL HU, not an LDCT-enhanced
display image. The rank model is an experimental local-feature model, not a new
CNN. Existing v3 and the web defaults are untouched.
"""
from dataclasses import asdict,replace
import hashlib
import json
from time import perf_counter
import numpy as np

from lung_candidates import detect_candidates,checked_inputs,CandidateConfig
from lung_candidate_structure import hessian_features
from lung_candidate_rules import (RulesConfig,FEATURE_NAMES,measure_pool,
                                  rule_scores,linear_scores,merge_regions,broad_pool)
from lung_candidate_feature_rank import feature_rank_scores


def rule_upper_bounds(hessian):
    """Exact bound for the stored float32 log-score and current rule contract.

    Contrast <= 1.5; continuation <= 1; shape <= 1. Thus no local measurement
    can make a point exceed 1.5 * encoded_blob_score. No annotations involved.
    """
    score=np.asarray([f['blob_score'] for f in hessian],float)
    encoded=np.log1p(score).astype(np.float32).astype(np.float64)
    return 1.5*np.expm1(encoded)


def _measure_with_bound(hu,candidates,features,geometry,cfg,threshold,enabled):
    pool=broad_pool(candidates,features,cfg)
    eligible=pool
    if enabled:
        eligible=pool[rule_upper_bounds([features[i] for i in pool])>=threshold]
    table=measure_pool(hu,[candidates[i] for i in eligible],[features[i] for i in eligible],geometry,cfg)
    table['source_indices']=eligible[table['source_indices']]
    return table,pool,len(pool)-len(eligible)


def extract_refined_regions(hu,lung_mask,geometry,rank_model,threshold=None,candidate_params=None,optimize_rule=True):
    """Return result/debug; candidate identity and current CT are caller-owned.

    rank_model: kind='rule' or 'linear', center_mode, optional rules_config and
    threshold. Linear models contain feature_names/weights/bias exported by the
    development runner. Scores have not been calibrated as clinical probability.
    """
    checked_inputs(hu,lung_mask,geometry)
    if rank_model.get('kind') not in ('rule','linear','boosted_rules'):
        raise ValueError('Need an explicitly identified rule or linear rank model')
    mode=rank_model.get('center_mode','refined_centers_zyx')
    if mode not in ('original_centers_zyx','refined_centers_zyx'):
        raise ValueError('Invalid center mode')
    threshold=rank_model.get('threshold') if threshold is None else threshold
    if threshold is None or not np.isfinite(threshold):raise ValueError('Need a finite explicit threshold')
    cfg=RulesConfig(**rank_model.get('rules_config',{})).checked()
    start=perf_counter()
    coarse,seed=detect_candidates(hu,lung_mask,geometry,candidate_params or CandidateConfig())
    features=hessian_features(hu,coarse['candidates'],geometry['spacing_zyx_mm'])
    bound_enabled=bool(optimize_rule and rank_model['kind']=='rule')
    table,original_pool,skipped=_measure_with_bound(hu,coarse['candidates'],features,geometry,cfg,threshold,bound_enabled)
    total_pool_count=len(original_pool)
    complementary=rank_model.get('complementary_sigma_mm')
    original_coarse_count=len(coarse['candidates'])
    if complementary is not None:
        if not np.isfinite(complementary) or complementary<=0:raise ValueError('Invalid complementary scale')
        single,single_seed=detect_candidates(hu,lung_mask,geometry,
            replace(candidate_params or CandidateConfig(),sigma_mm=(float(complementary),)))
        del single_seed
        existing={tuple(coarse['candidates'][i]['center_index_zyx']) for i in original_pool
                  if np.isclose(features[i]['sigma_mm'],complementary,atol=1e-5)}
        extra=[dict(c,candidate_id='scale2_'+c['candidate_id']) for c in single['candidates']
               if tuple(c['center_index_zyx']) not in existing]
        extra_features=hessian_features(hu,extra,geometry['spacing_zyx_mm'])
        added,extra_pool,extra_skipped=_measure_with_bound(hu,extra,extra_features,geometry,cfg,threshold,bound_enabled)
        total_pool_count+=len(extra_pool);skipped+=extra_skipped
        added['source_indices']+=original_coarse_count
        table={key:np.concatenate([table[key],added[key]],axis=0) for key in table}
        coarse['candidates'].extend(extra);features.extend(extra_features)
    scores=rule_scores(table['features'].astype(np.float64)) if rank_model['kind']=='rule' else feature_rank_scores(table['features'],rank_model)
    indices=np.flatnonzero(scores>=threshold)
    indices=indices[merge_regions(table[mode][indices],table['boxes_zyx'][indices],scores[indices],
                                 geometry['spacing_zyx_mm'],cfg)]
    candidates=[]
    for j,k in enumerate(indices):
        source=int(table['source_indices'][k]);candidate=dict(coarse['candidates'][source])
        p=table[mode][k]
        candidate.update(candidate_id=f'refined_{j+1:05d}',source_candidate_id=candidate['candidate_id'],
            original_peak_index_zyx=candidate['center_index_zyx'],center_index_zyx=p.tolist(),
            center_lps_xyz_mm=(np.array(geometry['origin_lps_mm'])+np.array(geometry['basis_lps_zyx'])@p).tolist(),
            bbox_zyx=table['boxes_zyx'][k].tolist(),score=float(scores[k]),
            score_kind='local_feature_rank_not_calibrated_probability',
            structure_features=features[source])
        candidates.append(candidate)
    result=dict(schema_version='lung_candidate_result_v1',algorithm_id='lung_candidate_refined_v4_experimental',
        backend='CPU_REFERENCE',status='success',source_shape_zyx=list(np.shape(hu)),
        geometry={k:np.asarray(geometry[k]).tolist() for k in ('spacing_zyx_mm','origin_lps_mm','basis_lps_zyx')},
        candidates=candidates,parameters=coarse['parameters'],rules_parameters=asdict(cfg),
        rank_model_kind=rank_model['kind'],rank_threshold=float(threshold),center_mode=mode,
        coarse_candidate_count=len(coarse['candidates']),broad_pool_count=total_pool_count,
        local_feature_candidates_measured=len(scores),local_feature_candidates_skipped_by_bound=skipped,
        safe_rule_bound_enabled=bound_enabled,
        original_coarse_count=original_coarse_count,complementary_sigma_mm=complementary,
        feature_names=FEATURE_NAMES,validation_scope='development experiment; no heldout or PL validation',
        rank_model_sha256=hashlib.sha256(json.dumps(rank_model,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        timings_ms=dict(candidate_total=(perf_counter()-start)*1000))
    return result,dict(seed=seed,feature_table=table,coarse_candidates=coarse['candidates'],
                       hessian_features=features,original_coarse_count=original_coarse_count)


def refined_roi_packet(result):
    """Keep rank identity when passing a compact packet to the ROI consumer."""
    from lung_candidate_structure import roi_candidate_packet
    packet=roi_candidate_packet(result)
    for key in ('rules_parameters','rank_model_sha256','rank_model_kind','rank_threshold',
                'center_mode','complementary_sigma_mm','feature_names'):
        if key in result:packet[key]=result[key]
    return packet
