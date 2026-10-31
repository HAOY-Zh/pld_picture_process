"""Small exported candidate ranker for PS/PC: NumPy only, no trainer imports.

This scores an already computed local feature table. It does not load CT, segment
lungs, run LDCT or claim hardware acceleration. Scores are uncalibrated ranks.
"""
import numpy as np

FEATURE_NAMES = (
    'log_response_2d', 'log_response_3d', 'log_blob_score', 'blobness',
    'log_region_diameter', 'log_scale_mm', 'log_elongation', 'mean_hu_scaled',
    'max_hu_scaled', 'local_contrast_scaled', 'local_core_std_scaled',
    'local_occupancy', 'local_elongation', 'axis_continuation',
    'center_shift_mm', 'z_extent_mm', 'xy_extent_mm', 'local_background_scaled',
)


def score_feature_table(features,model,feature_names=FEATURE_NAMES):
    kind=model.get('kind')
    declared=model.get('feature_names',FEATURE_NAMES if kind=='rule' else ())
    if tuple(feature_names)!=FEATURE_NAMES or tuple(declared)!=FEATURE_NAMES:
        raise ValueError('Candidate feature order differs from model contract')
    x=np.asarray(features,np.float64 if kind in ('linear','rule') else np.float32)
    if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('Need finite local feature table')
    if kind=='linear':
        weights=np.asarray(model['weights'],float)
        if weights.shape!=(len(FEATURE_NAMES),):raise ValueError('Invalid weight count')
        return x@weights+float(model['bias'])
    if kind=='rule':
        contrast=np.clip(x[:,9]/.3,.15,1.5)
        continuation=np.clip(1.-.65*np.maximum(x[:,13],0),.15,1.)
        return np.expm1(x[:,2])*contrast*continuation/np.sqrt(np.maximum(x[:,12],1))
    if kind!='boosted_rules':raise ValueError('Unknown candidate rank model')
    result=np.full(len(x),float(model['bias']),np.float64)
    for tree in model['trees']:
        left=np.array(tree['left'],int);right=np.array(tree['right'],int)
        column=np.array(tree['feature'],int);threshold=np.array(tree['threshold'])
        values=np.array(tree['value']);nodes=np.zeros(len(x),int)
        for _ in range(model['maximum_depth']+1):
            active=np.flatnonzero(left[nodes]>=0)
            if not len(active):break
            current=nodes[active]
            nodes[active]=np.where(x[active,column[current]]<=threshold[current],left[current],right[current])
        if (left[nodes]>=0).any():raise ValueError('Tree depth contract differs')
        result+=model['learning_rate']*values[nodes]
    return result
