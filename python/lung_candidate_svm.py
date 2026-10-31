"""Optional weighted-RBF-SVM second stage; existing rule candidates stay intact.

NumPy-only inference. This is an adaptation of published CAD feature/SVM
approaches, not a reproduction of a complete paper or an FPGA implementation.
"""
import numpy as np

FEATURE_NAMES = ('log_rule_score','log_blob_score','blobness','log_scale_mm',
    'log_region_diameter','log_region_elongation','mean_hu_scaled','max_hu_scaled',
    'log_box_z_mm','log_box_y_mm','log_box_x_mm','box_axis_ratio','region_box_fill')


def _assemble(base, boxes, spacing):
    base=np.asarray(base,float).reshape(-1,8)
    boxes=np.asarray(boxes,float).reshape(-1,6)
    spacing=np.asarray(spacing,float)
    if (spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any()
            or len(base)!=len(boxes)):
        raise ValueError('Feature geometry/count mismatch')
    extent=(boxes[:,1::2]-boxes[:,::2])*spacing
    if (extent<=0).any() or not np.isfinite(base).all():raise ValueError('Invalid feature/box')
    diameter=np.expm1(base[:,4])
    fill=(np.pi/6*diameter**3)/np.prod(extent,axis=1)
    return np.column_stack((base,np.log1p(extent),extent.max(1)/extent.min(1),fill))


def from_local_table(table, indices, primary_scores, spacing):
    """Reuse already saved measurements; no image access or feature extraction."""
    x=table['features'][indices].astype(float)
    base=np.column_stack((np.log1p(primary_scores),x[:,2],x[:,3],x[:,5],
                          x[:,4],x[:,6],x[:,7],x[:,8]))
    return _assemble(base,table['boxes_zyx'][indices],spacing)


def from_packet(candidates, spacing):
    """Full runtime packets already contain this shared structural descriptor."""
    values=[];boxes=[]
    for c in candidates:
        h=c['structure_features']
        # Match the existing local table's float32 descriptor encoding.
        raw=np.array([np.log1p(h['blob_score']),h['blobness'],np.log1p(h['sigma_mm']),
            np.log1p(c['equivalent_region_diameter_mm']),np.log1p(c['elongation']),
            c['mean_hu']/1000,c['max_hu']/1000],np.float32).astype(float)
        values.append([np.log1p(c['score']),*raw]);boxes.append(c['bbox_zyx'])
    return _assemble(values,boxes,spacing)


def decision_scores(features, model, chunk=256):
    if model.get('kind')!='weighted_rbf_svm_second_stage_v1' or tuple(model['feature_names'])!=FEATURE_NAMES:
        raise ValueError('Wrong second-stage model/feature order')
    x=np.asarray(features,float)
    if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('Need finite candidate features')
    mean=np.asarray(model['mean'],float);scale=np.asarray(model['scale'],float)
    support=np.asarray(model['support_vectors'],float).reshape(-1,len(FEATURE_NAMES))
    dual=np.asarray(model['dual_coefficients'],float)
    gamma=float(model['gamma']);bias=float(model['bias'])
    if (mean.shape!=(len(FEATURE_NAMES),) or scale.shape!=mean.shape or (scale<=0).any()
            or dual.shape!=(len(support),) or gamma<=0 or not np.isfinite(gamma)
            or not np.isfinite(bias) or not all(np.isfinite(v).all() for v in (mean,scale,support,dual))):
        raise ValueError('Invalid SVM parameters')
    if type(chunk)!=int or chunk<1:raise ValueError('Invalid scoring chunk')
    support_norm=(support*support).sum(1);scores=np.empty(len(x))
    for start in range(0,len(x),chunk):
        a=(x[start:start+chunk]-mean)/scale
        distance=np.maximum(0,(a*a).sum(1)[:,None]+support_norm[None]-2*a@support.T)
        scores[start:start+len(a)]=np.exp(-gamma*distance)@dual+bias
    return scores


def fit(features, labels, weights, c, gamma):
    """Training-only imports. Fit normalization on this training fold only."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    x=np.asarray(features,float);y=np.asarray(labels,int)
    if set(y)!={0,1}:raise ValueError('Both training classes required')
    scaler=StandardScaler().fit(x)
    classifier=SVC(C=c,gamma=gamma,kernel='rbf',class_weight='balanced',
                   probability=False,cache_size=128,max_iter=20000)
    classifier.fit(scaler.transform(x),y,sample_weight=np.asarray(weights,float))
    if classifier.fit_status_!=0:raise RuntimeError('SVM did not converge')
    model=dict(kind='weighted_rbf_svm_second_stage_v1',feature_names=FEATURE_NAMES,
        mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
        support_vectors=classifier.support_vectors_.tolist(),
        dual_coefficients=classifier.dual_coef_[0].tolist(),gamma=float(classifier._gamma),
        bias=float(classifier.intercept_[0]),C=c,gamma_setting=gamma,
        score_kind='uncalibrated_svm_margin_not_probability',
        stage='filter existing final rule candidates; cannot recover upstream misses')
    np.testing.assert_allclose(decision_scores(x,model),classifier.decision_function(scaler.transform(x)),
                               atol=1e-9,rtol=1e-9)
    return model
