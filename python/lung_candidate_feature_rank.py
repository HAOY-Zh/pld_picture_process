"""Tiny feature-ranker export/inference; NumPy only at runtime, no new CNN."""
import numpy as np
from lung_candidate_rank_runtime import FEATURE_NAMES,score_feature_table as feature_rank_scores


def fit_feature_rank(features,labels,weights):
    """Development trainer only; sklearn is not imported by inference."""
    from sklearn.ensemble import GradientBoostingClassifier
    x=np.asarray(features,np.float32);y=np.asarray(labels)
    weights=np.asarray(weights,float).copy()
    for value in [0,1]:
        mask=y==value
        if not mask.any():raise ValueError('Need both candidate classes in training scans')
        weights[mask]*=len(y)/(2*mask.sum())
    clf=GradientBoostingClassifier(n_estimators=32,max_depth=2,min_samples_leaf=20,
        learning_rate=.08,init='zero',random_state=20261018)
    clf.fit(x,y,sample_weight=weights)
    trees=[]
    for estimator in clf.estimators_[:,0]:
        tree=estimator.tree_
        trees.append(dict(left=tree.children_left.tolist(),right=tree.children_right.tolist(),
            feature=tree.feature.tolist(),threshold=tree.threshold.tolist(),value=tree.value[:,0,0].tolist()))
    model=dict(kind='boosted_rules',feature_names=FEATURE_NAMES,bias=0.,learning_rate=clf.learning_rate,
        maximum_depth=2,trees=trees,score_kind='feature_rule_rank_not_calibrated_probability',
        inference='32 depth-two feature trees; comparisons and additions; no neural network',
        training='development scan-grouped evaluation required')
    return model,clf
