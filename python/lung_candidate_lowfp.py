"""Portable small candidate rankers using existing structural/local descriptors.

No image loading, annotations, Torch or sklearn in the inference path. These
CPU reference rankers are optional; no existing detector or web default changes.
"""
import numpy as np
from lung_candidate_svm import FEATURE_NAMES as STRUCTURAL_NAMES

LOCAL_NAMES = ('local_contrast_scaled', 'local_core_std_scaled', 'local_occupancy',
    'local_elongation', 'axis_continuation', 'center_shift_mm', 'local_background_scaled')

FEATURE_NAMES = tuple(STRUCTURAL_NAMES)+tuple(LOCAL_NAMES)


def fit(x, y, weights, parameters, feature_names):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
    x = np.asarray(x, float); y = np.asarray(y, int); weights = np.asarray(weights, float)
    if set(y) != {0, 1} or x.shape[1] != len(feature_names): raise ValueError('Invalid training data')
    positive = y == 1
    weights = weights.copy()
    weights[positive] *= len(y)/(2*positive.sum())
    weights[~positive] *= len(y)/(2*(~positive).sum())
    family = parameters['family']
    model = dict(kind='local_candidate_ranker_v1', family=family, feature_names=list(feature_names),
        parameters=parameters, score_kind='uncalibrated_candidate_rank_not_clinical_probability')
    if family == 'svm':
        scaler = StandardScaler().fit(x)
        clf = SVC(C=parameters['C'], gamma='scale', kernel='rbf', cache_size=128, max_iter=30000)
        clf.fit(scaler.transform(x), y, sample_weight=weights)
        if clf.fit_status_: raise RuntimeError('SVM did not converge')
        model.update(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
            support_vectors=clf.support_vectors_.tolist(), dual=clf.dual_coef_[0].tolist(),
            gamma=float(clf._gamma), bias=float(clf.intercept_[0]))
        expected = clf.decision_function(scaler.transform(x))
    else:
        if family == 'boosted_trees':
            clf = GradientBoostingClassifier(n_estimators=80, learning_rate=.05,
                max_depth=parameters['depth'], min_samples_leaf=8, random_state=20261018)
        elif family == 'extra_trees':
            clf = ExtraTreesClassifier(n_estimators=120, max_depth=parameters['depth'],
                min_samples_leaf=3, max_features=1., random_state=20261018, n_jobs=1)
        else: raise ValueError('Unknown ranker')
        clf.fit(x, y, sample_weight=weights)
        estimators = clf.estimators_.ravel() if family == 'boosted_trees' else clf.estimators_
        trees = []
        for estimator in estimators:
            tree = estimator.tree_
            values = tree.value.reshape(tree.node_count, -1)
            value = values[:, 0] if family == 'boosted_trees' else values[:, 1]/values.sum(axis=1)
            trees.append(dict(left=tree.children_left.tolist(), right=tree.children_right.tolist(),
                feature=tree.feature.tolist(), threshold=tree.threshold.tolist(), value=value.tolist()))
        model.update(trees=trees)
        if family == 'boosted_trees':
            prior = clf.init_.class_prior_
            model.update(bias=float(np.log(prior[1]/prior[0])), learning_rate=clf.learning_rate)
            expected = clf.decision_function(x)
        else: expected = clf.predict_proba(x)[:, 1]
    np.testing.assert_allclose(score(x, model), expected, rtol=1e-9, atol=1e-9)
    return model


def score(features, model):
    x = np.asarray(features, float)
    if (model.get('kind') != 'local_candidate_ranker_v1' or x.ndim != 2
            or x.shape[1] != len(model['feature_names']) or not np.isfinite(x).all()):
        raise ValueError('Wrong ranker/feature contract')
    if model['family'] == 'svm':
        mean = np.asarray(model['mean']); scale = np.asarray(model['scale'])
        support = np.asarray(model['support_vectors']); dual = np.asarray(model['dual'])
        a = (x-mean)/scale; result = np.empty(len(x)); norm = (support**2).sum(1)
        for begin in range(0, len(x), 256):
            part = a[begin:begin+256]
            d = np.maximum(0, (part**2).sum(1)[:, None]+norm[None]-2*part@support.T)
            result[begin:begin+len(part)] = np.exp(-model['gamma']*d)@dual+model['bias']
        return result
    # sklearn decision trees compare float32 input with double thresholds.
    a = x.astype(np.float32); result = np.zeros(len(x))
    for tree in model['trees']:
        left = np.asarray(tree['left']); right = np.asarray(tree['right'])
        feature = np.asarray(tree['feature']); threshold = np.asarray(tree['threshold'])
        node = np.zeros(len(x), int); active = np.flatnonzero(left[node] >= 0)
        while len(active):
            here = node[active]
            node[active] = np.where(a[active, feature[here]] <= threshold[here], left[here], right[here])
            active = active[left[node[active]] >= 0]
        result += np.asarray(tree['value'])[node]
    if model['family'] == 'boosted_trees': return result*model['learning_rate']+model['bias']
    if model['family'] == 'extra_trees': return result/len(model['trees'])
    raise ValueError('Unknown model family')
