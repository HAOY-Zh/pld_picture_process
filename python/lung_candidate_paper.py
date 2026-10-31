"""Gu 2019-inspired feature selection / boundary undersampling / weighted SVM.

Adaptation, not a reproduction: fixed shell sampling, random-subset stability
ranking, scan-grouped inner checks and a hard/random negative mixture differ
from the paper. No annotation enters the inference path.
"""
import numpy as np
from lung_candidate_lowfp import fit as fit_svm, score as score_svm


def negative_subset(x, y, ratio, seed=20261018):
    """Retain all positives, 75% near-positive negatives and 25% random negatives.

    Selection is ONLY within training scans. It does not discard predictions,
    truncate per-scan outputs, or read labels at inference.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    if not len(pos) or not len(neg):raise ValueError('Both training classes required')
    count=min(len(neg),int(ratio*len(pos)))
    if count>=len(neg):return np.arange(len(y))
    z=StandardScaler().fit_transform(x)
    distance=NearestNeighbors(n_neighbors=min(3,len(pos))).fit(z[pos]).kneighbors(z[neg])[0].mean(1)
    ordered=neg[np.argsort(distance,kind='stable')]
    hard=ordered[:int(np.ceil(.75*count))]
    random=np.random.default_rng(seed).choice(ordered[len(hard):],size=count-len(hard),replace=False)
    return np.sort(np.r_[pos,hard,random])


def select_features(x,y,groups,weights,feature_names,subsets=64,keep=16):
    """RSFS-inspired random subset ranking; all inner scans belong to training.

    Each subset is tested by 3-NN on three scan-grouped inner folds. Feature
    relevance is mean Gmean when present minus when absent. This is explicitly
    not the paper's exact RSFS/random-walk implementation.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    import hashlib
    uids=sorted(set(groups),key=lambda s:hashlib.sha256(('rsfs:'+s).encode()).hexdigest())
    inner=[set(uids[i::3]) for i in range(3)]
    rng=np.random.default_rng(20261018)
    selections=np.array([np.sort(rng.choice(x.shape[1],size=min(12,x.shape[1]),replace=False)) for _ in range(subsets)])
    totals=np.zeros((subsets,4))
    for validation in inner:
        test=np.array([uid in validation for uid in groups]);train=~test
        if len(set(y[train]))!=2 or len(set(y[test]))!=2:raise ValueError('Inner grouped fold needs both classes')
        scaler=StandardScaler().fit(x[train]);a=scaler.transform(x[train]);b=scaler.transform(x[test])
        selected=negative_subset(a,y[train],10)
        for j,columns in enumerate(selections):
            clf=KNeighborsClassifier(n_neighbors=3,weights='distance').fit(a[selected][:,columns],y[train][selected])
            guess=clf.predict(b[:,columns]);positive=y[test]==1;w=weights[test]
            totals[j]+=np.array([w[positive & (guess==1)].sum(),w[positive].sum(),
                int((~positive & (guess==0)).sum()),int((~positive).sum())])
    gmean=np.sqrt(totals[:,0]/totals[:,1]*totals[:,2]/totals[:,3])
    relevance=[]
    for column in range(x.shape[1]):
        present=np.any(selections==column,axis=1)
        relevance.append(float(gmean[present].mean()-gmean[~present].mean()))
    chosen=np.sort(np.argsort(-np.array(relevance),kind='stable')[:min(keep,x.shape[1])])
    return dict(method='RSFS-inspired present-minus-absent mean inner grouped Gmean',
        exact_paper_RSFS=False,random_subsets=subsets,subset_size=selections.shape[1],selected=chosen.tolist(),
        selected_names=[feature_names[i] for i in chosen],feature_relevance=relevance,
        inner_validation_scans=[sorted(s) for s in inner],seed=20261018)


def fit(x,y,weights,parameters,feature_names,selection):
    columns=selection['selected'];a=x[:,columns]
    keep=(negative_subset(a,y,parameters['negative_ratio']) if parameters['negative_ratio'] else np.arange(len(y)))
    inner=fit_svm(a[keep],y[keep],weights[keep],dict(family='svm',C=parameters['C']),
        [feature_names[i] for i in columns])
    return dict(kind='paper_candidate_ranker_v1',feature_names=list(feature_names),
        selection=selection,selected_features=columns,svm=inner,parameters=parameters,
        training_candidates=len(y),retained_training_candidates=len(keep),
        retained_training_positives=int((y[keep]==1).sum()),retained_training_negatives=int((y[keep]==0).sum()),
        literature='https://doi.org/10.1371/journal.pone.0210551',
        exact_paper_reproduction=False,formal_deployment=False)


def score(features,model):
    x=np.asarray(features,float)
    if model.get('kind')!='paper_candidate_ranker_v1' or x.ndim!=2 or x.shape[1]!=len(model['feature_names']):
        raise ValueError('Wrong paper-inspired descriptor contract')
    return score_svm(x[:,model['selected_features']],model['svm'])
