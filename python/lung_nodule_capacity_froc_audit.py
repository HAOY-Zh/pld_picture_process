"""Independent ROC audit with integer count recovery before per-scan division.

Multiplying sklearn's floating FPR by the negative count can turn 3 FP into
3.0000000000000004. At an exact budget plateau, np.interp can then choose the
wrong side of a vertical FROC step. Restore integer counts, not scores/budgets.
"""
import numpy as np
from sklearn.metrics import roc_curve

from lung_candidates import point_metrics
from lung_candidate_rules_run import labels_for


def roc_counts(positive,negative,scans,references):
    p,n=len(positive),len(negative)
    if not p or not n:
        raise ValueError('This independent ROC path needs positive and negative proposals')
    fpr,tpr,_=roc_curve([1]*p+[0]*n,list(positive)+list(negative),drop_intermediate=False)
    fp=np.rint(fpr*n).astype(np.int64)
    tp=np.rint(tpr*p).astype(np.int64)
    np.testing.assert_allclose(fpr*n,fp,rtol=0,atol=1e-8)
    np.testing.assert_allclose(tpr*p,tp,rtol=0,atol=1e-8)
    return fp/scans,tp/references,fpr*n/scans,tpr*p/references


def check_froc_stable(rows,scores,expected):
    positive=[];negative=[];references=0
    for row in rows:
        prediction=scores[row['uid']]
        metric=point_metrics(row['world'],row['record']['nodules'],row['excluded'])
        for detail in metric['reference_details']:
            references+=1
            indices=detail['matching_candidate_indices']
            if indices:
                positive.append(float(prediction[indices].max()))
        negative.extend(prediction[labels_for(row['world'],row['record'],row['excluded'])==0].tolist())
    fp,tpr,old_fp,old_tpr=roc_counts(positive,negative,len(rows),references)
    budgets=[p['fp_budget'] for p in expected['operating_points']]
    stable=np.interp(budgets,fp,tpr)
    old=np.interp(budgets,old_fp,old_tpr)
    np.testing.assert_allclose(stable,[p['interpolated_sensitivity'] for p in expected['operating_points']],
                               rtol=0,atol=1e-12)
    for point in expected['operating_points']:
        detected=false_positive=0
        for row in rows:
            keep=scores[row['uid']]>=point['threshold'] if point['threshold'] is not None else np.zeros(len(row['world']),bool)
            m=point_metrics(row['world'][keep],row['record']['nodules'],row['excluded'])
            detected+=m['detected']
            false_positive+=m['false_positive_candidates']
        if (detected,false_positive/len(rows))!=(point['detected'],point['actual_fp_per_scan']):
            raise AssertionError('Empirical FROC count mismatch')
    return dict(status='PASS',integer_roc_counts_recovered=True,
        previous_float_conversion_different_points=[dict(fp_budget=b,old_float_value=float(a),stable_value=float(c))
            for b,a,c in zip(budgets,old,stable) if abs(a-c)>1e-12])
