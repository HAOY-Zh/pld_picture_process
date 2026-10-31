import unittest
import numpy as np

from lung_nodule_capacity_froc_audit import roc_counts,check_froc_stable
from lung_candidate_froc import evaluate


class RocPrecisionTests(unittest.TestCase):
    def test_exact_budget_at_vertical_step(self):
        # Some FPR denominators round just above the true three-FP count.
        n=next(n for n in range(4,10000) if (3/n)*n>3)
        positive=[.95,.70]
        negative=[.90,.85,.80]+np.linspace(.60,.01,n-3).tolist()
        x,y,old_x,old_y=roc_counts(positive,negative,12,2)
        self.assertEqual(float(np.interp(.25,x,y)),1.)
        self.assertLess(float(np.interp(.25,old_x,old_y)),.75)

    def test_whole_curve_independent_matching_with_ties_and_exclusion(self):
        nodule=dict(center_lps_xyz_mm=[0.,0.,0.],diameter_mm=2.)
        excluded=dict(center_lps_xyz_mm=[10.,0.,0.],diameter_mm=2.)
        row=dict(uid='a',world=np.array([[0.,0.,0.],[.2,0.,0.],[10.,0.,0.],[20.,0.,0.],[30.,0.,0.]]),
            record=dict(nodules=[nodule]),excluded=[excluded])
        scores={'a':np.array([.7,.8,.99,.8,.1])}
        curve=evaluate([row],scores)
        self.assertEqual(curve['detectable'],1)
        self.assertEqual(check_froc_stable([row],scores,curve)['status'],'PASS')


if __name__=='__main__':
    unittest.main()
