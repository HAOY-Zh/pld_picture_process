import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_candidate_froc import evaluate
from lung_nodule_five_bundle_evaluation import verify_rows


class PairedEvaluationTests(unittest.TestCase):
    def test_matching_and_ignored_duplicate_counts(self):
        r=dict(uid='one',world=np.array([[0,0,0],[.1,0,0],[10,0,0],[20,0,0]]),
            record=dict(nodules=[dict(center_lps_xyz_mm=[0,0,0],diameter_mm=4.)]),
            excluded=[dict(center_lps_xyz_mm=[10,0,0],diameter_mm=4.)])
        empty=dict(uid='two',world=np.zeros((0,3)),record=dict(nodules=[]),excluded=[])
        scores=dict(one=np.array([.9,.8,.7,.6]),two=np.array([]))
        curve=evaluate([r,empty],scores)
        verify_rows([r,empty],scores,curve)
        self.assertEqual(curve['reference_nodules'],1)
        self.assertEqual(curve['scans'],2)
        self.assertEqual(curve['operating_points'][-1]['actual_fp_per_scan'],.5)

    def test_independent_counting_rejects_wrong_summary(self):
        r=dict(uid='a',world=np.array([[0,0,0]]),record=dict(nodules=[dict(center_lps_xyz_mm=[0,0,0],diameter_mm=4.)]),excluded=[])
        scores=dict(a=np.array([.9]));curve=evaluate([r],scores)
        curve['operating_points'][-1]['detected']=0
        with self.assertRaises(ValueError):verify_rows([r],scores,curve)


if __name__=='__main__':unittest.main()
