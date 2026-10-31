import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_candidate_froc import evaluate


class FrocTest(unittest.TestCase):
    def test_duplicates_excluded_missed_strict_boundary_and_ties(self):
        n = lambda x: dict(center_lps_xyz_mm=[x, 0, 0], diameter_mm=2.)
        rows = [dict(uid='a', world=np.array([[0,0,0],[.2,0,0],[1,0,0],[5,0,0],[9,0,0]]),
            record=dict(nodules=[n(0), n(20)]), excluded=[n(9)])]
        # Boundary at 1 mm is FP; duplicate .2 and excluded 9 are ignored.
        result = evaluate(rows, {'a': np.array([.9,.8,.9,.1,10.])}, points=(0.,1.,2.))
        self.assertEqual(result['detectable'], 1)
        self.assertEqual(result['reference_nodules'], 2)
        self.assertEqual(result['ignored_excluded_candidates'], 1)
        self.assertEqual([p['sensitivity'] for p in result['operating_points']], [0.,.5,.5])
        self.assertEqual(result['curve']['fp_per_scan'], [0.,1.,2.])

    def test_no_candidates_and_no_false_positives(self):
        row=dict(uid='a',world=np.empty((0,3)),record=dict(nodules=[dict(
            center_lps_xyz_mm=[0,0,0],diameter_mm=2)]),excluded=[])
        self.assertEqual(evaluate([row],{'a':np.array([])})['cpm_interpolated'],0)
        row['world']=np.zeros((1,3))
        self.assertEqual(evaluate([row],{'a':np.array([1.])})['cpm_interpolated'],1)


if __name__=='__main__': unittest.main()
