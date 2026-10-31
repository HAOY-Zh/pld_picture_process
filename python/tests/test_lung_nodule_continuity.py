"""Geometry and failure-mode tests, not claims about clinical detection."""
import unittest
import numpy as np
from lung_nodule_continuity import (ContinuityConfig, component_statistics, feature_names,
                                    local_evidence, rerank_scores)


class ContinuityTests(unittest.TestCase):
    def test_stable_vessel_is_not_equivalent_to_round_nodule(self):
        z, y, x = np.ogrid[-10:11, -10:11, -10:11]
        sphere = z*z+y*y+x*x <= 16
        tube = np.broadcast_to(y*y+x*x <= 16, sphere.shape)
        a = component_statistics(sphere, np.array([10,10,10]), np.ones(3), 8)
        b = component_statistics(tube, np.array([10,10,10]), np.ones(3), 8)
        self.assertAlmostEqual(a[5], 0)
        self.assertAlmostEqual(b[5], 0)
        self.assertGreater(b[2], a[2]*2)
        self.assertGreater(b[4], a[4])

    def test_native_plane_count_and_physical_spacing(self):
        hu = np.full((9,31,31), -1000, np.int16)
        hu[3:6,13:18,13:18] = 0
        c = dict(center_index_zyx=[4,15,15], estimated_scale_diameter_mm=6)
        before = hu.copy()
        values, detail = local_evidence(hu, [2,1,1], c)
        self.assertEqual(len(values), len(feature_names()))
        self.assertEqual([d['native_planes'] for d in detail['supports']], [3,3,3])
        np.testing.assert_array_equal(hu, before)
        self.assertAlmostEqual(values[1], 2/6)
        # Increasing native spacing changes physical extent, not slice evidence count.
        b, bd = local_evidence(hu, [3,1,1], c)
        self.assertEqual([d['native_planes'] for d in bd['supports']], [3,3,3])
        self.assertGreater(b[4], values[4])

    def test_drift_and_missing_components_are_finite(self):
        hu = np.full((9,31,31), -1000, np.int16)
        c = dict(center_index_zyx=[4,15,15], estimated_scale_diameter_mm=6)
        a, d = local_evidence(hu, [1,1,1], c)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(sum(s['native_planes'] for s in d['supports']), 0)
        stable = np.zeros((9,31,31), bool)
        moving = np.zeros_like(stable)
        for z in range(9):
            stable[z,13:18,13:18] = True
            moving[z,13:18,9+z:14+z] = True
        sa = component_statistics(stable, np.array([4,15,15]), np.ones(3), 6)
        sb = component_statistics(moving, np.array([4,15,15]), np.ones(3), 6)
        self.assertGreater(sb[5], sa[5])
        self.assertGreater(sb[6], sa[6])
        self.assertLess(sb[7], sa[7])

    def test_bounded_soft_change_and_low_score_preservation(self):
        n = len(feature_names())
        r = dict(alpha=1,mean=[0]*n,scale=[1]*n,coef=[0]*n,intercept=100)
        scores = np.array([.02,.1,.8])
        result = rerank_scores(scores, np.zeros((3,n)), r)
        self.assertEqual(result[0], scores[0])
        self.assertTrue((result[1:] > scores[1:]).all())
        r['alpha'] = 0
        np.testing.assert_array_equal(rerank_scores(scores,np.zeros((3,n)),r),scores)
        with self.assertRaises(ValueError):
            rerank_scores(scores,np.zeros((2,n)),r)

    def test_training_selection_ignores_excluded_and_guards_cohorts(self):
        from lung_nodule_continuity_experiment import select_training, eligible
        centers = [[0,0,0],[.5,0,0],[10,0,0],[20,0,0]]
        candidates = [dict(center_lps_xyz_mm=p,score=.9,candidate_id=str(i)) for i,p in enumerate(centers)]
        reference = dict(nodules=[dict(center_lps_xyz_mm=[0,0,0],diameter_mm=4)])
        ignored = [dict(center_lps_xyz_mm=[10,0,0],diameter_mm=4)]
        chosen, labels = select_training(candidates,reference,ignored,np.random.default_rng(1))
        self.assertEqual(labels.tolist(),[1,1,0])
        self.assertEqual([c['candidate_id'] for c in chosen],['0','1','3'])
        plan = dict(fit_scans=['a'],calibration_scans=['a'],review_12=['c'],reused_6=['d'])
        with self.assertRaises(ValueError):eligible(plan,'fit_scans')
        with self.assertRaises(ValueError):eligible(plan,'heldout')


if __name__ == '__main__':
    unittest.main()
