"""Check frozen-threshold review counts, duplicate/exclusion handling and gates."""
import unittest
import numpy as np
from lung_nodule_cnn_review18 import eligible_uids,summarize_fixed,independent_counts


class FixedReviewTests(unittest.TestCase):
    def test_fixed_threshold_duplicates_exclusions_and_strict_boundary(self):
        row=dict(uid='synthetic',record=dict(nodules=[dict(center_lps_xyz_mm=[0,0,0],diameter_mm=4)]),
            excluded=[dict(center_lps_xyz_mm=[10,0,0],diameter_mm=4)],
            world=np.array([[0,0,0],[.5,0,0],[10,0,0],[2,0,0],[20,0,0]]),
            scores=np.array([.8,.7,.95,.85,.4]))
        m=summarize_fixed([row],.7)
        self.assertEqual(m['detected'],1)
        self.assertEqual(m['false_positive_candidates'],1)
        self.assertEqual(m['selected_candidates'],4)
        self.assertEqual(m['actual_fp_per_scan'],1.)
        independent=independent_counts([row],.7)
        self.assertEqual(independent['detected'],m['detected'])
        self.assertEqual(independent['false_positive_candidates'],m['false_positive_candidates'])

    def test_reference_free_cases_stay_in_false_positive_denominator(self):
        row=dict(uid='negative',record=dict(nodules=[]),excluded=[],
                 world=np.array([[1,2,3]]),scores=np.array([.8]))
        m=summarize_fixed([row,row],.8)
        self.assertIsNone(m['sensitivity'])
        self.assertEqual(m['actual_fp_per_scan'],1.)
        self.assertEqual(independent_counts([row,row],.8)['false_positive_candidates'],2)

    def test_no_training_calibration_or_heldout_entrypoint(self):
        p=dict(fit_scans=['fit'],calibration_scans=['cal'],review_12=['new'],reused_6=['old'])
        self.assertEqual(eligible_uids(p,'review_12'),['new'])
        for name in ['heldout','fit_scans','calibration_scans']:
            with self.assertRaises(ValueError):eligible_uids(p,name)
        p['review_12']=['fit']
        with self.assertRaises(ValueError):eligible_uids(p,'review_12')


if __name__=='__main__':
    unittest.main()
