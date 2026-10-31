import unittest
import numpy as np

from lung_nodule_heldout_report import bootstrap_fixed,fixed_metrics


class FinalReportTests(unittest.TestCase):
    def test_negative_scans_stay_in_fp_denominator(self):
        nodules=[dict(center_lps_xyz_mm=[0.,0.,0.],diameter_mm=2.),
                 dict(center_lps_xyz_mm=[5.,0.,0.],diameter_mm=2.)]
        def row(uid,world,nodules,scores):
            return dict(uid=uid,world=np.array(world),record=dict(nodules=nodules),excluded=[],
                raw_scores=np.array(scores),cross_scores=np.array(scores),
                reference_center_in_search=[True]*len(nodules),
                packet=dict(candidates=[dict(candidate_id=str(i)) for i in range(len(world))]))
        rows=[row('positive',[[0.,0.,0.],[5.,0.,0.],[20.,0.,0.]],nodules,[.95,.7,.9]),
              row('no_reference',[[30.,0.,0.],[40.,0.,0.]],[],[.8,.1])]
        metric=fixed_metrics(rows,.85,'cross',123)
        self.assertEqual(metric['detected'],1)
        self.assertEqual(metric['reference_nodules'],2)
        self.assertEqual(metric['sensitivity'],.5)
        self.assertEqual(metric['actual_fp_per_scan'],.5)
        self.assertEqual(metric['missed_reference_reasons'],{'matching_candidate_exists_below_frozen_threshold':1})
        self.assertEqual(metric['size_groups']['under6']['detected'],1)
        metric=fixed_metrics(rows,.65,'cross',123)
        self.assertEqual(metric['sensitivity'],1.)
        self.assertEqual(metric['actual_fp_per_scan'],1.)

    def test_case_bootstrap_keeps_ct_as_sampling_unit(self):
        cases=[dict(metric=dict(reference_nodules=1,detected=1,false_positive_candidates=2)) for _ in range(3)]
        result=bootstrap_fixed(cases,draws=100,seed=4)
        self.assertEqual(result['sensitivity_95_percentile_interval'],[1.,1.])
        self.assertEqual(result['mean_fp_per_scan_95_percentile_interval'],[2.,2.])
        self.assertEqual(result['nonzero_reference_draws'],100)


if __name__=='__main__':
    unittest.main()
