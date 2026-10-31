import unittest
from lung_nodule_context_audit import reference_hits


class ContextAuditTests(unittest.TestCase):
    def test_duplicate_hits_are_one_nodule_and_radius_boundary_is_strict(self):
        record = {'nodules':[{'center_lps_xyz_mm':[0, 0, 0], 'diameter_mm':4.},
                             {'center_lps_xyz_mm':[10, 0, 0], 'diameter_mm':4.}]}
        candidates = [{'center_lps_xyz_mm':p} for p in ([0, 0, 0], [1, 0, 0], [12, 0, 0])]
        self.assertEqual(reference_hits(candidates, [.8, .9, .9], record, .8), {0})
        self.assertEqual(reference_hits(candidates, [.8, .9, .9], record, None), set())

    def test_below_threshold_localized_candidate_is_not_a_detection(self):
        record = {'nodules':[{'center_lps_xyz_mm':[0, 0, 0], 'diameter_mm':4.}]}
        candidates = [{'center_lps_xyz_mm':[0, 0, 0]}]
        self.assertEqual(reference_hits(candidates, [.7], record, .8), set())
        self.assertEqual(reference_hits(candidates, [.8], record, .8), {0})


if __name__ == '__main__':
    unittest.main()
