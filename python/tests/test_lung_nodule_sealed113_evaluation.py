import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_nodule_sealed113_evaluation import test_records, load_test_volume, checked_summary


class Sealed113Tests(unittest.TestCase):
    def manifest(self):
        rows = [dict(series_uid=str(i), role='sealed_test', raw_sha256=str(i), nodules=[{}]) for i in range(113)]
        for r in rows[:62]:
            r['nodules'].append({})
        return dict(cases=rows)

    def test_requires_complete_cohort_and_no_fit_calibration_ranker_overlap(self):
        m = self.manifest()
        self.assertEqual(len(test_records(m, [], [], [])), 113)
        for args in ((['0'], [], []), ([], ['0'], []), ([], [], ['0'])):
            with self.assertRaises(ValueError):
                test_records(m, *args)
        with self.assertRaises(ValueError):
            test_records(dict(cases=m['cases'][:-1]), [], [], [])
        bad = copy.deepcopy(m)
        bad['cases'].append(dict(series_uid='fit', role='train', raw_sha256='0', nodules=[]))
        with self.assertRaises(ValueError):
            test_records(bad, ['fit'], [], [])

    def test_reader_refuses_non_protocol_or_non_test_uid_before_io(self):
        for plan, uid in ((dict(version='wrong', test_uids=['a']), 'a'),
                          (dict(version='frozen_sealed113_20261022_v1', test_uids=['a']), 'b')):
            with self.assertRaises(ValueError):
                load_test_volume({}, uid, plan)

    def test_fixed_threshold_counts_negatives_duplicates_excluded_and_boundary(self):
        rows = [dict(uid='p', scores=np.array([.9, .8, .9, .7, .6]),
                     world=np.array([[0,0,0], [.1,0,0], [10,0,0], [2,0,0], [30,0,0]]),
                     record=dict(nodules=[dict(center_lps_xyz_mm=[0,0,0], diameter_mm=4.)]),
                     excluded=[dict(center_lps_xyz_mm=[10,0,0], diameter_mm=4.)]),
                dict(uid='negative', scores=np.array([.8]), world=np.array([[9,9,9]]),
                     record=dict(nodules=[]), excluded=[])]
        r = checked_summary(rows, .7)
        self.assertEqual((r['detected'], r['reference_nodules'], r['false_positive_candidates']), (1,1,2))
        self.assertEqual(r['actual_fp_per_scan'], 1.)
        self.assertEqual(r['threshold'], .7)
        self.assertEqual(checked_summary(rows, None)['false_positive_candidates'], 0)


if __name__ == '__main__':
    unittest.main()
