"""Guard cohort isolation and expanded patch-label semantics without real CT IO."""
import unittest
import numpy as np
from lung_nodule_cnn import DetectorConfig, relative_labels
from lung_nodule_expanded_training import fit_records, rng_for, requests_for, label_group


class ExpandedTrainingTests(unittest.TestCase):
    def test_fit_only_and_duplicate_rejection(self):
        manifest = {'cases':[dict(series_uid='fit', role='train'), dict(series_uid='sealed', role='sealed_test'),
                             dict(series_uid='cal', role='calibration')]}
        self.assertEqual([r['series_uid'] for r in fit_records(manifest)], ['fit'])
        manifest['cases'].append(dict(series_uid='fit', role='train'))
        with self.assertRaises(ValueError):
            fit_records(manifest)

    def test_sampling_reproducible_and_reference_label_consistent(self):
        cfg = DetectorConfig()
        record = dict(nodules=[dict(center_index_zyx=[10., 100., 80.], diameter_mm=5.,
                    center_lps_xyz_mm=[64., 80., 12.5])])
        spacing = np.array([1.25, .8, .8])
        jobs = [(12.5, 0, 0), (15., 64, 64)]
        a = requests_for(record, jobs, spacing, cfg, rng_for(11, 'fit'))
        b = requests_for(record, jobs, spacing, cfg, rng_for(11, 'fit'))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 40)
        geometry = dict(origin_lps_mm=[0., 0., 0.], basis_lps_zyx=[[0., 0., .8], [0., .8, 0.], [1.25, 0., 0.]],
                        spacing_zyx_mm=spacing.tolist())
        for job in a[:16]:
            pos, ignored = relative_labels(record['nodules'], [], geometry, job, cfg)
            self.assertEqual(len(pos), 1)
            label = dict(job=job, positive=pos, ignored=ignored)
            self.assertEqual(label_group(label, [dict(record['nodules'][0], edge=True)], spacing, cfg), 'under6_edge')

    def test_empty_roi_not_discarded(self):
        cfg = DetectorConfig()
        jobs = requests_for(dict(nodules=[]), [], np.ones(3), cfg, rng_for(11, 'empty'))
        self.assertEqual(jobs, [(0., 0, 0)])
        self.assertEqual(label_group(dict(positive=[], job=jobs[0]), [], np.ones(3), cfg), 'background')

    def test_group_requires_real_matching_reference(self):
        label = dict(job=(0., 0, 0), positive=[[20., 20., 0., 5.]])
        with self.assertRaises(ValueError):
            label_group(label, [], np.ones(3), DetectorConfig())


if __name__ == '__main__':
    unittest.main()
