import unittest
import numpy as np
from lung_nodule_expanded_evaluation import calibration_records, validate_cached


class ExpandedEvaluationTests(unittest.TestCase):
    def test_only_calibration_and_overlap_rejected(self):
        m = dict(cases=[dict(series_uid='a', role='calibration'), dict(series_uid='b', role='train'),
                        dict(series_uid='c', role='sealed_test')])
        self.assertEqual([r['series_uid'] for r in calibration_records(m)], ['a'])
        m['cases'].append(dict(series_uid='a', role='train'))
        with self.assertRaises(ValueError):
            calibration_records(m)

    def test_changed_model_or_mask_rejected(self):
        r = dict(series_uid='a', raw_sha256='pixels', shape_zyx=[3, 4, 5])
        p = dict(series_uid='a', input_raw_sha256='pixels', source_shape_zyx=[3, 4, 5],
                 model_sha256='model', lung_mask_sha256='body', features=[], candidates=[])
        self.assertEqual(validate_cached(p, r, 'model', 'body').shape, (0, 38))
        with self.assertRaises(ValueError):
            validate_cached(p, r, 'other', 'body')
        with self.assertRaises(ValueError):
            validate_cached(p, r, 'model', 'otherbody')

    def test_feature_candidate_misalignment_rejected(self):
        r = dict(series_uid='a', raw_sha256='pixels', shape_zyx=[3, 4, 5])
        p = dict(series_uid='a', input_raw_sha256='pixels', source_shape_zyx=[3, 4, 5],
                 model_sha256='model', lung_mask_sha256='body', features=np.zeros((1, 38)).tolist(), candidates=[])
        with self.assertRaises(ValueError):
            validate_cached(p, r, 'model')


if __name__ == '__main__':
    unittest.main()
