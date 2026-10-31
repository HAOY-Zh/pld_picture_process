import unittest
from lung_nodule_context_timing import control_ratio


class TimingTests(unittest.TestCase):
    def test_median_handles_outlier_and_equal_architecture_passes(self):
        r = control_ratio({'baseline5':[1., 1., 10.], 'near_far5':[1.01, 1.02, 10.]})
        self.assertTrue(r['acceptable'])
        self.assertAlmostEqual(r['ratio'], 1.02)

    def test_original_fivefold_anomaly_is_rejected(self):
        self.assertFalse(control_ratio({'baseline5':[5., 5.], 'near_far5':[1., 1.]})['acceptable'])
        with self.assertRaises(ValueError):
            control_ratio({'baseline5':[0.], 'near_far5':[1.]})


if __name__ == '__main__':
    unittest.main()
