"""Check mining never turns annotated/ignored findings into hard negatives."""
import unittest
import numpy as np
from lung_nodule_cnn import DetectorConfig
from lung_nodule_cnn_hardneg import background_mask, select_patches


class HardBackgroundTests(unittest.TestCase):
    def test_reference_excluded_and_halo_not_mined(self):
        mask = background_mask(dict(positive=[[60., 60., 0., 8.]],
                                    ignored=[[80., 80., 8.]]), DetectorConfig())
        self.assertFalse(mask[30, 30])
        self.assertFalse(mask[40, 40])
        self.assertFalse(mask[2, 2])
        self.assertTrue(mask[20, 20])

    def test_selection_stable_disjoint_and_preserves_random_diversity(self):
        a, b = select_patches([.1, .9, .9, .3, .2, -np.inf], 2, 3,
                               np.random.default_rng(7))
        self.assertEqual(a, [1, 2])
        self.assertEqual(len(b), 3)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(set(a + b)), 5)

    def test_small_pool_does_not_duplicate_samples(self):
        a, b = select_patches([.4, .1], 32, 16, np.random.default_rng(1))
        self.assertEqual(a, [0, 1])
        self.assertFalse(b)


if __name__ == '__main__':
    unittest.main()
