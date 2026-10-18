"""Geometric diagnostics tests, not medical segmentation accuracy tests."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_error_audit import (binary, check_training_labels, density_counts,
                              enclosed_background, error_maps, parser)
from lung_unet import resize_stack


class ErrorAuditTests(unittest.TestCase):
    def test_hole_and_open_notch_are_distinct_without_modifying_input(self):
        reference = np.zeros((16, 16), bool)
        reference[2:14, 2:14] = True
        prediction = reference.copy()
        prediction[6:8, 6:8] = False
        prediction[2:7, 10] = False
        prediction[0, 0] = True
        before = prediction.copy()
        stats, maps = error_maps(prediction, reference)
        self.assertEqual(stats['closed_fn_pixels'], 4)
        self.assertEqual(stats['open_fn_pixels'], 5)
        self.assertEqual(stats['fn_pixels'], 9)
        self.assertEqual(stats['closed_fn_components'], 1)
        self.assertEqual(stats['fp_pixels'], 1)
        self.assertEqual(stats['deep_fn_pixels']+stats['edge_fn_pixels'], 9)
        self.assertFalse((maps['closed_fn'] & maps['open_fn']).any())
        np.testing.assert_array_equal(before, prediction)

    def test_real_reference_hole_is_not_false_negative(self):
        reference = np.ones((10, 10), bool)
        reference[4:6, 4:6] = False
        stats, _ = error_maps(reference, reference)
        self.assertEqual(stats['reference_enclosed_background_pixels'], 4)
        self.assertEqual(stats['closed_fn_pixels'], 0)
        self.assertEqual(stats['fn_pixels'], 0)
        self.assertFalse(enclosed_background(np.zeros((4,4), bool))[0].any())
        self.assertFalse(enclosed_background(np.ones((4,4), bool))[0].any())

    def test_spacing_and_padded_distance_on_all_white(self):
        reference = np.ones((8,8), bool)
        prediction = reference.copy()
        prediction[1,3] = False
        a, _ = error_maps(prediction, reference, (.5,.5))
        b, _ = error_maps(prediction, reference, (1.,1.))
        self.assertEqual(a['deep_fn_pixels'], 0)
        self.assertEqual(b['deep_fn_pixels'], 1)
        with self.assertRaises(ValueError):
            error_maps(prediction, reference, (-1,1))
        with self.assertRaises(ValueError):
            enclosed_background(np.zeros((2,2,2)))
        with self.assertRaises(ValueError):
            binary(np.array([.3]))

    def test_density_bins_are_counts_not_disease_labels(self):
        hu = np.array([[-1024,-800,-500,-200,0,300,900]])
        reference = np.ones_like(hu, bool)
        prediction = reference.copy()
        prediction[0,4] = False
        counts, missed = density_counts(hu, reference, prediction)
        np.testing.assert_array_equal(counts, np.ones(7))
        self.assertEqual(missed.sum(),1)
        self.assertEqual(missed[4],1)

    def test_training_resize_preserved_and_diagnosis_has_no_training_switch(self):
        reference = np.zeros((2,64,64), np.uint8)
        reference[:,10:50,12:51] = 1
        labels = resize_stack(reference,32,0)>=.5
        metrics = check_training_labels(reference,labels,32)
        self.assertGreater(metrics['dice'], .9)
        labels[0,0,0] = True
        with self.assertRaisesRegex(ValueError,'nearest-resized'):
            check_training_labels(reference,labels,32)
        options = {flag for action in parser()._actions for flag in action.option_strings}
        self.assertNotIn('--train',options)
        self.assertNotIn('--fill-holes',options)


if __name__ == '__main__':
    unittest.main()
