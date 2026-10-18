"""Soft single-slice context, unchanged initialization and real paired CPU smoke test."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_layer_context import (LEVELS, LevelDataset, coarse_tags, initialize,
    make_layer_model, paired, parser, predict_slice, selection_key, training_class_weights)
from lung_unet import file_sha256, make_model, model_summary, torch_module
from test_lung_outer_contour import fixture


class CoarseTests(unittest.TestCase):
    def test_physical_extent_not_scan_length_and_reversed_order(self):
        mask = np.zeros((11, 8, 8), np.uint8)
        mask[2:9, 2:6, 2:6] = 255
        before = mask.copy()
        z = np.arange(11) * 3. + 180
        tags = coarse_tags(mask, z)
        np.testing.assert_array_equal(tags, [0, 0, 1, 1, 2, 2, 3, 3, 3, 0, 0])
        np.testing.assert_array_equal(coarse_tags(mask[::-1], z[::-1]), tags[::-1])
        np.testing.assert_array_equal(coarse_tags(mask, z + 300), tags)
        np.testing.assert_array_equal(mask, before)
        np.testing.assert_array_equal(coarse_tags(np.zeros_like(mask), z), np.zeros(11))
        mask[:] = 0
        mask[5] = 1
        self.assertEqual(coarse_tags(mask, z)[5], 2)
        with self.assertRaises(ValueError):
            coarse_tags(mask, np.zeros(11))

    def test_training_weights_never_include_validation(self):
        rows = [dict(case='train', slices=4)]
        counts, weights = training_class_weights(rows, {'train': [0, 1, 2, 3], 'val': [0] * 100})
        self.assertEqual(counts, [1] * 4)
        np.testing.assert_array_equal(weights, np.ones(4))
        changed = {'train': [0, 1, 2, 3], 'val': [3] * 999}
        np.testing.assert_array_equal(training_class_weights(rows, changed)[1], weights)

    def test_no_hard_dice_floor_in_contour_selection(self):
        base = dict(mean_case_dice=.967, outer=dict(main_missing_foreground=0,
                    mean_case_bad5_fraction=.04, worst5_percent_bad5_mean=.3))
        lower_dice = copy.deepcopy(base)
        lower_dice['mean_case_dice'] = .95
        lower_dice['outer']['mean_case_bad5_fraction'] = .03
        self.assertLess(selection_key(lower_dice), selection_key(base))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch_module().set_num_threads(1)

    def test_initial_segmentation_exact_and_resource_count(self):
        torch = torch_module()
        torch.manual_seed(1)
        original = make_model(8, 1).eval()
        model = make_layer_model(8, True).eval()
        initialize(model, original.state_dict())
        image = torch.rand(2, 1, 32, 32)
        with torch.no_grad():
            expected = original(image)
            actual, level = model(image, return_level=True)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(tuple(level.shape), (2, 4))
        self.assertEqual(model_summary(model, 256)['parameters'], 123221)
        self.assertEqual(model_summary(model, 256)['conv_MACs_per_slice'], 665845760)
        with self.assertRaises(ValueError):
            initialize(model, {'head.bias': original.state_dict()['head.bias']})

    def test_unknown_single_slice_no_gt_no_forced_empty_finite_gradients(self):
        torch = torch_module()
        model = make_layer_model(2, True)
        result = predict_slice(model, np.full((40, 48), -750.), torch.device('cpu'), 32)
        self.assertEqual(result['mask'].shape, (40, 48))
        self.assertAlmostEqual(float(result['level_probability'].sum()), 1., places=6)
        self.assertTrue(np.isfinite(result['level_probability']).all())
        model.train()
        image = torch.rand(2, 1, 32, 32)
        seg, level = model(image, return_level=True)
        loss = seg.square().mean() + level.square().mean()
        loss.backward()
        self.assertGreater(float(model.context_film.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            model.level_head.weight.zero_()
            model.level_head.bias[:] = torch.tensor([100., 0., 0., 0.])
            model.head.weight.zero_()
            model.head.bias.fill_(10.)
        result = predict_slice(model, np.full((40, 48), -750.), torch.device('cpu'), 32)
        self.assertEqual(int(result['level_probability'].argmax()), 0)
        self.assertTrue(result['mask'].all())  # 'none' does not override the segmentation.

    def test_dataset_same_flips_no_validation_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, manifest = fixture(Path(tmp))
            rows = [manifest['cases'][0]]
            tags = {rows[0]['case']: [1, 3]}
            with LevelDataset(path.parent, rows, tags) as dataset:
                with patch('lung_unet.random.random', return_value=1.):
                    original = dataset[0]
                with patch('lung_unet.random.random', return_value=0.):
                    flipped = dataset[0]
                np.testing.assert_array_equal(flipped[0].numpy(), original[0].numpy()[:, :, ::-1])
                np.testing.assert_array_equal(flipped[1].numpy(), original[1].numpy()[:, :, ::-1])
                self.assertEqual(int(flipped[2]), int(original[2]))
                self.assertEqual(int(dataset[1][2]), 3)
            with self.assertRaises(ValueError):
                LevelDataset(path.parent, [manifest['cases'][1]], tags)

    def test_real_one_epoch_paired_cpu_original_protection_compressed_masks(self):
        torch = torch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = fixture(root)
            for row in manifest['cases']:
                row['source_ct_files_sorted'] = []
            dataset.write_text(json.dumps(manifest), encoding='utf-8')
            checkpoint = root / 'original.pt'
            torch.save(dict(model=make_model(2).state_dict(), base_channels=2, image_size=64,
                context_slices=1, hu_clip=[-1000, 400], threshold=.5, dataset_fingerprint='original'), checkpoint)
            before = file_sha256(checkpoint), file_sha256(dataset)
            tags = {r['case']: [1, 3] for r in manifest['cases']}
            args = parser().parse_args(['--dataset', str(dataset), '--checkpoint', str(checkpoint),
                '--outdir', str(root / 'screen'), '--epochs', '1', '--batch-size', '2', '--device', 'cpu', '--threads', '1'])
            with patch('lung_layer_context.prepare_tags', return_value=(tags, {'status': 'fixture'})):
                paired(args)
            self.assertEqual(before, (file_sha256(checkpoint), file_sha256(dataset)))
            report = json.loads((root / 'screen/evaluation.json').read_text())
            self.assertEqual(report['status'], 'complete_paired_layer_context_screen')
            self.assertEqual(set(report['arms']), {'auxiliary_only', 'soft_context'})
            self.assertTrue(report['protected_inputs_unchanged'])
            for arm in report['arms']:
                for kind in ['best_contour', 'best_dice']:
                    selected = report['arms'][arm]['selected'][kind]
                    self.assertEqual(selected['epoch'], 1)
                    confusion = selected['levels']['summary']['confusion_rows_reference_columns_prediction']
                    self.assertEqual(np.sum(confusion), 2)
                    case = manifest['cases'][1]['case']
                    with np.load(root / 'screen' / arm / kind / case / 'prediction_native.npz') as archive:
                        self.assertEqual(archive['mask'].shape, (2, 64, 64))
                        self.assertEqual(archive['mask'].dtype, np.bool_)
            self.assertFalse(list((root / 'screen').rglob('*.npy')))


if __name__ == '__main__':
    unittest.main()
