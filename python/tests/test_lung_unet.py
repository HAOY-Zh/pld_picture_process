"""Engineering checks for the small learned-model pipeline, not medical validation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_unet import (DEVELOPMENT_SEEN, SliceDataset, make_model, model_summary,
                       patient_split, read_manifest, resize_stack, segmentation_loss)


class DatasetTests(unittest.TestCase):
    def test_patient_split_no_test_or_previously_seen_validation(self):
        names = [f'LCTSC-Train-S{s}-{i:03d}' for s in (2, 3) for i in range(1, 13)]
        split = patient_split(names)
        self.assertEqual(sum(v == 'train' for v in split.values()), 18)
        self.assertEqual(sum(v == 'validation' for v in split.values()), 6)
        for site in ('S2', 'S3'):
            self.assertEqual(sum(site in k and v == 'validation' for k, v in split.items()), 3)
        self.assertTrue(all(split[n] == 'train' for n in DEVELOPMENT_SEEN))
        self.assertEqual(split, patient_split(names[::-1]))
        with self.assertRaises(ValueError):
            patient_split(names + ['LCTSC-Test-S2-101'])

    def test_resize_binary_labels_and_constant_hu(self):
        image = np.full((2, 64, 64), -800., dtype=np.float32)
        np.testing.assert_allclose(resize_stack(image, 32, 1), -800)
        label = np.zeros((2, 64, 64), dtype=np.uint8)
        label[:, 16:48, 16:48] = 1
        small = resize_stack(label, 32, 0)
        self.assertEqual(set(np.unique(small)), {0, 1})
        self.assertEqual(int(small[0].sum()), 256)

    def test_manifest_rejects_partial_cache_and_patient_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.json'
            value = {'status': 'incomplete', 'cases': [], 'failures': []}
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaises(ValueError):
                read_manifest(path)
            value = {'status': 'complete', 'failures': [], 'cases': [
                {'patient_key': 'same', 'case': 'LCTSC-Train-S2-001', 'split': 'train'},
                {'patient_key': 'same', 'case': 'LCTSC-Train-S2-002', 'split': 'validation'}]}
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate-patient'):
                read_manifest(path)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Run with .venv-lung after installing PyTorch')
class ModelTests(unittest.TestCase):
    def test_forward_backward_and_checkpoint(self):
        import torch
        torch.set_num_threads(2)
        model = make_model(4)
        image = torch.rand(2, 1, 32, 32)
        target = (image > .5).float()
        logits = model(image)
        self.assertEqual(tuple(logits.shape), tuple(target.shape))
        loss = segmentation_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'model': model.state_dict(), 'base_channels': 4}, path)
            state = torch.load(path, weights_only=True)
            restored = make_model(state['base_channels'])
            restored.load_state_dict(state['model'])
            model.eval()
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(image), restored(image))

    def test_summary_and_slice_indexing(self):
        import torch
        torch.set_num_threads(2)
        summary = model_summary(make_model(8), 64)
        self.assertGreater(summary['parameters'], 0)
        self.assertGreater(summary['conv_MACs_per_slice'], 0)
        with tempfile.TemporaryDirectory() as directory:
            records = []
            for i in range(2):
                folder = Path(directory) / str(i)
                folder.mkdir()
                np.save(folder / 'images.npy', np.full((2, 32, 32), i, dtype=np.float16))
                np.save(folder / 'labels.npy', np.full((2, 32, 32), i, dtype=np.uint8))
                records.append({'slices': 2, 'cache_folder': str(i)})
            dataset = SliceDataset(directory, records)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(float(dataset[1][0].sum()), 0)
            self.assertEqual(float(dataset[2][0].sum()), 1024)
            with self.assertRaises(IndexError):
                dataset[4]


if __name__ == '__main__':
    unittest.main()
