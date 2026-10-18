"""Engineering checks for the small learned-model pipeline, not medical validation."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_unet import (DEVELOPMENT_SEEN, SliceDataset, make_model, model_summary,
                       EarlyStopping, aggregate_slice_quality, build_parser, mapped_npy,
                       patient_split, read_manifest, recover_best_checkpoint, resize_stack,
                       restore_training_controls, save_training_curves, segmentation_loss,
                       slice_quality_metrics, file_sha256, load_comparison_prediction, small_lung_sampling,
                       context_batch, audit_context_geometry, _case_prediction)
from lungseg.metrics import overlap_metrics


class DatasetTests(unittest.TestCase):
    def test_context_batch_order_edges_single_slice_and_owned_copy(self):
        images = np.stack([np.full((4, 4), i) for i in range(4)])
        result = context_batch(images, [0, 2, 3], 3)
        np.testing.assert_array_equal(result[:, :, 0, 0], [[0, 0, 1], [1, 2, 3], [2, 3, 3]])
        self.assertEqual(result.shape, (3, 3, 4, 4))
        self.assertEqual(result.dtype, np.float32)
        result[0] = 99
        self.assertEqual(int(images[0].sum()), 0)
        np.testing.assert_array_equal(context_batch(images, [2], 1)[0, 0], images[2])
        np.testing.assert_array_equal(context_batch(images[:1], [0], 3), 0)
        for channels in (0, 2, 5):
            with self.assertRaises(ValueError):
                context_batch(images, [0], channels)
        with self.assertRaises(IndexError):
            context_batch(images, [-1], 3)
        with self.assertRaises(IndexError):
            context_batch(images, [4], 3)
        with self.assertRaises(ValueError):
            context_batch(images, [.5], 3)

    def test_context_geometry_checks_physical_order_identity_and_spacing(self):
        from types import SimpleNamespace
        headers = [SimpleNamespace(PatientID='patient', SeriesInstanceUID='series', FrameOfReferenceUID='frame',
                   SOPInstanceUID=str(i), Rows=32, Columns=32, ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
                   ImagePositionPatient=[0, 0, i*2.], PixelSpacing=[1., 1.]) for i in range(3)]
        row = {'case': 'synthetic', 'patient_key': hashlib.sha256(b'patient').hexdigest(),
               'frame_uid': 'frame', 'slices': 3, 'native_shape': [3, 32, 32],
               'spacing_zyx_mm': [2., 1., 1.], 'source_ct_files_sorted': ['a', 'b', 'c']}
        with patch('lung_unet.pydicom.dcmread', side_effect=headers) as read:
            result = audit_context_geometry([row])
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['cases'][0]['slice_spacing_mm'], 2.)
            self.assertTrue(all(call.kwargs['stop_before_pixels'] for call in read.call_args_list))
        with patch('lung_unet.pydicom.dcmread', side_effect=headers[::-1]):
            with self.assertRaisesRegex(ValueError, 'unordered'):
                audit_context_geometry([row])
        with patch('lung_unet.pydicom.dcmread', side_effect=headers):
            with self.assertRaisesRegex(ValueError, 'identity'):
                audit_context_geometry([dict(row, patient_key='another')])
        with patch('lung_unet.pydicom.dcmread', side_effect=headers):
            with self.assertRaisesRegex(ValueError, 'nonuniform'):
                audit_context_geometry([dict(row, spacing_zyx_mm=[3., 1., 1.])])

    def test_small_lung_sampling_uses_training_only_and_does_not_change_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self.make_resolution_cache(root)
            record = manifest['cases'][0]
            folder = root / record['cache_folder']
            labels = np.zeros((2, 32, 32), dtype=np.uint8)
            labels[0, 15, 15] = 1
            np.save(folder / 'labels.npy', labels)
            before = file_sha256(folder / 'labels.npy')
            weights, info = small_lung_sampling(root, [record], 4)
            np.testing.assert_array_equal(weights, [4, 1])
            self.assertEqual(info['native_small_lung_slices'], 1)
            self.assertEqual(info['boosted_slices'], 1)
            self.assertEqual(info['expected_boosted_draw_fraction'], .8)
            self.assertEqual(before, file_sha256(folder / 'labels.npy'))
            uniform, metadata = small_lung_sampling(root, [record], 1)
            self.assertIsNone(uniform)
            self.assertEqual(metadata['strategy'], 'uniform_shuffle_no_replacement')
            with self.assertRaisesRegex(ValueError, 'training patients only'):
                small_lung_sampling(root, manifest['cases'], 4)
            with self.assertRaises(ValueError):
                small_lung_sampling(root, [record], float('nan'))

    def test_small_lung_sampling_does_not_boost_lost_low_resolution_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self.make_resolution_cache(root)
            weights, info = small_lung_sampling(root, manifest['cases'][:1], 4)
            np.testing.assert_array_equal(weights, [1, 1])
            self.assertEqual(info['boosted_slices'], 0)
            self.assertEqual(info['small_slices_without_low_resolution_foreground'], 1)

    def test_previous_prediction_comparison_checks_expert_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'LCTSC-Train-S2-001'
            folder.mkdir()
            reference = np.zeros((2, 8, 8), dtype=bool)
            reference[:, 3:5, 3:5] = True
            prediction = reference.copy()
            prediction[0, 0, 0] = True
            np.save(folder / 'prediction_native.npy', prediction.astype(np.uint8) * 255)
            row = {'case': folder.name, 'learned_metrics_native': overlap_metrics(prediction, reference)}
            loaded, scores = load_comparison_prediction(directory, row, reference)
            np.testing.assert_array_equal(loaded, prediction)
            self.assertEqual(scores, row['learned_metrics_native'])
            altered = reference.copy()
            altered[0, 0, 0] = True
            with self.assertRaisesRegex(ValueError, 'expert reference'):
                load_comparison_prediction(directory, row, altered)

    def make_resolution_cache(self, root):
        records = []
        for index, split in enumerate(('train', 'validation')):
            name = f'LCTSC-Train-S2-{index+1:03d}'
            folder = root / name
            folder.mkdir()
            native = np.zeros((2, 64, 64), dtype=np.uint8)
            native[0, 31, 31] = 1  # retain a tiny original reference even if low-res labels lost it
            np.save(folder / 'reference_native.npy', native)
            np.save(folder / 'images.npy', np.zeros((2, 32, 32), dtype=np.float16))
            np.save(folder / 'labels.npy', np.zeros((2, 32, 32), dtype=np.uint8))
            records.append({'case': name, 'split': split, 'patient_key': str(index),
                            'frame_uid': f'frame-{index}', 'cache_folder': name, 'slices': 2,
                            'native_shape': [2, 64, 64], 'spacing_zyx_mm': [3, 1, 1],
                            'source_ct_files_sorted': ['synthetic-a', 'synthetic-b']})
        value = {'status': 'complete', 'failures': [], 'cases': records, 'seed': 20261009,
                 'image_size': 32, 'hu_clip': [-1000, 400], 'fingerprint': 'original',
                 'case_split': {r['case']: r['split'] for r in records}}
        path = root / 'dataset.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        return path, value

    def test_resolution_rebuild_rereads_ct_and_preserves_native_labels_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, original = self.make_resolution_cache(root)
            args = build_parser().parse_args(['rebuild-resolution', '--dataset', str(path),
                                             '--outdir', str(root / 'high_resolution'), '--size', '64'])
            with patch('lung_unet.load_cached_source', return_value=np.full((2, 64, 64), 400., dtype=np.float32)) as source:
                args.func(args)
            self.assertEqual(source.call_count, 2)
            new_root, rebuilt = read_manifest(root / 'high_resolution' / 'dataset.json')
            self.assertEqual(rebuilt['case_split'], original['case_split'])
            self.assertEqual(rebuilt['seed'], original['seed'])
            self.assertEqual(rebuilt['image_size'], 64)
            self.assertNotEqual(rebuilt['fingerprint'], original['fingerprint'])
            for before, after in zip(original['cases'], rebuilt['cases']):
                src, dst = root / before['cache_folder'], new_root / after['cache_folder']
                self.assertEqual(file_sha256(src / 'reference_native.npy'), file_sha256(dst / 'reference_native.npy'))
                self.assertEqual(after['reference_native_sha256'], file_sha256(src / 'reference_native.npy'))
                image = np.load(dst / 'images.npy', allow_pickle=False)
                self.assertEqual(image.shape, (2, 64, 64))
                np.testing.assert_array_equal(image, 1.)  # original CT, not upsampled zero-filled images.npy
                np.testing.assert_array_equal(np.load(dst / 'labels.npy'), np.load(src / 'reference_native.npy'))
                self.assertEqual(int(np.load(src / 'labels.npy').sum()), 0)  # old cache untouched

    def test_resolution_rebuild_failure_blocks_training_without_changing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, original = self.make_resolution_cache(root)
            args = build_parser().parse_args(['rebuild-resolution', '--dataset', str(path),
                                             '--outdir', str(root / 'failed'), '--size', '64'])
            with patch('lung_unet.load_cached_source', side_effect=[np.zeros((2, 64, 64)), OSError('missing CT')]):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    args.func(args)
            report = json.loads((root / 'failed' / 'dataset.json').read_text())
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['case_split'], original['case_split'])
            self.assertEqual(report['failures'][0]['case'], original['cases'][1]['case'])
            with self.assertRaises(ValueError):
                read_manifest(root / 'failed' / 'dataset.json')

    def test_resolution_rebuild_rejects_unchanged_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.make_resolution_cache(root)
            args = build_parser().parse_args(['rebuild-resolution', '--dataset', str(path),
                                             '--outdir', str(root / 'unchanged'), '--size', '32'])
            with self.assertRaises(ValueError):
                args.func(args)
            self.assertFalse((root / 'unchanged').exists())

    def test_slice_quality_separates_small_and_empty_regions(self):
        reference = np.zeros((4, 32, 32), dtype=np.uint8)
        reference[0, :2, :2] = 1  # small reference, half missed
        reference[1] = 1  # 1024 reference pixels, medium group
        prediction = reference.copy()
        prediction[0, 0] = 0
        prediction[2, 0, 0] = 255  # empty reference with one false pixel
        quality = slice_quality_metrics(prediction, reference)
        self.assertEqual(quality['groups']['small']['slices'], 1)
        self.assertAlmostEqual(quality['groups']['small']['mean_slice_dice'], 2/3)
        self.assertEqual(quality['groups']['medium']['mean_slice_dice'], 1)
        self.assertIsNone(quality['groups']['large']['mean_slice_dice'])
        self.assertEqual(quality['empty_reference_slices'], 2)
        self.assertEqual(quality['empty_slices_with_prediction'], 1)
        self.assertEqual(quality['empty_false_positive_pixels'], 1)
        combined = aggregate_slice_quality([quality, quality])
        self.assertEqual(combined['groups']['small']['slices'], 2)
        self.assertAlmostEqual(combined['groups']['small']['mean_slice_dice'], 2/3)
        self.assertEqual(combined['empty_reference_slices'], 4)
        self.assertEqual(combined['empty_false_positive_pixels'], 2)
        perfect_small = np.repeat(reference[:1], 2, axis=0)
        weighted = aggregate_slice_quality([quality, slice_quality_metrics(perfect_small, perfect_small)])
        self.assertEqual(weighted['groups']['small']['slices'], 3)
        self.assertAlmostEqual(weighted['groups']['small']['mean_slice_dice'], 8/9)
        boundaries = np.zeros((4, 128, 128), dtype=bool)
        for index, area in enumerate((1000, 1001, 10000, 10001)):
            boundaries[index].flat[:area] = True
        grouped = slice_quality_metrics(boundaries, boundaries)['groups']
        self.assertEqual([grouped[name]['slices'] for name in ('small', 'medium', 'large')], [1, 2, 1])
        with self.assertRaises(ValueError):
            slice_quality_metrics(prediction[0], reference[0])

    def test_early_stopping_min_delta_state_and_disabled_mode(self):
        stopper = EarlyStopping(patience=2, min_delta=.001)
        self.assertFalse(stopper.update(.9, 1))
        self.assertFalse(stopper.update(.9004, 2))
        restored = EarlyStopping(patience=2, min_delta=.001)
        restored.load_state_dict(stopper.state_dict())
        self.assertTrue(restored.update(.9009, 3))
        self.assertEqual(restored.best_epoch, 1)
        self.assertFalse(stopper.update(.9011, 3))
        self.assertEqual(stopper.bad_epochs, 0)
        disabled = EarlyStopping(patience=0)
        self.assertTrue(all(not disabled.update(.8, epoch) for epoch in range(12)))
        with self.assertRaises(ValueError):
            stopper.update(float('nan'), 4)
        with self.assertRaises(ValueError):
            EarlyStopping(patience=-1)
        with self.assertRaises(ValueError):
            EarlyStopping(min_delta=float('inf'))

    def test_training_curves_accept_legacy_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'curves.png'
            save_training_curves([{'epoch': 1, 'train_loss': 1.2,
                                   'mean_validation_case_dice_native': .8}], path)
            self.assertGreater(path.stat().st_size, 1000)

    def test_mapping_closes_on_success_and_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'images.npy'
            np.save(path, np.ones((2, 8, 8), dtype=np.float16))
            with mapped_npy(path) as image:
                mapping = image._mmap
                self.assertEqual(float(image.sum()), 128)
            self.assertTrue(mapping.closed)
            with self.assertRaisesRegex(ValueError, 'consumer failed'):
                with mapped_npy(path) as image:
                    mapping = image._mmap
                    raise ValueError('consumer failed')
            self.assertTrue(mapping.closed)
            path.rename(Path(directory) / 'closed.npy')  # fails on Windows if still mapped

    def test_partial_dataset_open_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory) / 'images.npy', np.zeros((2, 8, 8), dtype=np.float16))
            mappings, load = [], np.load

            def tracked_load(*args, **kwargs):
                array = load(*args, **kwargs)
                mappings.append(array._mmap)
                return array

            with SliceDataset(directory, [{'slices': 2, 'cache_folder': '.'}]) as dataset:
                with patch('lung_unet.np.load', side_effect=tracked_load):
                    with self.assertRaises(FileNotFoundError):
                        dataset[0]
                self.assertEqual(dataset.arrays, {})
                self.assertEqual(len(mappings), 1)
                self.assertTrue(mappings[0].closed)
            dataset.close()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                dataset[0]

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
    def test_context_dataset_never_crosses_cases_and_augments_all_channels_together(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for case in range(2):
                folder = root / str(case)
                folder.mkdir()
                images = np.stack([np.arange(32*32).reshape(32, 32) + 10000*case + 2000*z for z in range(3)])
                labels = np.zeros((3, 32, 32), dtype=np.uint8)
                labels[1, 4, 5] = 1
                np.save(folder / 'images.npy', images.astype(np.float32))
                np.save(folder / 'labels.npy', labels)
                records.append({'slices': 3, 'cache_folder': str(case)})
            with SliceDataset(root, records, context_slices=3) as dataset:
                first, first_label = dataset[0]
                last, _ = dataset[2]
                other, _ = dataset[3]
                np.testing.assert_array_equal(first[:, 0, 0], [0, 0, 2000])
                np.testing.assert_array_equal(last[:, 0, 0], [2000, 4000, 4000])
                np.testing.assert_array_equal(other[:, 0, 0], [10000, 10000, 12000])
                self.assertEqual(float(first_label.sum()), 0)  # no neighboring target leakage
                middle, label = dataset[1]
            self.assertEqual(float(other[0, 0, 0]), 10000)  # independent of closed maps
            with SliceDataset(root, records, augment=True, context_slices=3) as dataset:
                with patch('lung_unet.random.random', return_value=0.):
                    flipped, flipped_label = dataset[1]
                torch.testing.assert_close(flipped, middle.flip(-1))
                torch.testing.assert_close(flipped_label, label.flip(-1))
            self.assertEqual(float(label.sum()), 1)

    def test_context_model_summary_forward_backward_and_batch_prediction(self):
        import torch
        torch.set_num_threads(2)
        model = make_model(8, 3)
        image = torch.rand(2, 3, 32, 32)
        target = (image[:, 1:2] > .5).float()
        loss = segmentation_loss(model(image), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        summary, old = model_summary(model, 32), model_summary(make_model(8), 32)
        self.assertEqual(summary['input_shape'], [1, 3, 32, 32])
        self.assertEqual(summary['parameters'] - old['parameters'], 144)
        self.assertEqual(summary['conv_MACs_per_slice'] - old['conv_MACs_per_slice'], 144*32*32)
        with self.assertRaises(ValueError):
            make_model(8, 2)
        with tempfile.TemporaryDirectory() as directory:
            images = np.random.default_rng(42).random((5, 32, 32)).astype(np.float16)
            np.save(Path(directory) / 'images.npy', images)
            record = {'cache_folder': '.', 'native_shape': [5, 32, 32]}
            predicted, _ = _case_prediction(model, directory, record, torch.device('cpu'), 2)
            with torch.inference_mode():
                expected = torch.sigmoid(model(torch.from_numpy(context_batch(images, np.arange(5), 3))))[:, 0] >= .5
            np.testing.assert_array_equal(predicted, expected.numpy())

    def test_context_train_resume_and_evaluation_restore_three_channel_mode(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            label = np.zeros((2, 32, 32), dtype=np.uint8)
            label[:, 8:24, 8:24] = 1
            for index, split in enumerate(('train', 'validation')):
                folder = root / str(index)
                folder.mkdir()
                for name, array in (('images', label.astype(np.float16)), ('labels', label), ('reference_native', label)):
                    np.save(folder / (name + '.npy'), array)
                records.append({'case': f'LCTSC-Train-S2-{index+1:03d}', 'patient_key': str(index),
                                'split': split, 'slices': 2, 'cache_folder': str(index),
                                'native_shape': [2, 32, 32], 'spacing_zyx_mm': [2., 1., 1.]})
            manifest = {'status': 'complete', 'failures': [], 'cases': records, 'fingerprint': 'synthetic',
                        'image_size': 32, 'hu_clip': [-1000, 400]}
            dataset_path = root / 'dataset.json'
            dataset_path.write_text(json.dumps(manifest), encoding='utf-8')
            common = ['train', '--dataset', str(dataset_path), '--device', 'cpu', '--base-channels', '2',
                      '--batch-size', '2', '--threads', '2', '--context-slices', '3']
            first = build_parser().parse_args(common + ['--outdir', str(root / 'first'), '--epochs', '1'])
            with patch('lung_unet.audit_context_geometry', side_effect=ValueError('invalid geometry')):
                with self.assertRaisesRegex(ValueError, 'invalid geometry'):
                    first.func(first)
            self.assertFalse((root / 'first').exists())
            quality = slice_quality_metrics(label, label)
            rows = [{'case': records[1]['case'], 'metrics': {'dice': .9}, 'slice_quality': quality}]
            with patch('lung_unet.audit_context_geometry', return_value={'status': 'synthetic_test'}), \
                    patch('lung_unet.validate_model', return_value=(rows, .9)):
                first.func(first)
                continued = build_parser().parse_args(common + ['--outdir', str(root / 'continued'), '--epochs', '2',
                          '--resume', str(root / 'first' / 'last.pt')])
                continued.func(continued)
            checkpoint = torch.load(root / 'continued' / 'best.pt', weights_only=True)
            self.assertEqual(checkpoint['context_slices'], 3)
            self.assertEqual(checkpoint['model']['enc1.0.weight'].shape[1], 3)
            evaluation = build_parser().parse_args(['evaluate', '--dataset', str(dataset_path), '--device', 'cpu',
                '--checkpoint', str(root / 'continued' / 'best.pt'), '--outdir', str(root / 'evaluation')])
            with patch('lung_unet.audit_context_geometry', return_value={'status': 'synthetic_test'}), \
                    patch('lung_unet.read_native_hu', return_value=np.zeros_like(label)), \
                    patch('lung_unet.segment_volume', return_value=(label != 0, [], 0.)):
                evaluation.func(evaluation)
            report = json.loads((root / 'evaluation' / 'evaluation.json').read_text())
            self.assertEqual(report['context_slices'], 3)  # no manual evaluate input-mode override
            self.assertTrue((root / 'evaluation' / records[1]['case'] / 'prediction_native.npy').is_file())

    def test_legacy_control_initialization_preserves_optimizer_lr(self):
        import torch
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
        stopper = EarlyStopping(8, .001)
        checkpoint = {'best_dice': .9, 'epoch': 2, 'history': [
            {'epoch': 1, 'mean_validation_case_dice_native': .9},
            {'epoch': 2, 'mean_validation_case_dice_native': .8}]}
        note = restore_training_controls(scheduler, stopper, checkpoint, {})
        self.assertIn('Legacy', note)
        self.assertEqual(optimizer.param_groups[0]['lr'], .001)
        self.assertEqual(scheduler.best, .9)
        self.assertEqual(stopper.bad_epochs, 1)
        with self.assertRaisesRegex(ValueError, 'sampling controls differ'):
            restore_training_controls(scheduler, stopper, checkpoint, {'small_lung_weight': 4.})
        config = {'early_stopping_patience': 8, 'small_lung_weight': 1.}
        legacy_config = {'early_stopping_patience': 8}
        saved = {'training_controls': {'config': legacy_config, 'scheduler': scheduler.state_dict(),
                                      'early_stopping': stopper.state_dict()}}
        self.assertIn('Restored', restore_training_controls(scheduler, stopper, saved, config))
        with self.assertRaisesRegex(ValueError, 'controls differ'):
            restore_training_controls(scheduler, stopper, saved, dict(config, small_lung_weight=4.))
        with self.assertRaisesRegex(ValueError, 'controls differ'):
            restore_training_controls(scheduler, stopper, {'training_controls': {'config': {'other': 1}}}, {})

    def test_resume_preserves_best_and_stops_without_resetting_state(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for name, split in (('001', 'train'), ('002', 'validation')):
                folder = root / name
                folder.mkdir()
                label = np.zeros((2, 32, 32), dtype=np.uint8)
                label[:, 8:24, 8:24] = 1
                np.save(folder / 'images.npy', label.astype(np.float16))
                np.save(folder / 'labels.npy', label)
                np.save(folder / 'reference_native.npy', label)
                records.append({'case': 'LCTSC-Train-S2-' + name, 'patient_key': name,
                                'split': split, 'slices': 2, 'cache_folder': name,
                                'native_shape': [2, 32, 32]})
            manifest = {'status': 'complete', 'failures': [], 'cases': records,
                        'fingerprint': 'synthetic', 'image_size': 32, 'hu_clip': [-1000, 400]}
            dataset_path = root / 'dataset.json'
            dataset_path.write_text(json.dumps(manifest), encoding='utf-8')
            common = ['train', '--dataset', str(dataset_path), '--device', 'cpu',
                      '--base-channels', '2', '--batch-size', '2', '--threads', '2',
                      '--early-stopping-patience', '3', '--lr-patience', '0']
            quality = slice_quality_metrics(label, label)

            def validation_result(score):
                return ([{'case': records[1]['case'], 'metrics': {'dice': score},
                          'slice_quality': quality}], score)

            first = build_parser().parse_args(common + ['--outdir', str(root / 'first'), '--epochs', '2'])
            with patch('lung_unet.validate_model', side_effect=[validation_result(.9), validation_result(.8)]):
                first.func(first)
            first_last = torch.load(root / 'first' / 'last.pt', weights_only=True)
            first_best = torch.load(root / 'first' / 'best.pt', weights_only=True)
            self.assertEqual(first_best['epoch'], 1)
            self.assertEqual(first_last['optimizer']['param_groups'][0]['lr'], .0005)
            wrong_context = build_parser().parse_args(common + ['--outdir', str(root / 'wrong_context'),
                '--epochs', '3', '--resume', str(root / 'first' / 'last.pt'), '--context-slices', '3'])
            with self.assertRaisesRegex(ValueError, 'context slices differ'):
                wrong_context.func(wrong_context)
            self.assertFalse((root / 'wrong_context').exists())
            second = build_parser().parse_args(common + ['--outdir', str(root / 'continued'), '--epochs', '30',
                '--resume', str(root / 'first' / 'last.pt')])
            with patch('lung_unet.validate_model', side_effect=[validation_result(.7), validation_result(.6)]):
                second.func(second)
            last = torch.load(root / 'continued' / 'last.pt', weights_only=True)
            best = torch.load(root / 'continued' / 'best.pt', weights_only=True)
            self.assertEqual(last['epoch'], 4)  # stops before 30, not 3 extra bad epochs after resume
            self.assertEqual(last['training_controls']['early_stopping']['bad_epochs'], 3)
            self.assertEqual(last['history'][2]['learning_rate'], .0005)
            self.assertEqual(best['epoch'], 1)  # no new improvement; historical best still available
            for key in best['model']:
                torch.testing.assert_close(best['model'][key], first_best['model'][key], rtol=0, atol=0)
            summary = json.loads((root / 'continued' / 'training_summary.json').read_text())
            self.assertEqual(summary['stop_reason'], 'early_stopping')
            self.assertEqual(summary['best_epoch'], 1)
            self.assertTrue((root / 'continued' / 'training_curves.png').is_file())
            resumed_stopped = build_parser().parse_args(common + ['--outdir', str(root / 'should_not_exist'),
                '--epochs', '30', '--resume', str(root / 'continued' / 'last.pt')])
            with self.assertRaisesRegex(ValueError, 'exhausted'):
                resumed_stopped.func(resumed_stopped)
            self.assertFalse((root / 'should_not_exist').exists())
            legacy = {key: value for key, value in first_last.items() if key != 'training_controls'}
            recovered = recover_best_checkpoint(root / 'first' / 'last.pt', legacy)
            self.assertEqual(recovered['epoch'], 1)
            with self.assertRaisesRegex(ValueError, 'sibling best.pt'):
                recover_best_checkpoint(root / 'missing' / 'last.pt', legacy)

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
        larger = model_summary(make_model(8), 128)
        self.assertEqual(larger['parameters'], summary['parameters'])
        self.assertEqual(larger['conv_MACs_per_slice'], summary['conv_MACs_per_slice'] * 4)
        with tempfile.TemporaryDirectory() as directory:
            records = []
            for i in range(2):
                folder = Path(directory) / str(i)
                folder.mkdir()
                np.save(folder / 'images.npy', np.full((2, 32, 32), i, dtype=np.float16))
                np.save(folder / 'labels.npy', np.full((2, 32, 32), i, dtype=np.uint8))
                records.append({'slices': 2, 'cache_folder': str(i)})
            with SliceDataset(directory, records) as dataset:
                self.assertEqual(len(dataset), 4)
                self.assertEqual(float(dataset[1][0].sum()), 0)
                copied_image = dataset[2][0]
                self.assertEqual(float(copied_image.sum()), 1024)
                mappings = [array._mmap for arrays in dataset.arrays.values() for array in arrays]
                with self.assertRaises(IndexError):
                    dataset[4]
            self.assertTrue(all(mapping.closed for mapping in mappings))
            self.assertEqual(float(copied_image.sum()), 1024)
            dataset.close()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                dataset[0]

            with self.assertRaisesRegex(ValueError, 'consumer failed'):
                with SliceDataset(directory, records) as dataset:
                    dataset[0]
                    mappings = [array._mmap for arrays in dataset.arrays.values() for array in arrays]
                    raise ValueError('consumer failed')
            self.assertTrue(all(mapping.closed for mapping in mappings))


if __name__ == '__main__':
    unittest.main()
