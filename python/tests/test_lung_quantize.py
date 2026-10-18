"""Synthetic PTQ engineering checks. Not evidence of medical accuracy."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_quantize import (build_parser, calibrate_training, convert_static,
                           export_core, fold_batchnorm, make_quantizable,
                           prepare_static, restore_quantized, run_quantization,
                           save_quantized)
from lung_unet import file_sha256, make_model


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class QuantizationTests(unittest.TestCase):
    def setUp(self):
        import torch
        if 'x86' not in torch.backends.quantized.supported_engines:
            self.skipTest('CPU x86 quantized engine unavailable')
        self.old_engine = torch.backends.quantized.engine
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(731)

    def tearDown(self):
        import torch
        torch.backends.quantized.engine = self.old_engine
        torch.set_num_threads(self.old_threads)

    def test_copy_and_fold_preserve_nontrivial_bn_and_original_weights(self):
        import torch
        original = make_model(2).eval()
        with torch.no_grad():
            for module in original.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.running_mean.copy_(torch.rand_like(module.running_mean)*.2)
                    module.running_var.copy_(torch.rand_like(module.running_var)+.3)
                    module.weight.copy_(torch.rand_like(module.weight)+.5)
                    module.bias.copy_(torch.rand_like(module.bias)*.1)
        state = {k: v.clone() for k, v in original.state_dict().items()}
        copy = make_quantizable(original)
        folded = fold_batchnorm(copy)
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in folded.modules()))
        image = torch.rand(2, 1, 32, 32)
        with torch.inference_mode():
            torch.testing.assert_close(copy(image), original(image), rtol=0, atol=0)
            torch.testing.assert_close(folded(image), original(image), rtol=2e-4, atol=2e-4)
        for key in state:
            torch.testing.assert_close(state[key], original.state_dict()[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            make_quantizable(make_model(2, 3).eval())
        with self.assertRaises(ValueError):
            make_quantizable(make_model(2).train())

    def test_calibration_rejects_validation_test_duplicate_and_bad_shape(self):
        import torch
        original = make_model(2).eval()
        prepared = prepare_static(fold_batchnorm(make_quantizable(original)))
        row = {'case': 'LCTSC-Train-S2-001', 'split': 'train', 'patient_key': 'synthetic-patient',
               'slices': 2, 'cache_folder': '.'}
        for records in ([dict(row, split='validation')], [dict(row, case='LCTSC-Test-S2-101')],
                        [row, row], []):
            with patch('lung_quantize.mapped_npy', side_effect=AssertionError('Must reject before reading inputs')):
                with self.assertRaisesRegex(ValueError, 'training patients only'):
                    calibrate_training(prepared, original, '.', records, torch.device('cpu'))
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory)/'images.npy', np.zeros((2, 32), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, 'shape'):
                calibrate_training(prepared, original, directory, [row], torch.device('cpu'))
            np.save(Path(directory)/'images.npy', np.full((2, 32, 32), np.nan, dtype=np.float16))
            with self.assertRaisesRegex(ValueError, 'finite normalized'):
                calibrate_training(prepared, original, directory, [row], torch.device('cpu'))

    def test_real_integer_core_export_concat_and_bit_identical_restore(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = make_model(2).eval()
            prepared = prepare_static(fold_batchnorm(make_quantizable(original)))
            with torch.inference_mode():
                for _ in range(2):
                    prepared(torch.rand(2, 1, 32, 32))
            quantized = convert_static(prepared)
            export = export_core(quantized, 32, root)
            self.assertEqual(export['integer_convolution_count'], 15)
            self.assertEqual(export['FP32_convolution_count'], 0)
            self.assertEqual(len(export['skip_buffers']), 3)
            self.assertLess(export['weight_and_bias_bytes_excluding_scales'],
                            export['same_folded_core_fp32_weight_and_bias_bytes'])
            self.assertTrue(all(r['input_dtype'] == 'torch.quint8' for r in export['layers']))
            self.assertTrue(all(r['proposed_int32_accumulator_max_abs_bound'] < 2**31 for r in export['layers']))
            observed_cats = []
            actual_cat = torch.ops.quantized.cat
            with patch('torch.ops.quantized.cat', side_effect=lambda *a, **k:
                       (observed_cats.append((a, k)), actual_cat(*a, **k))[1]), torch.inference_mode():
                quantized(torch.rand(1, 1, 32, 32))
            self.assertEqual(len(observed_cats), 3)
            with np.load(root/'int8_core_parameters.npz', allow_pickle=False) as arrays:
                self.assertEqual(arrays['head_weight_int8'].dtype, np.int8)
                self.assertEqual(arrays['head_proposed_bias_int32'].dtype, np.int32)
            checkpoint = {'base_channels': 2, 'image_size': 32, 'dataset_fingerprint': 'synthetic-only'}
            save_quantized(quantized, root/'int8.pt', checkpoint, {'checkpoint_sha256': 'synthetic-hash'})
            restored, saved = restore_quantized(root/'int8.pt')
            image = torch.rand(2, 1, 32, 32)
            with torch.inference_mode():
                torch.testing.assert_close(restored(image), quantized(image), rtol=0, atol=0)
            self.assertEqual(saved['threshold'], .5)

    def test_gpu_calibration_observers_convert_to_cpu_without_reset(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for cross-device conversion regression')
        prepared = prepare_static(fold_batchnorm(make_quantizable(make_model(2).eval()))).cuda()
        with torch.inference_mode():
            prepared(torch.rand(2, 1, 32, 32, device='cuda'))
        before_scale, before_zero = prepared.head.activation_post_process.calculate_qparams()
        quantized = convert_static(prepared)
        self.assertAlmostEqual(quantized.head.scale, float(before_scale), places=8)
        self.assertEqual(quantized.head.zero_point, int(before_zero))
        self.assertEqual(quantized.head.weight().device.type, 'cpu')
        with torch.inference_mode():
            self.assertEqual(quantized(torch.rand(1, 1, 32, 32)).device.type, 'cpu')

    def test_end_to_end_uses_unlabeled_training_only_and_preserves_source(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            rng = np.random.default_rng(52)
            for number, split in enumerate(('train', 'validation')):
                folder = root/str(number)
                folder.mkdir()
                np.save(folder/'images.npy', rng.uniform(0, 1, (2, 32, 32)).astype(np.float16))
                if split == 'validation':
                    reference = np.zeros((2, 32, 32), dtype=np.uint8)
                    reference[:, 8:24, 8:24] = 1
                    np.save(folder/'reference_native.npy', reference)
                # No labels.npy anywhere; no expert data in calibration folder.
                records.append({'case': f'LCTSC-Train-S2-{number+1:03d}', 'split': split,
                                'patient_key': f'synthetic-{number}', 'frame_uid': f'frame-{number}',
                                'slices': 2, 'native_shape': [2, 32, 32],
                                'spacing_zyx_mm': [1, 1, 1], 'cache_folder': str(number)})
            manifest = {'status': 'complete', 'failures': [], 'fingerprint': 'synthetic-only',
                        'image_size': 32, 'hu_clip': [-1000, 400],
                        'input_normalization': '(clip(HU,-1000,400)+1000)/1400', 'cases': records}
            dataset = root/'dataset.json'
            dataset.write_text(json.dumps(manifest), encoding='utf-8')
            source = make_model(2).eval()
            weights = root/'best.pt'
            torch.save({'model': source.state_dict(), 'dataset_fingerprint': manifest['fingerprint'],
                        'base_channels': 2, 'image_size': 32, 'epoch': 1, 'threshold': .5}, weights)
            before = file_sha256(weights)
            args = build_parser().parse_args(['--dataset', str(dataset), '--checkpoint', str(weights),
                '--outdir', str(root/'results'), '--device', 'cpu', '--threads', '2'])
            tf32_before = torch.backends.cudnn.allow_tf32
            with patch('lung_quantize.load_cached_source', return_value=np.zeros((2,32,32))), \
                 patch('lung_quantize.save_learned_preview'):
                run_quantization(args)
            report = json.loads((root/'results'/'evaluation.json').read_text())
            protocol = json.loads((root/'results'/'protocol.json').read_text())
            self.assertEqual(report['status'], 'complete_validation_ptq')
            self.assertEqual(report['validation_patients'], 1)
            self.assertEqual(report['validation_slices'], 2)
            self.assertEqual(report['calibration']['patients'], 1)
            self.assertEqual(report['calibration']['slices'], 2)
            self.assertEqual(report['calibration']['cases'][0]['case'], records[0]['case'])
            self.assertFalse(protocol['test_used'])
            self.assertEqual(file_sha256(weights), before)
            self.assertEqual(torch.backends.cudnn.allow_tf32, tf32_before)
            self.assertTrue((root/'results'/'case_metrics.csv').is_file())
            with self.assertRaisesRegex(ValueError, 'not empty'):
                run_quantization(args)
            self.assertEqual(torch.backends.cudnn.allow_tf32, tf32_before)


if __name__ == '__main__':
    unittest.main()
