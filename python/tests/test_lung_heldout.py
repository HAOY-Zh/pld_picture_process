"""Synthetic engineering checks, not a medical accuracy claim."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_heldout import (add_independent, build_parser, freeze_protocol, run_heldout,
                          source_identities, summarize)
from lung_unet import make_model, read_manifest, slice_quality_metrics
from lungseg.metrics import overlap_metrics
from test_lung_segmentation import make_case


class HeldoutTests(unittest.TestCase):
    def test_independence_rejects_each_identity_without_partial_mutation(self):
        for collision in ('patients', 'frames', 'series', 'sops'):
            used = {k: {f'old-{k}'} for k in ('patients', 'frames', 'series', 'sops')}
            identity = {k: {f'new-{k}'} for k in used}
            identity[collision] = used[collision].copy()
            before = {k: v.copy() for k, v in used.items()}
            with self.assertRaisesRegex(ValueError, 'independence'):
                add_independent(identity, used)
            self.assertEqual(used, before)

    def test_summary_counts_empty_slices_and_bad_cases_not_only_good_cases(self):
        reference = np.zeros((2, 8, 8), dtype=np.uint8)
        reference[0, 2:4, 2:4] = 1
        bad = np.zeros_like(reference)
        bad[1, 0, 0] = 1
        rows = [{'slices': 2, 'learned_metrics_native': overlap_metrics(p, reference),
                 'slice_quality': slice_quality_metrics(p, reference)} for p in (reference, bad)]
        result = summarize(rows)
        self.assertEqual(result['evaluated_cases'], 2)
        self.assertEqual(result['evaluated_slices'], 4)
        self.assertEqual(result['dice']['mean'], .5)
        self.assertEqual(result['dice']['min'], 0.)
        self.assertEqual(result['slice_quality']['empty_reference_slices'], 2)
        self.assertEqual(result['slice_quality']['empty_false_positive_pixels'], 1)

    def test_no_threshold_or_case_selection_options(self):
        parser = build_parser()
        self.assertNotIn('threshold', {a.dest for a in parser._actions})
        self.assertNotIn('case', {a.dest for a in parser._actions})

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
    def test_frozen_roundtrip_real_synthetic_dicom_missing_and_duplicate_cases(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for number, split in enumerate(('train', 'validation'), 1):
                folder = root / f'development-{number}'
                folder.mkdir()
                make_case(folder)
                files = sorted(folder.glob('slice*.dcm'))
                for path in files:
                    ds = pydicom.dcmread(path)
                    ds.PatientID = f'dev-{number}'
                    ds.save_as(path, enforce_file_format=True)
                identity = source_identities(files)
                records.append({'case': f'LCTSC-Train-S2-{number:03d}', 'split': split,
                    'patient_key': next(iter(identity['patients'])), 'frame_uid': next(iter(identity['frames'])),
                    'source_ct_files_sorted': list(map(str, files)), 'slices': 2})
            reserved = ['LCTSC-Test-S2-101', 'LCTSC-Test-S2-102']
            missing = 'LCTSC-Test-S2-103'
            for name, patient in zip(reserved, ('test-independent', 'dev-1')):
                folder = root / 'input' / name
                folder.mkdir(parents=True)
                make_case(folder)
                for path in folder.glob('slice*.dcm'):
                    ds = pydicom.dcmread(path)
                    ds.PatientID = patient
                    ds.save_as(path, enforce_file_format=True)
            manifest = {'status': 'complete', 'failures': [], 'cases': records,
                        'fingerprint': 'synthetic-only', 'image_size': 32,
                        'hu_clip': [-1000, 400], 'input_normalization': '(clip(HU,-1000,400)+1000)/1400',
                        'held_out_test_cases': reserved}
            dataset = root / 'dataset.json'
            dataset.write_text(json.dumps(manifest), encoding='utf-8')
            audit = root / 'audit.json'
            audit.write_text(json.dumps({'cases': [{'case': n, 'split': 'test'}
                                                   for n in reserved + [missing]]}), encoding='utf-8')
            torch.manual_seed(42)
            model = make_model(2)
            checkpoint = {'model': model.state_dict(), 'dataset_fingerprint': 'synthetic-only',
                          'image_size': 32, 'base_channels': 2, 'epoch': 1, 'threshold': .5}
            weights = root / 'best.pt'
            torch.save(checkpoint, weights)
            weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
            args = build_parser().parse_args(['--dataset', str(dataset), '--checkpoint', str(weights),
                '--audit', str(audit), '--input-root', str(root / 'input'),
                '--outdir', str(root / 'results'), '--device', 'cpu', '--batch-size', '1'])
            for changed in ({'dataset_fingerprint': 'another'}, {'image_size': 64},
                            {'threshold': .4}, {'context_slices': 3}):
                with self.assertRaises(ValueError):
                    freeze_protocol(args, manifest, dict(checkpoint, **changed))
            self.assertFalse((root / 'results').exists())
            with patch('lung_heldout.save_learned_preview'):
                run_heldout(args)
            report = json.loads((root / 'results' / 'evaluation.json').read_text())
            self.assertEqual(report['evaluated_cases'], 1)
            self.assertEqual(report['evaluated_slices'], 2)
            self.assertEqual(report['expected_test_cases'], 3)
            self.assertEqual(report['reserved_candidates_evaluated'], 1)
            self.assertEqual(report['status'], 'complete_available_subset')
            self.assertEqual(len(report['data_failures']), 2)
            self.assertIn('independence', report['data_failures'][0]['error'])
            self.assertTrue(report['checkpoint_unchanged'])
            self.assertEqual(hashlib.sha256(weights.read_bytes()).hexdigest(), weight_hash)
            self.assertTrue((root / 'results' / 'case_metrics.csv').is_file())
            cache_path = root / 'results' / 'test_dataset.json'
            with self.assertRaises(ValueError):
                read_manifest(cache_path)  # Separate test cache cannot enter training.
            with self.assertRaisesRegex(ValueError, 'not empty'):
                run_heldout(args)  # Do not overwrite scores/protocol.
            args.outdir = str(root / 'aborted')
            with patch('lung_heldout._case_prediction', side_effect=RuntimeError('synthetic inference failure')):
                with self.assertRaisesRegex(RuntimeError, 'inference failure'):
                    run_heldout(args)
            aborted = json.loads((root / 'aborted' / 'evaluation.json').read_text())
            self.assertEqual(aborted['status'], 'aborted_incomplete_evaluation')
            self.assertEqual(aborted['cases'], [])
            self.assertEqual(aborted['execution_failure']['case'], reserved[0])


if __name__ == '__main__':
    unittest.main()
