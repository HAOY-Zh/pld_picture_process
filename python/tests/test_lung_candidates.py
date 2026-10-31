"""Candidate geometry, blob extraction, matching and held-out isolation checks."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_candidates import (CandidateConfig, detect_candidates, generate_seed_reference,
                              log_kernels, point_metrics, response_slice)
from lung_candidate_data import (development_records, freeze_split, load_development_case,
                                  read_split, write_json)
from lung_candidate_structure import (hessian_features, select_structural_candidates, StructureConfig,
                                      extract_suspicious_regions, roi_candidate_packet)


def phantom():
    shape, spacing = (24, 64, 64), np.array([1.5, .7, .7])
    origin = np.array([100., 200., -30.])
    basis = np.diag([-spacing[2], -spacing[1], spacing[0]])[:, ::-1]
    geometry = dict(spacing_zyx_mm=spacing.tolist(), origin_lps_mm=origin.tolist(), basis_lps_zyx=basis.tolist())
    grid = np.ogrid[:shape[0], :shape[1], :shape[2]]
    hu = np.full(shape, -850., np.float32)
    centers = [np.array([8, 20, 20]), np.array([16, 43, 42])]
    for center in centers:
        distance2 = sum(((g-c)*s)**2 for g, c, s in zip(grid, center, spacing))
        hu += 500*np.exp(-distance2/(2*1.5**2))
    mask = np.broadcast_to(((grid[1]-32)**2+(grid[2]-32)**2) < 25**2, shape).copy()
    references = [dict(center_lps_xyz_mm=(origin+basis@p).tolist(), diameter_mm=5.) for p in centers]
    return hu, mask, geometry, references


class CandidateTests(unittest.TestCase):
    def test_constant_image_zero_dc_at_anisotropic_spacings(self):
        for sigma in (1., 2., 4.):
            kernels = log_kernels(sigma, (.6, .9), 3.5)
            for g, derivative in kernels:
                self.assertAlmostEqual(float(g.sum()), 1.)
                self.assertLess(abs(float(derivative.sum())), 1e-12)
            response = response_slice(np.full((32, 48), 300., np.float32), kernels, sigma)
            self.assertLess(float(abs(response).max()), 1e-3)

    def test_two_blobs_world_coordinates_and_coarse_boxes(self):
        hu, mask, geometry, references = phantom()
        cfg = CandidateConfig(sigma_mm=(1., 2.), workers=1, search_margin_mm=0.)
        result, seed = detect_candidates(hu, mask, geometry, cfg)
        scores = point_metrics([c['center_lps_xyz_mm'] for c in result['candidates']], references)
        self.assertEqual(scores['detected'], 2)
        self.assertGreater(seed['seed_mask'].sum(), 0)
        for c in result['candidates']:
            p = c['center_index_zyx']
            for i in range(3):
                self.assertLessEqual(c['bbox_zyx'][2*i], p[i])
                self.assertGreater(c['bbox_zyx'][2*i+1], p[i])
            self.assertGreater(c['voxel_count'], 0)

    def test_empty_lung_does_not_fall_back_to_whole_body(self):
        hu, mask, geometry, _ = phantom()
        result, seed = detect_candidates(hu, np.zeros_like(mask), geometry)
        self.assertEqual(result['candidates'], [])
        self.assertEqual(seed['search_mask'].sum(), 0)

    def test_invalid_mask_geometry_and_config_rejected(self):
        hu, mask, geometry, _ = phantom()
        with self.assertRaises(ValueError):
            generate_seed_reference(hu, mask.astype(np.uint8)*255, geometry)
        with self.assertRaises(ValueError):
            generate_seed_reference(hu, mask, dict(geometry, spacing_zyx_mm=[1., 1., 1.]))
        for cfg in [replace(CandidateConfig(), response_threshold_hu=0),
                    replace(CandidateConfig(), sigma_mm=(0.,)), replace(CandidateConfig(), workers=0)]:
            with self.assertRaises(ValueError):
                cfg.checked()

    def test_parallel_filtering_reproducible_and_no_mask_mutation(self):
        hu, mask, geometry, _ = phantom()
        original = mask.copy()
        first = generate_seed_reference(hu, mask, geometry, CandidateConfig(sigma_mm=(1., 2.), workers=1))
        second = generate_seed_reference(hu, mask, geometry, CandidateConfig(sigma_mm=(1., 2.), workers=2))
        np.testing.assert_array_equal(first['response'], second['response'])
        np.testing.assert_array_equal(mask, original)

    def test_matching_exclusions_duplicates_and_strict_radius(self):
        refs = [dict(center_lps_xyz_mm=[0., 0., 0.], diameter_mm=4.)]
        excluded = [dict(center_lps_xyz_mm=[20., 0., 0.], diameter_mm=-1.)]
        result = point_metrics([[0, 0, 0], [1, 0, 0], [2, 0, 0], [24, 0, 0], [25, 0, 0]], refs, excluded)
        self.assertEqual(result['detected'], 1)
        self.assertEqual(result['duplicate_reference_hits'], 1)
        self.assertEqual(result['ignored_excluded_candidates'], 1)
        self.assertEqual(result['false_positive_candidates'], 2)
        empty = point_metrics([], refs)
        self.assertEqual(empty['detected'], 0)
        self.assertIsNone(point_metrics([], [])['sensitivity'])


class SplitTests(unittest.TestCase):
    def make_audit(self):
        rows = []
        for i in range(135):
            rows.append(dict(series_uid=f'1.2.3.{i}', raw_sha256=f'{i:064x}', spacing_zyx_mm=[1. if i%2 else 2., .7, .7],
                nodules=[dict(diameter_mm=4.+i%12)] if i < 87 else []))
        return dict(status='PASS', errors=[], source_changed=False, cases=rows,
            source_archive='not_opened_by_split', source_bytes=0, source_mtime_ns=0, annotations_sha256='frozen',
            prepared_case=dict(series_uid='1.2.3.0'))

    def test_split_counts_inspected_case_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root/'audit.json'
            write_json(audit, self.make_audit())
            a = freeze_split(audit, root/'a')
            b = freeze_split(audit, root/'b')
            self.assertEqual(a, b)
            self.assertEqual(a['summary']['development']['scans'], 90)
            self.assertEqual(a['summary']['heldout']['scans'], 45)
            self.assertEqual(a['summary']['heldout']['with_reference'], 29)
            self.assertEqual(a['summary']['heldout']['without_reference'], 16)
            self.assertEqual(len(a['pilot_development_uids']), 12)
            check = development_records(a, 'check-development')
            self.assertEqual(len(check), 12)
            self.assertFalse(set(a['pilot_development_uids']) & {r['series_uid'] for r in check})
            self.assertTrue(all(r['split'] == 'development' for r in check))
            self.assertIn('1.2.3.0', [r['series_uid'] for r in development_records(a)])
            self.assertEqual(read_split(root/'a/split.json'), json.loads((root/'a/split.json').read_text()))
            with self.assertRaises(ValueError):
                development_records(a, 'heldout')
            heldout = next(r for r in a['cases'] if r['split'] == 'heldout')
            with self.assertRaisesRegex(ValueError, 'Held-out'):
                load_development_case(a, heldout)
            with self.assertRaises(ValueError):
                freeze_split(audit, root/'a')
            with (root/'a/split.json').open('a') as f:
                f.write(' ')
            with self.assertRaisesRegex(ValueError, 'changed'):
                read_split(root/'a/split.json')

    def test_duplicate_payload_rejected_before_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.make_audit()
            a['cases'][1]['raw_sha256'] = a['cases'][0]['raw_sha256']
            write_json(root/'audit.json', a)
            with self.assertRaisesRegex(ValueError, 'Identical'):
                freeze_split(root/'audit.json', root/'split')


class StructureTests(unittest.TestCase):
    def test_three_dimensional_blob_and_long_tube_are_different(self):
        z, y, x = np.ogrid[:25, :49, :49]
        sphere = -850.+500*np.exp(-((z-12)**2+(y-24)**2+(x-24)**2)/(2*2.**2))
        tube = np.broadcast_to(-850.+500*np.exp(-((y-24)**2+(x-24)**2)/(2*2.**2)), sphere.shape).copy()
        candidate = dict(center_index_zyx=[12, 24, 24], estimated_scale_diameter_mm=2*np.sqrt(2)*2)
        spherical = hessian_features(sphere.astype(np.float32), [candidate], [1., 1., 1.])[0]
        tubular = hessian_features(tube.astype(np.float32), [candidate], [1., 1., 1.])[0]
        self.assertGreater(spherical['blobness'], .9)
        self.assertLess(tubular['blobness'], .01)
        self.assertGreater(spherical['blob_score'], tubular['blob_score'])

    def test_structure_selection_keeps_identity_and_does_not_read_labels(self):
        hu, mask, geometry, _ = phantom()
        result, _ = detect_candidates(hu, mask, geometry, CandidateConfig(sigma_mm=(1., 2.), workers=1))
        features = hessian_features(hu, result['candidates'], geometry['spacing_zyx_mm'])
        selected = select_structural_candidates(result['candidates'], features, geometry,
            StructureConfig(minimum_region_diameter_mm=1.5))
        self.assertGreaterEqual(len(selected), 2)
        self.assertTrue(all(c['source_candidate_id'] for c in selected))
        with self.assertRaises(ValueError):
            StructureConfig(minimum_blobness=2).checked()

    def test_callable_result_packet_contains_coordinates_without_debug_arrays(self):
        hu, mask, geometry, _ = phantom()
        result, debug = extract_suspicious_regions(hu, mask, geometry,
            CandidateConfig(sigma_mm=(1., 2.), workers=1), StructureConfig(minimum_region_diameter_mm=1.5))
        packet = roi_candidate_packet(result)
        self.assertGreaterEqual(len(packet['candidates']), 2)
        self.assertEqual(packet['source_shape_zyx'], list(hu.shape))
        self.assertEqual(packet['geometry'], geometry)
        self.assertNotIn('stage_centers_index_zyx', packet)
        self.assertNotIn('reference', packet)
        json.dumps(packet, allow_nan=False)
        self.assertIn('seed_mask', debug)


if __name__ == '__main__':
    unittest.main()
