"""Synthetic engineering tests; passing does not establish medical accuracy."""
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, RTStructureSetStorage, ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungseg.baseline import BaselineConfig, segment_slice
from lungseg.dicom_io import (CTVolume, discover, load_ct, patient_to_pixel,
                             polygon_mask, rasterize_lungs)
from lungseg.metrics import overlap_metrics
from lung_unet import load_cached_source


def file_dataset(path, sop_class, uid):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = uid
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, uid
    return ds


def make_case(directory):
    series_uid, frame_uid, study_uid = generate_uid(), generate_uid(), generate_uid()
    slices = []
    for z in range(2):
        # Filename and InstanceNumber deliberately contradict physical ordering.
        path = Path(directory) / f'slice_{1-z}.dcm'
        ds = file_dataset(path, CTImageStorage, generate_uid())
        ds.Modality = 'CT'
        ds.SeriesInstanceUID, ds.FrameOfReferenceUID, ds.StudyInstanceUID = series_uid, frame_uid, study_uid
        ds.Rows, ds.Columns = 20, 30
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [10, 20, 30 + z * 3]
        ds.PixelSpacing, ds.SliceThickness = [2, 1], 3
        ds.InstanceNumber = 2 - z
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept, ds.RescaleType = 2, -1000, 'HU'
        pixels = np.full((20, 30), 550, dtype='<i2')
        pixels[:2] = 0
        pixels[-2:] = 0
        pixels[:, :2] = 0
        pixels[:, -2:] = 0
        pixels[4:15, 4:12] = 100 + z
        pixels[4:15, 18:26] = 100 + z
        ds.PixelData = pixels.tobytes()
        ds.save_as(path, enforce_file_format=True)
        slices.append(ds)
    path = Path(directory) / 'structures.dcm'
    rt = file_dataset(path, RTStructureSetStorage, generate_uid())
    rt.Modality, rt.StudyInstanceUID, rt.SeriesInstanceUID = 'RTSTRUCT', study_uid, generate_uid()
    ref_series, study, frame = Dataset(), Dataset(), Dataset()
    ref_series.SeriesInstanceUID = series_uid
    study.RTReferencedSeriesSequence = Sequence([ref_series])
    frame.FrameOfReferenceUID = frame_uid
    frame.RTReferencedStudySequence = Sequence([study])
    rt.ReferencedFrameOfReferenceSequence = Sequence([frame])
    rt.StructureSetROISequence, rt.ROIContourSequence = Sequence([]), Sequence([])
    for number, name, x0, x1 in [(1, 'Lung_L', 18, 25), (2, 'Lung_R', 4, 11)]:
        roi = Dataset()
        roi.ROINumber, roi.ROIName, roi.ReferencedFrameOfReferenceUID = number, name, frame_uid
        rt.StructureSetROISequence.append(roi)
        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = number
        roi_contour.ContourSequence = Sequence([])
        for ds in slices:
            contour, image_ref = Dataset(), Dataset()
            image_ref.ReferencedSOPInstanceUID, image_ref.ReferencedSOPClassUID = ds.SOPInstanceUID, CTImageStorage
            contour.ContourImageSequence = Sequence([image_ref])
            contour.ContourGeometricType = 'CLOSED_PLANAR'
            points = [[10 + x, 20 + y * 2, float(ds.ImagePositionPatient[2])]
                      for y, x in [(4, x0), (4, x1), (14, x1), (14, x0)]]
            contour.NumberOfContourPoints = 4
            contour.ContourData = np.ravel(points).tolist()
            roi_contour.ContourSequence.append(contour)
        rt.ROIContourSequence.append(roi_contour)
    rt.save_as(path, enforce_file_format=True)
    return series_uid, rt


class GeometryTests(unittest.TestCase):
    def test_cached_source_revalidates_identity_order_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            uid, _ = make_case(directory)
            series, _, _ = discover(directory)
            for path, _ in series[uid]:
                ds = pydicom.dcmread(path)
                ds.PatientID = 'synthetic-patient'
                ds.save_as(path, enforce_file_format=True)
            volume = load_ct(series[uid])
            path_by_uid = {str(ds.SOPInstanceUID): str(path) for path, ds in series[uid]}
            record = {'source_ct_files_sorted': [path_by_uid[sop] for sop in volume.sop_uids],
                      'patient_key': hashlib.sha256(b'synthetic-patient').hexdigest(),
                      'frame_uid': volume.frame_uid, 'native_shape': list(volume.hu.shape),
                      'spacing_zyx_mm': volume.spacing.tolist(), 'slices': 2}
            np.testing.assert_array_equal(load_cached_source(record), volume.hu)
            with self.assertRaisesRegex(ValueError, 'file order'):
                load_cached_source(dict(record, source_ct_files_sorted=record['source_ct_files_sorted'][::-1]))
            with self.assertRaisesRegex(ValueError, 'patient identity'):
                load_cached_source(dict(record, patient_key='different'))
            with self.assertRaisesRegex(ValueError, 'frame/shape'):
                load_cached_source(dict(record, frame_uid='different'))
            with self.assertRaisesRegex(ValueError, 'spacing'):
                load_cached_source(dict(record, spacing_zyx_mm=[2, 1, 1]))

    def test_reference_overlap_requires_explicit_permission_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            uid, rt = make_case(directory)
            series, _, _ = discover(directory)
            volume = load_ct(series[uid])
            contour = rt.ROIContourSequence[0].ContourSequence[0]
            # One reference voxel is shared at row 14, col 11. No relabeling allowed.
            z = float(contour.ContourData[2])
            contour.ContourData = np.ravel([[21, 48, z], [28, 48, z],
                                          [28, 56, z], [21, 56, z]]).tolist()
            with self.assertRaisesRegex(ValueError, 'overlap at 1 voxels'):
                rasterize_lungs(volume, rt)
            masks, info = rasterize_lungs(volume, rt, allow_overlap=True)
            self.assertEqual(info['overlap_check']['voxels'], 1)
            self.assertTrue(masks['left'][0, 14, 11])
            self.assertTrue(masks['right'][0, 14, 11])

    def test_anisotropic_and_oblique_coordinates(self):
        orientation = np.array([0, 1, 0, 0, 0, 1.])
        origin = np.array([12, 25, -8])
        rc = np.array([[1, 2], [5, 7]], dtype=float)
        xyz = origin + rc[:, 1, None] * orientation[:3] * .7 + rc[:, 0, None] * orientation[3:] * 2
        np.testing.assert_allclose(patient_to_pixel(xyz, origin, orientation, [2, .7]), rc)
        with self.assertRaisesRegex(ValueError, 'plane'):
            patient_to_pixel(xyz + [1, 0, 0], origin, orientation, [2, .7])

    def test_polygon_orientation_boundary_and_hole(self):
        vertices = np.array([[1, 1], [1, 4], [4, 4], [4, 1.]])
        filled = polygon_mask(vertices, (6, 6))
        self.assertEqual(int(filled.sum()), 16)
        np.testing.assert_array_equal(filled, polygon_mask(vertices[::-1], (6, 6)))
        diagonal = polygon_mask([[1, 1], [1, 3], [3, 1]], (5, 5))
        self.assertEqual(int(diagonal.sum()), 6)
        inner = polygon_mask([[2, 2], [2, 3], [3, 3], [3, 2]], (6, 6))
        self.assertEqual(int((filled ^ inner).sum()), 12)

    def test_subpixel_contour_without_pixel_centers_is_valid_empty(self):
        mask = polygon_mask([[2.1, 2.1], [2.1, 2.4], [2.3, 2.4], [2.3, 2.1]], (8, 8))
        self.assertFalse(mask.any())

    def test_ct_order_hu_and_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            uid, rt = make_case(directory)
            series, structures, _ = discover(directory)
            self.assertEqual(len(structures), 1)
            volume = load_ct(series[uid])
            np.testing.assert_allclose(volume.spacing, [3, 2, 1])
            np.testing.assert_allclose(volume.positions[:, 2], [30, 33])
            np.testing.assert_allclose(volume.hu[:, 5, 5], [-800, -798])
            masks, metadata = rasterize_lungs(volume, rt)
            self.assertEqual(int(masks['left'].sum()), 176)
            self.assertEqual(int(masks['right'].sum()), 176)
            self.assertEqual(metadata['left']['annotated_slices'], [0, 1])
            self.assertTrue(masks['left'][0, 5, 20])
            self.assertFalse(masks['left'][0, 5, 5])

    def test_reject_missing_slice_wrong_frame_and_plane(self):
        with tempfile.TemporaryDirectory() as directory:
            uid, rt = make_case(directory)
            series, _, _ = discover(directory)
            volume = load_ct(series[uid])
            contour = rt.ROIContourSequence[0].ContourSequence[0]
            original_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
            contour.ContourImageSequence[0].ReferencedSOPInstanceUID = generate_uid()
            with self.assertRaisesRegex(ValueError, 'SOP reference'):
                rasterize_lungs(volume, rt)
            contour.ContourImageSequence[0].ReferencedSOPInstanceUID = original_uid
            original_frame = rt.StructureSetROISequence[0].ReferencedFrameOfReferenceUID
            rt.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = generate_uid()
            with self.assertRaisesRegex(ValueError, 'frames of reference'):
                rasterize_lungs(volume, rt)
            rt.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = original_frame
            contour.ContourData[2] = float(contour.ContourData[2]) + 1
            with self.assertRaisesRegex(ValueError, 'plane'):
                rasterize_lungs(volume, rt)

    def test_xor_holes_and_mixed_types(self):
        with tempfile.TemporaryDirectory() as directory:
            uid, rt = make_case(directory)
            series, _, _ = discover(directory)
            volume = load_ct(series[uid])
            import copy
            contours = rt.ROIContourSequence[0].ContourSequence
            for contour in contours:
                contour.ContourGeometricType = 'CLOSEDPLANAR_XOR'
            inner = copy.deepcopy(contours[0])
            z = float(inner.ContourData[2])
            inner.ContourData = np.ravel([[30, 32, z], [32, 32, z], [32, 36, z], [30, 36, z]]).tolist()
            contours.append(inner)
            masks, _ = rasterize_lungs(volume, rt)
            self.assertFalse(masks['left'][0, 7, 21])
            self.assertEqual(int(masks['left'].sum()), 176 - 9)
            contours[0].ContourGeometricType = 'CLOSED_PLANAR'
            with self.assertRaisesRegex(ValueError, 'consistent'):
                rasterize_lungs(volume, rt)

    def test_duplicate_nonuniform_and_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            uid, _ = make_case(directory)
            series, _, _ = discover(directory)
            entries = series[uid]
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_ct(entries + [entries[0]])
            entries[1][1].ImageOrientationPatient = [1, 0, 0, 0, -1, 0]
            with self.assertRaisesRegex(ValueError, 'orientations'):
                load_ct(entries)


class BaselineTests(unittest.TestCase):
    def test_external_air_rejected_hole_filled(self):
        hu = np.full((64, 64), -1000., dtype=float)
        hu[6:58, 6:58] = 40
        hu[15:45, 12:27] = -800
        hu[15:45, 37:52] = -800
        hu[25:28, 18:21] = 100
        hu[10, 32] = -800  # tiny third candidate should be rejected
        pred, stages, info = segment_slice(hu, config=BaselineConfig(closing_iterations=0))
        self.assertFalse(pred[0, 0])
        self.assertFalse(pred[10, 32])
        self.assertTrue(pred[26, 19])
        self.assertEqual(int(pred.sum()), 900)
        self.assertEqual(info['retained_components'], 2)
        self.assertTrue(stages['candidate'][0, 0])

    def test_empty_and_zero_iteration(self):
        for image in [np.full((8, 8), 40.), np.full((8, 8), -1000.)]:
            pred, _, info = segment_slice(image)
            self.assertFalse(pred.any())
            self.assertIsNotNone(info['warning'])
        hu = np.full((9, 9), 40.)
        hu[2:7, 2:7] = -800
        hu[2, 4] = 40  # notch connects to outside; fill-holes should not fill it
        a, _, _ = segment_slice(hu, config=BaselineConfig(min_area_mm2=1, closing_iterations=0))
        b, _, _ = segment_slice(hu, config=BaselineConfig(min_area_mm2=1, closing_iterations=1))
        self.assertFalse(a[2, 4])
        self.assertTrue(b[2, 4])

    def test_invalid_config(self):
        for config in [BaselineConfig(min_area_mm2=-1), BaselineConfig(closing_iterations=-1),
                       BaselineConfig(threshold_hu=float('nan'))]:
            with self.assertRaises(ValueError):
                segment_slice(np.zeros((8, 8)), config=config)


class MetricTests(unittest.TestCase):
    def test_counts(self):
        scores = overlap_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
        self.assertEqual(scores['dice'], .5)
        self.assertAlmostEqual(scores['iou'], 1 / 3)
        self.assertEqual((scores['true_positive'], scores['false_positive'], scores['false_negative']), (1, 1, 1))

    def test_empty_not_inflated(self):
        scores = overlap_metrics(np.zeros((2, 2)), np.zeros((2, 2)))
        self.assertIsNone(scores['dice'])
        self.assertTrue(scores['both_empty'])
        self.assertEqual(overlap_metrics(np.ones((2, 2)), np.zeros((2, 2)))['dice'], 0)
        with self.assertRaises(ValueError):
            overlap_metrics(np.array([.5]), np.array([1]))


class CliTests(unittest.TestCase):
    def test_audit_missing_ct_and_partial_download(self):
        cli = Path(__file__).resolve().parents[1] / 'lung_segmentation.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'download'
            full = root / 'LCTSC-Train-S2-001'
            full.mkdir(parents=True)
            make_case(full)
            import shutil
            no_ct = root / 'LCTSC-Train-S1-001'
            no_ct.mkdir()
            shutil.copy2(full / 'structures.dcm', no_ct / 'structures.dcm')
            partial = root / 'LCTSC-Test-S2-101'
            partial.mkdir()
            shutil.copy2(full / 'structures.dcm', partial / 'structures.dcm')
            shutil.copy2(full / 'slice_0.dcm', partial / 'slice_0.dcm')
            output = Path(directory) / 'audit'
            result = subprocess.run([sys.executable, str(cli), 'audit', '--input', str(root),
                                     '--outdir', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            import json
            manifest = json.loads((output / 'audit.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['available_train_cases'], ['LCTSC-Train-S2-001'])
            self.assertEqual(len(manifest['exclude_for_now']), 2)
            rows = {row['case']: row for row in manifest['cases']}
            self.assertEqual(rows['LCTSC-Test-S2-101']['missing_referenced_slices'], 1)
            self.assertEqual(len(list(full.glob('*.dcm'))), 3)

    def test_prepare_evaluate_and_protect_outputs(self):
        cli = Path(__file__).resolve().parents[1] / 'lung_segmentation.py'
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data'
            data.mkdir()
            make_case(data)
            prepared, result = Path(directory) / 'prepared', Path(directory) / 'result'
            def run(*args):
                return subprocess.run([sys.executable, str(cli), *map(str, args)],
                                      capture_output=True, text=True)
            inspect = run('inspect', '--input', data)
            self.assertEqual(inspect.returncode, 0, inspect.stderr)
            prep = run('prepare', '--input', data, '--outdir', prepared)
            self.assertEqual(prep.returncode, 0, prep.stderr)
            self.assertTrue((prepared / 'alignment_preview.png').exists())
            evaluation = run('evaluate', '--case', prepared / 'case.npz', '--outdir', result)
            self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
            import json
            metrics = json.loads((result / 'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(metrics['volume_metrics']['dice'], 1)
            self.assertTrue((result / 'preview.png').exists())
            self.assertEqual(np.load(result / 'candidate_mask.npy').shape, (20, 30))
            rejected = run('evaluate', '--case', prepared / 'case.npz', '--outdir', result)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('not empty', rejected.stderr)
            invalid_slice = run('evaluate', '--case', prepared / 'case.npz',
                                '--outdir', Path(directory) / 'bad', '--slice', 20)
            self.assertNotEqual(invalid_slice.returncode, 0)


if __name__ == '__main__':
    unittest.main()
