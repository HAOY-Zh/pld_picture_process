"""Expert overlays on synthetic CT; fake hardware proves routing, not FPGA accuracy."""
import base64
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pydicom

import test_lung_live as live
from test_lung_segmentation import make_case
from lung_live_data import read_ct
from lung_live_reference import read_reference
from lung_volume_viewer import nifti_export
from lungseg.dicom_io import discover, load_ct, rasterize_lungs
from lungseg.metrics import overlap_metrics


def unpack(value, shape):
    return np.unpackbits(np.frombuffer(base64.b64decode(value), np.uint8),
                         bitorder='little')[:np.prod(shape)].reshape(shape).astype(bool)


class ReferenceDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.folder = Path(self.temp.name)
        self.uid, self.rt = make_case(self.folder)

    def tearDown(self): self.temp.cleanup()

    def test_auto_rtstruct_matches_original_reference_and_never_decodes_unselected_ct(self):
        series, _, _ = discover(self.folder)
        masks, _ = rasterize_lungs(load_ct(series[self.uid]), self.rt)
        read = pydicom.dcmread; decoded = []
        def track(path, *args, **kwargs):
            ds = read(path, *args, **kwargs)
            if ds.Modality == 'CT' and not kwargs.get('stop_before_pixels'): decoded.append(Path(path).name)
            return ds
        with patch.object(pydicom, 'dcmread', side_effect=track):
            hu, geom, details = read_ct(self.folder, slices='1')
            expert, info = read_reference('', hu.shape, geom, details)
        np.testing.assert_array_equal(expert, (masks['left'] | masks['right'])[1:2])
        self.assertEqual(decoded, ['slice_0.dcm'])
        self.assertTrue(info['automatic']); self.assertEqual(info['source_slice_indices'], [1])
        self.assertEqual(info['geometry'], geom)

    def test_ambiguous_rtstruct_is_not_guessed_and_explicit_disable_still_loads_ct(self):
        shutil.copyfile(self.folder/'structures.dcm', self.folder/'other.dcm')
        hu, geom, details = read_ct(self.folder)
        with self.assertRaisesRegex(ValueError, '多份'):
            read_reference('', hu.shape, geom, details)
        self.assertEqual(read_reference('', hu.shape, geom, details, enabled=False), (None, None))
        expert, _ = read_reference(str(self.folder/'structures.dcm'), hu.shape, geom, details)
        self.assertTrue(expert.any())

    def test_rtstruct_wrong_frame_unknown_sop_and_wrong_plane_are_rejected(self):
        hu, geom, details = read_ct(self.folder)
        rt_path = self.folder/'structures.dcm'
        for problem in ('frame', 'sop', 'plane'):
            rt = pydicom.dcmread(rt_path)
            if problem == 'frame': rt.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = '1.2.3'
            elif problem == 'sop':
                rt.ROIContourSequence[0].ContourSequence[0].ContourImageSequence[0].ReferencedSOPInstanceUID = '1.2.3'
            else: rt.ROIContourSequence[0].ContourSequence[0].ContourData[2] += 1
            target = self.folder/(problem+'.dcm'); rt.save_as(target, enforce_file_format=True)
            with self.assertRaises(ValueError): read_reference(str(target), hu.shape, geom, details)

    def test_nifti_axis_flips_and_partial_crop_preserve_ct_world_alignment(self):
        hu, geom, details = read_ct(self.folder)
        mask = np.zeros(hu.shape, np.uint8); mask[:, 4:14, 5:11] = 1; mask[1, 6:10, 20:25] = 2
        path = self.folder/'expert.nii.gz'; nifti_export(mask, geom, path)
        # Export was LPS storage; canonicalize to reverse X/Y storage explicitly.
        image = nib.as_closest_canonical(nib.load(path)); nib.save(image, path)
        phu, pg, pd = read_ct(self.folder, slices='1')
        actual, _ = read_reference(str(path), phu.shape, pg, pd)
        np.testing.assert_array_equal(actual, mask[1:2] != 0)
        # Also align CT loaded through canonical NIfTI rather than DICOM.
        ct = self.folder/'ct.nii.gz'; nifti_export(hu, geom, ct)
        chu, cg, cd = read_ct(ct, True, slices='1')
        actual, _ = read_reference(str(path), chu.shape, cg, cd)
        np.testing.assert_array_equal(actual, mask[1:2, ::-1, ::-1] != 0)

    def test_reference_rejects_wrong_spacing_origin_label_and_conflicting_affines(self):
        hu, geom, details = read_ct(self.folder)
        mask = np.zeros(hu.shape, np.uint8); mask[:, 4:12, 4:11] = 1
        path = self.folder/'expert.nii'
        for problem in ('spacing', 'origin', 'fractional_labels', 'qform'):
            nifti_export(mask, geom, path); image = nib.load(path)
            affine = image.affine.copy()
            if problem == 'spacing': affine[:3, 0] *= 2
            elif problem == 'origin': affine[0, 3] += .25
            if problem in ('spacing', 'origin'):
                image.set_sform(affine, 1); image.set_qform(affine, 1)
            elif problem == 'qform':
                affine[0, 3] += 1; image.set_qform(affine, 1)
            else:
                data = mask.astype(np.float32).transpose(2, 1, 0); data[0, 0, 0] = .5
                image = nib.Nifti1Image(data, affine); image.header.set_xyzt_units('mm')
            target = self.folder/(problem+'.nii.gz'); nib.save(image, target)
            del image
            with self.assertRaises(ValueError): read_reference(str(target), hu.shape, geom, details)

    def test_explicit_npz_requires_mask_geometry_not_just_array_shape(self):
        hu, geom, details = read_ct(self.folder)
        path = self.folder/'expert.npz'
        values = dict(mask=np.ones(hu.shape, np.uint8), spacing_zyx_mm=geom['spacing_zyx_mm'],
            origin_lps_mm=geom['origin_lps_mm'], orientation=[1, 0, 0, 0, 1, 0])
        np.savez(path, **values)
        expert, _ = read_reference(str(path), hu.shape, geom, details); self.assertTrue(expert.all())
        values['origin_lps_mm'] = [11, 20, 30]; np.savez(path, **values)
        with self.assertRaises(ValueError): read_reference(str(path), hu.shape, geom, details)


class ReferenceWorkflowTests(unittest.TestCase):
    setUp = live.WorkflowTests.setUp
    tearDown = live.WorkflowTests.tearDown
    connect = live.WorkflowTests.connect

    def load_expert(self, mask=None, slices=''):
        if mask is None:
            mask = np.zeros((3, 32, 40), np.uint8); mask[:, 5:25, 8:30] = 1
        self.reference = self.folder/'expert.npz'
        np.savez(self.reference, mask=mask, spacing_zyx_mm=[2.5, .8, .7],
            origin_lps_mm=[0, 0, 0], orientation=[1, 0, 0, 0, 1, 0])
        self.control.submit('load', dict(path=str(self.path), confirm_hu=True,
            expert_path=str(self.reference), load_slices=slices))
        state = live.wait(self.control); self.assertEqual(state['phase'], 'idle', state['error'])
        return state

    def test_expert_available_before_inference_in_all_mpr_planes_without_fake_predictions(self):
        state = self.load_expert(); self.assertTrue(state['expert_present'])
        self.assertEqual(state['comparison'], {}); self.assertEqual(self.hardware.calls, 0)
        for axis in range(3):
            p = self.control.plane(state['case_id'], axis, 0, 'body')
            np.testing.assert_array_equal(unpack(p['expert'], p['shape']), np.take(self.control.expert, 0, axis))
            self.assertFalse(unpack(p['known'], p['shape']).any())
            self.assertIsNone(p['expert_comparison'])

    def test_single_and_sparse_dice_exclude_unknown_planes_and_survive_recomputation(self):
        self.load_expert(); self.connect()
        for selection, expected in [('1', [1]), ('0,2', [0, 1, 2]), ('1', [0, 1, 2])]:
            self.control.submit('run', dict(mode='selection', index=0, slices=selection))
            state = live.wait(self.control); self.assertEqual(state['phase'], 'done', state['error'])
            row = state['comparison']['body']; self.assertEqual(row['source_slice_indices'], expected)
            self.assertEqual(row['dice'], overlap_metrics(self.control.body[expected], self.control.expert[expected])['dice'])
            if expected == [1]: initial_report = next(self.control.case_folder.glob('*_comparison.json'))
        self.assertEqual(self.hardware.calls, 4)
        report = json.loads(initial_report.read_text(encoding='utf-8'))
        self.assertEqual(report['stages']['body']['source_slice_indices'], [1])
        self.assertEqual(report['stages']['body']['scope'], 'computed_slices')

    def test_partial_expert_mesh_and_saved_report_use_same_region_as_actual_predictions(self):
        self.load_expert(slices='1-2'); self.connect()
        self.control.submit('run', dict(mode='volume', index=0)); state = live.wait(self.control)
        self.assertEqual(state['phase'], 'done', state['error'])
        self.assertEqual(state['comparison']['body']['scope'], 'computed_slices')
        self.assertEqual(state['comparison']['body']['source_slice_indices'], [1, 2])
        expert = self.control.surface(state['case_id'], 'expert')
        body = self.control.surface(state['case_id'], 'body')
        self.assertEqual(expert['reconstruction'], body['reconstruction'])
        self.assertEqual(expert['geometry'], body['geometry'])
        self.assertAlmostEqual(np.array(expert['vertices'])[:, 2].min(), 1.25)
        self.assertAlmostEqual(np.array(expert['vertices'])[:, 2].max(), 6.25)
        report = json.loads(next(self.control.case_folder.glob('*_volume.json')).read_text(encoding='utf-8'))
        self.assertTrue(report['expert_present']); self.assertEqual(report['expert_comparison'], state['comparison'])
        self.assertIsNone(report['volume_ml'])
        receipt = json.loads(next(self.control.case_folder.glob('*/receipt.json')).read_text(encoding='utf-8'))
        self.assertEqual(receipt['expert_comparison']['reference_sha256'], state['expert_source']['mask_sha256'])

    def test_expert_never_changes_network_input_or_any_prediction_stage_and_new_case_clears_it(self):
        self.connect(); self.control.submit('run', dict(mode='volume', index=0)); live.wait(self.control)
        original = [a.copy() for a in (self.hardware.capture, self.control.raw, self.control.body, self.control.cleaned)]
        self.load_expert(np.ones((3, 32, 40), np.uint8))
        self.control.submit('run', dict(mode='volume', index=0)); state = live.wait(self.control)
        self.assertEqual(state['phase'], 'done', state['error'])
        for a, b in zip(original, (self.hardware.capture, self.control.raw, self.control.body, self.control.cleaned)):
            np.testing.assert_array_equal(a, b)
        self.assertIn('cleaned', state['comparison'])
        self.control.submit('load', dict(path=str(self.path), confirm_hu=True)); state = live.wait(self.control)
        self.assertFalse(state['expert_present']); self.assertIsNone(state['comparison'])
        self.assertIsNone(self.control.plane(state['case_id'], 0, 0, 'body')['expert'])

    def test_failure_on_recomputed_slice_removes_stale_score(self):
        self.load_expert(); self.connect()
        self.control.submit('run', dict(mode='slice', index=1)); live.wait(self.control)
        self.hardware.fail = True
        self.control.submit('run', dict(mode='slice', index=1)); state = live.wait(self.control)
        self.assertEqual(state['phase'], 'error'); self.assertEqual(state['comparison'], {})
        self.assertEqual(state['known_slices'], 0)

    def test_empty_reference_and_empty_prediction_have_undefined_dice(self):
        self.load_expert(np.zeros((3, 32, 40), np.uint8))
        with self.control.lock:
            self.control.known[0] = True
            self.control.scores['body'][0] = overlap_metrics(self.control.body[0], self.control.expert[0])
        state = self.control.snapshot()
        self.assertIsNone(state['comparison']['body']['dice'])
        self.assertTrue(state['comparison']['body']['both_empty'])
        p = self.control.plane(state['case_id'], 0, 0, 'body')
        self.assertIsNone(p['expert_comparison']['dice']); self.assertTrue(p['expert_comparison']['both_empty'])


if __name__ == '__main__': unittest.main()
