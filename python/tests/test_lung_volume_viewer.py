"""Native geometry, exact transport, unmodified masks and offline export tests."""
import base64
import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_volume_viewer import (build, geometry, html_export, hu_storage, mesh,
                               nifti_export, packed, parser, payload, slice_metrics, to_world,
                               html_payload, refresh_html, unpack_viewer, cleanup_export)
from lung_unet import file_sha256, make_model, torch_module
from test_lung_outer_contour import fixture


def decoded(item):
    raw = gzip.decompress(base64.b64decode(item['gzip_base64']))
    if item['dtype'] == 'bits':
        return np.unpackbits(np.frombuffer(raw, np.uint8), bitorder='little')[:item['elements']].reshape(item['shape'])
    types = {'int16': '<i2', 'float32': '<f4', 'uint32': '<u4'}
    return np.frombuffer(raw, types[item['dtype']]).reshape(item['shape'])


class ViewerTests(unittest.TestCase):
    def test_world_and_nifti_ras_mapping_no_axis_flip_error(self):
        geom = geometry([-100, -200, 30], [1, 0, 0, 0, 1, 0], [2, .8, .6])
        np.testing.assert_allclose(to_world([[2, 3, 4]], geom), [[-97.6, -197.6, 34]])
        array = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ct.nii.gz'
            nifti_export(array, geom, path)
            image = nib.load(path)
            np.testing.assert_array_equal(np.asarray(image.dataobj).transpose(2, 1, 0), array)
            xyz = nib.affines.apply_affine(image.affine, [4, 3, 2])
            np.testing.assert_allclose(xyz, [97.6, 197.6, 34], atol=1e-5)
            self.assertEqual(image.header.get_xyzt_units()[0], 'mm')
        with self.assertRaises(ValueError):
            geometry([0, 0, 0], [1, 0, 0, 0, 1, 0], [0, 1, 1])
        with self.assertRaises(ValueError):
            geometry([0, 0, 0], [0, 1, 0, 0, 0, 1], [1, 1, 1])

    def test_mesh_preserves_inner_cavity_and_tiny_island_source_unchanged(self):
        geom = geometry([0, 0, 0], [1, 0, 0, 0, 1, 0], [2, 1, 1])
        array = np.zeros((24, 24, 24), np.uint8)
        array[4:20, 4:20, 4:20] = 1
        array[10:13, 10:13, 10:13] = 0
        array[1, 1, 1] = 1
        before = array.copy()
        result = mesh(array, geom)
        np.testing.assert_array_equal(array, before)
        self.assertTrue(np.any(np.all((result['voxels'] >= 9.5) & (result['voxels'] <= 12.5), axis=1)))
        self.assertTrue(np.any(np.all(result['voxels'] <= 1.5, axis=1)))
        np.testing.assert_allclose(result['world'], to_world(result['voxels'], geom))
        self.assertEqual(mesh(np.zeros_like(array), geom)['faces'].shape, (0, 3))

    def test_transport_exact_native_values_bit_order_and_odd_dimensions(self):
        for array in [np.array([-1024, 1359], np.int16), np.array([.2, 1.3], np.float32),
                      np.array([0, 170000], np.uint32)]:
            np.testing.assert_array_equal(decoded(packed(array)), array)
        mask = np.zeros((3, 7, 5), np.uint8)
        mask.ravel()[[0, 7, 8, 104]] = 255
        np.testing.assert_array_equal(decoded(packed(mask, True)), mask != 0)
        with self.assertRaises(ValueError):
            packed(np.array([2], np.uint8), True)

    def test_hu_storage_no_lossy_windowing(self):
        hu = np.array([[[-1024., 5000.]]], np.float32)
        self.assertEqual(hu_storage(hu).dtype, np.int16)
        np.testing.assert_array_equal(hu_storage(hu), hu)
        hu = np.array([[[-1000.25, 90000.]]], np.float32)
        self.assertEqual(hu_storage(hu).dtype, np.float32)
        np.testing.assert_array_equal(hu_storage(hu), hu)

    def test_native_slice_metrics_and_payload_alignment(self):
        reference = np.zeros((4, 6, 8), bool)
        reference[1:3, 2:5, 2:6] = True
        prediction = reference.copy(); prediction[1, 2, 2] = False
        geom = geometry([0, 0, 0], [1, 0, 0, 0, 1, 0], [2, 1, 1])
        meshes = {'expert': mesh(reference, geom), 'prediction': mesh(prediction, geom)}
        data = payload(np.zeros_like(reference, dtype=np.int16), reference, prediction, geom, meshes, {})
        np.testing.assert_array_equal(decoded(data['expert']), reference)
        np.testing.assert_array_equal(decoded(data['prediction']), prediction)
        self.assertEqual(slice_metrics(prediction, reference, 0, 1)['false_negative'], 1)
        self.assertEqual(slice_metrics(prediction, reference, 1, 2)['false_negative'], 1)
        with self.assertRaises(ValueError):
            payload(np.zeros((4, 5, 8)), reference, prediction, geom, meshes, {})

    def test_offline_html_json_safe_no_cdn_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'viewer.html'
            html_export({'case': '</script><script>alert(1)</script>'}, path)
            text = path.read_text(encoding='utf-8')
            self.assertNotIn('__PAYLOAD__', text)
            self.assertNotIn('__PLOTLY__', text)
            self.assertNotIn('<script src=', text)
            self.assertNotIn('</script><script>alert(1)</script>', text)
            self.assertIn('DecompressionStream', text)
            self.assertIn('viewerInfo', text)

    def test_real_small_export_protects_original_files(self):
        torch = torch_module(); torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); dataset, manifest = fixture(root)
            for r in manifest['cases']:
                source = root / (r['case'] + '.dcm'); source.write_bytes(b'fixture header')
                r['source_ct_files_sorted'] = [str(source)] * 2
            dataset.write_text(json.dumps(manifest), encoding='utf-8')
            original = root / 'original.pt'
            torch.save(dict(model=make_model(2).state_dict(), base_channels=2, image_size=64,
                context_slices=1, hu_clip=[-1000, 400], threshold=.5, epoch=12, dataset_fingerprint='original'), original)
            before = file_sha256(original), file_sha256(dataset)
            args = parser().parse_args(['--dataset', str(dataset), '--checkpoint', str(original), '--case',
                manifest['cases'][1]['case'], '--slice', '0', '--device', 'cpu', '--threads', '1', '--outdir', str(root / 'view')])
            header = SimpleNamespace(ImagePositionPatient=[0, 0, 0], ImageOrientationPatient=[1, 0, 0, 0, 1, 0])
            with patch('lung_volume_viewer.load_cached_source', return_value=np.zeros((2, 64, 64), np.float32)), \
                 patch('lung_volume_viewer.pydicom.dcmread', return_value=header), patch('lung_volume_viewer.preview'):
                build(args)
            self.assertEqual(before, (file_sha256(original), file_sha256(dataset)))
            report = json.loads((root / 'view/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'complete_native_3d_comparison')
            self.assertTrue(report['protected_inputs_unchanged'])
            self.assertFalse(report['metadata']['smoothing'])
            self.assertFalse(report['metadata']['component_removal'])
            pred = nib.load(root / 'view/prediction_mask.nii.gz')
            ref = nib.load(root / 'view/expert_mask.nii.gz')
            np.testing.assert_array_equal(pred.affine, ref.affine)
            self.assertEqual(pred.shape, (64, 64, 2))

    def test_layout_refresh_preserves_embedded_data_and_other_artifacts(self):
        geom = geometry([0, 0, 0], [1, 0, 0, 0, 1, 0], [2, 1, 1])
        mask = np.zeros((4, 6, 8), np.uint8)
        surfaces = {'expert': mesh(mask, geom), 'prediction': mesh(mask, geom)}
        data = payload(mask.astype(np.int16), mask, mask, geom, surfaces, {'case': 'fixture'})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'viewer.html'
            artifact = Path(tmp) / 'expert_mask.nii.gz'
            artifact.write_bytes(b'untouched original mask')
            before = file_sha256(artifact)
            html_export(data, path)
            raw, _ = html_payload(path.read_text(encoding='utf-8'))
            report = refresh_html(path)
            text = path.read_text(encoding='utf-8')
            self.assertEqual(raw, html_payload(text)[0])
            self.assertEqual(before, file_sha256(artifact))
            self.assertTrue(report['embedded_dataset_byte_identical'])
            self.assertFalse(report['inference_rerun'])
            self.assertIn('id="workspace"', text)
            self.assertIn('ResizeObserver', text)
            self.assertIn('id="fullscreen"', text)
            self.assertIn('<option value="alpha">CT＋肺区色块</option>', text)
            self.assertIn('id="maskOpacity"', text)
            self.assertIn('id="sliceLegend"', text)
            self.assertIn('id="sliceSource"', text)
            self.assertEqual(report['html_sha256_after'], file_sha256(path))

    def test_layout_refresh_refuses_unrecognized_html_without_replacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'viewer.html'
            path.write_text('<html>not a viewer</html>', encoding='utf-8')
            before = file_sha256(path)
            with self.assertRaises(ValueError):
                refresh_html(path)
            self.assertEqual(before, file_sha256(path))

    def test_unpack_viewer_exact_native_transport(self):
        for a in [np.array([0,1,0,1,1],np.uint8),np.array([-1024,100],np.int16),np.array([.25,1.],np.float32)]:
            np.testing.assert_array_equal(unpack_viewer(packed(a,a.dtype==np.uint8)),a)
        bad=packed(np.array([0,1],np.uint8),True);bad['shape']=[3]
        with self.assertRaises(ValueError):unpack_viewer(bad)

    def test_optional_cleanup_export_preserves_original_and_rejects_unaccepted(self):
        from lungseg.metrics import overlap_metrics
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case='LCTSC-Train-S3-008';screen=root/'screen';folder=screen/case;folder.mkdir(parents=True)
            g=np.zeros((8,8,8),bool);g[2:6,2:6,2:6]=True
            original=g.copy();original[0,0,0]=True;cleaned=g.copy()
            geom=geometry([0,0,0],[1,0,0,0,1,0],[2,1,1])
            surfaces={name:mesh(a,geom) for name,a in [('expert',g),('prediction',original)]}
            source=root/'original.html'
            metadata=dict(case=case,split='validation',initial_slice=3,checkpoint_sha256='frozen',
                expert_volume_ml=float(g.sum()*.002),metrics=overlap_metrics(original,g))
            html_export(payload(np.zeros_like(g,dtype=np.int16),g,original,geom,surfaces,metadata),source)
            before=file_sha256(source)
            artifact=folder/'predictions_native.npz';np.savez_compressed(artifact,original=original,primary=cleaned)
            scores={'original':overlap_metrics(original,g),'primary':dict(overlap_metrics(cleaned,g),removed_components=1)}
            row=dict(case=case,split='validation',spacing_zyx_mm=[2,1,1],prediction_sha256=file_sha256(artifact),
                     cpu_analysis_ms=1.,scores=scores)
            report=dict(status='complete_development_screen',primary_accepted_for_optional_development_view=True,
                protected_inputs_unchanged=True,protocol=dict(model_sha256='frozen',primary={'max_island_ml':5.,'min_distance_mm':20.}),cases=[row])
            report_path=screen/'evaluation.json';report_path.write_text(json.dumps(report),encoding='utf-8')
            cleanup_export(source,report_path,root/'view')
            self.assertEqual(before,file_sha256(source))
            _,exported=html_payload((root/'view/viewer.html').read_text(encoding='utf-8'))
            np.testing.assert_array_equal(unpack_viewer(exported['prediction']),original)
            np.testing.assert_array_equal(unpack_viewer(exported['cleaned']),cleaned)
            np.testing.assert_array_equal(unpack_viewer(exported['expert']),g)
            self.assertEqual(exported['metadata']['cleanup']['metrics']['dice'],1.)
            self.assertTrue((root/'view/preview_3d.png').exists())
            self.assertTrue((root/'view/cleaned_mask.nii.gz').exists())
            report['primary_accepted_for_optional_development_view']=False
            report_path.write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaises(ValueError):cleanup_export(source,report_path,root/'rejected')
            self.assertFalse((root/'rejected').exists())


if __name__ == '__main__':
    unittest.main()
