"""All-validation export, immutable inputs, no inference and safe local URLs."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_case_browser import build, case_payload, checked_inputs, checked_body_screen, parser, portal_export
from lung_volume_viewer import html_payload, unpack_viewer
from lung_unet import file_sha256
from lungseg.metrics import overlap_metrics
from test_lung_outer_contour import fixture


def browser_fixture(root):
    dataset,manifest=fixture(root)
    checkpoint=root/'frozen.pt';checkpoint.write_bytes(b'immutable checkpoint fixture; no inference')
    r=manifest['cases'][1]
    source=root/'source.dcm';source.write_bytes(b'fixture header')
    r['source_ct_files_sorted']=[str(source)]*r['slices']
    dataset.write_text(json.dumps(manifest),encoding='utf-8')
    screen=root/'screen';folder=screen/r['case'];folder.mkdir(parents=True)
    expert=np.load(root/r['cache_folder']/'reference_native.npy')!=0
    original=expert.copy();original[0,0,0]=1;cleaned=expert.copy()
    artifact=folder/'predictions_native.npz';np.savez_compressed(artifact,original=original,primary=cleaned)
    tracked=[dataset,checkpoint,root/r['cache_folder']/'images.npy',root/r['cache_folder']/'reference_native.npy']
    row=dict(case=r['case'],split='validation',shape=r['native_shape'],spacing_zyx_mm=r['spacing_zyx_mm'],
        prediction_sha256=file_sha256(artifact),reference_sha256=file_sha256(tracked[-1]),
        inference_ms=12.,cpu_analysis_ms=3.,scores={'original':overlap_metrics(original,expert),
        'primary':dict(overlap_metrics(cleaned,expert),removed_components=1)})
    report=dict(status='complete_development_screen',protected_inputs_unchanged=True,
        primary_accepted_for_optional_development_view=True,cases=[row],
        protocol=dict(validation_cases=[r['case']],model_sha256=file_sha256(checkpoint),model_epoch=12,
            primary={'max_island_ml':5.,'min_distance_mm':20.},input_sha256={str(p.resolve()):file_sha256(p) for p in tracked}))
    report_path=screen/'evaluation.json';report_path.write_text(json.dumps(report),encoding='utf-8')
    header=SimpleNamespace(ImagePositionPatient=[0,0,0],ImageOrientationPatient=[1,0,0,0,1,0])
    return dataset,checkpoint,report_path,manifest,report,header


class CaseBrowserTests(unittest.TestCase):
    def test_body_stage_transport_and_native_roi_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,checkpoint,screen,manifest,old,header=browser_fixture(root)
            record=manifest['cases'][1];case=record['case'];folder=root/'body'/case;folder.mkdir(parents=True)
            with np.load(screen.parent/case/'predictions_native.npz') as a:
                original,body=a['original'],a['primary']
            artifact=folder/'predictions_native.npz';np.savez_compressed(artifact,original=original,body=body,body_cleaned=body)
            plan=dict(native_yx=[64,64],box_y0_y1_x0_x1=[2,62,2,62],pad_top_bottom_left_right=[0,0,0,0],fallback_full_frame=False)
            geometry=folder/'body_geometry.json';geometry.write_text(json.dumps(dict(plans=[plan,plan])),encoding='utf-8')
            report=dict(status='complete_body_gate_development_screen',accepted_for_optional_view=True,protected_inputs_unchanged=True,
                protocol=dict(model_sha256=file_sha256(checkpoint),input_sha256={},parameters={'margin_mm':20,'closing_mm':3}),
                cases=[dict(case=case,split='validation',expert_outside_envelope_voxels=0,prediction_sha256=file_sha256(artifact),
                    body_geometry_sha256=file_sha256(geometry),body_gate_cpu_ms=1.,scores={'body':old['cases'][0]['scores']['primary'],'body_cleaned':old['cases'][0]['scores']['primary']})])
            path=root/'body/evaluation.json';path.write_text(json.dumps(report),encoding='utf-8')
            context=checked_body_screen(path,[record],file_sha256(checkpoint),{})
            with patch('lung_case_browser.load_cached_source',return_value=np.zeros((2,64,64),np.float32)),patch('lung_case_browser.pydicom.dcmread',return_value=header):
                data=case_payload(root,record,old['cases'][0],screen,old,context)
                np.testing.assert_array_equal(unpack_viewer(data['variants']['body']),body)
                np.testing.assert_array_equal(unpack_viewer(data['variants']['body_cleaned']),body)
                self.assertEqual(data['metadata']['variants']['body']['removed_voxels'],1)
                self.assertEqual(data['body_geometry']['plans'][0]['box_y0_y1_x0_x1'],[2,62,2,62])
                # A different ROI must not silently display the same constrained mask.
                plan['box_y0_y1_x0_x1']=[0,64,0,64]
                geometry.write_text(json.dumps(dict(plans=[plan,plan])),encoding='utf-8')
                with self.assertRaises(ValueError):case_payload(root,record,old['cases'][0],screen,old,context)

    def test_unaccepted_body_screen_never_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'evaluation.json'
            path.write_text(json.dumps(dict(status='complete_body_gate_development_screen',accepted_for_optional_view=False)),encoding='utf-8')
            with self.assertRaises(ValueError):checked_body_screen(path,[],'checkpoint',{})

    def test_portal_only_local_urls_and_safe_inline_json(self):
        name='LCTSC-Train-S3-008'
        catalog=dict(default_case=name,cases=[dict(case=name,url=f'cases/{name}/viewer.html',bytes=1)],
                     warning='</script><script>bad()</script>')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.html';portal_export(catalog,path)
            text=path.read_text(encoding='utf-8')
            self.assertNotIn('</script><script>bad()</script>',text)
            self.assertNotIn('__CATALOG__',text)
            self.assertNotIn('<script src=',text)
            self.assertIn('event.source!==currentFrame.contentWindow',text)
            self.assertIn('host.replaceChildren()',text)
            catalog['cases'][0]['url']='https://untrusted.invalid/viewer.html'
            with self.assertRaises(ValueError):portal_export(catalog,path)

    def test_rejects_omitted_validation_rows_and_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,checkpoint,screen,manifest,report,_=browser_fixture(root)
            checked_inputs(dataset,screen,checkpoint)
            report['cases']=[];screen.write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaises(ValueError):checked_inputs(dataset,screen,checkpoint)
            report['cases']=[dict(case=manifest['cases'][0]['case'],split='train')]
            screen.write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaises(ValueError):checked_inputs(dataset,screen,checkpoint)

    def test_changed_checkpoint_and_unaccepted_screen_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,checkpoint,screen,_,report,_=browser_fixture(root)
            report['primary_accepted_for_optional_development_view']=False
            screen.write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaises(ValueError):checked_inputs(dataset,screen,checkpoint)
            report['primary_accepted_for_optional_development_view']=True
            screen.write_text(json.dumps(report),encoding='utf-8');checkpoint.write_bytes(b'changed weights')
            with self.assertRaises(ValueError):checked_inputs(dataset,screen,checkpoint)

    def test_real_export_reuses_masks_native_geometry_and_preserves_all_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,checkpoint,screen,manifest,report,header=browser_fixture(root)
            before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            case=manifest['cases'][1]['case']
            args=parser().parse_args(['--dataset',str(dataset),'--checkpoint',str(checkpoint),'--screen',str(screen),
                '--default-case',case,'--outdir',str(root/'browser')])
            with patch('lung_case_browser.load_cached_source',return_value=np.zeros((2,64,64),np.float32)), \
                 patch('lung_case_browser.pydicom.dcmread',return_value=header), \
                 patch('lung_unet._case_prediction',side_effect=AssertionError('Must not rerun model')):
                build(args)
            for p,h in before.items():self.assertEqual(file_sha256(p),h)
            result=json.loads((root/'browser/report.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'complete_offline_validation_case_browser')
            self.assertTrue(result['protected_inputs_unchanged']);self.assertFalse(result['inference_rerun'])
            self.assertEqual([e['case'] for e in result['cases']],[case])
            _,data=html_payload((root/'browser/cases'/case/'viewer.html').read_text(encoding='utf-8'))
            with np.load(screen.parent/case/'predictions_native.npz') as values:
                np.testing.assert_array_equal(unpack_viewer(data['prediction']),values['original'])
                np.testing.assert_array_equal(unpack_viewer(data['cleaned']),values['primary'])
            self.assertEqual(data['metadata']['cleanup']['removed_voxels'],1)
            self.assertEqual(data['metadata']['initial_slice'],0)
            self.assertFalse((root/'browser/cases'/manifest['cases'][0]['case']).exists())

    def test_native_payload_rejects_score_or_subset_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,_,screen,manifest,report,header=browser_fixture(root)
            record=manifest['cases'][1];row=report['cases'][0]
            row['scores']['primary']['dice']=.1
            with patch('lung_case_browser.load_cached_source',return_value=np.zeros((2,64,64),np.float32)), \
                 patch('lung_case_browser.pydicom.dcmread',return_value=header):
                with self.assertRaises(ValueError):case_payload(root,record,row,screen,report)


if __name__=='__main__':unittest.main()
