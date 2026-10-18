"""Geometry and paired-experiment safeguards, not clinical accuracy tests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_body_crop import (_plan, acceptance, body_plan, checked_plan, crop_resize,
    crop_prediction, paired, parser, prepare, restore_probability)
from lung_unet import file_sha256, make_model, read_manifest


class GeometryTests(unittest.TestCase):
    def test_body_detector_uses_ct_only_excludes_detached_bed_and_preserves_input(self):
        hu=np.full((128,128),-1000.,np.float32)
        hu[22:96,31:99]=30
        hu[35:80,42:88]=-850  # Lung interior remains within temporary filled body.
        hu[118:122,5:123]=100
        before=hu.copy()
        plan=body_plan(hu,[1,1],margin_mm=4,closing_mm=2)
        np.testing.assert_array_equal(hu,before)
        y0,y1,x0,x1=plan['box_y0_y1_x0_x1']
        self.assertLessEqual(y0,22);self.assertGreaterEqual(y1,96)
        self.assertLessEqual(x0,31);self.assertGreaterEqual(x1,99)
        self.assertLess(y1,118)
        self.assertEqual(y1-y0,x1-x0)
        self.assertFalse(plan['fallback_full_frame'])
        self.assertNotIn('reference',body_plan.__code__.co_varnames)

    def test_uniform_empty_tiny_and_clipped_edge_are_conservative(self):
        for value in [-1000.,0.]:
            plan=body_plan(np.full((32,48),value),[1,1])
            self.assertEqual(plan['box_y0_y1_x0_x1'],[0,32,0,48])
            self.assertTrue(plan['fallback_full_frame'])
        hu=np.full((64,64),-1000.);hu[30,30]=0
        self.assertTrue(body_plan(hu,[1,1])['fallback_full_frame'])
        hu[:35,:36]=0
        self.assertEqual(body_plan(hu,[1,1],margin_mm=2)['box_y0_y1_x0_x1'][:1],[0])
        for settings in [dict(spacing_yx=[0,1]),dict(spacing_yx=[1,1],margin_mm=-1),
                         dict(spacing_yx=[1,1],closing_mm=float('nan'))]:
            with self.assertRaises(ValueError):body_plan(hu,**settings)
        hu[0,0]=np.nan
        with self.assertRaises(ValueError):body_plan(hu,[1,1])

    def test_rectangular_letterbox_and_restore_coordinates_no_label_hole_filling(self):
        plan=_plan((32,48),[0,32,0,48],True)
        self.assertEqual(plan['pad_top_bottom_left_right'],[8,8,0,0])
        image=np.arange(32*48).reshape(32,48).astype(np.float32)
        square=crop_resize(image,plan,48,0,-1000)
        np.testing.assert_array_equal(square[8:40],image)
        np.testing.assert_array_equal(square[:8],-1000)
        mask=np.zeros((32,48),np.uint8);mask[9:24,12:37]=1;mask[15:19,20:24]=0
        small=crop_resize(mask,plan,48,0)
        restored=restore_probability(small,plan)
        np.testing.assert_array_equal(restored,mask)
        sub=_plan((64,80),[12,44,23,55],False)
        restored=restore_probability(np.ones((32,32)),sub)
        expected=np.zeros((64,80));expected[12:44,23:55]=1
        np.testing.assert_array_equal(restored,expected)
        invalid=dict(sub,box_y0_y1_x0_x1=[-1,44,23,55])
        with self.assertRaises(ValueError):checked_plan(invalid,(64,80))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
    def test_restore_matches_torch_pixel_centers_and_native_crop_prediction(self):
        import torch
        rng=np.random.default_rng(11)
        probability=rng.random((32,32)).astype(np.float32)
        for side in [32,43,65]:
            plan=_plan((side,side),[0,side,0,side],True)
            expected=torch.nn.functional.interpolate(torch.from_numpy(probability)[None,None],
                size=(side,side),mode='bilinear',align_corners=False)[0,0].numpy()
            np.testing.assert_allclose(restore_probability(probability,plan),expected,atol=4e-6)
        class One(torch.nn.Module):
            def forward(self,x):return torch.full_like(x,10)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);np.save(root/'images.npy',np.zeros((2,32,32),np.float16))
            plans=[_plan((64,80),[12,44,23,55],False),_plan((64,80),[8,40,30,62],False)]
            (root/'crop_geometry.json').write_text(json.dumps(plans),encoding='utf-8')
            row=dict(cache_folder='.',crop_geometry_file='crop_geometry.json',native_shape=[2,64,80])
            result,_=crop_prediction(One(),root,row,torch.device('cpu'),2)
            self.assertEqual(int(result.sum()),2*32*32)
            self.assertTrue(result[0,12:44,23:55].all());self.assertFalse(result[0,:12].any())

    def test_native_coverage_guard_and_incomplete_cache_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cases=[]
            hu=np.full((1,64,64),-1000.,np.float32);hu[:,20:44,20:44]=0
            for i,split in enumerate(['train','validation']):
                folder=root/f'case{i}';folder.mkdir()
                ref=np.zeros((1,64,64),np.uint8);ref[:,25:35,25:35]=1
                if i==0:ref[:,0,0]=1  # Force crop failure; do not correct box using label.
                np.save(folder/'reference_native.npy',ref)
                fake=folder/'ct.dcm';fake.write_bytes(b'synthetic protected source')
                cases.append(dict(case=f'LCTSC-Train-S2-{i+1:03d}',split=split,patient_key=str(i),
                    frame_uid=str(i),cache_folder=folder.name,source_ct_files_sorted=[str(fake)],
                    spacing_zyx_mm=[1,1,1],native_shape=[1,64,64],slices=1))
            manifest=root/'dataset.json'
            manifest.write_text(json.dumps(dict(status='complete',failures=[],cases=cases,
                image_size=32,hu_clip=[-1000,400],fingerprint='original',case_split={r['case']:r['split'] for r in cases})),encoding='utf-8')
            before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            args=parser().parse_args(['prepare','--dataset',str(manifest),'--outdir',str(root/'bad'),
                '--margin-mm','1'])
            with patch('lung_body_crop.load_cached_source',return_value=hu),patch('lung_body_crop.save_crop_preview'):
                with self.assertRaisesRegex(ValueError,'excluded'):prepare(args)
            report=json.loads((root/'bad'/'dataset.json').read_text())
            self.assertEqual(report['status'],'incomplete')
            with self.assertRaises(ValueError):read_manifest(root/'bad'/'dataset.json',allow_body_crop=True)
            for p,digest in before.items():self.assertEqual(file_sha256(p),digest)
            # Remove only the intentionally failing label, create a NEW experiment directory.
            ref=np.zeros((1,64,64),np.uint8);ref[:,25:35,25:35]=1
            np.save(root/'case0'/'reference_native.npy',ref)
            args.outdir=str(root/'good')
            with patch('lung_body_crop.load_cached_source',return_value=hu),patch('lung_body_crop.save_crop_preview'):
                prepare(args)
            output=root/'good'/'dataset.json'
            with self.assertRaisesRegex(ValueError,'full-frame readers'):read_manifest(output)
            _,report=read_manifest(output,allow_body_crop=True)
            self.assertEqual(report['case_split'],json.loads(manifest.read_text())['case_split'])
            self.assertTrue(report['protected_inputs_unchanged'])
            for row in report['cases']:
                self.assertEqual(row['crop_audit']['excluded_expert_pixels'],0)
                self.assertEqual(file_sha256(root/'good'/row['cache_folder']/'reference_native.npy'),row['native_reference_sha256'])

    def test_fixed_gates_and_no_external_or_teacher_switch(self):
        args=parser().parse_args(['paired','--cropped-dataset','cache','--outdir','new'])
        self.assertEqual(args.epochs,4);self.assertEqual(args.lr,.0001)
        base=dict(mean_case_dice=.96,false_negative=1000,false_positive=1000,
            slice_quality=dict(groups=dict(small=dict(mean_slice_dice=.5)),empty_false_positive_pixels=100,
                empty_slices_with_prediction=2))
        better=json.loads(json.dumps(base));better['false_negative']=900
        self.assertTrue(acceptance(better,base)['passed'])
        better['slice_quality']['empty_slices_with_prediction']=3
        self.assertFalse(acceptance(better,base)['passed'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
class ExperimentTests(unittest.TestCase):
    def test_paired_real_training_keeps_original_and_rejects_changed_expert_target(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cases=[]
            full=root/'full';crop=root/'crop';full.mkdir();crop.mkdir()
            for i,split in enumerate(['train','validation']):
                name=f'LCTSC-Train-S2-{i+1:03d}'
                row=dict(case=name,split=split,patient_key=str(i),frame_uid=str(i),cache_folder=name,
                    slices=2,native_shape=[2,32,32],spacing_zyx_mm=[1,1,1])
                cases.append(row)
                mask=np.zeros((2,32,32),np.uint8);mask[:,8:24,8:24]=1
                for cache in [full,crop]:
                    folder=cache/name;folder.mkdir()
                    np.save(folder/'images.npy',np.full((2,32,32),.6,np.float16))
                    np.save(folder/'labels.npy',mask);np.save(folder/'reference_native.npy',mask)
                plans=[_plan((32,32),[0,32,0,32],True)]*2
                (crop/name/'crop_geometry.json').write_text(json.dumps(plans),encoding='utf-8')
            manifest=dict(status='complete',failures=[],image_size=32,hu_clip=[-1000,400],
                fingerprint='full',case_split={r['case']:r['split'] for r in cases},cases=cases)
            (full/'dataset.json').write_text(json.dumps(manifest),encoding='utf-8')
            manifest=dict(manifest,source_fingerprint='full',fingerprint='crop',preprocessing=dict(kind='body_crop_v1'))
            manifest['cases']=[dict(r,crop_geometry_file='crop_geometry.json',
                native_reference_sha256=file_sha256(full/r['case']/'reference_native.npy'),
                crop_audit=dict(excluded_expert_pixels=0)) for r in cases]
            (crop/'dataset.json').write_text(json.dumps(manifest),encoding='utf-8')
            torch.manual_seed(31)
            checkpoint=root/'original.pt'
            torch.save(dict(model=make_model(2,1).state_dict(),dataset_fingerprint='full',context_slices=1,
                image_size=32,hu_clip=[-1000,400],threshold=.5,base_channels=2,epoch=12),checkpoint)
            before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            args=parser().parse_args(['paired','--dataset',str(full/'dataset.json'),
                '--cropped-dataset',str(crop/'dataset.json'),'--checkpoint',str(checkpoint),
                '--outdir',str(root/'run'),'--epochs','1','--device','cpu','--threads','2'])
            original_tf32=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
            paired(args)
            self.assertEqual(original_tf32,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))
            report=json.loads((root/'run'/'evaluation.json').read_text())
            self.assertTrue(report['protected_inputs_unchanged']);self.assertTrue(report['original_model_retained'])
            self.assertEqual(set(report['arms']),{'full_control','body_crop'})
            a=torch.load(root/'run'/'full_control'/'last.pt',weights_only=True)
            b=torch.load(root/'run'/'body_crop'/'last.pt',weights_only=True)
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            for p,digest in before.items():self.assertEqual(file_sha256(p),digest)
            np.save(crop/cases[0]['case']/'reference_native.npy',np.zeros((2,32,32),np.uint8))
            args.outdir=str(root/'rejected')
            with self.assertRaisesRegex(ValueError,'target changed'):paired(args)
            self.assertFalse((root/'rejected').exists())


if __name__=='__main__':unittest.main()
