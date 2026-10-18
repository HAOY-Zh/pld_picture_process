"""Outer geometry, train-only distance maps and real paired-run safeguards."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_outer_contour import (ContourDataset,OuterValidator,boundary_loss,checked_distance,
    contour_gate,contour_metrics,distance_weight,outer_edge,paired,parser,prepare,reference_contour)
from lung_distill import KIND as TEACHER_KIND,TEACHER_SHA256
from lung_unet import file_sha256,make_model


class GeometryTests(unittest.TestCase):
    def test_closed_holes_ignored_only_diagnostically_open_notches_retained(self):
        ref=np.zeros((64,64),np.uint8);ref[8:56,8:56]=1
        pred=ref.copy();pred[25:35,25:35]=0;before=pred.copy()
        info=reference_contour(ref,[1,1]);m=contour_metrics(pred,info,[1,1])
        self.assertEqual(m['bad5_fraction'],0);self.assertEqual(m['hd95_mm'],0)
        np.testing.assert_array_equal(pred,before)
        pred[8:37,28:36]=0  # Open notch is NOT filled or ignored.
        m=contour_metrics(pred,info,[1,1])
        self.assertGreater(m['bad5_fraction'],0);self.assertGreater(m['hd95_mm'],5)
        a,_=outer_edge(pred);b,_=outer_edge(ref);self.assertGreater(np.count_nonzero(a&~b),0)

    def test_physical_distance_directional_percentile_missing_and_extra_components(self):
        ref=np.zeros((80,80),bool);ref[10:40,10:40]=1
        pred=np.zeros_like(ref);pred[20:50,10:40]=1
        a=contour_metrics(pred,reference_contour(ref,[1,1]),[1,1])
        b=contour_metrics(pred,reference_contour(ref,[2,2]),[2,2])
        self.assertAlmostEqual(b['hd95_mm'],2*a['hd95_mm'])
        self.assertGreater(b['bad5_fraction'],a['bad5_fraction'])
        empty=contour_metrics(np.zeros_like(ref),reference_contour(ref,[1,1]),[1,1])
        self.assertEqual(empty['status'],'missing_foreground');self.assertIsNone(empty['hd95_mm'])
        self.assertEqual(empty['bad5_fraction'],1)
        self.assertEqual(contour_metrics(pred,reference_contour(np.zeros_like(ref),[1,1]),[1,1])['status'],'empty_reference')
        pred=ref.copy();pred[65:75,65:75]=1
        self.assertGreater(contour_metrics(pred,reference_contour(ref,[1,1]),[1,1])['bad5_fraction'],0)
        with self.assertRaises(ValueError):reference_contour(ref,[0,1])

    def test_distance_weights_human_holes_empty_border_and_physical_units(self):
        ref=np.zeros((64,64),np.uint8);ref[8:56,8:56]=1;ref[25:30,25:30]=0
        before=ref.copy();w=distance_weight(ref,[1,1],10)
        np.testing.assert_array_equal(ref,before)
        self.assertEqual(w[8,30],0);self.assertEqual(w[35,35],1)
        self.assertEqual(w[26,26],0);self.assertEqual(w[0,0],1)
        w2=distance_weight(ref,[2,2],10);self.assertAlmostEqual(w2[10,40],2*w[10,40],places=6)
        self.assertEqual(distance_weight(np.zeros_like(ref),[1,1]).sum(),0)
        white=distance_weight(np.ones_like(ref),[1,1]);self.assertEqual(white[0,0],0)
        self.assertEqual(white[32,32],1)
        with self.assertRaises(ValueError):distance_weight(ref,[1,1],0)

    def test_new_dice_tolerance_and_outer_selection_gates(self):
        from copy import deepcopy
        base=dict(mean_case_dice=.966575711,false_positive=1000,
            slice_quality=dict(groups=dict(small=dict(mean_slice_dice=.5)),empty_false_positive_pixels=100,
                empty_slices_with_prediction=2),outer=dict(mean_case_bad5_fraction=.2,worst5_percent_bad5_mean=.5,
                main_missing_foreground=0,cases=[dict(case='one',bad5_fraction=.2)]))
        c=deepcopy(base);c['mean_case_dice']=base['mean_case_dice']*.995
        c['outer'].update(mean_case_bad5_fraction=.17,worst5_percent_bad5_mean=.45)
        c['outer']['cases'][0]['bad5_fraction']=.17
        self.assertTrue(contour_gate(c,base)['passed'])
        c['mean_case_dice']-=.000001;self.assertFalse(contour_gate(c,base)['passed'])
        with self.assertRaises(ValueError):contour_gate(c,base,.006)
        a=parser().parse_args(['paired','--distance-dataset','cache','--outdir','new'])
        self.assertEqual((a.epochs,a.max_dice_drop,a.boundary_weight),(8,.005,.5))
        with self.assertRaises(ValueError):OuterValidator('.', [dict(split='train')])


def fixture(root):
    rows=[]
    for i,split in enumerate(['train','validation']):
        name=f'LCTSC-Train-S2-{i+1:03d}';folder=root/name;folder.mkdir()
        mask=np.zeros((2,64,64),np.uint8);mask[:,10:54,12:52]=1;mask[:,25:29,25:29]=0
        image=np.broadcast_to(np.arange(64)/64,(2,64,64)).astype(np.float16).copy()
        np.save(folder/'images.npy',image);np.save(folder/'labels.npy',mask);np.save(folder/'reference_native.npy',mask)
        rows.append(dict(case=name,split=split,patient_key=str(i),frame_uid=str(i),cache_folder=name,
            slices=2,native_shape=[2,64,64],spacing_zyx_mm=[1,1,1]))
    m=dict(status='complete',failures=[],image_size=64,hu_clip=[-1000,400],fingerprint='original',cases=rows,
        case_split={r['case']:r['split'] for r in rows},held_out_test_cases=['LCTSC-Test-S1-101'])
    p=root/'dataset.json';p.write_text(json.dumps(m),encoding='utf-8')
    return p,m


class CacheTests(unittest.TestCase):
    def test_prepare_only_original_train_preserves_targets_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,m=fixture(root)
            before={str(x):file_sha256(x) for x in root.rglob('*') if x.is_file()}
            args=parser().parse_args(['prepare','--dataset',str(p),'--outdir',str(root/'distance')]);prepare(args)
            out=root/'distance'/'distance_dataset.json';_,cache=checked_distance(out,m,p)
            self.assertEqual(len(cache['cases']),1);self.assertEqual(cache['cases'][0]['split'],'train')
            self.assertEqual(cache['cases'][0]['student_spacing_yx_mm'],[1,1])
            for x,digest in before.items():self.assertEqual(file_sha256(x),digest)
            self.assertFalse((root/'distance'/m['cases'][1]['case']).exists())
            cache['cases'][0]['split']='validation';out.write_text(json.dumps(cache),encoding='utf-8')
            with self.assertRaises(ValueError):checked_distance(out,m,p)


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
class TrainingTests(unittest.TestCase):
    def test_gradient_spatial_weighting_and_zero_empty_auxiliary_loss(self):
        import torch
        labels=torch.tensor([[[[1.,1.,0.,0.]]]])
        weights=torch.tensor([[[[1.,.1,1.,0.]]]])
        z=torch.zeros_like(labels,requires_grad=True);loss=boundary_loss(z,labels,weights);loss.backward()
        self.assertLess(z.grad[0,0,0,0],0);self.assertGreater(z.grad[0,0,0,2],0)
        self.assertAlmostEqual(abs(z.grad[0,0,0,0].item()),10*abs(z.grad[0,0,0,1].item()),places=6)
        self.assertEqual(z.grad[0,0,0,3].item(),0)
        zero=torch.zeros_like(z,requires_grad=True);loss=boundary_loss(zero,labels,torch.zeros_like(weights));loss.backward()
        self.assertEqual(loss.item(),0);self.assertEqual(zero.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):boundary_loss(z,labels,weights*float('nan'))

    def test_real_paired_training_zero_coefficient_equivalence_and_test_exclusion(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,m=fixture(root);rows=m['cases']
            prepare(parser().parse_args(['prepare','--dataset',str(dataset),'--outdir',str(root/'distance')]))
            _,distance=checked_distance(root/'distance'/'distance_dataset.json',m,dataset)
            teacher_root=root/'teacher';teacher_root.mkdir()
            from lung_unet import mapped_npy
            with mapped_npy(root/rows[0]['case']/'labels.npy') as a:probability=a.astype(np.float16)
            np.save(teacher_root/'probability.npy',probability)
            tr=dict(rows[0],probability_file='probability.npy',probability_sha256=file_sha256(teacher_root/'probability.npy'))
            teacher=dict(status='complete',kind=TEACHER_KIND,teacher_sha256=TEACHER_SHA256,dataset_fingerprint='original',
                dataset_sha256=file_sha256(dataset),image_size=64,temperature=1,cases=[tr],input_sha256={})
            tp=teacher_root/'teacher_dataset.json';tp.write_text(json.dumps(teacher),encoding='utf-8')
            with ContourDataset(root,[rows[0]],teacher_root,[tr],root/'distance',distance['cases']) as data:
                with patch('lung_outer_contour.random.random',return_value=1):raw=data[0]
                with patch('lung_outer_contour.random.random',return_value=0):flipped=data[0]
                for a,b in zip(raw,flipped):torch.testing.assert_close(a.flip(-1),b)
            data.close();self.assertGreater(flipped[3].sum().item(),0)
            torch.manual_seed(51);checkpoint=root/'original.pt'
            torch.save(dict(model=make_model(2).state_dict(),base_channels=2,image_size=64,hu_clip=[-1000,400],
                threshold=.5,context_slices=1,dataset_fingerprint='original',epoch=12),checkpoint)
            before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            args=parser().parse_args(['paired','--dataset',str(dataset),'--teacher-dataset',str(tp),
                '--distance-dataset',str(root/'distance'/'distance_dataset.json'),'--checkpoint',str(checkpoint),
                '--outdir',str(root/'run'),'--epochs','1','--boundary-weight','0','--device','cpu','--threads','2'])
            previous=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
            with patch('lung_outer_contour.previews',return_value=[]):paired(args)
            self.assertEqual(previous,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))
            report=json.loads((root/'run'/'evaluation.json').read_text())
            self.assertTrue(report['protected_inputs_unchanged']);self.assertTrue(report['original_model_retained'])
            self.assertEqual(report['protocol']['held_out_test_cases'],['LCTSC-Test-S1-101'])
            self.assertEqual(report['baseline']['outer']['main_slices'],2)
            a=torch.load(root/'run'/'teacher_control'/'last.pt',weights_only=True)
            b=torch.load(root/'run'/'distance'/'last.pt',weights_only=True)
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            for p,digest in before.items():self.assertEqual(file_sha256(p),digest)
            teacher['cases'].append(dict(rows[1],probability_file='probability.npy'))
            tp.write_text(json.dumps(teacher),encoding='utf-8');args.outdir=str(root/'rejected')
            with self.assertRaises(ValueError):paired(args)
            self.assertFalse((root/'rejected').exists())


if __name__=='__main__':unittest.main()
