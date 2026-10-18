"""Teacher split/geometry/gradient/paired-training safeguards, not clinical tests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_distill import (GuidedDataset,KIND,TEACHER_SHA256,checked_teacher,
    guidance_loss,paired,parser,restore_teacher,teacher_probabilities,train_records)
from lung_unet import file_sha256,make_model


class GeometryTests(unittest.TestCase):
    def test_vendor_box_order_soft_endpoint_restore_and_no_mutation(self):
        p=np.arange(16,dtype=np.float32).reshape(4,4)/15
        before=p.copy();box=[2,7,9,18]
        restored=restore_teacher(p,box,(12,21))
        np.testing.assert_array_equal(p,before)
        np.testing.assert_allclose(restored[2:9,7:18],ndi.zoom(p,[7/4,11/4],order=1))
        self.assertEqual(restored[2,7],p[0,0]);self.assertEqual(restored[8,17],p[-1,-1])
        self.assertEqual(restored[:2].sum(),0);self.assertTrue(((restored>0)&(restored<1)).any())
        for b in [[2,7,13,18],[2,7,9.5,18],[9,7,2,18]]:
            with self.assertRaises(ValueError):restore_teacher(p,b,(12,21))
        with self.assertRaises(ValueError):restore_teacher(p*np.nan,box,(12,21))

    def test_original_training_only_no_external_or_cropped_input(self):
        m=dict(hu_clip=[-1000,400],cases=[dict(case='LCTSC-Train-S2-001',split='train'),
            dict(case='LCTSC-Train-S2-002',split='validation')])
        self.assertEqual(len(train_records(m)),1)
        m['cases'][0]['case']='LCTSC-Test-S1-101'
        with self.assertRaises(ValueError):train_records(m)
        m['preprocessing']=dict(kind='body_crop_v1')
        with self.assertRaises(ValueError):train_records(m)
        args=parser().parse_args(['paired','--teacher-dataset','new','--outdir','new'])
        self.assertEqual((args.weight,args.confidence,args.epochs),(.25,.8,4))


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
class GuidanceTests(unittest.TestCase):
    def test_gradient_ignores_teacher_errors_and_ambiguous_pixels(self):
        import torch
        logits=torch.zeros((1,1,1,6),requires_grad=True)
        labels=torch.tensor([[[[1.,0.,1.,0.,1.,0.]]]])
        teacher=torch.tensor([[[[.95,.05,.05,.95,.6,.4]]]],requires_grad=True)
        loss=guidance_loss(logits,labels,teacher);loss.backward()
        self.assertLess(logits.grad[0,0,0,0],0);self.assertGreater(logits.grad[0,0,0,1],0)
        torch.testing.assert_close(logits.grad[...,2:],torch.zeros_like(logits.grad[...,2:]))
        self.assertIsNone(teacher.grad)
        empty=torch.zeros_like(logits,requires_grad=True)
        loss=guidance_loss(empty,labels,1-labels);loss.backward()
        self.assertEqual(loss.item(),0);self.assertEqual(empty.grad.abs().sum().item(),0)

    def test_class_balance_self_kl_and_invalid_inputs(self):
        import torch
        labels=torch.tensor([[[[1.,0.,0.,0.]]]])
        q=torch.tensor([[[[.9,.1,.1,.1]]]])
        z=torch.zeros_like(q,requires_grad=True)
        guidance_loss(z,labels,q).backward()
        self.assertAlmostEqual(abs(z.grad[0,0,0,0].item()),z.grad[...,1:].sum().item(),places=6)
        self.assertAlmostEqual(guidance_loss(torch.logit(q),labels,q).item(),0,places=6)
        with self.assertRaises(ValueError):guidance_loss(z,labels,q*float('nan'))
        with self.assertRaises(ValueError):guidance_loss(z,labels,q,confidence=.5)

    def test_paired_training_flip_ownership_hashes_and_leak_rejection(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cases=[];teacher_root=root/'teacher';teacher_root.mkdir()
            for i,split in enumerate(['train','validation']):
                name=f'LCTSC-Train-S2-{i+1:03d}';folder=root/name;folder.mkdir()
                mask=np.zeros((2,32,32),np.uint8);mask[:,8:24,7:21]=1
                image=np.broadcast_to(np.arange(32)/32,(2,32,32)).astype(np.float16).copy()
                np.save(folder/'images.npy',image);np.save(folder/'labels.npy',mask)
                np.save(folder/'reference_native.npy',mask)
                cases.append(dict(case=name,split=split,patient_key=str(i),frame_uid=str(i),
                    cache_folder=name,slices=2,native_shape=[2,32,32],spacing_zyx_mm=[1,1,1]))
            manifest=dict(status='complete',failures=[],image_size=32,hu_clip=[-1000,400],
                fingerprint='original',cases=cases,case_split={r['case']:r['split'] for r in cases})
            dataset=root/'dataset.json';dataset.write_text(json.dumps(manifest),encoding='utf-8')
            # Deliberately wrong teacher everywhere: gating must make both training arms IDENTICAL.
            probability=1-mask.astype(np.float16);probability_path=teacher_root/'probability.npy'
            np.save(probability_path,probability)
            teacher_record=dict(cases[0],probability_file='probability.npy',probability_sha256=file_sha256(probability_path))
            teacher=dict(status='complete',kind=KIND,teacher_sha256=TEACHER_SHA256,
                dataset_fingerprint='original',dataset_sha256=file_sha256(dataset),image_size=32,temperature=1,
                cases=[teacher_record],input_sha256={})
            teacher_path=teacher_root/'teacher_dataset.json';teacher_path.write_text(json.dumps(teacher),encoding='utf-8')
            checked_teacher(teacher_path,manifest)
            with GuidedDataset(root,[cases[0]],teacher_root,[teacher_record]) as data:
                with patch('lung_distill.random.random',return_value=0):x,y,q=data[1]
                np.testing.assert_array_equal(x.numpy()[0],image[1,:,::-1])
                np.testing.assert_array_equal(y.numpy()[0],mask[1,:,::-1])
                np.testing.assert_array_equal(q.numpy()[0],probability[1,:,::-1])
            self.assertEqual(q.sum().item(),probability[1].sum())
            data.close()
            with self.assertRaises(RuntimeError):data[0]
            torch.manual_seed(41);checkpoint=root/'original.pt'
            torch.save(dict(model=make_model(2).state_dict(),base_channels=2,image_size=32,
                hu_clip=[-1000,400],threshold=.5,context_slices=1,dataset_fingerprint='original',epoch=12),checkpoint)
            before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            args=parser().parse_args(['paired','--dataset',str(dataset),'--teacher-dataset',str(teacher_path),
                '--checkpoint',str(checkpoint),'--outdir',str(root/'run'),'--epochs','1','--device','cpu','--threads','2'])
            original_tf32=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
            paired(args)
            self.assertEqual(original_tf32,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))
            report=json.loads((root/'run'/'evaluation.json').read_text())
            self.assertTrue(report['protected_inputs_unchanged']);self.assertTrue(report['original_model_retained'])
            a=torch.load(root/'run'/'control'/'last.pt',weights_only=True)
            b=torch.load(root/'run'/'distilled'/'last.pt',weights_only=True)
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            for path,digest in before.items():self.assertEqual(file_sha256(path),digest)
            # Injecting a validation teacher record must fail before creating any run directory.
            teacher['cases'].append(dict(cases[1],probability_file='probability.npy'))
            teacher_path.write_text(json.dumps(teacher),encoding='utf-8');args.outdir=str(root/'rejected')
            with self.assertRaises(ValueError):paired(args)
            self.assertFalse((root/'rejected').exists())
            teacher['cases'].pop();teacher_path.write_text(json.dumps(teacher),encoding='utf-8')
            np.save(probability_path,np.zeros_like(probability))
            with self.assertRaisesRegex(ValueError,'hash'):checked_teacher(teacher_path,manifest)


@unittest.skipUnless(importlib.util.find_spec('lungmask') and importlib.util.find_spec('torch'),'Teacher dependency required')
class AdapterTests(unittest.TestCase):
    def test_teacher_soft_union_not_argmax_and_restore_vendor_geometry(self):
        import torch
        from types import SimpleNamespace
        from lungmask import utils
        from lung_unet import resize_stack
        class Fake(torch.nn.Module):
            def forward(self,x):
                logits=torch.zeros((len(x),3,256,256));logits[:,0]=.4
                return logits
        inferer=SimpleNamespace(model=Fake(),batch_size=1,device=torch.device('cpu'))
        hu=np.full((2,48,64),-1000,np.float32);hu[:,8:42,10:53]=30
        before=hu.copy();prepared,boxes=utils.preprocess(hu,resolution=[256,256])
        probabilities,saved_boxes=teacher_probabilities(inferer,hu,32)
        value=2/(np.exp(.4)+2)
        for i,box in enumerate(boxes):
            native=utils.reshape_mask(np.ones((256,256))*value,box,hu.shape[1:])
            np.testing.assert_allclose(probabilities[i],resize_stack(native[None],32,1)[0],atol=.0003)
            self.assertEqual(saved_boxes[i],np.asarray(box).tolist())
        np.testing.assert_array_equal(hu,before)
        self.assertTrue(((probabilities>0)&(probabilities<1)).any())


if __name__=='__main__':unittest.main()
