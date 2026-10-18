"""Train-only emphasis tests; these are not medical accuracy tests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_refine import acceptance,emphasis_weights,local_brightness,parser,refinement_loss,run
from lung_unet import file_sha256,make_model,segmentation_loss


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class RefineTests(unittest.TestCase):
    def test_brightness_patch_preserves_labels_background_and_boundary_with_private_rng(self):
        import torch
        images=torch.full((3,1,64,64),.2)
        labels=torch.zeros_like(images)
        labels[0,0,8:56,8:56]=1
        labels[1,0,10:54,10:54]=1
        original_images,original_labels=images.clone(),labels.clone()
        rng_before=torch.get_rng_state().clone()
        result,counts=local_brightness(images,labels,torch.Generator().manual_seed(42),1,(300,300),(16,16))
        repeat,_=local_brightness(images,labels,torch.Generator().manual_seed(42),1,(300,300),(16,16))
        torch.testing.assert_close(result,repeat)
        torch.testing.assert_close(images,original_images)
        torch.testing.assert_close(labels,original_labels)
        torch.testing.assert_close(torch.get_rng_state(),rng_before)
        torch.testing.assert_close(result[labels==0],images[labels==0])
        protected=torch.nn.functional.avg_pool2d(labels,9,stride=1,padding=4)<1-1e-6
        torch.testing.assert_close(result[protected],images[protected])
        self.assertEqual(counts['augmented_slices'],2)
        self.assertGreater(counts['augmented_pixels'],0)
        self.assertTrue(((result>=images)&(result<=1)).all())
        unchanged,counts=local_brightness(images,labels,torch.Generator().manual_seed(42),0)
        torch.testing.assert_close(unchanged,images)
        self.assertEqual(counts['augmented_pixels'],0)

    def test_brightness_clipping_thin_empty_masks_and_invalid_parameters(self):
        import torch
        images=torch.full((2,1,32,32),.95)
        labels=torch.zeros_like(images)
        labels[0,0,4:28,4:28]=1
        labels[1,0,16,:]=1
        result,counts=local_brightness(images,labels,torch.Generator().manual_seed(2),1)
        self.assertEqual(float(result.max()),1.)
        torch.testing.assert_close(result[1],images[1])
        self.assertEqual(counts['augmented_slices'],1)
        for settings in [{'probability':2},{'delta_hu':(-10,300)},{'radius':(20,10)},
                         {'probability':float('nan')}]:
            with self.assertRaises(ValueError):
                local_brightness(images,labels,torch.Generator(),**settings)

    def test_only_bright_expert_positive_is_emphasized_without_label_mutation(self):
        import torch
        images=torch.tensor([[[[.1,.8],[.8,.2]]]])
        target=torch.tensor([[[[1.,1.],[0.,1.]]]])
        before=target.clone()
        weights=emphasis_weights(images,target)
        torch.testing.assert_close(weights,torch.tensor([[[[1.,2.],[1.,1.]]]]))
        torch.testing.assert_close(target,before)
        loss=refinement_loss(torch.zeros_like(target,requires_grad=True),target,images)
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            emphasis_weights(images,target,0)
        with self.assertRaises(ValueError):
            emphasis_weights(images,target,2,float('nan'))

    def test_weight_one_matches_original_loss_and_unchanged_architecture_backpropagates(self):
        import torch
        torch.manual_seed(123)
        model=make_model(4,1)
        images=torch.rand(2,1,32,32)
        labels=(torch.rand_like(images)>.4).float()
        logits=model(images)
        torch.testing.assert_close(refinement_loss(logits,labels,images,1),segmentation_loss(logits,labels))
        before=next(model.parameters()).detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        refinement_loss(logits,labels,images).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before,next(model.parameters())))
        self.assertEqual(model.context_slices,1)

    def test_predeclared_gates_reject_foreground_expansion_and_empty_layer_regression(self):
        summary=dict(mean_case_dice=.96,false_positive=1000,false_negative=2000,
            bright_false_negative=1000,slice_quality={'groups':{'small':{'mean_slice_dice':.5}},
            'empty_false_positive_pixels':100,'empty_slices_with_prediction':2})
        import copy
        better=copy.deepcopy(summary)
        better['bright_false_negative']=800
        better['false_negative']=1800
        self.assertTrue(acceptance(better,summary)['passed'])
        better['false_positive']=1100
        self.assertFalse(acceptance(better,summary)['passed'])
        better['false_positive']=1000
        better['slice_quality']['empty_slices_with_prediction']=3
        self.assertFalse(acceptance(better,summary)['passed'])

    def test_bounded_defaults_and_no_test_case_selection(self):
        args=parser().parse_args(['--outdir','new'])
        self.assertEqual(args.epochs,4)
        self.assertEqual(args.weight,2)
        options={flag for action in parser()._actions for flag in action.option_strings}
        self.assertNotIn('--test',options)
        self.assertNotIn('--case',options)

    def test_real_small_training_keeps_initial_checkpoint_and_labels_and_restores_backend(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            model=make_model(4,1)
            checkpoint=root/'original.pt'
            torch.save(dict(model=model.state_dict(),dataset_fingerprint='synthetic',image_size=32,
                context_slices=1,hu_clip=[-1000,400],threshold=.5,base_channels=4,epoch=12,seed=123),checkpoint)
            records=[]
            for number,split in enumerate(['train','validation'],1):
                name=f'LCTSC-Train-S2-{number:03d}'
                folder=root/name
                folder.mkdir()
                labels=np.zeros((2,32,32),np.uint8)
                labels[:,7:25,9:23]=1
                images=np.full_like(labels,.1,dtype=np.float16)
                images[labels!=0]=.7
                np.save(folder/'images.npy',images)
                np.save(folder/'labels.npy',labels)
                np.save(folder/'reference_native.npy',labels)
                records.append(dict(case=name,split=split,patient_key=f'patient-{number}',
                    cache_folder=name,slices=2,native_shape=[2,32,32]))
            manifest=root/'dataset.json'
            manifest.write_text(json.dumps(dict(status='complete',failures=[],cases=records,
                fingerprint='synthetic',image_size=32,hu_clip=[-1000,400])),encoding='utf-8')
            hashes={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            settings=parser().parse_args(['--dataset',str(manifest),'--checkpoint',str(checkpoint),
                '--outdir',str(root/'new'),'--epochs','1','--device','cpu','--threads','2'])
            original_tf32=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
            run(settings)
            self.assertEqual(original_tf32,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))
            report=json.loads((root/'new'/'evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'complete_exploratory_validation')
            self.assertEqual(report['selected_epoch'],1)
            self.assertTrue(report['original_model_retained'])
            self.assertEqual(report['previews'],[])
            for path,digest in hashes.items():
                self.assertEqual(file_sha256(path),digest)
            with self.assertRaises(ValueError):
                run(settings)
            settings.outdir=str(root/'augmented')
            settings.weight=1
            settings.augmentation='local-brightness'
            settings.augmentation_probability=1
            run(settings)
            augmented=json.loads((root/'augmented'/'evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(augmented['history'][0]['augmentation_counts']['augmented_slices'],2)
            for path,digest in hashes.items():
                self.assertEqual(file_sha256(path),digest)


if __name__=='__main__':
    unittest.main()
