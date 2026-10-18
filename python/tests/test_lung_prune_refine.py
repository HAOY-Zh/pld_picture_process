"""Train-only importance and exact channel provenance for new candidates."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_prune_refine import PLANS,FEATURES,retained_channels,importance_indices,feature_distillation
from lung_prune import prune_model,workload
from lung_unet import make_model,torch_module


class RefineTests(unittest.TestCase):
    def setUp(self):
        self.t=torch_module();self.t.set_num_threads(2);self.original=make_model(8,1).eval()

    def importance(self):
        return {n:np.arange(1,self.original.get_submodule(n).out_channels+1,dtype=float)
                for n in {n for p in PLANS.values() for n in p}}

    def test_all_plans_save_real_hardware_tiles_and_preserve_full_width_order(self):
        baseline=workload(self.original)
        for widths in PLANS.values():
            kept=retained_channels(self.original,widths,self.importance())
            model,actual=prune_model(self.original,widths,kept)
            self.assertEqual(kept,actual)
            self.assertEqual(kept['dec1.3'],list(range(8)))
            self.assertGreaterEqual(1-workload(model)['padded_input_beats']/baseline['padded_input_beats'],.10)
            with self.t.inference_mode():self.assertEqual(tuple(model(self.t.zeros(1,1,32,32)).shape),(1,1,32,32))

    def test_invalid_or_uninformative_importance_rejected(self):
        for value in [np.zeros(64),np.full(64,np.nan),np.full(64,-1.),np.ones(3)]:
            scores=self.importance();scores['neck.0']=value
            with self.assertRaises(ValueError):retained_channels(self.original,PLANS['retain_detail'],scores)

    def test_selected_features_match_original_channel_indices_and_backpropagate(self):
        kept=retained_channels(self.original,PLANS['fisher_moderate'],self.importance())
        teacher={};student={}
        for feature,conv in FEATURES.items():
            teacher[feature]=self.t.randn(2,self.original.get_submodule(conv).out_channels,8,8)
            student[feature]=(teacher[feature][:,kept[conv]].clone()+.1).requires_grad_()
        loss=feature_distillation(student,teacher,kept);loss.backward()
        self.assertGreater(loss.item(),0.)
        self.assertTrue(all(a.grad is not None and self.t.isfinite(a.grad).all() for a in student.values()))

    def test_importance_sampling_rejects_development_and_test_patients(self):
        for r in [dict(case='LCTSC-Test-S1-101',split='train'),dict(case='validation',split='validation')]:
            with self.assertRaises(ValueError):importance_indices(Path('.'),[r])
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);folder=root/'case';folder.mkdir()
            a=np.zeros((4,32,32),np.uint8);a[1,0,0]=1;a[2:]=1
            np.save(folder/'reference_native.npy',a)
            r=dict(case='LCTSC-Train-S2-001',split='train',cache_folder='case',slices=4)
            indices,weights,rows=importance_indices(root,[r])
            self.assertEqual(set(indices),{0,1,2,3});self.assertEqual(weights,[3.,6.,1.,1.])
            self.assertEqual(rows[0]['empty'],1)


if __name__=='__main__':unittest.main()
