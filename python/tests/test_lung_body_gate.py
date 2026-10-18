"""2D output-only gate: margins, unknown slice inference and frozen data safety."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_body_gate import gate_slice,gate_volume,predict_slice_with_body,evaluate_native,parser,run
from lung_unet import file_sha256,torch_module
from test_lung_case_browser import browser_fixture


class BodyGateTests(unittest.TestCase):
    def test_table_prediction_removed_body_kept_inputs_unchanged(self):
        hu=np.full((256,256),-1000.,np.float32)
        hu[35:160,65:190]=0;hu[70:130,90:160]=-900
        hu[225:230,20:236]=100;hu[210:225,100:150]=-700
        p=np.zeros_like(hu,bool);p[80:120,100:150]=1;p[212:220,105:145]=1
        before_hu,before_p=hu.copy(),p.copy()
        result,plan=gate_slice(hu,p,[1,1])
        self.assertTrue(result[90,120]);self.assertFalse(result[215,120])
        self.assertFalse(plan['fallback_full_frame'])
        np.testing.assert_array_equal(hu,before_hu);np.testing.assert_array_equal(p,before_p)

    def test_fallback_uniform_and_tiny_slice_fail_open(self):
        for hu in [np.full((64,64),-1000.),np.zeros((64,64))]:
            p=np.ones((64,64),bool);result,plan=gate_slice(hu,p,[1,1])
            self.assertTrue(plan['fallback_full_frame']);np.testing.assert_array_equal(result,p)
        hu=np.full((64,64),-1000.);hu[30:32,30:32]=0
        p=np.ones((64,64),bool);result,plan=gate_slice(hu,p,[1,1])
        self.assertTrue(plan['fallback_full_frame']);np.testing.assert_array_equal(result,p)

    def test_margin_physical_units_and_no_prediction_expansion(self):
        hu=np.full((128,128),-1000.);hu[35:85,35:85]=0
        p=np.ones((128,128),bool)
        a,_=gate_slice(hu,p,[1,1],margin_mm=20)
        b,_=gate_slice(hu,p,[2,2],margin_mm=20)
        self.assertGreater(a.sum(),b.sum())
        sparse=np.zeros((128,128),bool);sparse[60,60]=1
        result,_=gate_slice(hu,sparse,[1,1]);self.assertEqual(result.sum(),1)

    def test_slice_stack_invalid_shape_values_and_gt_only_scoring(self):
        hu=np.zeros((2,64,64));p=np.ones_like(hu,bool)
        result,plans,_=gate_volume(hu,p,[1,1])
        self.assertEqual(len(plans),2);np.testing.assert_array_equal(result,p)
        with self.assertRaises(ValueError):gate_slice(hu[0],np.ones((32,32)),[1,1])
        with self.assertRaises(ValueError):gate_slice(hu[0],np.ones((64,64))*.5,[1,1])
        ref=np.zeros_like(p);ref[:,5:10,5:10]=1
        constrained=p.copy();constrained[:,5,5]=0
        score=evaluate_native(constrained,ref,p)
        self.assertEqual(score['removed_true_lung_voxels'],2)

    def test_unknown_single_slice_api_preserves_normalization_and_no_gt(self):
        torch=torch_module();torch.set_num_threads(1)
        class Fixed(torch.nn.Module):
            context_slices=1
            def forward(self,x):
                self.seen=x.detach().cpu().numpy()
                return torch.ones((1,1,x.shape[-2],x.shape[-1]),device=x.device)*10
        model=Fixed().eval()
        hu=np.full((64,64),-1000.);hu[10:54,10:54]=0
        out=predict_slice_with_body(model,hu,[1,1],torch.device('cpu'),64)
        self.assertEqual(model.seen.shape,(1,1,64,64));self.assertEqual(float(model.seen.min()),0.)
        self.assertTrue(out['original'].all());self.assertEqual(out['body'].shape,hu.shape)
        model.train()
        with self.assertRaises(ValueError):predict_slice_with_body(model,hu,[1,1],torch.device('cpu'),64)

    def test_real_development_run_retains_all_original_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset,checkpoint,screen,manifest,_,_=browser_fixture(root)
            before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            args=parser().parse_args(['--dataset',str(dataset),'--screen',str(screen),'--checkpoint',str(checkpoint),'--outdir',str(root/'gate')])
            with patch('lung_body_gate.load_cached_source',return_value=np.zeros((2,64,64),np.float32)),patch('lung_body_gate.save_preview'):
                run(args)
            for p,h in before.items():self.assertEqual(file_sha256(p),h)
            report=json.loads((root/'gate/evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'complete_body_gate_development_screen')
            self.assertTrue(report['protected_inputs_unchanged'])
            self.assertFalse(report['protocol']['input_crop']);self.assertFalse(report['protocol']['inference_rerun'])
            self.assertEqual([r['case'] for r in report['cases']],[manifest['cases'][1]['case']])


if __name__=='__main__':unittest.main()
