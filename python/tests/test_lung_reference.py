"""Numerical boundary profiles and pinned vendor-adapter tests, not clinical validation."""
import importlib.util
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_reference import TARGETS,boundary_profiles,parser,reference_outputs,refresh_profiles
from lung_unet import file_sha256
from lungseg.metrics import overlap_metrics


class ProfileTests(unittest.TestCase):
    def test_native_hu_jump_and_internal_wrong_edge_without_mask_mutation(self):
        reference=np.zeros((80,80),bool);reference[10:70,10:70]=1
        prediction=reference.copy();prediction[32:48,32:48]=0
        hu=np.full(reference.shape,50.,np.float32);hu[reference]=-800;hu[32:48,32:48]=-200
        original_ref,original_pred,original_hu=reference.copy(),prediction.copy(),hu.copy()
        groups=boundary_profiles(hu,reference,prediction,[1.,1.])
        self.assertGreater(groups['expert_outer']['summary']['sampled_points'],0)
        self.assertGreater(groups['wrong_internal']['summary']['sampled_points'],0)
        self.assertAlmostEqual(groups['expert_outer']['summary']['median_signed_jump_hu'],850,delta=1)
        self.assertAlmostEqual(groups['wrong_internal']['summary']['median_signed_jump_hu'],600,delta=1)
        np.testing.assert_array_equal(reference,original_ref)
        np.testing.assert_array_equal(prediction,original_pred)
        np.testing.assert_array_equal(hu,original_hu)

    def test_expert_hole_not_misclassified_and_samples_physical_not_pixel_distance(self):
        reference=np.zeros((80,80),bool);reference[10:70,10:70]=1;reference[30:40,30:40]=0
        hu=np.where(reference,-800.,50.).astype(np.float32)
        groups=boundary_profiles(hu,reference,reference,[.5,2.],max_points=12)
        self.assertEqual(groups['wrong_internal']['summary']['sampled_points'],0)
        self.assertEqual(groups['expert_outer']['summary']['sampled_points'],12)
        coords=groups['expert_outer']['coordinates_yx']
        self.assertFalse(((coords[:,0]>25)&(coords[:,0]<45)&(coords[:,1]>25)&(coords[:,1]<45)).any())
        np.testing.assert_allclose(np.linalg.norm(groups['expert_outer']['normals_yx'],axis=1),1)
        empty=boundary_profiles(hu,reference,np.zeros_like(reference),[1,1])
        self.assertIsNone(empty['wrong_internal']['summary']['median_profile_hu'])
        with self.assertRaises(ValueError):boundary_profiles(hu,reference,reference,[0,1])
        with self.assertRaises(ValueError):boundary_profiles(hu*np.nan,reference,reference,[1,1])

    def test_fixed_targets_no_training_case_or_threshold_search(self):
        self.assertEqual(len(TARGETS),4)
        self.assertEqual(TARGETS[0],('external','coronacases_003',80))
        options={flag for action in parser()._actions for flag in action.option_strings}
        for flag in ['--train','--case','--threshold','--epochs']:self.assertNotIn(flag,options)

    def test_internal_profiles_cannot_cross_into_true_exterior(self):
        from scipy import ndimage as ndi
        from lung_reference import DISTANCES_MM
        reference=np.zeros((64,64),bool);reference[5:59,5:59]=1
        prediction=reference.copy();prediction[:,56:]=0
        hu=np.where(reference,-800.,50.).astype(np.float32)
        group=boundary_profiles(hu,reference,prediction,[1.,1.])['wrong_internal']
        coords=group['coordinates_yx'][:,None,:]+DISTANCES_MM[None,:,None]*group['normals_yx'][:,None,:]
        values=ndi.map_coordinates(reference.astype(np.float32),[coords[:,:,0],coords[:,:,1]],order=1)
        self.assertTrue((values>=1-1e-6).all())

    def test_replot_uses_saved_metrics_protects_sources_and_rejects_incomplete_report(self):
        import lung_reference
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir()
            original=root/'original.pt';original.write_bytes(b'protected baseline')
            source_report=source/'evaluation.json'
            report={'status':'complete_fixed_difficult_slice_comparison','cases':[],
                'original_model_retained':True,'protocol':{'targets':[list(t) for t in TARGETS],
                'input_sha256':{str(original.resolve()):file_sha256(original),
                    str(Path(lung_reference.__file__).resolve()):'historical-code-sha'}}}
            expert=np.zeros((64,64),bool);expert[8:56,8:56]=1
            old=expert.copy();old[24:40,24:40]=0
            hu=np.where(expert,-800,50).astype(np.float32)
            for cohort,name,index in TARGETS:
                folder=source/name/f'slice_{index:03d}';folder.mkdir(parents=True)
                np.savez_compressed(folder/'slice_masks.npz',hu=hu,expert=expert,original=old,
                                    r231_raw=expert,r231_official=expert)
                report['cases'].append({'case':name,'slice':index,'cohort':cohort,
                    'spacing_zyx_mm':[1,1,1],'original':overlap_metrics(old,expert),
                    'r231_raw':overlap_metrics(expert,expert),'r231_official':overlap_metrics(expert,expert)})
            source_report.write_text(json.dumps(report),encoding='utf-8')
            hashes={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            args=parser().parse_args(['--reuse-outputs',str(source),'--outdir',str(root/'new')])
            with patch('lung_reference.save_figures'):
                refresh_profiles(args)
            result=json.loads((root/'new'/'evaluation.json').read_text(encoding='utf-8'))
            self.assertTrue(result['protocol']['no_new_inference'])
            self.assertEqual(len(result['cases']),4)
            self.assertEqual(result['protocol']['historical_inference_code_sha256'],'historical-code-sha')
            for p,digest in hashes.items():self.assertEqual(file_sha256(p),digest)
            report['status']='aborted_incomplete_comparison'
            source_report.write_text(json.dumps(report),encoding='utf-8')
            args.outdir=str(root/'bad')
            with self.assertRaises(ValueError):refresh_profiles(args)


@unittest.skipUnless(importlib.util.find_spec('lungmask') and importlib.util.find_spec('torch'),
                     'Optional lungmask/PyTorch required')
class VendorAdapterTests(unittest.TestCase):
    def test_single_forward_adapter_exactly_matches_official_raw_and_postprocessed_api(self):
        import torch
        from lungmask.mask import LMInferer
        class FakeNetwork(torch.nn.Module):
            def forward(self,x):
                lung=1-torch.abs(x-.2)*5
                background=torch.zeros_like(lung)
                return torch.cat([background,lung,lung-2],dim=1)
        inferer=LMInferer.__new__(LMInferer)
        inferer.model=FakeNetwork().eval();inferer.device=torch.device('cpu')
        inferer.batch_size=2;inferer.tqdm_disable=True;inferer.fillmodel=None
        hu=np.full((3,64,64),-1024.,np.float32);hu[:,5:59,5:59]=20
        hu[:,16:48,16:48]=-700;hu[:,30:34,30:34]=20
        before=hu.copy()
        raw,processed,_=reference_outputs(inferer,hu)
        inferer.volume_postprocessing=False
        np.testing.assert_array_equal(raw,inferer.apply(hu))
        inferer.volume_postprocessing=True
        np.testing.assert_array_equal(processed,inferer.apply(hu))
        np.testing.assert_array_equal(hu,before)
        self.assertTrue(np.isin(raw,[0,1,2]).all())


if __name__=='__main__':unittest.main()
