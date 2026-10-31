from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from nodule_slice_benchmark import plane_once,summary
from lung_nodule_cnn import DetectorConfig


class SliceTimingTests(unittest.TestCase):
    def fixture(self):
        hu=np.full((7,12,12),-800,np.int16);search=np.ones_like(hu,bool)
        geometry=dict(spacing_zyx_mm=[1.25,.8,.8],origin_lps_mm=[0.,0.,0.],
            basis_lps_zyx=[[0.,0.,.8],[0.,.8,0.],[1.25,0.,0.]])
        cfg=DetectorConfig(patch=8,halo=2)
        return hu,search,[(3.75,0,0),(3.75,0,4)],geometry,cfg

    def test_every_tile_of_the_same_plane_is_executed_and_summed(self):
        hu,search,jobs,g,cfg=self.fixture();calls=[]
        def predict(patch):
            calls.append(patch.copy());p=np.zeros((5,4,4),np.float32)
            p[0]=-10;p[0,2,2]=2
            return p,dict(host_total_ms=3.,patch_total_ms=5.,arm_quantize_stride2_pack_ms=1.)
        row,preds=plane_once(predict,hu,search,jobs,g,cfg)
        self.assertEqual(len(calls),2);self.assertEqual(preds.shape,(2,5,4,4))
        self.assertEqual(row['network_ms'],6.);self.assertEqual(row['patch_ms'],10.)
        self.assertEqual(row['input_pack_ms'],2.);self.assertEqual(row['tiles'],2)
        self.assertGreater(row['plane_candidate_count'],0)
        self.assertGreaterEqual(row['slice_sampling_to_candidates_ms'],row['sample_ms']+row['decode_ms']+row['plane_merge_ms'])

    def test_mixed_z_and_empty_jobs_do_not_become_fake_single_slice_time(self):
        hu,search,jobs,g,cfg=self.fixture()
        for bad in [[],[jobs[0],(5.,0,0)]]:
            with self.assertRaises(ValueError):plane_once(lambda _:None,hu,search,bad,g,cfg)

    def test_statistic_averages_complete_planes_not_context_channels(self):
        keys=['network_ms','input_pack_ms','patch_ms','sample_ms','decode_ms','plane_merge_ms','slice_sampling_to_candidates_ms']
        a={k:10. for k in keys};b={k:30. for k in keys}
        result=summary([a,b])
        self.assertEqual(result['network_ms']['mean_ms'],20.)
        self.assertEqual(result['slice_sampling_to_candidates_ms']['mean_ms'],20.)


if __name__=='__main__':unittest.main()
