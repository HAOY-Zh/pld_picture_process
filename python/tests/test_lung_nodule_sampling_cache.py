import unittest
import numpy as np
from lung_nodule_whole_background import FloatPlaneCache
from lung_nodule_sampling_cache import sample_patch_reuse
from lung_nodule_cnn import DetectorConfig,sample_patch


class SamplingReuseTests(unittest.TestCase):
    def test_original_input_values_preserved_for_overlapping_z_xy_and_padding(self):
        hu=np.random.default_rng(17).integers(-2000,3000,(17,51,57),dtype=np.int16)
        cfg=DetectorConfig(patch=32,halo=8)
        s=np.array([1.37,.63,.79]);cache=FloatPlaneCache(hu)
        for z in [0.,1.25,2.5,3.75,5.,20.,25.]:
            for y,x in [(-8,-8),(8,8),(24,24),(40,40)]:
                job=(z,y,x)
                np.testing.assert_array_equal(sample_patch(hu,s,job,cfg),sample_patch_reuse(cache,s,job,cfg))
        with self.assertRaises(ValueError):sample_patch_reuse(cache,s*2,(0,0,0),cfg)


if __name__=='__main__':unittest.main()
