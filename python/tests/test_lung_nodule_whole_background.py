import unittest
import numpy as np
from lung_nodule_cnn import DetectorConfig,sample_patch
from lung_nodule_whole_background import FloatPlaneCache,fit_linear


class WholeBackgroundTests(unittest.TestCase):
    def test_cached_float_planes_preserve_original_samples_exactly(self):
        hu=np.random.default_rng(6).integers(-1200,2000,(13,43,47),dtype=np.int16)
        config=DetectorConfig(patch=32,halo=8)
        spacing=np.array([1.7,.73,.81])
        cached=FloatPlaneCache(hu,limit=4)
        for job in [(0.,-8,-8),(3.75,0,0),(12.5,17,13),(25.,32,32)]:
            np.testing.assert_array_equal(sample_patch(hu,spacing,job,config),
                                          sample_patch(cached,spacing,job,config))
        self.assertLessEqual(len(cached.planes),4)

    def test_excluded_and_below_floor_not_used_as_background(self):
        from lung_nodule_continuity import feature_names
        n=len(feature_names())
        features=np.zeros((4,n));features[0,0]=1;features[1,0]=-1
        row=dict(scores=np.array([.9,.8,.95,.02]),labels=np.array([1,0,-1,0]),features=features,
            world=np.array([[0,0,0],[10,0,0],[20,0,0],[30,0,0]]),
            record=dict(nodules=[dict(center_lps_xyz_mm=[0,0,0],diameter_mm=4)]))
        a=fit_linear([row],.1,'scan_balanced')
        row['features'][2:]=100000
        b=fit_linear([row],.1,'scan_balanced')
        np.testing.assert_array_equal(a['coef'],b['coef'])
        self.assertEqual(a['intercept'],b['intercept'])


if __name__=='__main__':unittest.main()
