"""Check exact image/label transforms and training sampler probability mass."""
import unittest
import numpy as np
from lung_nodule_cnn_balanced import orient,balanced_weights
from lung_nodule_cnn import DetectorConfig,targets


class BalancedTests(unittest.TestCase):
    def test_all_orientations_keep_marker_and_offsets_aligned(self):
        cfg=DetectorConfig()
        image=np.zeros((5,128,128),np.int16)
        image[1,45,70]=100
        for rotation in range(4):
            for fx in (False,True):
                for fy in (False,True):
                    for fz in (False,True):
                        transformed,pos,ignored=orient(image,[[70.,45.,.2,6.]],[[72.,49.,9.]],rotation,fx,fy,fz)
                        x,y,z,d=pos[0]
                        self.assertEqual(transformed[3 if fz else 1,int(y),int(x)],100)
                        self.assertEqual(z,-.2 if fz else .2)
                        self.assertEqual(d,6.)
                        target=targets(pos,ignored,cfg)
                        self.assertEqual(target[3].sum(),1)
                        iy,ix=np.argwhere(target[3])[0]
                        self.assertAlmostEqual(ix+target[2][0,iy,ix],x/2)
                        self.assertAlmostEqual(iy+target[2][1,iy,ix],y/2)
        self.assertEqual(image[1,45,70],100)

    def test_weights_balance_positive_strata_without_dropping_backgrounds(self):
        groups=['under6_edge']*2+['over10_interior']*8+['background']*30
        w=balanced_weights(groups)
        np.testing.assert_allclose(w.sum(),1.)
        self.assertAlmostEqual(w[:2].sum(),.25)
        self.assertAlmostEqual(w[2:10].sum(),.25)
        self.assertAlmostEqual(w[10:].sum(),.5)
        self.assertTrue((w>0).all())

    def test_four_rotations_round_trip_fractional_labels(self):
        image=np.arange(5*16*16,dtype=np.int16).reshape(5,16,16)
        pos=[[6.25,8.75,-.3,6.]]
        ignored=[[2.25,3.75,8.]]
        actual=image
        for _ in range(4):
            actual,pos,ignored=orient(actual,pos,ignored,1)
        np.testing.assert_array_equal(actual,image)
        np.testing.assert_allclose(pos,[[6.25,8.75,-.3,6.]])
        np.testing.assert_allclose(ignored,[[2.25,3.75,8.]])


if __name__=='__main__':
    unittest.main()
