import unittest
import numpy as np
from lung_nodule_continuity import local_evidence
from lung_nodule_continuity_boundary import compatible_evidence


class BoundaryTests(unittest.TestCase):
    def test_half_voxel_edge_matches_existing_decode_gate(self):
        hu=np.full((9,17,17),-1000,np.int16)
        hu[:3,6:11,6:11]=0
        c=dict(center_index_zyx=[-.2,8,8],estimated_scale_diameter_mm=5)
        x,d=compatible_evidence(hu,[1,1,1],c)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(d['native_z_range'][0],0)
        self.assertEqual(d['supports'][0]['native_planes'],3)
        self.assertEqual(c['center_index_zyx'],[-.2,8,8])
        c['center_index_zyx'][0]=-.6
        with self.assertRaises(ValueError):compatible_evidence(hu,[1,1,1],c)

    def test_interior_is_bitwise_identical(self):
        hu=np.random.default_rng(3).integers(-1000,100,(11,21,21),dtype=np.int16)
        c=dict(center_index_zyx=[5.1,10,10],estimated_scale_diameter_mm=6)
        a,da=local_evidence(hu,[1.5,.7,.7],c)
        b,db=compatible_evidence(hu,[1.5,.7,.7],c)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(da,db)


if __name__=='__main__':unittest.main()
