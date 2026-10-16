import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from ct_registration import (rigid_matrix, quantize_matrix, warp_fixed, warp_float,
    score, register_rigid, hardware_packet, hardware_expected, phantom, Q)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.image=np.random.default_rng(41).integers(0,256,(19,23),dtype=np.uint8)

    def test_identity_and_exact_border(self):
        actual,valid=warp_fixed(self.image,quantize_matrix(np.eye(3)[:2]))
        np.testing.assert_array_equal(actual,self.image)
        self.assertTrue(valid.all())

    def test_inverse_mapping_direction_and_negative_border(self):
        matrix=np.array([[1,0,-1],[0,1,2.]])
        actual,valid=warp_fixed(self.image,quantize_matrix(matrix))
        np.testing.assert_array_equal(actual[:-2,1:],self.image[2:,:-1])
        self.assertFalse(valid[:,0].any())
        self.assertFalse(valid[-2:].any())

    def test_fractional_interpolation_independent_scalar(self):
        matrix=np.array([[.97,-.03125,.5],[.015625,1.,-.25]])
        q=quantize_matrix(matrix)
        actual,valid=warp_fixed(self.image,q)
        for y in range(self.image.shape[0]):
            for x in range(self.image.shape[1]):
                sx,sy=q@np.array([x,y,1])
                if not (0<=sx<=(self.image.shape[1]-1)*Q and 0<=sy<=(self.image.shape[0]-1)*Q):
                    self.assertFalse(valid[y,x]); self.assertEqual(actual[y,x],0); continue
                ix,rx=divmod(int(sx),Q); iy,ry=divmod(int(sy),Q)
                fx,fy=rx>>8,ry>>8
                jx,jy=min(ix+1,self.image.shape[1]-1),min(iy+1,self.image.shape[0]-1)
                weights=[(256-fx)*(256-fy),fx*(256-fy),(256-fx)*fy,fx*fy]
                pixels=[self.image[iy,ix],self.image[iy,jx],self.image[jy,ix],self.image[jy,jx]]
                expected=(sum(int(p)*w for p,w in zip(pixels,weights))+32768)//65536
                self.assertEqual(int(actual[y,x]),expected)

    def test_fixedpoint_rounding_relative_float(self):
        q=quantize_matrix(rigid_matrix(self.image.shape,3.2,.75,-.25))
        actual,valid=warp_fixed(self.image,q)
        expected,mask=warp_float(self.image,q/Q)
        np.testing.assert_array_equal(valid,mask)
        # Geometry is Q16.16; interpolator deliberately drops 8 low fraction bits.
        self.assertLessEqual(np.abs(actual.astype(float)-expected).max(),2.)

    def test_no_overlap_rejected_and_empty_ssd(self):
        q=quantize_matrix(np.array([[1,0,999],[0,1,999.]]))
        actual,valid=warp_fixed(self.image,q)
        self.assertTrue(np.isinf(score(self.image,actual,valid)))
        trailer=hardware_expected(self.image,self.image,q)[-4:]
        np.testing.assert_array_equal(trailer,[0,0,0,0])

    def test_cost_and_packet_word_encoding(self):
        image=np.zeros((3,4),np.uint8)
        moving=np.full_like(image,255)
        words=hardware_expected(image,moving,quantize_matrix(np.eye(3)[:2]))
        self.assertEqual(len(words),16)
        self.assertEqual(int(words[0]),255|(1<<16))
        self.assertEqual(int(words[-4]),12*255**2)
        self.assertEqual(int(words[-2]),12)
        np.testing.assert_array_equal(hardware_packet(image,moving),np.full(12,255<<8))

    def test_bad_dtype_shape_and_coordinate_overflow(self):
        with self.assertRaises(ValueError):
            warp_fixed(self.image.astype(float),quantize_matrix(np.eye(3)[:2]))
        with self.assertRaises(ValueError):
            quantize_matrix(np.ones((3,3)))
        with self.assertRaises(ValueError):
            warp_fixed(self.image,np.full((2,3),2**30,dtype=np.int64))

    def test_known_rigid_transform_recovery(self):
        moving=phantom(96)
        truth=rigid_matrix(moving.shape,3.,4.,-3.)
        sampled,_=warp_float(moving,truth)
        fixed=np.floor(sampled+.5).astype(np.uint8)
        result=register_rigid(fixed,moving)
        matrix=np.array(result['matrix_fixed_to_moving'])
        points=np.array([[0,0,1],[95,0,1],[0,95,1],[95,95,1]])
        error=np.linalg.norm(points@matrix.T-points@truth.T,axis=1).max()
        self.assertLess(error,.75)
        before,bmask=warp_float(moving,np.eye(3)[:2])
        after,mask=warp_float(moving,matrix)
        self.assertLess(score(fixed,after,mask&bmask,0),score(fixed,before,mask&bmask,0)*.1)


if __name__=='__main__':
    unittest.main()
