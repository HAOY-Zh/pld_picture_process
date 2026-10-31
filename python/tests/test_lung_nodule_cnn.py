"""Geometry, coverage, supervision and network checks for the new detector."""
import unittest
import numpy as np

from lung_nodule_cnn import (DetectorConfig,model,sample_patch,tiles,targets,decode,
                            consolidate,relative_labels,loss,search_mask)


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.cfg=DetectorConfig()
        self.geometry=dict(spacing_zyx_mm=[1.25,.8,.8],origin_lps_mm=[10,20,30],
                           basis_lps_zyx=[[0,0,.8],[0,-.8,0],[1.25,0,0]])

    def test_native_patch_interpolation(self):
        hu=np.broadcast_to(np.arange(12,dtype=np.int16)[:,None,None]*100,(12,140,140))
        patch=sample_patch(hu,np.array(self.geometry['spacing_zyx_mm']),(6.25,0,0),self.cfg)
        np.testing.assert_array_equal(patch[:,30,30],[300,400,500,600,700])

    def test_tiling_coverage_and_skip(self):
        search=np.zeros((5,150,150),bool);search[:,80,90]=1
        jobs,stats=tiles(search,np.array(self.geometry['spacing_zyx_mm']),self.cfg)
        self.assertGreater(stats['skipped_tiles'],0)
        for zm in range(5):
            self.assertTrue(any(j[0]==zm*1.25 and j[1]+32<=80<j[1]+96 and
                                j[2]+32<=90<j[2]+96 for j in jobs))

    def test_label_decode_roundtrip_with_rotated_basis(self):
        cfg=self.cfg;job=(6.25,0,0)
        point=np.array([5.2,55.2,40.4]);basis=np.array(self.geometry['basis_lps_zyx'])
        xyz=np.array(self.geometry['origin_lps_mm'])+basis@point
        nodules=[dict(center_lps_xyz_mm=xyz.tolist(),diameter_mm=6.)]
        positive,ignored=relative_labels(nodules,[],self.geometry,job,cfg)
        heat,valid,reg,center=targets(positive,ignored,cfg)
        prediction=np.zeros((1,5,64,64),np.float32);prediction[:,0]=-20
        iy,ix=np.argwhere(center)[0]
        prediction[0,0,iy,ix]=10;prediction[0,1:,iy,ix]=reg[:,iy,ix]
        points,scores,diameters=decode(prediction,[job],self.geometry,cfg)
        np.testing.assert_allclose(points,[point],atol=1e-5)
        packet=consolidate(points,scores,diameters,self.geometry,cfg)
        np.testing.assert_allclose(packet[0]['center_lps_xyz_mm'],xyz,atol=1e-5)
        self.assertAlmostEqual(packet[0]['estimated_scale_diameter_mm'],6.,places=5)

    def test_excluded_not_negative_and_neighbor_z_not_positive(self):
        heat,valid,reg,center=targets([],[[64,64,10]],self.cfg)
        self.assertEqual(valid[32,32],0);self.assertEqual(center.sum(),0)
        heat,valid,reg,center=targets([[64,64,0,6]],[[64,64,10]],self.cfg)
        self.assertEqual(valid[32,32],1);self.assertEqual(center.sum(),1)
        point=np.array([6.,64.,64.]);xyz=np.array(self.geometry['origin_lps_mm'])+np.array(self.geometry['basis_lps_zyx'])@point
        pos,ignore=relative_labels([dict(center_lps_xyz_mm=xyz.tolist(),diameter_mm=6)],[],self.geometry,(6.25,0,0),self.cfg)
        self.assertFalse(pos);self.assertTrue(ignore)

    def test_margin_reaches_adjacent_slices(self):
        lung=np.zeros((12,20,20),bool);lung[5,10,10]=1
        search=search_mask(lung,np.array([1.25,.8,.8]),self.cfg)
        self.assertTrue(search[2,10,10]);self.assertFalse(lung[2,10,10])

    def test_single_plane_envelope_matches_volume(self):
        from dataclasses import replace
        from lung_nodule_cnn_single_slice import plane_envelope
        cfg=replace(self.cfg,margin_mm=2.)
        spacing=np.array([1.25,.8,.8]);lung=np.zeros((7,32,45),bool)
        lung[2:5,6:23,9:27]=1;lung[3,10:15,14:19]=0;lung[0,:4,30:]=1
        expected=search_mask(lung,spacing,cfg)
        for z in range(len(lung)):
            np.testing.assert_array_equal(plane_envelope(lung,spacing,z,cfg.margin_mm),expected[z])

    def test_network_shape_and_finite_gradient(self):
        import torch
        torch.set_num_threads(2)
        net=model(self.cfg)
        prediction=net(torch.zeros((2,5,128,128)))
        self.assertEqual(tuple(prediction.shape),(2,5,64,64))
        one=targets([[64.2,63.7,.1,5.]],[],self.cfg)
        target=tuple(torch.from_numpy(np.stack([a,a])) for a in one)
        value,_,_=loss(prediction,target);value.backward()
        self.assertTrue(torch.isfinite(value).item())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))


if __name__=='__main__':unittest.main()
