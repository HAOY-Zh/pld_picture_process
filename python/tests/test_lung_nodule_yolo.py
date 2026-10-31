import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_nodule_yolo import (cohort, square_roi, plane_boxes, context_input,
                              orient, requests, z_grid, ignored_objectness)


class YoloGeometryTests(unittest.TestCase):
    def test_metadata_serializes_numpy_counts(self):
        from lung_nodule_yolo import save,read
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'metadata.json'
            save(p,dict(references=np.int32(3),covered=np.bool_(True)))
            self.assertEqual(read(p),dict(references=3,covered=True))

    def test_cohort_rejects_sealed_and_overlapping_uids(self):
        m={'cases':[dict(series_uid='a',role='train'),dict(series_uid='b',role='sealed_test')]}
        self.assertEqual(len(cohort(m,'train')),1)
        with self.assertRaises(ValueError):cohort(m,'sealed_test')
        m['cases'].append(dict(series_uid='a',role='calibration'))
        with self.assertRaises(ValueError):cohort(m,'train')

    def test_roi_uses_only_lungs_and_spacing(self):
        body=np.zeros((20,80,100),bool);body[3:17,15:65,20:80]=1
        roi=square_roi(body,[2.,1.,1.])
        self.assertLessEqual(roi['x0_mm'],15)
        self.assertGreaterEqual(roi['x0_mm']+roi['side_mm'],84)
        self.assertEqual(roi['z_limits_mm'],[1.,37.])
        empty=square_roi(np.zeros_like(body),[2.,1.,1.])
        self.assertTrue(empty['empty_predicted_lung'])
        self.assertEqual(empty['z_limits_mm'],[0.,38.])

    def test_reference_becomes_multiple_slice_boxes(self):
        roi=dict(x0_mm=0.,y0_mm=0.,side_mm=100.,pixel_mm=.25)
        n=[dict(center_index_zyx=[10.,50.,40.],diameter_mm=10.)]
        center=plane_boxes(n,10.,roi,np.ones(3))
        near=plane_boxes(n,13.,roi,np.ones(3))
        np.testing.assert_allclose(center,[[0,.4,.5,.1,.1]])
        np.testing.assert_allclose(near,[[0,.4,.5,.08,.08]])
        self.assertEqual(len(plane_boxes(n,16.,roi,np.ones(3))),0)

    def test_physical_context_interpolation_and_boundaries(self):
        planes=np.arange(8,dtype=np.uint8)[:,None,None]*np.ones((8,2,2),np.uint8)
        x=context_input(planes,4.,2.,(-2.,-1.,0.,1.,2.))
        self.assertEqual(x.dtype,np.float32)
        np.testing.assert_allclose(x[:,0,0]*255,[1.,1.5,2.,2.5,3.])
        edge=context_input(planes,0.,2.)
        self.assertEqual(edge[0,0,0],0.)

    def test_rotation_flips_labels_and_ignored_identically(self):
        x=np.arange(5*4*4).reshape(5,4,4)
        a=np.array([[0,.2,.3,.1,.15]],np.float32)
        y,b,e=orient(x,a,a,1,True,True)
        np.testing.assert_allclose(b,[[0,.7,.8,.15,.1]],atol=1e-7)
        np.testing.assert_array_equal(b,e)
        self.assertTrue(y.flags.c_contiguous)

    def test_requests_reproducible_without_validation(self):
        r=dict(series_uid='train_only',shape_zyx=[20,80,100],spacing_zyx_mm=[1.,1.,1.])
        meta=dict(positive_planes=[5.,6.],roi=dict(z_limits_mm=[0.,19.]))
        plan=dict(seed=123,positive_repeats=4,random_planes_per_scan=32,z_step_mm=1.25)
        a=requests(r,meta,plan,1)
        self.assertEqual(a,requests(r,meta,plan,1))
        self.assertNotEqual(a,requests(r,meta,plan,2))
        self.assertEqual(len(a),40)

    def test_ignored_background_does_not_erase_positive(self):
        import torch
        base=torch.nn.BCEWithLogitsLoss()
        masked=ignored_objectness(base)
        masked.ignored=torch.tensor([[0.,0.,.5,.5,1.,1.]])
        p=torch.zeros((1,3,4,4),requires_grad=True)
        target=torch.zeros_like(p);target[0,0,1,1]=1.
        value=masked(p,target)
        value.backward()
        self.assertAlmostEqual(float(value.detach()),np.log(2),places=6)
        self.assertLess(float(p.grad[0,0,1,1]),0.)
        self.assertEqual(float(p.grad[0,1,0,0]),0.)

    def test_decoded_center_roundtrip_with_nontrivial_lps_basis(self):
        import torch
        from lung_nodule_yolo import detections_to_points
        roi=dict(y0_mm=10.,x0_mm=20.,pixel_mm=.5)
        geometry=dict(spacing_zyx_mm=[2.,.75,.8],origin_lps_mm=[100.,-80.,-100.],
                      basis_lps_zyx=[[0.,0.,-.8],[0.,.75,0.],[-2.,0.,0.]])
        predictions=torch.tensor([[40.,60.,12.,12.,.9,.8],[40.,60.,12.,12.,.8,.7]])
        p,s,d=detections_to_points(predictions,24.,roi,geometry)
        self.assertEqual(len(s),1)
        np.testing.assert_allclose(p[0],[12.,40/.75,40/.8])
        np.testing.assert_allclose(np.array(geometry['origin_lps_mm'])+np.array(geometry['basis_lps_zyx'])@p[0],
                                   [60.,-40.,-124.])
        self.assertAlmostEqual(float(d[0]),6.)


if __name__=='__main__':unittest.main()
