"""The cropped mesh must exactly match the original full-grid extraction."""
from pathlib import Path
import sys,unittest
import numpy as np
from skimage.measure import marching_cubes
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_volume_viewer import mesh,geometry,to_world

class ExactCropTests(unittest.TestCase):
    def test_empty_edges_random_and_255_masks_keep_exact_full_grid_geometry(self):
        geom=geometry([12,-20,44],[1,0,0,0,1,0],[2.5,.7,.8])
        rng=np.random.default_rng(314)
        for shape in ((3,4,5),(12,16,20)):
            for kind in range(7):
                a=np.zeros(shape,bool)
                if kind==1:a[:]=1
                elif kind==2:a[0,0,0]=1
                elif kind==3:a[-1,-1,-1]=1
                elif kind==4:a[1:-1,1:-1,1:-1]=1
                elif kind>=5:a=rng.random(shape)<.12
                for data in (a,a.astype(np.uint8)*255):
                    with self.subTest(shape=shape,kind=kind,dtype=str(data.dtype)):
                        before=data.copy();actual=mesh(data,geom)
                        np.testing.assert_array_equal(data,before)
                        if not a.any():
                            for key in ('voxels','world','faces'):self.assertEqual(actual[key].shape,(0,3))
                            continue
                        vertices,faces,_,_=marching_cubes(np.pad(a,1).astype(np.uint8),level=.5,step_size=1,allow_degenerate=False)
                        vertices-=1
                        np.testing.assert_array_equal(actual['voxels'],vertices.astype(np.float32))
                        np.testing.assert_array_equal(actual['world'],to_world(vertices,geom).astype(np.float32))
                        np.testing.assert_array_equal(actual['faces'],faces.astype(np.uint32))
    def test_invalid_values_still_rejected(self):
        geom=geometry([0,0,0],[1,0,0,0,1,0],[1,1,1])
        for value in (.5,2,float('nan')):
            with self.subTest(value=value),self.assertRaises(ValueError):mesh(np.full((3,4,5),value),geom)

if __name__=='__main__':unittest.main()
