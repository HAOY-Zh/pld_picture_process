import sys
from pathlib import Path
import subprocess
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_candidate_lowfp import fit,score
from lung_candidate_shell import measure,directions,FEATURE_NAMES


class LowFPTests(unittest.TestCase):
    def test_portable_classifiers_and_empty(self):
        rng=np.random.default_rng(25);x=rng.normal(size=(100,3));y=(x[:,0]*x[:,1]>.6).astype(int)
        for params in [dict(family='svm',C=.1),dict(family='boosted_trees',depth=2),dict(family='extra_trees',depth=4)]:
            # fit performs strict sklearn/portable inference agreement checks.
            model=fit(x,y,np.ones(len(y)),params,['a','b','c'])
            self.assertEqual(score(x[:0],model).shape,(0,))
            np.testing.assert_allclose(score(x[:3],model),score(x,model)[:3],atol=1e-10)

    def test_shell_ball_vs_long_tube_and_physical_spacing(self):
        z,y,x=np.mgrid[-16:17,-16:17,-16:17]
        ball=(-900+900*np.exp(-(x*x+y*y+z*z)/8)).astype(np.float32)
        tube=(-900+900*np.exp(-(x*x+y*y)/8)).astype(np.float32)
        a=measure(ball,[16]*3,[1]*3,2);b=measure(tube,[16]*3,[1]*3,2)
        self.assertEqual(len(a),len(FEATURE_NAMES));self.assertTrue(np.isfinite(a).all())
        # A spherical shell has more uniformly inward-facing gradients.
        self.assertGreater(a[3],b[3]);self.assertGreater(a[9],b[9])
        np.testing.assert_allclose(directions()[:21],-directions()[21:])
        elongated=np.repeat(ball,2,axis=0)
        c=measure(elongated,[32.5,16,16],[.5,1,1],2)
        np.testing.assert_allclose(a,c,atol=.035,rtol=.06)

    def test_inference_does_not_import_training_or_torch(self):
        script="import sys;sys.path.insert(0,'python');import lung_candidate_lowfp;assert 'torch' not in sys.modules;assert 'sklearn' not in sys.modules"
        subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[2],check=True)


if __name__=='__main__':unittest.main()
