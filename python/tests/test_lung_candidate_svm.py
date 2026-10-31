"""Cached descriptor and SVM export checks; no CT/network inference."""
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_candidate_svm import FEATURE_NAMES,from_packet,from_local_table,fit,decision_scores


class CandidateSVMTests(unittest.TestCase):
    def test_packet_descriptor_matches_saved_local_encoding(self):
        c=dict(score=300.7,bbox_zyx=[3,6,10,15,20,27],equivalent_region_diameter_mm=4.2,
            elongation=1.8,mean_hu=-450.,max_hu=35.,
            structure_features=dict(blob_score=290.,blobness=.55,sigma_mm=2.))
        x=np.zeros((1,18),np.float32)
        x[0,[2,3,5,4,6,7,8]]=[np.log1p(290.),.55,np.log1p(2.),np.log1p(4.2),np.log1p(1.8),-.45,.035]
        table=dict(features=x,boxes_zyx=np.array([c['bbox_zyx']]))
        np.testing.assert_array_equal(from_packet([c],[2.,.7,.7]),from_local_table(table,[0],[300.7],[2.,.7,.7]))

    def test_export_matches_training_scores_and_handles_empty_input(self):
        from threadpoolctl import threadpool_limits
        rng=np.random.default_rng(22);x=rng.normal(size=(240,len(FEATURE_NAMES)))
        y=((x[:,0]*x[:,1])>.2).astype(int)
        with threadpool_limits(limits=2):
            model=fit(x,y,np.ones(len(y)),1.,'scale')
            a=decision_scores(x,model,chunk=7);b=decision_scores(x,model,chunk=1000)
        np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
        self.assertEqual(decision_scores(np.empty((0,len(FEATURE_NAMES))),model).shape,(0,))
        with self.assertRaises(ValueError):decision_scores(x,dict(model,feature_names=[]))

    def test_runtime_import_is_numpy_only(self):
        script="import sys;sys.path.insert(0,sys.argv[1]);import lung_candidate_svm;assert not any(x in sys.modules for x in ('torch','sklearn','scipy'))"
        r=subprocess.run([sys.executable,'-c',script,str(Path(__file__).resolve().parents[1])],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_application_binds_primary_identity_and_preserves_input(self):
        import json
        from lung_candidate_svm_apply import filter_result,canonical_sha
        c=dict(score=300.7,bbox_zyx=[3,6,10,15,20,27],center_index_zyx=[4,12,23],
            equivalent_region_diameter_mm=4.2,elongation=1.8,mean_hu=-450.,max_hu=35.,
            structure_features=dict(blob_score=290.,blobness=.55,sigma_mm=2.))
        primary=dict(kind='rule',threshold=240.)
        packet=dict(algorithm_id='lung_candidate_refined_v4_experimental',
            rank_model_sha256=canonical_sha(primary),candidates=[c],geometry=dict(spacing_zyx_mm=[2.,.7,.7]))
        model=dict(kind='weighted_rbf_svm_second_stage_v1',feature_names=FEATURE_NAMES,
            mean=[0.]*len(FEATURE_NAMES),scale=[1.]*len(FEATURE_NAMES),
            support_vectors=[[0.]*len(FEATURE_NAMES)],dual_coefficients=[1.],gamma=.1,bias=0.,
            threshold=-1.,source_primary_model_sha256='a'*64)
        before=json.dumps(packet,sort_keys=True)
        out=filter_result(packet,model,primary,'a'*64)
        self.assertEqual(json.dumps(packet,sort_keys=True),before)
        self.assertEqual(out['candidates'][0]['center_index_zyx'],c['center_index_zyx'])
        self.assertEqual(out['candidates'][0]['bbox_zyx'],c['bbox_zyx'])
        with self.assertRaises(ValueError):filter_result(packet,model,primary,'b'*64)


if __name__=='__main__':unittest.main()
