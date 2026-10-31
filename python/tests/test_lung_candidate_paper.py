import sys
import copy
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_candidate_paper import negative_subset,select_features,fit,score
from lung_candidate_lowfp_apply import filter_features,apply
from lung_candidate_froc import evaluate


class PaperTests(unittest.TestCase):
    def test_training_undersampling_preserves_positives_and_is_deterministic(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(120,4));y=np.r_[np.ones(6,int),np.zeros(114,int)]
        a=negative_subset(x,y,5);b=negative_subset(x,y,5)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(a),36);self.assertTrue(set(range(6))<=set(a));self.assertEqual(len(set(a)),36)

    def test_selected_features_export_and_filter_geometry_unchanged(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(180,20));y=np.zeros(180,int)
        groups=np.repeat([f'case{i}' for i in range(9)],20)
        for i in range(9):y[20*i:20*i+2]=1
        x[y==1,0]+=4
        names=[f'x{i}' for i in range(20)]
        selection=select_features(x,y,groups,np.ones(len(y)),names,subsets=24,keep=8)
        self.assertEqual(len(selection['selected']),8)
        model=fit(x,y,np.ones(len(y)),dict(C=1.,negative_ratio=5),names,selection)
        self.assertEqual(score(x[:0],model).shape,(0,))
        model.update(high_recall_threshold=-.5,operating_thresholds={'4.0':0.})
        packet=dict(candidates=[dict(candidate_id=str(i),center_index_zyx=[i,2,3],
            center_lps_xyz_mm=[3,2,i],bbox_zyx=[i,i+1,1,4,1,4],score=5.) for i in range(3)])
        original=copy.deepcopy(packet);out=filter_features(packet,x[:3],model,'4.0','model-sha')
        self.assertEqual(packet,original)
        for c in out['candidates']:
            before=packet['candidates'][int(c['candidate_id'])]
            for key in ('center_index_zyx','center_lps_xyz_mm','bbox_zyx'):self.assertEqual(c[key],before[key])

    def test_froc_agrees_with_reference_roc_conversion(self):
        from sklearn.metrics import roc_curve
        nodules=[dict(center_lps_xyz_mm=[0,0,0],diameter_mm=2),
                 dict(center_lps_xyz_mm=[20,0,0],diameter_mm=2)]
        rows=[dict(uid='a',world=np.array([[0,0,0],[.2,0,0],[5,0,0],[8,0,0]]),
                   record=dict(nodules=nodules),excluded=[])]
        result=evaluate(rows,{'a':np.array([.8,.7,.8,.1])})
        # One best score per detected reference, one undetected reference remains
        # in the denominator; duplicate candidate .7 contributes neither TP nor FP.
        fpr,tpr,_=roc_curve([1,0,0],[.8,.8,.1],drop_intermediate=False)
        expected=np.interp([p['fp_budget'] for p in result['operating_points']],fpr*2,tpr/2)
        np.testing.assert_allclose(expected,[p['interpolated_sensitivity'] for p in result['operating_points']])

    def test_runtime_adapter_binds_primary_model_and_preserves_input(self):
        import hashlib,json
        from lung_candidate_lowfp import fit as fit_local
        from lung_candidate_svm import FEATURE_NAMES
        canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        rng=np.random.default_rng(7);x=rng.normal(size=(40,13));y=np.r_[np.zeros(20,int),np.ones(20,int)]
        model=fit_local(x,y,np.ones(40),dict(family='svm',C=.1),FEATURE_NAMES)
        model.update(high_recall_threshold=-1e9,operating_thresholds={'4.0':-1e9})
        primary=dict(kind='rule',threshold=240.)
        c=dict(candidate_id='one',center_index_zyx=[2,2,2],center_lps_xyz_mm=[2,2,2],
            bbox_zyx=[1,4,1,4,1,4],score=300.,equivalent_region_diameter_mm=3.,elongation=1.,
            mean_hu=-400.,max_hu=50.,structure_features=dict(blob_score=250.,blobness=.8,sigma_mm=1.))
        result=dict(algorithm_id='lung_candidate_refined_v4_experimental',rank_model_sha256=canonical(primary),
            source_shape_zyx=[5,5,5],geometry=dict(spacing_zyx_mm=[1,1,1]),candidates=[c])
        original=copy.deepcopy(result)
        contract=dict(primary_file_sha256='primary',model_canonical_sha256=canonical(model),model_file_sha256='new')
        out=apply(np.zeros((5,5,5),np.int16),result,model,primary,'primary',contract,'4.0')
        self.assertEqual(out['candidates'][0]['candidate_id'],'one');self.assertEqual(result,original)
        with self.assertRaises(ValueError):apply(np.zeros((5,5,5)),result,model,primary,'wrong',contract,'4.0')


if __name__=='__main__':unittest.main()
