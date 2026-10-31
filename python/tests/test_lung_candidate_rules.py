"""Refinement geometry and score export; synthetic data, no clinical claims."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_candidate_rules import (RulesConfig,FEATURE_NAMES,local_measurement,
    merge_regions,linear_scores,measure_pool)


class RuleTests(unittest.TestCase):
    def test_center_refines_asymmetric_peak_in_physical_units_without_mutating_hu(self):
        z,y,x=np.ogrid[:21,:31,:31];spacing=np.array([2.,.7,.7]);target=np.array([10.,15.,15.])
        r=sum(((g-v)*s)**2 for g,v,s in zip([z,y,x],target,spacing))
        hu=(-850+500*np.exp(-r/(2*2.**2))).astype(np.float32);old=hu.copy()
        c=dict(center_index_zyx=[10,15,13],estimated_scale_diameter_mm=2*np.sqrt(2)*2)
        center,f=local_measurement(hu,c,spacing)
        self.assertLess(np.linalg.norm((center-target)*spacing),np.linalg.norm((np.array(c['center_index_zyx'])-target)*spacing))
        self.assertLessEqual(f['shift'],2.000001)
        np.testing.assert_array_equal(hu,old)

    def test_overlap_merges_same_region_but_preserves_neighbor_and_nontransitive_chain(self):
        p=np.array([[10,10,10],[10,10,13],[10,10,20]])
        b=np.array([[8,13,7,14,7,17],[8,13,7,14,8,18],[8,13,7,14,18,24]])
        keep=merge_regions(p,b,[3,2,1],[1,1,1],replace(RulesConfig(),merge_distance_mm=2.))
        np.testing.assert_array_equal(keep,[0,2])
        p=np.array([[0,0,0],[0,0,2],[0,0,4]])
        b=np.array([[-1,1,-1,1,v-1,v+1] for v in [0,2,4]])
        np.testing.assert_array_equal(merge_regions(p,b,[3,2,1],[1,1,1]),[0,2])

    def test_threshold_prefix_matches_full_merge(self):
        rng=np.random.default_rng(7);p=rng.normal(size=(50,3))*5
        lo=np.floor(p-2).astype(int);hi=np.ceil(p+2).astype(int)
        boxes=np.stack([lo,hi],axis=2).reshape(-1,6);scores=rng.normal(size=50)
        all_keep=merge_regions(p,boxes,scores,[1,1,1])
        for threshold in [-1.,0.,1.]:
            subset=np.flatnonzero(scores>=threshold)
            direct=subset[merge_regions(p[subset],boxes[subset],scores[subset],[1,1,1])]
            np.testing.assert_array_equal(direct,all_keep[scores[all_keep]>=threshold])

    def test_exported_linear_contract_and_empty_input(self):
        model=dict(feature_names=FEATURE_NAMES,weights=[.5]*len(FEATURE_NAMES),bias=2.)
        np.testing.assert_allclose(linear_scores(np.ones((2,len(FEATURE_NAMES))),model),11.)
        self.assertEqual(len(merge_regions([],[],[],[1,1,1])),0)
        with self.assertRaises(ValueError):linear_scores([[float('nan')]*len(FEATURE_NAMES)],model)
        with self.assertRaises(ValueError):RulesConfig(center_blend=2).checked()

    def test_tiny_tree_export_matches_training_implementation(self):
        from lung_candidate_feature_rank import fit_feature_rank,feature_rank_scores
        from threadpoolctl import threadpool_limits
        rng=np.random.default_rng(19);x=rng.normal(size=(400,len(FEATURE_NAMES))).astype(np.float32)
        y=((x[:,0]>.4)&(x[:,1]>-.8)).astype(int)
        with threadpool_limits(limits=2):model,trained=fit_feature_rank(x,y,np.ones(len(y)))
        np.testing.assert_allclose(feature_rank_scores(x,model),trained.decision_function(x),atol=1e-12,rtol=1e-12)
        from lung_candidate_rank_runtime import score_feature_table
        np.testing.assert_allclose(score_feature_table(x,model),trained.decision_function(x),atol=1e-12,rtol=1e-12)
        self.assertEqual(len(model['trees']),32)
        self.assertEqual(model['maximum_depth'],2)

    def test_rank_runtime_does_not_import_network_or_training_libraries(self):
        import subprocess
        script=("import sys;sys.path.insert(0,sys.argv[1]);import lung_candidate_rank_runtime;"
                "assert not any(x in sys.modules for x in ['torch','sklearn','scipy'])")
        run=subprocess.run([sys.executable,'-c',script,str(Path(__file__).resolve().parents[1])],
                           capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_rule_bound_cannot_remove_a_passing_point(self):
        from lung_candidate_refined import rule_upper_bounds
        from lung_candidate_rules import rule_scores
        rng=np.random.default_rng(23);blob=np.exp(rng.uniform(0,7,size=1000))
        x=np.zeros((len(blob),len(FEATURE_NAMES)),np.float32)
        x[:,2]=np.log1p(blob);x[:,9]=rng.uniform(-1,2,size=len(blob))
        x[:,12]=rng.uniform(.1,20,size=len(blob));x[:,13]=rng.uniform(-2,2,size=len(blob))
        scores=rule_scores(x.astype(float));bounds=rule_upper_bounds([dict(blob_score=float(v)) for v in blob])
        self.assertTrue(np.all(scores<=bounds))
        for threshold in [0.,40.,240.,1000.]:
            self.assertFalse(np.any((scores>=threshold)&(bounds<threshold)))

    def test_standalone_entry_reuses_lung_mask_and_exports_native_coordinates(self):
        from test_lung_candidates import phantom
        from lung_candidate_refined import extract_refined_regions
        from lung_candidates import CandidateConfig,point_metrics
        hu,mask,geometry,refs=phantom();original=mask.copy()
        model=dict(kind='rule',threshold=0.,center_mode='refined_centers_zyx')
        result,debug=extract_refined_regions(hu,mask,geometry,model,
            candidate_params=CandidateConfig(sigma_mm=(1.,2.),workers=1))
        self.assertEqual(point_metrics([c['center_lps_xyz_mm'] for c in result['candidates']],refs)['detected'],2)
        np.testing.assert_array_equal(mask,original)
        for c in result['candidates']:
            np.testing.assert_allclose(c['center_lps_xyz_mm'],np.array(geometry['origin_lps_mm'])+
                np.array(geometry['basis_lps_zyx'])@c['center_index_zyx'])
        self.assertNotIn('references',result)
        result2,_=extract_refined_regions(hu,mask,geometry,dict(model,complementary_sigma_mm=2.),
            candidate_params=CandidateConfig(sigma_mm=(1.,2.),workers=1))
        self.assertEqual(point_metrics([c['center_lps_xyz_mm'] for c in result2['candidates']],refs)['detected'],2)
        original_result,_=extract_refined_regions(hu,mask,geometry,dict(model,complementary_sigma_mm=2.),
            candidate_params=CandidateConfig(sigma_mm=(1.,2.),workers=1),optimize_rule=False)
        self.assertEqual(result2['candidates'],original_result['candidates'])
        with self.assertRaises(ValueError):extract_refined_regions(hu,mask,geometry,dict(kind='rule'))


if __name__=='__main__':unittest.main()
