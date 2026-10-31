"""Check the memory-bounded extraction against the existing v4 oracle."""
import sys,json
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_candidate_expand import memory_bounded_primary,PRIMARY,FEATURE_NAMES
from lung_candidate_refined import extract_refined_regions
from lung_candidate_lowfp_apply import extract_features


class ExpandTests(unittest.TestCase):
    def test_memory_bounded_primary_keeps_exact_candidates_and_descriptors(self):
        z,y,x=np.mgrid[:16,:48,:48]
        hu=(-900+1100*np.exp(-((z-8)**2+(y-22)**2+(x-25)**2)/6)).astype(np.int16)
        body=np.zeros_like(hu,bool);body[2:14,7:42,6:42]=True
        geometry=dict(spacing_zyx_mm=[1.,1.,1.],origin_lps_mm=[0.,0.,0.],basis_lps_zyx=[[0,0,1],[0,1,0],[1,0,0]])
        primary=json.loads(PRIMARY.read_text('utf-8'))
        old,_=extract_refined_regions(hu,body,geometry,primary)
        new,features=memory_bounded_primary(hu,body,geometry,primary)
        self.assertGreater(len(new['candidates']),0)
        self.assertEqual(new['candidates'],old['candidates'])
        np.testing.assert_array_equal(features,extract_features(hu,old,FEATURE_NAMES))


if __name__=='__main__':unittest.main()
