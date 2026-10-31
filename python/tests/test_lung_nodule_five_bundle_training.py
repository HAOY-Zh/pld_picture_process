import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_nodule_five_bundle_training import diverse_hard,fit_records
from lung_nodule_cnn_hardneg import background_mask
from lung_nodule_cnn import DetectorConfig


class TrainingTests(unittest.TestCase):
    def test_physical_diversity_prevents_repeated_same_structure(self):
        scores=[9.,8.,7.,6.]
        points=[[0.,10.,10.],[1.25,10.,10.],[10.,10.,10.],[20.,10.,10.]]
        self.assertEqual(diverse_hard(scores,points,3),[0,2,3])

    def test_invalid_peaks_not_hard_negatives(self):
        result=diverse_hard([np.inf,-np.inf,np.nan,2.],[[0,0,0],[0,0,0],[0,0,0],[1,1,1]],4)
        self.assertEqual(result,[3])

    def test_score_ties_are_deterministic(self):
        self.assertEqual(diverse_hard([1.,1.,1.],[[0,0,0],[20,0,0],[40,0,0]],2),[0,1])

    def test_reference_and_indeterminate_disks_excluded_from_mining(self):
        cfg=DetectorConfig()
        label=dict(positive=[[48.,48.,0.,8.]],ignored=[[80.,80.,10.]])
        mask=background_mask(label,cfg)
        self.assertFalse(mask[24,24]);self.assertFalse(mask[40,40])
        self.assertTrue(mask[24,40]);self.assertFalse(mask[0,0])

    def test_no_calibration_or_sealed_uids_in_fit(self):
        m=dict(cases=[dict(series_uid='a',role='train'),dict(series_uid='b',role='sealed_test')])
        self.assertEqual([r['series_uid'] for r in fit_records(m)],['a'])
        m['cases'].append(dict(series_uid='a',role='calibration'))
        with self.assertRaises(ValueError):fit_records(m)


if __name__=='__main__':unittest.main()
