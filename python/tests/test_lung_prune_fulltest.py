"""Check shared CT envelope equivalence and true-overlap failure accounting."""
import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from lung_body_gate import gate_volume
from lung_int8_final import final_masks,score_mask
from lung_prune_fulltest import STAGES,paired_slices,postprocess,summaries


class FullPruningTestProtocolTests(unittest.TestCase):
    def test_shared_envelope_equals_original_postprocess(self):
        hu=np.full((5,96,96),-1000,np.int16);hu[:,12:85,10:88]=20
        prediction=np.zeros(hu.shape,bool);prediction[:,24:70,20:76]=True
        prediction[0,0:4,0:4]=True
        spacing=[3.,1.,1.]
        envelope,_,_=gate_volume(hu,np.ones(hu.shape,bool),spacing[1:])
        body,final,_=postprocess(prediction,envelope,spacing)
        old_body,old_final,_,_,_=final_masks(hu,prediction,spacing)
        np.testing.assert_array_equal(body,old_body);np.testing.assert_array_equal(final,old_final)

    def test_false_positive_nonempty_is_not_a_successful_detection(self):
        truth=np.zeros((4,8,8),bool);truth[:3,2:4,2:4]=True
        old=np.zeros_like(truth);new=np.zeros_like(truth)
        old[0,0,0]=True  # Zero overlap already: emptying it is not a new missed detection.
        old[1,2,2]=True  # A true detection is lost.
        new[2,2,2]=True  # A previously missed detection is recovered.
        rows=paired_slices(old,new,truth)
        self.assertFalse(rows[0]['newly_zero_overlap'])
        self.assertTrue(rows[1]['newly_zero_overlap']);self.assertTrue(rows[2]['recovered_overlap'])
        self.assertIsNone(rows[3]['original_dice']);self.assertIsNone(rows[3]['pruned_dice'])

    def test_missing_small_slice_group_stays_undefined(self):
        truth=np.ones((2,64,64),bool)
        scores={stage:score_mask(truth,truth) for stage in STAGES}
        result=summaries([dict(cohort='lctsc_heldout',slices=2,scores=scores)])
        self.assertIsNone(result['lctsc_heldout']['pairs']['fixed_final']['small_slice_mean_dice_delta'])


if __name__=='__main__':unittest.main()
