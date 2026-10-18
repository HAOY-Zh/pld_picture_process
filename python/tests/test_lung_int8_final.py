"""Final postprocess CPU screen guards; never claims hardware deployment."""
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_int8_final import acceptance,removed_counts,score_mask,summarize_rows,final_masks,read_body_masks
from lung_unet import file_sha256


class FinalPipelineTests(unittest.TestCase):
    def test_accepted_body_artifact_hash_and_subset_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'masks.npz';raw=np.zeros((2,4,4),bool);raw[0,1,1]=1
            np.savez(path,original=raw,body_cleaned=raw)
            a,b=read_body_masks(path,file_sha256(path))
            np.testing.assert_array_equal(a,raw);np.testing.assert_array_equal(b,raw)
            with self.assertRaises(ValueError):read_body_masks(path,'wrong-hash')
            final=np.ones_like(raw);np.savez(path,original=raw,body_cleaned=final)
            with self.assertRaises(ValueError):read_body_masks(path,file_sha256(path))
    def test_subset_removed_tp_and_shape_guard(self):
        raw=np.ones((2,4,4),bool);final=raw.copy();final[0,0,0]=0
        reference=np.zeros_like(raw);reference[0,0,0]=1
        self.assertEqual(removed_counts(raw,final,reference),dict(true_positive=1,false_positive=0))
        with self.assertRaises(ValueError):removed_counts(final,raw,reference)

    def test_order_body_then_recomputed_3d(self):
        hu=np.zeros((2,16,16));raw=np.ones_like(hu,bool);body=raw.copy();body[0]=0
        combined=body.copy();combined[1,0]=0
        with patch('lung_int8_final.gate_volume',return_value=(body,[],1.)) as gate,patch('lung_int8_final.clean_prediction',return_value=(combined,{})) as clean:
            result=final_masks(hu,raw,[2,1,1])
            self.assertIs(clean.call_args.args[0],body);self.assertIs(result[1],combined)
            self.assertEqual(gate.call_args.args[2],[1,1])

    def test_metrics_empty_slice_handling_and_aggregate(self):
        ref=np.zeros((2,16,16),bool);ref[0,3:6,3:6]=1
        scores=score_mask(ref,ref)
        rows=[dict(scores={name:scores for name in ['fp32_final','int8_final']})]
        summaries=summarize_rows(rows,['fp32_final','int8_final'])
        self.assertEqual(summaries['int8_final']['mean_case_dice'],1)
        self.assertEqual(summaries['int8_final']['slice_quality']['empty_reference_slices'],1)

    def test_screen_rejects_true_lung_removal_and_dice_drop(self):
        ref=np.zeros((2,16,16),bool);ref[0,3:6,3:6]=1
        scores=score_mask(ref,ref)
        row=dict(scores={name:scores for name in ['fp32_final','int8_final']},removed=dict(true_positive=0),expert_outside_envelope_voxels=0)
        summaries=summarize_rows([row],['fp32_final','int8_final'])
        self.assertTrue(acceptance([row],summaries)['accepted_for_hardware_design'])
        row['removed']['true_positive']=1
        self.assertFalse(acceptance([row],summaries)['accepted_for_hardware_design'])
        row['removed']['true_positive']=0;summaries['int8_final']['mean_case_dice']=.99
        self.assertFalse(acceptance([row],summaries)['accepted_for_hardware_design'])


if __name__=='__main__':unittest.main()
