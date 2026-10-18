from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_board_preview import native_mask
from lung_unet import torch_module


class BoardPreviewTests(unittest.TestCase):
    def test_same_size_uses_frozen_affine_head_grid(self):
        image=np.array([[[71,72,73],[0,255,72]]],np.uint8)
        result=native_mask(image,.3118481934070587,72,(2,3))
        np.testing.assert_array_equal(result,image[0]>=72)

    def test_matches_sigmoid_then_native_interpolation_not_logits_first(self):
        torch=torch_module()
        codes=np.array([[[0,74],[76,69]]],np.uint8)
        logits=(torch.from_numpy(codes).to(torch.float32)[None]-72)*.3118481934070587
        expected=torch.nn.functional.interpolate(torch.sigmoid(logits),size=(7,9),mode='bilinear',align_corners=False)[0,0].numpy()>=.5
        wrong=torch.sigmoid(torch.nn.functional.interpolate(logits,size=(7,9),mode='bilinear',align_corners=False))[0,0].numpy()>=.5
        self.assertTrue(np.any(expected!=wrong))
        np.testing.assert_array_equal(native_mask(codes,.3118481934070587,72,(7,9)),expected)

    def test_rejects_non_codes_multiple_channels_and_invalid_grid(self):
        for codes in [np.zeros((1,2,2),np.float32),np.zeros((2,2,2),np.uint8),np.zeros((2,2),np.uint8)]:
            with self.assertRaises(ValueError): native_mask(codes,1.,0,(4,4))
        codes=np.zeros((1,2,2),np.uint8)
        for scale,zero in [(0,0),(float('nan'),0),(1.,256),(1.,-1)]:
            with self.assertRaises(ValueError): native_mask(codes,scale,zero,(4,4))


if __name__=='__main__': unittest.main()
