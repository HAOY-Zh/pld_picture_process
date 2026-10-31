"""Width mapping must preserve both decoder skip paths and permit new learning."""
from dataclasses import replace
import unittest

import torch

from lung_nodule_cnn import DetectorConfig, model
from lung_nodule_capacity import doubled_state, architecture_counts, eligible


class CapacityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)

    def test_initial_function_and_non_symmetric_gradient(self):
        cfg = DetectorConfig(patch=32, halo=8)
        small, wide = model(cfg), model(replace(cfg, base=16))
        wide.load_state_dict(doubled_state(small.state_dict(), 456))
        x = torch.rand(3,5,32,32)
        a, b = small(x), wide(x)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        # Nonzero head output weights and unequal split yield unequal incoming
        # gradients; equal half-weights would trap the network in duplicate pairs.
        b.square().mean().backward()
        grad = wide.stem[0][0].weight.grad
        self.assertGreater(float((grad[0::2]-grad[1::2]).abs().max()), 1e-12)
        optimizer = torch.optim.SGD(wide.parameters(), lr=1.)
        optimizer.step()
        weight = wide.stem[0][0].weight
        self.assertGreater(float((weight[0::2]-weight[1::2]).detach().abs().max()), 0.)

    def test_real_size_counts(self):
        cfg = DetectorConfig()
        self.assertEqual(architecture_counts(cfg)['parameters'],13157)
        self.assertEqual(architecture_counts(replace(cfg,base=16))['parameters'],51653)
        self.assertEqual(architecture_counts(cfg)['conv_macs_per_patch'],19038208)
        self.assertEqual(architecture_counts(replace(cfg,base=16))['conv_macs_per_patch'],72876032)

    def test_seed_and_pair_sum(self):
        small = model(DetectorConfig()).state_dict()
        a, b = doubled_state(small,5), doubled_state(small,5)
        for key in a:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        head = a['head.weight']
        torch.testing.assert_close(head[:,0::2]+head[:,1::2],small['head.weight'],atol=1e-10,rtol=1e-7)
        self.assertFalse(torch.equal(head[:,0::2],head[:,1::2]))

    def test_forbid_overlap_and_heldout_modes_before_loading_ct(self):
        plan = dict(fit_scans=['a'],calibration_scans=['a'],review_12=[],reused_6=[])
        with self.assertRaisesRegex(ValueError,'overlap'):
            eligible(plan,'fit_scans')
        with self.assertRaisesRegex(ValueError,'heldout'):
            eligible(plan,'heldout_scans')


if __name__ == '__main__':
    unittest.main()
