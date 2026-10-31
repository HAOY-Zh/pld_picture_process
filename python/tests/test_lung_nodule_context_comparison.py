import unittest
import numpy as np
import torch

from lung_nodule_cnn import DetectorConfig, sample_patch, model
from lung_nodule_context_comparison import context_model, sample_context, BASE_CONTEXT, ALL_CONTEXT, VARIANTS
from lung_nodule_whole_background import FloatPlaneCache
from lung_nodule_cnn_balanced import orient


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectorConfig()

    def test_native_resampling_matches_original_interior_and_padding(self):
        hu = np.random.default_rng(42).integers(-1100, 900, (17, 55, 63), dtype=np.int16)
        spacing = np.array([.7, .68, .91])
        for job in ((0., -32, -31), (4.375, 2, 1), (18.75, 44, 50)):
            actual = sample_context(FloatPlaneCache(hu), spacing, job, self.cfg, BASE_CONTEXT)
            expected = sample_patch(hu, spacing, job, self.cfg)
            # The original reference uses a different order of linear arithmetic;
            # comparison is to the frozen reuse sampler used for baseline training.
            from lung_nodule_sampling_cache import sample_patch_reuse
            exact = sample_patch_reuse(FloatPlaneCache(hu), spacing, job, self.cfg)
            np.testing.assert_array_equal(actual, exact)
            self.assertLessEqual(np.max(np.abs(actual.astype(int)-expected.astype(int))), 1)

    def test_shared_context_subsets_and_z_reversal(self):
        hu = np.arange(21, dtype=np.int16)[:, None, None]*np.ones((1, 7, 9), dtype=np.int16)
        cache = FloatPlaneCache(hu)
        full = sample_context(cache, (1., 1., 1.), (10., 0, 0), self.cfg, ALL_CONTEXT)
        for context in VARIANTS.values():
            actual = sample_context(cache, (1., 1., 1.), (10., 0, 0), self.cfg, context)
            np.testing.assert_array_equal(actual, full[[ALL_CONTEXT.index(c) for c in context]])
            image, positive, _ = orient(actual, [[2, 3, .25, 5]], [], flip_z=True)
            np.testing.assert_array_equal(image, actual[::-1])
            self.assertEqual(positive[0][2], -.25)

    def test_expansion_preserves_five_plane_function_at_initialization(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        base = model(self.cfg).eval()
        x = torch.rand(2, 5, 128, 128)
        with torch.inference_mode():
            expected = base(x)
            for context in (VARIANTS['context7'], VARIANTS['context9']):
                net = context_model(self.cfg, context, base.state_dict()).eval()
                expanded = torch.rand(2, len(context), 128, 128)
                for i, offset in enumerate(BASE_CONTEXT):
                    expanded[:, context.index(offset)] = x[:, i]
                torch.testing.assert_close(net(expanded), expected, atol=1e-6, rtol=1e-6)
                for k, offset in enumerate(context):
                    if offset not in BASE_CONTEXT:
                        self.assertEqual(net.stem[0][0].weight[:, k].count_nonzero().item(), 0)

    def test_near_far_preserves_channel_weights_and_parameter_budget(self):
        base = model(self.cfg)
        state = base.state_dict()
        for name, context in VARIANTS.items():
            net = context_model(self.cfg, context, state)
            self.assertEqual(sum(p.numel() for p in net.parameters()),
                             {'near_far5':13157, 'context7':13301, 'context9':13445}[name])
            if name == 'near_far5':
                for key in state:
                    torch.testing.assert_close(net.state_dict()[key], state[key], atol=0, rtol=0)

    def test_cache_geometry_mismatch_rejected(self):
        cache = FloatPlaneCache(np.zeros((3, 7, 7), dtype=np.int16))
        sample_context(cache, (1., 1., 1.), (0., 0, 0), self.cfg, BASE_CONTEXT)
        with self.assertRaises(ValueError):
            sample_context(cache, (2., 1., 1.), (0., 0, 0), self.cfg, ALL_CONTEXT)


if __name__ == '__main__':
    unittest.main()
