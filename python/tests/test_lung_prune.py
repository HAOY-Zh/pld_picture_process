"""Check channel identity across skip edges, exact restoration and real tiling."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from lung_prune import PLANS, prune_model, restore_candidate, workload
from lung_unet import make_model, torch_module


class PruningGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch=torch_module();cls.torch.set_num_threads(2)

    def test_identity_and_restore_preserve_output(self):
        torch=self.torch;torch.manual_seed(7);source=make_model().eval()
        candidate,kept=prune_model(source,{})
        image=torch.randn(2,1,32,32)
        with torch.inference_mode():torch.testing.assert_close(source(image),candidate(image),rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.pt'
            torch.save(dict(schema='lung_pruned_fp32_v1',widths={},kept=kept,model=candidate.state_dict()),path)
            restored,_=restore_candidate(path)
            with torch.inference_mode():torch.testing.assert_close(restored(image),source(image),rtol=0,atol=0)

    def test_concat_indices_use_original_branch_offsets(self):
        torch=self.torch;torch.manual_seed(12);source=make_model().eval()
        candidate,kept=prune_model(source,PLANS['balanced'])
        for block,previous,skip,offset in [('dec3','neck.3','enc3.3',64),('dec2','dec3.3','enc2.3',32),('dec1','dec2.3','enc1.3',16)]:
            indices=kept[previous]+[offset+i for i in kept[skip]]
            expected=getattr(source,block)[0].weight[kept[block+'.0']][:,indices]
            torch.testing.assert_close(getattr(candidate,block)[0].weight,expected,rtol=0,atol=0)
        with torch.inference_mode():self.assertEqual(tuple(candidate(torch.zeros(1,1,32,32)).shape),(1,1,32,32))

    def test_workload_counts_padded_tiles_not_dense_zeros(self):
        source=make_model().eval();baseline=workload(source,32)
        with self.torch.no_grad():source.dec2[0].weight.zero_()
        self.assertEqual(workload(source,32)['padded_input_beats'],baseline['padded_input_beats'])
        candidate,_=prune_model(source,PLANS['balanced']);reduced=workload(candidate,32)
        self.assertLess(reduced['padded_input_beats'],baseline['padded_input_beats']*.8)
        self.assertLess(reduced['commands'],baseline['commands'])
        self.assertTrue(all(r['channels']%8==0 for r in reduced['concat']))
        first=reduced['layers'][0]
        self.assertEqual(first['padded_input_beats'],32*32*9)

    def test_bad_tile_width_and_channel_identity_are_rejected(self):
        source=make_model().eval()
        with self.assertRaises(ValueError):prune_model(source,{'dec2.0':6})
        _,kept=prune_model(source,{})
        kept['dec2.0'][1]=kept['dec2.0'][0]
        with self.assertRaises(ValueError):prune_model(source,{},kept)


if __name__=='__main__':unittest.main()
