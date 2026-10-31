from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from nodule_plan import load_package,lower_stem,quantize_hu
from nodule_int8_cuda_reference import IntegerNodule
from nodule_integer_reference import forward


class IntegerQualityReferenceTests(unittest.TestCase):
    def test_all_nodes_of_four_real_board_vectors_are_exact_on_cuda(self):
        self.assertTrue(torch.cuda.is_available(),'CUDA reference was not validated')
        folder,_,plan,p=load_package(ROOT/'data/nodule_board_20261023_v1/nodule_board')
        net=IntegerNodule(plan,p,'cuda')
        with np.load(folder/'nodule_reference.npz') as z:
            hu=np.stack([z[f'case{i}_hu'] for i in range(4)])
            q=net.lut[hu.astype(np.int32)+32768]
            head,trace=net.forward_codes(torch.from_numpy(q),True)
            for node in plan['nodes']:
                for i in range(4):np.testing.assert_array_equal(trace[node['name']][i],z[f'case{i}_'+node['name']])

    def test_extreme_and_random_inputs_match_independent_fp64_integer_reference(self):
        _,_,plan,p=load_package(ROOT/'data/nodule_board_20261023_v1/nodule_board')
        net=IntegerNodule(plan,p,'cuda');rng=np.random.default_rng(1023)
        samples=[np.zeros((5,128,128),np.uint8),np.full((5,128,128),255,np.uint8),
            rng.integers(0,256,(5,128,128),dtype=np.uint8)]
        _,trace=net.forward_codes(torch.from_numpy(np.stack(samples)),True)
        for i,q in enumerate(samples):
            _,old,_=forward(plan,p,lower_stem(q),'cuda',True)
            for node in plan['nodes']:np.testing.assert_array_equal(trace[node['name']][i],old[node['name']])
        self.assertLess(max(net.bounds.values()),2**24)


if __name__=='__main__':unittest.main()
