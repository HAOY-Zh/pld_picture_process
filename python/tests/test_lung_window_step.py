from pathlib import Path
import random
import re
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_window_step_prepare import EXTRA, checked_old


class WindowStepTests(unittest.TestCase):
    def test_public_interface_unchanged(self):
        root=Path(__file__).resolve().parents[2]/'fpga'
        def ports(name):
            return re.sub(r'\s+','',(root/name).read_text(encoding='utf-8').split('input wire clk',1)[1].split(');',1)[0])
        self.assertEqual(ports('lung_feature_window_fast.v'),ports('lung_feature_window_step.v'))

    def test_no_live_coordinate_multiply_or_new_math(self):
        text=(Path(__file__).resolve().parents[2]/'fpga/lung_feature_window_step.v').read_text(encoding='utf-8')
        self.assertNotIn('sy[9:0]*',text)
        self.assertNotIn('source_x*',text)
        self.assertNotIn('lung_requant',text)
        self.assertIn('state==ISSUE&&!outside',text)
        self.assertIn('address<=kernel_row_address+row_stride',text)

    def test_stepping_recurrence_matches_coordinate_oracle(self):
        # Supplements actual RTL tests; this is NOT an RTL simulation substitute.
        rng=random.Random(1610)
        for _ in range(100):
            h,w,c=rng.randrange(3,15),rng.randrange(3,15),rng.randrange(1,97)
            k=rng.choice([1,3]);pad=rng.choice([0,1]) if k==3 else 0;b=(c+7)//8
            oh,ow=(h-2,w-2) if k==3 and not pad else (h,w)
            base=rng.choice([0,13,2**32-h*w*b]);pixel=base-pad*(w*b+b);row=pixel+w*b
            for y in range(oh):
                for x in range(ow):
                    address=kernel_row=pixel
                    for ky in range(k):
                        for kx in range(k):
                            for cb in range(b):
                                sy,sx=y+ky-pad,x+kx-pad
                                if 0<=sy<h and 0<=sx<w:
                                    self.assertEqual(address,base+(sy*w+sx)*b+cb)
                                address+=1
                        kernel_row+=w*b;address=kernel_row
                    if x<ow-1: pixel+=b
                    else: pixel=row;row+=w*b

    def test_independent_rtl_oracle_and_accepted_baseline_present(self):
        text=(Path(__file__).resolve().parents[2]/'fpga/tb_lung_window_step.sv').read_text(encoding='utf-8')
        self.assertIn(EXTRA,text)
        self.assertIn('addr+(sy*w+sx)*b+cb',text)
        self.assertIn('Output changed under backpressure',text)
        old=checked_old()
        self.assertEqual(old['status'],'built_routed_tensor_burst_not_board_tested')


if __name__=='__main__': unittest.main()
