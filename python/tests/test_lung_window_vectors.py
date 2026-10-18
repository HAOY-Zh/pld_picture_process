from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_window_vectors import feature_words,window_plan,emit
from lung_tile8_vectors import pack_frame
from lung_fpga_contract import multiplier,conv_accumulator
from lung_weight_reuse_vectors import read_bank


class WindowTests(unittest.TestCase):
    def test_hwc8_address_order_tails_padding_and_original_packet_agreement(self):
        q=np.arange(11*3*5,dtype=np.uint8).reshape(11,3,5);w=np.ones((13,11,3,3),np.int8)
        words=feature_words(q);plan=list(window_plan(q,101,3,1,255))
        self.assertEqual(len(words),30);self.assertEqual(len(plan),270)
        for group in range(15):
            packets,_=pack_frame(q,w,0,group//5,group%5,1,255)
            for i,(address,data,keep,first,last,x,y,final) in enumerate(plan[group*18:group*18+18]):
                self.assertEqual(keep,packets[i][2]);self.assertEqual(first,i==0);self.assertEqual(last,i==17)
                for lane in range(8):
                    if keep>>lane&1:self.assertEqual(data>>(8*lane)&255,packets[i][0]>>(8*lane)&255)
                if address is not None:self.assertEqual(data,words[address-101])
        self.assertIsNone(plan[0][0]);self.assertTrue(plan[-1][-1])

    def test_degenerate_single_pixel_has_one_memory_read_eight_padding_beats(self):
        q=np.array([[[17]]],np.uint8);plan=list(window_plan(q,2**32-1,3,1,128))
        self.assertEqual(len(plan),9);self.assertEqual([p[0] for p in plan if p[0] is not None],[2**32-1])
        self.assertEqual(sum(p[0] is None for p in plan),8)
        with self.assertRaisesRegex(ValueError,'range'):list(window_plan(q,2**32,3,1,128))

    def test_valid_convolution_and_kernel_one_no_padding(self):
        q=np.arange(9*5*7,dtype=np.uint16).astype(np.uint8).reshape(9,5,7)
        plan=list(window_plan(q,19,3,0,0));self.assertEqual(len(plan),3*5*18)
        self.assertTrue(all(p[0] is not None for p in plan))
        single=list(window_plan(q,19,1,0,0));self.assertEqual(len(single),70)
        self.assertEqual([p[0] for p in single],list(range(19,89)))
        with self.assertRaises(ValueError):list(window_plan(q,0,1,1,0))

    def test_emitted_frame_offsets_and_golden_output_order(self):
        q=np.arange(3*3*5,dtype=np.uint8).reshape(3,3,5);w=np.ones((9,3,3,3),np.int8)
        m,s=multiplier(np.full(9,.01));b=np.arange(9,dtype=np.int32)
        case=dict(name='small',q=q,w=w,bias=b,m=m,shift=s,iz=128,oz=128,relu=True,pad=1)
        expected=conv_accumulator(q,w,b,128,1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=emit([case],root);frames=read_bank(root,'frames');acc=read_bank(root,'accumulators')
            self.assertEqual(report['frames'],2);self.assertEqual(report['outputs'],135)
            self.assertEqual(frames[0],frames[16]);self.assertEqual(frames[8],255);self.assertEqual(frames[24],1)
            for f in range(2):
                offset=frames[f*16+13];oc=f*8
                for y in range(3):
                    for x in range(5):
                        for lane in range(min(8,9-oc)):
                            got=acc[offset];got=got-(1<<32) if got>>31 else got
                            self.assertEqual(got,int(expected[oc+lane,y,x]));offset+=1


if __name__=='__main__':unittest.main()
