"""Independent 8x8 packet reduction, masks, frame tags, signed bank encoding."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_tile8_vectors import packed_bytes,pack_frame,emit_frame_banks
from lung_fpga_contract import conv_accumulator,requantize,multiplier


def reduce_packets(packets,mask,zero):
    partial=np.zeros((8,8),np.int64)
    for a,w,keep in packets:
        for o in range(8):
            for i in range(8):
                if (mask>>o)&1 and (keep>>i)&1:
                    q=(a>>(8*i))&255;weight=(w>>(8*(o*8+i)))&255
                    if weight>=128:weight-=256
                    partial[o,i]+=(q-zero)*weight
    return partial.sum(1)


class TileVectorTests(unittest.TestCase):
    def test_packed_bytes_lane_order_and_signed_weights(self):
        self.assertEqual(packed_bytes([1,2,3]),0x030201)
        self.assertEqual(packed_bytes([-128,-1,127]),0x7fff80)

    def test_tail_channels_and_nonzero_padding_match_scalar_reference(self):
        rng=np.random.default_rng(665)
        q=rng.integers(0,256,(11,3,5),dtype=np.uint8);w=rng.integers(-128,128,(13,11,3,3),dtype=np.int8)
        bias=np.arange(13,dtype=np.int32)-7;expected=conv_accumulator(q,w,bias,255,1)
        for y,x in [(0,0),(1,2),(2,4)]:
            for start in [0,8]:
                packets,mask=pack_frame(q,w,start,y,x,1,255)
                self.assertEqual(len(packets),18);self.assertEqual(mask,255 if start==0 else 31)
                self.assertEqual([p[2] for p in packets[:2]],[255,7])
                sums=reduce_packets(packets,mask,255)
                n=min(8,13-start)
                np.testing.assert_array_equal(sums[:n]+bias[start:start+n],expected[start:start+n,y,x])
                np.testing.assert_array_equal(sums[n:],np.zeros(8-n))

    def test_max_frozen_channel_group_fits_128_packet_cache(self):
        q=np.zeros((96,1,1),np.uint8);w=np.ones((8,96,3,3),np.int8)
        packets,mask=pack_frame(q,w,0,0,0,1,128)
        self.assertEqual(len(packets),108);self.assertLessEqual(len(packets),128);self.assertEqual(mask,255)

    def test_frame_bank_layout_output_tags_and_scalar_golden(self):
        rng=np.random.default_rng(17)
        q=rng.integers(0,256,(3,2,2),dtype=np.uint8);w=rng.integers(-5,6,(13,3,3,3),dtype=np.int8)
        b=np.arange(13,dtype=np.int32)-4;m,s=multiplier(np.full(13,.01))
        case=dict(name='synthetic',q=q,w=w,bias=b,m=m,shift=s,iz=128,oz=128,relu=True,pad=1)
        expected_acc=conv_accumulator(q,w,b,128,1);expected_code=requantize(expected_acc,m,s,128,True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=emit_frame_banks([case],root)
            read=lambda name:[int(x,16) for x in (root/(name+'.mem')).read_text().splitlines()]
            header=read('header');desc=read('descriptors');packets=read('packets');acc=read('accumulators');code=read('codes');lanes=read('lanes');lasts=read('lasts')
            self.assertEqual(header,report['header']);self.assertEqual(header[0],8);self.assertEqual(header[3],52)
            for g in range(header[0]):
                row=desc[g*12:g*12+12];start,n,mask,iz,oz,relu,pbase,outbase,_,y,x,oc=row
                self.assertEqual(packets[start]>>584&1,1);self.assertEqual(packets[start+n-1]>>585&1,1)
                unpacked=[(p&((1<<64)-1),(p>>64)&((1<<512)-1),(p>>576)&255) for p in packets[start:start+n]]
                reduced=reduce_packets(unpacked,mask,iz)
                for lane in range(min(8,13-oc)):
                    signed=acc[outbase+lane];signed=signed-(1<<32) if signed&(1<<31) else signed
                    self.assertEqual(signed,int(expected_acc[oc+lane,y,x]));self.assertEqual(reduced[lane]+int(b[oc+lane]),signed)
                    self.assertEqual(code[outbase+lane],int(expected_code[oc+lane,y,x]))
                    self.assertEqual(lanes[outbase+lane],lane);self.assertEqual(lasts[outbase+lane],lane==min(8,13-oc)-1)


if __name__=='__main__':unittest.main()
