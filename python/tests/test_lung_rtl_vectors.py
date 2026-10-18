"""Readmemh bank layout, nonempty counts, frozen numeric-vector identity."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_rtl_vectors import hex_bank,quant_word,generate,parser
from lung_fpga_contract import conv_accumulator,requantize,multiplier
from lung_unet import file_sha256


class RtlVectorTests(unittest.TestCase):
    def test_packed_word_layout_signed_codes_and_invalid_metadata(self):
        word=quant_word(-7,1073741824,31,128,True,124,1)
        self.assertEqual((word>>56)&0xffffffff,0xfffffff9)
        self.assertEqual((word>>24)&0xffffffff,1073741824)
        self.assertEqual((word>>18)&63,31);self.assertEqual((word>>10)&255,128)
        self.assertEqual((word>>9)&1,1);self.assertEqual((word>>1)&255,124);self.assertEqual(word&1,1)

    def test_hex_twos_complement_and_fixed_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'codes.mem';hex_bank(path,[-128,-1,0,127,255],8)
            self.assertEqual(path.read_text(encoding='ascii').splitlines(),['80','ff','00','7f','ff'])
            hex_bank(path,[-2147483648,2147483647],32)
            self.assertEqual(path.read_text(encoding='ascii').splitlines(),['80000000','7fffffff'])

    def test_frozen_synthetic_contract_bank_round_trip_and_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();m,s=multiplier([.5])
            w=np.array([[[[3]]]],np.int8);b=np.array([7],np.int32)
            p=dict(head_weight_int8=w,head_bias_int32=b,head_multiplier_int32=m,head_right_shift_int32=s)
            np.savez(source/'parameters.npz',**p);v={}
            for n,tag in enumerate(['real_zero','ramp127','ramp255']):
                q=np.full((1,8,8),127,np.uint8) if n==0 else (np.arange(64).reshape(1,8,8)*n).astype(np.uint8)
                out=requantize(conv_accumulator(q,w,b,127,0),m,s,128)
                v['head_'+tag+'_input_uint8']=q;v['head_'+tag+'_output_uint8']=out
            np.savez(source/'operator_vectors.npz',**v)
            path=source/'contract.json';path.write_text('{}',encoding='utf-8')
            contract=dict(layers=[dict(name='head',input_zero_point=127,output_zero_point=128,fused_relu=False,padding=[0,0])])
            tracked={str(p):file_sha256(p) for p in source.iterdir()}
            args=parser().parse_args(['--contract',str(path),'--outdir',str(root/'vectors')])
            with patch('lung_rtl_vectors.checked_contract',return_value=(path,contract,tracked)):report=generate(args)
            header=[int(x,16) for x in (root/'vectors/header.mem').read_text().splitlines()]
            self.assertEqual(header[0],6);self.assertGreater(header[4],0);self.assertGreater(header[5],header[4])
            self.assertEqual(header,report['header'])
            for name,h in report['outputs'].items():self.assertEqual(file_sha256(root/'vectors'/name),h)
            for p,h in tracked.items():self.assertEqual(file_sha256(p),h)
            with patch('lung_rtl_vectors.checked_contract',return_value=(path,contract,tracked)):
                with self.assertRaises(ValueError):generate(args)


if __name__=='__main__':unittest.main()
