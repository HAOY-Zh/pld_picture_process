"""Resident weights preserve golden coverage and reject unsafe deduplication."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_tile8_vectors import emit_frame_banks
from lung_weight_reuse_vectors import convert_banks,read_bank
from lung_fpga_contract import multiplier
from lung_rtl_vectors import hex_bank


class WeightReuseTests(unittest.TestCase):
    def fixture(self,root):
        rng=np.random.default_rng(25);m,s=multiplier(np.full(13,.01))
        case=dict(name='tail',q=rng.integers(0,256,(11,3,5),dtype=np.uint8),
            w=rng.integers(-128,128,(13,11,3,3),dtype=np.int8),bias=np.arange(13,dtype=np.int32),
            m=m,shift=s,iz=255,oz=128,relu=True,pad=1)
        return emit_frame_banks([case],root)

    def test_reuses_weights_across_pixels_preserves_results_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'src';dst=Path(tmp)/'dst';src.mkdir();dst.mkdir()
            old=self.fixture(src);report=convert_banks(src,dst)
            self.assertEqual(report['weight_blocks'],2);self.assertEqual(report['weight_load_beats'],36)
            self.assertEqual(report['old_weight_beats'],540);self.assertEqual(report['outputs'],195)
            self.assertEqual(report['weight_transfer_reduction'],14/15)
            rows=read_bank(src,'descriptors');new=read_bank(dst,'descriptors');acts=read_bank(dst,'activations')
            source_packets=read_bank(src,'packets');weights=read_bank(dst,'weights');wdesc=read_bank(dst,'weight_descriptors')
            order=sorted(range(old['frames']),key=lambda g:(rows[g*12+8],rows[g*12+11],rows[g*12+9],rows[g*12+10]))
            for group,original in enumerate(order):
                a=new[group*12:group*12+12];b=rows[original*12:original*12+12]
                for i in range(a[1]):
                    packet=source_packets[b[0]+i];self.assertEqual(acts[a[0]+i]&((1<<64)-1),packet&((1<<64)-1))
                    self.assertEqual(acts[a[0]+i]>>64&255,packet>>576&255)
                    self.assertEqual(weights[wdesc[a[8]*3]+i],packet>>64&((1<<512)-1))
                count=b[2].bit_count()
                for bank in ['accumulators','codes','lanes','lasts']:
                    self.assertEqual(read_bank(dst,bank)[a[7]:a[7]+count],read_bank(src,bank)[b[7]:b[7]+count])

    def test_reject_changed_weight_inside_resident_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);packets=read_bank(root,'packets');packets[36]^=1<<64
            hex_bank(root/'packets.mem',packets,586)
            with self.assertRaisesRegex(ValueError,'changed inside resident'):convert_banks(root,root)

    def test_reject_bad_marker_or_missing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);packets=read_bank(root,'packets');packets[0]^=1<<584
            hex_bank(root/'packets.mem',packets,586)
            with self.assertRaisesRegex(ValueError,'marker'):convert_banks(root,root)


if __name__=='__main__':unittest.main()
