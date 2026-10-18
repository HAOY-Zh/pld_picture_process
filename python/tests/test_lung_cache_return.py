from pathlib import Path
import re
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_cache_return_prepare import checked_previous


class CacheReturnTests(unittest.TestCase):
    def test_same_ports_synchronous_ram_and_policy(self):
        root=Path(__file__).resolve().parents[2]/'fpga'
        a=(root/'lung_feature_cache_fast.v').read_text(encoding='utf-8');b=(root/'lung_feature_cache_return.v').read_text(encoding='utf-8')
        def ports(t):return re.sub(r'\s+','',t.split('input wire clk',1)[1].split(');',1)[0].replace('output reg','output wire'))
        self.assertEqual(ports(a),ports(b))
        self.assertEqual(b.count('ram_style="block"'),2)
        self.assertIn('tag_q<=tags[s_addr[AB-1:4]]',b)
        self.assertIn('state<=m_ready?IDLE:RETURN',b)
        self.assertIn('wire direct_hit=state==CHECK&&(|hit)',b)
        self.assertIn('wire commit=fill_end&&!final_error',b)
    def test_accepted_physical_baseline_and_independent_hit_bench(self):
        p=checked_previous();self.assertEqual(p['routed_design']['clock_mhz'],100)
        t=(Path(__file__).resolve().parents[2]/'fpga/tb_lung_cache_return.sv').read_text(encoding='utf-8')
        for text in ['Missing one-cycle synchronous hit response','Stalled direct response changed','Hit counters counted stalls/duplicates']:
            self.assertIn(text,t)


if __name__=='__main__':unittest.main()
