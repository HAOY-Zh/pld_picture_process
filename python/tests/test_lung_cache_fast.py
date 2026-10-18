from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_cache_fast_prepare import cycle_rows


class FastCacheTests(unittest.TestCase):
    def test_all_frame_counters_required_in_order(self):
        text='\n'.join('CACHE_FAST_CYCLES frame={} cycles={}'.format(i,i+100) for i in range(147))
        rows=cycle_rows(text)
        self.assertEqual(len(rows),147)
        self.assertEqual(rows[-1],dict(frame=146,cycles=246))

    def test_partial_duplicate_zero_or_faulty_cycle_log_rejected(self):
        lines=['CACHE_FAST_CYCLES frame={} cycles=100'.format(i) for i in range(147)]
        for text in ['\n'.join(lines[:-1]),'\n'.join(lines+[lines[-1]]),
                     '\n'.join(lines).replace('cycles=100','cycles=0',1),
                     '\n'.join(lines)+'\nFatal: mismatch',
                     '\n'.join(lines)+'\nWARNING: readmemh missing data']:
            with self.assertRaises(ValueError): cycle_rows(text)

    def test_same_public_cache_and_window_interfaces(self):
        root=Path(__file__).resolve().parents[2]/'fpga'
        for name in ['lung_feature_cache','lung_feature_window']:
            def ports(path):
                text=path.read_text(encoding='utf-8').split('input wire clk',1)[1].split(');',1)[0]
                return re.sub(r'\s+','',text)
            self.assertEqual(ports(root/(name+'.v')),ports(root/(name+'_fast.v')))

    def test_no_new_rounding_or_dram_protocol_implementation(self):
        root=Path(__file__).resolve().parents[2]/'fpga'
        text=(root/'lung_feature_cache_fast.v').read_text(encoding='utf-8')
        self.assertIn('ram_style="block"',text)
        self.assertIn('tag_q<=tags[s_addr[AB-1:4]]',text)
        self.assertIn('data_q<=data[s_addr[AB-1:0]]',text)
        self.assertNotIn('lung_requant',text)
        self.assertNotIn('m_axi_',text)
        window=(root/'lung_feature_window_fast.v').read_text(encoding='utf-8')
        self.assertIn('address<=address+1\'b1',window)
        self.assertIn('if(zero_group)state<=OUTPUT',window)


if __name__=='__main__': unittest.main()
