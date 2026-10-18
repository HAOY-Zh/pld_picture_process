from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_tensor_burst_prepare import transactions,engine_source
from verify_lung_prepared_board import commands


class BurstTests(unittest.TestCase):
    def test_independent_logical_and_prefetched_traffic(self):
        self.assertEqual(transactions(0,2,2,8,0,4096,1048576,2097152),(4,1,4,1))
        self.assertEqual(transactions(1,1,1,1,0,4096,1048576,2097152),(4,1,1,1))
        self.assertEqual(transactions(3,2,2,8,8,4096,1048576,2097152),(32,2,20,2))

    def test_output_chunks_clip_at_4k_word_boundary(self):
        a=transactions(3,2,2,8,8,4096,1048576,2097663)
        self.assertEqual(a,(32,2,20,3))

    def test_derivative_pipeline_and_unchanged_arithmetic(self):
        path=Path(__file__).resolve().parents[2]/'fpga/lung_tensor_word_engine.v'
        old=path.read_text(encoding='utf-8');new=engine_source(old)
        section=lambda text:text[text.index('    wire rq_ready'):text.index('    assign c_ready')]
        self.assertEqual(section(old),section(new))
        self.assertLess(new.index('wire[31:0]source_base'),new.index('reg[32:0]branch_end'))
        self.assertIn('LINE_LENGTH:begin clipped_length<=clipped_hi-clipped_lo',new)
        self.assertIn('written_words<=written_words+buffer_count',new)
        self.assertIn('FINISH:if(rd_ready&&wr_ready)',new)

    def test_returned_frame_rejects_command_count_or_cycle_tamper(self):
        model=dict(nodes=[dict(name='conv',kind='conv',shape=[8,1,1]),dict(name='pool',kind='pool',shape=[8,1,1])])
        row=dict(command_count=2,pl_command_cycles=15,commands=[dict(node='conv',engine='conv',block=0,written_words=1,cycles=10,completed=90),
            dict(node='pool',engine='tensor',block=None,written_words=1,cycles=5,completed=12)])
        self.assertEqual(len(commands(row,model)),2)
        row['commands'][0]['written_words']=2
        with self.assertRaises(ValueError): commands(row,model)
        row['commands'][0]['written_words']=1;row['pl_command_cycles']=16
        with self.assertRaises(ValueError): commands(row,model)


if __name__=='__main__': unittest.main()
