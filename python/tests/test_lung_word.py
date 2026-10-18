from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_word_prepare import replace_once
from lung_word_deploy import parse_cycles


class WordProofTests(unittest.TestCase):
    def test_derivative_anchor_must_be_unique(self):
        self.assertEqual(replace_once('old text','old','new'),'new text')
        for value in ['missing','old old']:
            with self.assertRaises(ValueError): replace_once(value,'old','new')

    def test_complete_cycle_log(self):
        text='WORD_CYCLES frame=0 op=3 ca=8 cycles=400 words=10\n'
        text+='Lung tensor AXI-Lite RTL: PASS (1 frames, 10 words, 10 reads, 24 error cases; all)'
        rows=parse_cycles(text)
        self.assertEqual(rows,[dict(frame=0,op=3,ca=8,cycles=400,words=10)])

    def test_incomplete_or_faulty_cycle_proof_rejected(self):
        text='WORD_CYCLES frame=0 op=3 ca=8 cycles=400 words=10\n'
        text+='Lung tensor AXI-Lite RTL: PASS (1 frames, 10 words, 10 reads, 24 error cases; all)'
        for bad in [text.replace('24 error','19 error'),text.replace('10 words','11 words'),
                    text.replace('frame=0','frame=1'),text.replace('cycles=400','cycles=0'),
                    text+'\nFatal: mismatch',text.replace('PASS','FAIL')]:
            with self.assertRaises(ValueError): parse_cycles(bad)


if __name__=='__main__': unittest.main()
