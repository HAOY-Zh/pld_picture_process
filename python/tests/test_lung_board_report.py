import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_lung_board_smoke import validate_report


def report():
    rows = []
    for completed, (case, block, words) in enumerate([(0,0,1),('same_address_update',None,1),
            (1,0,63),(2,0,63),(2,1,63),(3,0,15),(3,1,15)], 1):
        row = dict(case=case, same=True, different_words=0, completed=completed,
                   written_words=words, cycles=100, cache_hits=0, cache_misses=1)
        if block is not None: row.update(output_block=block, cpu_call_ms=1.0)
        rows.append(row)
    return dict(status='passed_real_board_single_block_smoke', bit_sha256='bit',
                hwh_sha256='hwh', full_network=False, results=rows)


class BoardReportTests(unittest.TestCase):
    def setUp(self): self.hashes = {'lung_axil.bit':'bit', 'lung_axil.hwh':'hwh'}

    def test_complete_report(self):
        self.assertEqual(validate_report(report(), self.hashes)['written_words'], 221)

    def test_wrong_bit_or_hwh_rejected(self):
        for key in ['bit_sha256', 'hwh_sha256']:
            value=report(); value[key]='wrong'
            with self.assertRaises(ValueError): validate_report(value, self.hashes)

    def test_missing_duplicate_or_reordered_command(self):
        for mode in ['missing','duplicate','reordered']:
            value=report()
            if mode=='missing': value['results'].pop(1)
            elif mode=='duplicate': value['results'][1]=copy.deepcopy(value['results'][0])
            else: value['results'][0],value['results'][2]=value['results'][2],value['results'][0]
            with self.assertRaises(ValueError): validate_report(value, self.hashes)

    def test_bad_counter_or_false_success_rejected(self):
        for field, wrong in [('same',False),('different_words',1),('completed',2),
                             ('written_words',True),('cycles',0),('cpu_call_ms',float('nan'))]:
            value=report(); value['results'][0][field]=wrong
            with self.assertRaises(ValueError): validate_report(value, self.hashes)


if __name__ == '__main__': unittest.main()
