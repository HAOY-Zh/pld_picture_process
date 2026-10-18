from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify_lung_axil_board import parse_utilization


class BoardReportTests(unittest.TestCase):
    def report(self, star=''):
        return ('| Slice LUTs' + star + ' | 10306 | 0 |\n'
                '| Slice Registers | 11933 | 0 |\n'
                '| DSPs | 73 | 0 |\n'
                '| Block RAM Tile | 31.5 | 0 |\n')

    def test_optional_lut_footnote(self):
        for star in ['', '*']:
            values = parse_utilization(self.report(star))
            self.assertEqual(values['Slice LUTs*'], 10306)
            self.assertEqual(values['Block RAM Tile'], 31.5)

    def test_missing_resource_rejected(self):
        with self.assertRaises(ValueError):
            parse_utilization(self.report().replace('DSPs', 'Other'))

    def test_repeated_identical_register_total(self):
        self.assertEqual(parse_utilization(self.report() +
                         '| Slice Registers | 11933 | 0 |\n')['Slice Registers'], 11933)

    def test_duplicate_resource_rejected(self):
        with self.assertRaises(ValueError):
            parse_utilization(self.report() + '| Slice LUTs* | 2 | 0 |\n')

    def test_over_budget_rejected(self):
        with self.assertRaises(ValueError):
            parse_utilization(self.report().replace('| 73 |', '| 221 |'))


if __name__ == '__main__':
    unittest.main()
