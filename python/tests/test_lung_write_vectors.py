from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_write_vectors import output_plan


class WriteTests(unittest.TestCase):
    def test_sparse_lanes_fill_and_hwc8_stride(self):
        stream = [(10, 1, 0, 0, False, False), (20, 6, 0, 0, True, False),
                  (30, 1, 1, 0, False, False), (40, 6, 1, 0, True, True)]
        got = output_plan(511, 1, 2, 12, 11, 0x42, 128, stream)
        self.assertEqual([a for a, _ in got], [522, 534])
        self.assertEqual(got[0][1].to_bytes(8, 'little'), bytes([128, 10, 128, 128, 128, 128, 20, 128]))
        self.assertEqual(got[1][1].to_bytes(8, 'little'), bytes([128, 30, 128, 128, 128, 128, 40, 128]))

    def test_last_physical_word_and_degenerate_image(self):
        self.assertEqual(output_plan(2**29-1, 1, 1, 1, 0, 1, 0, [(255, 0, 0, 0, True, True)]), [(2**29-1, 255)])
        with self.assertRaises(ValueError):
            output_plan(2**29, 1, 1, 1, 0, 1, 0, [])

    def test_invalid_configs(self):
        good = [0, 1, 1, 1, 0, 1, 0, [(1, 0, 0, 0, True, True)]]
        for index, value in [(0, -1), (1, 0), (1, 513), (2, 0), (2, 513),
                             (3, 0), (3, 13), (4, 1), (5, 0), (5, 256), (6, -1), (6, 256)]:
            args = good.copy(); args[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                output_plan(*args)

    def test_reject_bad_lengths_order_coordinates_markers_and_codes(self):
        good = (5, 0, 0, 0, True, True)
        for stream in [[], [good, good]] + [[tuple(value if i == field else old for i, old in enumerate(good))]
                for field, value in [(0, 256), (1, 1), (2, 1), (3, 1), (4, False), (5, False)]]:
            with self.assertRaises(ValueError):
                output_plan(0, 1, 1, 1, 0, 1, 128, stream)


if __name__ == '__main__':
    unittest.main()
