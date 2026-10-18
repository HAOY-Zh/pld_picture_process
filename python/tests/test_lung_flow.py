from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_flow_prepare import traffic
from lung_tensor_burst_prepare import transactions


class FlowTrafficTests(unittest.TestCase):
    def test_pool_rows_retain_both_lines(self):
        args = (0, 4, 32, 8, 0, 4096, 1048576, 2097152)
        self.assertEqual(traffic(*args, False), (128, 64, 1024, 2))
        self.assertEqual(traffic(*args, True), (128, 8, 128, 2))

    def test_tiny_rows_share_one_physical_line(self):
        args = (0, 2, 2, 8, 0, 4096, 1048576, 2097152)
        self.assertEqual(traffic(*args, False), (4, 1, 4, 1))
        self.assertEqual(traffic(*args, True), (4, 1, 4, 1))

    def test_frozen_and_unaligned_geometry_preserve_logical_io(self):
        for h, w, c in [(256,256,8), (128,128,16), (64,64,32), (6,10,9), (2,2,1)]:
            for offset in [0, 1, 509, 511]:
                args = (0,h,w,c,0,4096+offset,1048576,2097152+offset)
                old = transactions(*args)
                self.assertEqual(traffic(*args,False), old)
                new = traffic(*args,True)
                self.assertEqual((new[0],new[3]), (old[0],old[3]))
                self.assertLessEqual(new[2], old[2])

    def test_non_pool_traffic_is_identical(self):
        for op in [1,2,3]:
            for c,cb in [(8,8),(9,7),(16,24),(1,1)]:
                if op==1:cb=0
                args=(op,4,6,c,cb,4097,1049087,2097663)
                self.assertEqual(traffic(*args,True),transactions(*args))


if __name__ == '__main__':
    unittest.main()
