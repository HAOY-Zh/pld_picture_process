from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_cache_vectors import FeatureCacheModel


class CacheTests(unittest.TestCase):
    def test_unaligned_span_clipping_and_hits(self):
        model=FeatureCacheModel(3,19)
        self.assertEqual(model.read(3),(3,13));self.assertIsNone(model.read(15))
        self.assertEqual(model.read(21),(16,6));self.assertIsNone(model.read(16))
        self.assertEqual((model.hits,model.misses),(2,2))
        with self.assertRaises(ValueError):model.read(2)
        with self.assertRaises(ValueError):model.read(22)

    def test_three_way_fifo_eviction_same_set(self):
        model=FeatureCacheModel(0,128,sets=2)
        for a in [0,32,64]:self.assertIsNotNone(model.read(a))
        for a in [0,32,64]:self.assertIsNone(model.read(a))
        model.read(96);self.assertIsNone(model.read(32));self.assertIsNone(model.read(64))
        self.assertIsNotNone(model.read(0))

    def test_failed_fill_is_not_committed_and_new_frame_is_clean(self):
        model=FeatureCacheModel(0,16)
        self.assertEqual(model.read(2,fill_ok=False),(0,16));self.assertEqual(model.read(2),(0,16))
        self.assertIsNone(model.read(2));self.assertIsNotNone(FeatureCacheModel(0,16).read(2))

    def test_address_end_and_4k_boundaries(self):
        model=FeatureCacheModel(2**32-3,3)
        self.assertEqual(model.read(2**32-1),(2**32-3,3))
        m=FeatureCacheModel(505,20)
        self.assertEqual(m.read(511),(505,7));self.assertEqual(m.read(512),(512,13))
        for a,n in m.bursts:self.assertLessEqual((a%512)+n,512)
        with self.assertRaises(ValueError):FeatureCacheModel(2**32-1,2)
        with self.assertRaises(ValueError):FeatureCacheModel(0,1,sets=3)


if __name__=='__main__':unittest.main()
