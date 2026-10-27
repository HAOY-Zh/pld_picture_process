"""Anchors, physical units and full-case safeguards of optional new cleanup."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lung_structural_cleanup import clean_structural,clean_for_display


class StructuralTests(unittest.TestCase):
    def test_large_lungs_protected_far_and_short_islands_removed_near_one_kept(self):
        a=np.zeros((64,96,128),bool)
        a[10:55,10:55,10:45]=1;a[10:55,10:55,55:90]=1
        a[20:35,20:35,120:126]=1 # Small relative to anchors and far.
        a[35:37,71:76,45:50]=1 # Two isolated planes, outside the near-main guard.
        a[25:29,25:29,46:48]=1 # Near-main disconnected piece is protected.
        before=a.copy();b,info=clean_structural(a,[2,1,1])
        np.testing.assert_array_equal(a,before)
        self.assertTrue(b[10:55,10:55,10:45].all());self.assertTrue(b[10:55,10:55,55:90].all())
        self.assertFalse(b[20:35,20:35,120:126].any());self.assertFalse(b[35:37,71:76,45:50].any())
        self.assertTrue(b[25:29,25:29,46:48].all());self.assertEqual(len(info['anchors']),2)

    def test_partial_input_and_unreliable_small_lung_do_not_remove(self):
        a=np.zeros((8,20,20),bool);a[2:6,3:15,4:15]=1;a[7,19,19]=1
        for full in (False,True):
            b,info=clean_structural(a,[1,1,1],whole_volume=full)
            np.testing.assert_array_equal(a,b);self.assertFalse(info['applied'])

    def test_few_layers_is_not_applied_without_mm_and_volume_constraints(self):
        a=np.zeros((30,70,90),bool);a[3:27,3:55,3:50]=1
        a[10:12,60:65,55:60]=1
        # Two 5-mm native planes exceed the declared 6-mm thin-slab criterion.
        b,info=clean_structural(a,[5,1,1]);self.assertTrue(b[10:12,60:65,55:60].all())

    def test_no_erosion_hole_filling_or_added_voxels(self):
        a=np.zeros((40,70,80),bool);a[4:36,4:60,4:60]=1;a[15:18,20:23,20:23]=0
        b,info=clean_structural(a,[1,1,1]);np.testing.assert_array_equal(a,b)
        self.assertFalse((b&~a).any())

    def test_preview_preserves_original_cleanup_removals_and_original_array(self):
        from lung_island_cleanup import clean_prediction,PRIMARY
        a=np.zeros((50,60,100),bool);a[5:45,5:45,5:45]=1;a[20,20,90]=1
        before=a.copy();legacy,_=clean_prediction(a,[1,1,1],**PRIMARY)
        preview,info=clean_for_display(a,[1,1,1],**PRIMARY)
        self.assertFalse((preview&~legacy).any());self.assertFalse(preview[20,20,90])
        self.assertEqual(info['kind'],'structural_preview_v2')
        np.testing.assert_array_equal(a,before)


if __name__=='__main__':unittest.main()
