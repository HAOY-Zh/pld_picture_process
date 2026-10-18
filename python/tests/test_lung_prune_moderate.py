"""Patient-level selection guards for the bounded moderate experiment."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_prune_moderate import development_rank


class DevelopmentSelectionTests(unittest.TestCase):
    def fixture(self, values):
        def score(dice):
            return dict(metrics=dict(dice=dice), slice_quality=dict(
                groups=dict(small=dict(mean_slice_dice=.5)), empty_false_positive_pixels=18))
        baseline = dict(cases=[dict(case=str(i), scores=dict(fp32_final=score(.96)))
                               for i in range(6)], summaries=dict(fp32_final=dict(
                                   mean_case_dice=.96, **{'slice_quality': score(.96)['slice_quality']})))
        rows = [dict(case=str(i), scores=dict(final=score(v))) for i, v in enumerate(values)]
        summary = dict(mean_case_dice=sum(values)/len(values),
                       slice_quality=copy.deepcopy(score(.96)['slice_quality']))
        return rows, summary, baseline

    def test_average_improvement_cannot_hide_one_bad_patient(self):
        args = self.fixture([.94, .98, .98, .98, .98, .98])
        rank, details = development_rank(*args)
        self.assertGreater(rank[1], .96)
        self.assertFalse(rank[0])
        self.assertAlmostEqual(details['worst_case_dice_delta'], -.02)

    def test_one_percentage_point_boundary_is_inclusive(self):
        rank, _ = development_rank(*self.fixture([.95]*6))
        self.assertTrue(rank[0])

    def test_missing_or_duplicate_patient_is_rejected(self):
        rows, summary, baseline = self.fixture([.96]*6)
        with self.assertRaises(ValueError): development_rank(rows[:-1], summary, baseline)
        rows[-1]['case'] = rows[0]['case']
        with self.assertRaises(ValueError): development_rank(rows, summary, baseline)

    def test_empty_false_positives_are_checked_independently(self):
        rows, summary, baseline = self.fixture([.97]*6)
        summary['slice_quality']['empty_false_positive_pixels'] = 1019
        self.assertFalse(development_rank(rows, summary, baseline)[0][0])


if __name__ == '__main__': unittest.main()
