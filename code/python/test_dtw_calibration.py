import unittest

import numpy as np

from dtw_calibration import coverage_check, pairwise_threshold, select_blend


class DtwCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.pairs = np.array([[0., 1., 3., 3.], [1., 0., 3., 3.],
                               [3., 3., 0., 1.1], [3., 3., 1.1, 0.]])

    def test_two_close_pairs_are_reported_despite_passing_all_leave_one_out_checks(self):
        report = coverage_check(self.pairs, "legacy")
        self.assertEqual(report["components"], [[0, 1], [2, 3]])
        self.assertTrue(report["warning"])
        self.assertTrue(all(report["loo_passed"]))
        self.assertGreater(report["pair_q75"], report["baseline_threshold"])

    def test_connecting_templates_removes_the_group_warning(self):
        report = coverage_check(np.ones((4, 4)) - np.eye(4), "legacy")
        self.assertEqual(report["components"], [[0, 1, 2, 3]])
        self.assertFalse(report["warning"])

    def test_coverage_respects_an_existing_non_min_aggregation(self):
        report = coverage_check(self.pairs, "legacy", "mean3")
        self.assertEqual(report["calibration_aggregate"], "mean3")
        self.assertGreater(report["baseline_threshold"], 2.)
        self.assertLess(coverage_check(self.pairs, "legacy")["baseline_threshold"], 2.)

    def test_zero_blend_preserves_original_calibration_and_does_not_mutate_pairs(self):
        original = self.pairs.copy()
        baseline = pairwise_threshold(self.pairs, "legacy", 0.)
        self.assertAlmostEqual(baseline, 1.15)
        self.assertGreater(pairwise_threshold(self.pairs, "legacy", .5), baseline)
        np.testing.assert_array_equal(original, self.pairs)
        with self.assertRaises(ValueError):
            pairwise_threshold(self.pairs, "legacy", 1.01)

    def test_selection_recovers_a_genuine_without_bypassing_the_voice_gate(self):
        rows = [self.row(True, 1.2), self.row(False, 1.3, voice=False)]
        policy, candidates = select_blend(rows)
        self.assertEqual(policy["blend"], .25)
        chosen = next(r for r in candidates if r["blend"] == policy["blend"])
        self.assertEqual(chosen["genuine_accepted"], 1)
        self.assertEqual(chosen["new_false_accepts"], 0)

    def test_selection_keeps_baseline_when_genuine_and_impostor_overlap(self):
        policy, candidates = select_blend([self.row(True, 1.2), self.row(False, 1.2)])
        self.assertEqual(policy["blend"], 0.)
        self.assertTrue(all(r["new_false_accepts"] for r in candidates if r["blend"] > 0))

    def test_empty_calibration_is_an_error_rather_than_a_selected_policy(self):
        with self.assertRaises(ValueError):
            select_blend([])

    @staticmethod
    def row(genuine, distance, voice=True):
        return dict(genuine=genuine, voice_passed=voice, dtw_distance=distance,
                    baseline_threshold=1., pair_q75=2.)


if __name__ == "__main__":
    unittest.main()
