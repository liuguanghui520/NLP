import unittest

import numpy as np

from scripts.build_comparison_report import holm, macro_f1, paired_bootstrap


class OfflineComparisonTests(unittest.TestCase):
    def test_invalid_output_counts_as_true_class_false_negative(self):
        # Class0 F1=1/2; class1 F1=2/3. Dropping invalid would inflate the score.
        self.assertAlmostEqual(float(macro_f1([0, 0, 1, 1], [0, -1, 1, 0])), 7 / 12)

    def test_constant_majority_does_not_receive_high_macro_f1(self):
        self.assertAlmostEqual(float(macro_f1([0, 1, 1, 1], [1, 1, 1, 1])), 3 / 7)

    def test_fixed_label_set_in_single_class_group(self):
        self.assertEqual(float(macro_f1([1, 1], [1, 1])), .5)

    def test_holm_enforces_monotone_adjustment_in_original_order(self):
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])

    def test_identical_models_have_exactly_zero_paired_difference(self):
        y = np.array([0, 0, 1, 1, 1])
        prediction = np.array([0, 1, 1, -1, 1])
        out = paired_bootstrap(y, {"a": [prediction], "b": [prediction.copy()]}, seed=67, resamples=31, batch=7)
        np.testing.assert_array_equal(out["a"] - out["b"], np.zeros(31))

    def test_resampling_reproducible_for_fixed_configuration(self):
        y = np.array([0, 0, 1, 1, 1])
        predictions = {"a": [np.array([0, 1, 1, 0, 1])]}
        a = paired_bootstrap(y, predictions, seed=67, resamples=21, batch=7)
        b = paired_bootstrap(y, predictions, seed=67, resamples=21, batch=7)
        np.testing.assert_array_equal(a["a"], b["a"])

    def test_stratified_bootstrap_keeps_perfect_model_f1_one(self):
        y = np.array([0, 1, 1])
        values = paired_bootstrap(y, {"perfect": [y.copy()]}, seed=67, resamples=21, batch=7)
        np.testing.assert_array_equal(values["perfect"], np.ones(21))

    def test_bootstrap_rejects_missing_true_class(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 1], {"a": [np.ones(2)]}, seed=67, resamples=2)


if __name__ == "__main__":
    unittest.main()
