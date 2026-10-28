"""Metric regression cases: invalid outputs must not disappear from recall."""
import unittest

from evaluate_qwen06 import metrics


class MetricsTests(unittest.TestCase):
    def test_invalid_and_wrong_predictions(self):
        result = metrics([
            dict(label=0, prediction=0), dict(label=0, prediction=None),
            dict(label=1, prediction=0), dict(label=1, prediction=1),
        ])
        self.assertEqual(result['confusion_matrix'], [[1, 0, 1], [1, 1, 0]])
        self.assertEqual(result['accuracy'], 0.5)
        self.assertAlmostEqual(result['macro_f1'], 7 / 12)
        self.assertEqual(result['invalid_rate'], 0.25)

    def test_all_invalid(self):
        result = metrics([dict(label=0, prediction=None), dict(label=1, prediction=None)])
        self.assertEqual(result['macro_f1'], 0)
        self.assertEqual(result['accuracy'], 0)
        self.assertEqual(result['invalid_rate'], 1)


if __name__ == '__main__':
    unittest.main()
