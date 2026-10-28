import unittest
from src.evaluation.metrics import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.truth = [dict(sample_id=str(i), label=i, split='test') for i in [0, 1]]
        self.records = [dict(run_id='r', method='qwen_zero_shot', config_id='c', seed=None,
            sample_id=str(i), split='test', true_label=i, parsed_label=i, raw_output=str(i),
            status='ok', model_id='m', checkpoint_id=None, source_artifact_sha256='a'*64) for i in [0, 1]]

    def test_valid(self):
        self.assertEqual(evaluate(self.records, self.truth)['metrics']['macro_f1'], 1)

    def test_all_invalid(self):
        for r in self.records:
            r.update(status='invalid_format', parsed_label=None, raw_output='unknown')
        result = evaluate(self.records, self.truth)
        self.assertTrue(result['complete'])
        self.assertEqual(result['metrics']['macro_f1'], 0)
        self.assertEqual(result['metrics']['confusion_matrix'], [[0, 0, 1], [0, 0, 1]])

    def test_illegal_is_false_negative(self):
        self.records[0].update(status='invalid_format', parsed_label=None, raw_output='10')
        result = evaluate(self.records, self.truth)['metrics']
        self.assertEqual(result['macro_f1'], .5)
        self.assertEqual(result['accuracy'], .5)

    def test_missing_duplicate_extra_order(self):
        for records in [self.records[:1], self.records+self.records[:1],
                        [dict(self.records[0], sample_id='extra'), self.records[1]], self.records[::-1]]:
            self.assertFalse(evaluate(records, self.truth)['complete'])

    def test_conflicts_and_failures(self):
        for changes in [dict(true_label=1), dict(raw_output='1'), dict(status='oom'),
                        dict(status='invalid_format'), dict(parsed_label=None), dict(status='other')]:
            records = [dict(self.records[0], **changes), self.records[1]]
            self.assertIsNone(evaluate(records, self.truth)['metrics'])

    def test_weighted_by_evaluation_support(self):
        truth = [dict(sample_id=str(i),label=0 if i<3 else 1,split='test') for i in range(4)]
        records = [dict(self.records[1],sample_id=str(i),true_label=t['label']) for i,t in enumerate(truth)]
        m = evaluate(records,truth)['metrics']
        self.assertAlmostEqual(m['macro_f1'],.2)
        self.assertAlmostEqual(m['weighted_f1'],.1)

    def test_mixed_identity(self):
        records = [self.records[0],dict(self.records[1],seed=43)]
        self.assertFalse(evaluate(records,self.truth)['complete'])


if __name__ == '__main__':
    unittest.main()
