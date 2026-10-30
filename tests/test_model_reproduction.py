"""Validate the full prediction plan and aggregation without loading models."""
from pathlib import Path
from copy import deepcopy
import unittest
from unittest.mock import patch

from src.model_reproduction import jobs, summarize, reference_matches, completion_policy
from src.common.io import load
from src.adapters.inference import source_context


class ModelReproductionTests(unittest.TestCase):
    def policy_records(self):
        scores = dict(macro_f1=.8, accuracy=.8, weighted_f1=.8, invalid_rate=0.)
        return [dict(method=method, seed=seed, run_id=f'{method}-{seed}', complete=True,
                     metrics=dict(n=777, **scores), metric_differences={k: 0. for k in scores})
                for method, seed in jobs()]

    def test_qwen_difference_is_successful_execution_but_not_exact_agreement(self):
        records = self.policy_records()
        records[-1]['metric_differences']['macro_f1'] = -.001
        differences = [dict(run_id=records[-1]['run_id'], sample_id='changed')]
        policy = completion_policy(records, differences, True)
        self.assertTrue(policy['execution_complete'])
        self.assertTrue(policy['baseline_reference_match'])
        self.assertTrue(policy['execution_policy_passed'])
        self.assertFalse(policy['qwen_reference_match'])
        self.assertFalse(policy['strict_reference_match'])
        self.assertEqual(policy['teacher_acceptance'], 'not_recorded')
        self.assertEqual(records[-1]['metric_differences']['macro_f1'], -.001)

    def test_baseline_metric_or_prediction_change_remains_unsuccessful(self):
        for method in ('tfidf', 'bert'):
            records = self.policy_records()
            item = next(r for r in records if r['method'] == method)
            item['metric_differences']['accuracy'] = .001
            self.assertFalse(completion_policy(records, [], True)['execution_policy_passed'])
            item['metric_differences']['accuracy'] = 0.
            self.assertFalse(completion_policy(records, [dict(run_id=item['run_id'])], True)['execution_policy_passed'])

    def test_missing_duplicate_or_incomplete_runs_cannot_qualify(self):
        original = self.policy_records()
        duplicate = deepcopy(original)
        duplicate[-1] = deepcopy(duplicate[0])
        short = deepcopy(original)
        short[0]['metrics']['n'] = 776
        unfinished = deepcopy(original)
        unfinished[0]['complete'] = False
        for records in (original[:-1], duplicate, short, unfinished):
            with self.subTest(records=records):
                self.assertFalse(completion_policy(records, [], True)['execution_complete'])

    def test_unverified_resources_nonfinite_metrics_and_unknown_differences_fail(self):
        records = self.policy_records()
        self.assertFalse(completion_policy(records, [], False)['execution_policy_passed'])
        self.assertFalse(completion_policy(records, [dict(run_id='unknown')], True)['execution_complete'])
        records[-1]['metrics']['macro_f1'] = float('nan')
        self.assertFalse(completion_policy(records, [], True)['execution_complete'])

    def test_exact_match_records_both_execution_and_reference_success(self):
        policy = completion_policy(self.policy_records(), [], True)
        self.assertTrue(policy['strict_reference_match'])
        self.assertTrue(policy['execution_policy_passed'])

    def test_equal_scores_cannot_hide_different_sample_predictions(self):
        self.assertTrue(reference_matches(0,0))
        self.assertFalse(reference_matches(0,1))
        self.assertFalse(reference_matches(.001,0))

    def test_covers_all_methods_and_three_training_seeds(self):
        plan=jobs()
        self.assertEqual(len(plan),11)
        self.assertEqual(len(set(plan)),11)
        for method in ('tfidf','bert','qwen_lora'):
            self.assertEqual({seed for item,seed in plan if item==method},{42,43,44})
        self.assertIn(('qwen_zero_shot',None),plan)
        self.assertIn(('qwen_few_shot',None),plan)

    def test_aggregates_real_run_metrics_against_immutable_reference(self):
        root=Path(__file__).resolve().parents[1]
        contract=load(root/'configs/reproduction.json')
        records=[dict(method=method,metrics=contract['expected_test_means'][method].copy()) for method,_ in jobs()]
        result=summarize(records,contract)
        self.assertEqual(len(result),5)
        self.assertTrue(all(row['macro_f1']['difference']==0 for row in result))
        records[0]['metrics']['macro_f1']-=0.03
        result=summarize(records,contract)
        self.assertAlmostEqual(result[0]['macro_f1']['difference'],-0.01)

    def test_extracted_source_has_provenance_without_git(self):
        with patch('src.adapters.inference.subprocess.check_output',side_effect=FileNotFoundError):
            state=source_context(Path('example'))
        self.assertIsNone(state['git_commit'])
        self.assertEqual(state['source_kind'],'extracted_submission')


if __name__=='__main__':
    unittest.main()
