"""Validate the full prediction plan and aggregation without loading models."""
from pathlib import Path
import unittest
from unittest.mock import patch

from src.model_reproduction import jobs, summarize
from src.common.io import load
from src.adapters.inference import source_context


class ModelReproductionTests(unittest.TestCase):
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
