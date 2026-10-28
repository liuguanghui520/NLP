from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.resource_figure import resource_breakdown

ROOT = Path(__file__).resolve().parents[1]


class ResourceFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.latency = json.loads((ROOT/'results/analysis/comparison-formal-20261028-03/latency.json').read_text(encoding='utf-8'))
        cls.storage = json.loads((ROOT/'results/summaries/deployment-storage.json').read_text(encoding='utf-8'))

    def test_adapter_and_configuration_reconcile_with_complete_deployment(self):
        data = resource_breakdown(self.latency,self.storage,12874640)
        self.assertEqual(data['adapter_weight_MB'],12.87464)
        self.assertEqual(data['adapter_config_and_other_bytes'],1008)
        self.assertEqual(data['full_increment_bytes'],12875648)
        self.assertEqual(data['base_deployment_bytes']+data['full_increment_bytes'],4092326388)
        self.assertIsNone(data['methods'][0]['allocated_peak_MiB'])

    def test_cannot_present_entire_model_as_adapter(self):
        with self.assertRaises(ValueError):
            resource_breakdown(self.latency,self.storage,4092326388)

    def test_different_base_versions_are_rejected(self):
        storage = deepcopy(self.storage)
        next(r for r in storage if r['method']=='qwen_few_shot')['deployment_bytes'] += 100
        with self.assertRaises(ValueError):
            resource_breakdown(self.latency,storage,12874640)
