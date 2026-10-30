"""Cross-GPU diagnostics cannot turn unequal results into a formal pass."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch

from scripts.diagnose_gpu_reproduction import choose_samples, compare, save, main, load, PROFILES


class GpuDiagnosticTests(unittest.TestCase):
    def test_validation_selection_independent_of_order_and_predictions(self):
        rows = [dict(sample_id=str(i), label=i % 2) for i in range(20)]
        selected = choose_samples(rows)
        self.assertEqual(selected, choose_samples(list(reversed(rows))))
        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(x['label'] == 0 for x in selected), 4)

    def test_small_validation_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_samples([dict(sample_id='1', label=0)])

    def test_default_runs_only_original_in_separate_process(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'new'
            with patch('scripts.diagnose_gpu_reproduction.subprocess.run', return_value=
                       subprocess.CompletedProcess([], 0)) as run:
                self.assertEqual(main(['--work-root', temp, '--output', str(output)]), 0)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs['timeout'], 600)
            self.assertEqual(load(output / 'status.json')['runs'][0]['profile'], 'original')

    def test_timeout_is_retained_as_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'timeout'
            with patch('scripts.diagnose_gpu_reproduction.subprocess.run', side_effect=
                       subprocess.TimeoutExpired('diagnostic', 1)):
                self.assertEqual(main(['--work-root', temp, '--output', str(output),
                                       '--profile-timeout', '1']), 1)
            self.assertEqual(load(output / 'original.json')['status'], 'timeout')
            self.assertEqual(load(output / 'status.json')['runs'][0]['returncode'], 124)

    def test_separates_input_score_and_output_differences(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right = root / 'left', root / 'right'
            left.mkdir()
            right.mkdir()
            base = dict(status='passed', model_files={'weights': 'frozen'},
                runtime={'buffer_fingerprints': {}}, samples=[dict(method='zero', sample_id='a',
                input_ids=[1, 2], generated_ids=[3], steps=[dict(scores_sha256='abc')])])
            for profile in PROFILES:
                save(left / (profile + '.json'), base)
                changed = deepcopy(base)
                changed['samples'][0]['steps'][0]['scores_sha256'] = 'def'
                if profile == 'original':
                    changed['samples'][0]['generated_ids'] = [4]
                save(right / (profile + '.json'), changed)
            result = compare(left, right)
            self.assertFalse(result['formal_acceptance_changed'])
            self.assertEqual(result['profiles']['original']['output_differences'], 1)
            self.assertEqual(result['profiles']['controlled_math']['output_differences'], 0)
            self.assertEqual(result['profiles']['controlled_math']['first_step_score_differences'], 1)
            self.assertEqual(result['profiles']['original']['input_differences'], 0)

    def test_missing_profile_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            result = compare(Path(temp), Path(temp))
            self.assertTrue(all(r['status'] == 'missing_profile' for r in result['profiles'].values()))

    def test_different_sample_sets_cannot_be_compared_as_equal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right = root / 'left', root / 'right'
            left.mkdir()
            right.mkdir()
            base = dict(status='passed', samples=[dict(method='zero', sample_id='a')])
            save(left / 'original.json', base)
            base['samples'][0]['sample_id'] = 'b'
            save(right / 'original.json', base)
            self.assertEqual(compare(left, right)['profiles']['original']['status'], 'sample_sets_differ')
