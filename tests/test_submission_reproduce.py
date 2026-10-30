"""Regression coverage from the returned Windows reproduction evidence."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import submission_reproduce as entry
from src.common.downloads import download_verified, safe_path
from src.model_preflight import inspect_environment


class PortableDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_windows_names_traversal_and_link_escape_are_rejected(self):
        for name in ("../escape", "/absolute", "C:/escape", "C:escape", "\\\\host\\share", "a/../b", "a\\b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_path(self.root, name)
        self.assertEqual(safe_path(self.root, "data/中文/test.json"), self.root.resolve() / "data/中文/test.json")
        link = self.root / "linked"
        try:
            link.symlink_to(self.root.parent, target_is_directory=True)
        except OSError:
            return  # Windows symlink privilege is not universally available.
        with self.assertRaises(ValueError):
            safe_path(self.root, "linked/escape")

    def test_creating_windows_targets_does_not_resolve_the_target(self):
        # This includes concurrent creation, the failure trigger in attempt 002.
        def create(number):
            target = safe_path(self.root, f"data/目录/{number}.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(number), encoding="utf-8")
            return target
        with ThreadPoolExecutor(max_workers=4) as pool:
            targets = list(pool.map(create, range(100)))
        self.assertEqual(len(targets), 100)

    def test_incomplete_response_is_retained_and_retry_is_verified(self):
        content = b"complete data"
        expected = hashlib.sha256(content).hexdigest()
        target = self.root / "input.json"
        with patch("src.common.downloads.shutil.which", return_value=None), patch(
                "src.common.downloads.urlopen", side_effect=[io.BytesIO(b"truncated"), io.BytesIO(content)]):
            download_verified("https://example.org/data", target, expected, log=self.root / "events.jsonl")
        self.assertEqual(target.read_bytes(), content)
        self.assertEqual(len(list(self.root.glob("*.failed-*"))), 1)
        self.assertEqual([json.loads(s)["status"] for s in (self.root / "events.jsonl").read_text().splitlines()], ["failed", "verified"])

    def test_corrupt_archive_is_quarantined_before_redownload(self):
        target = self.root / "model.zip"
        target.write_bytes(b"corrupt")
        content = b"correct archive"
        with patch("src.common.downloads.urlopen", return_value=io.BytesIO(content)):
            download_verified("https://example.org/archive", target, hashlib.sha256(content).hexdigest())
        self.assertEqual(target.read_bytes(), content)
        self.assertEqual(next(self.root.glob("*.failed-*")).read_bytes(), b"corrupt")

    def test_hash_checked_local_cache_needs_no_network(self):
        cached = self.root / "cached.json"
        cached.write_bytes(b"known bytes")
        with patch("src.common.downloads.urlopen", side_effect=AssertionError("network must not be called")):
            target = download_verified("https://example.org/data", self.root / "downloaded.json",
                                       hashlib.sha256(cached.read_bytes()).hexdigest(), cache_file=cached)
        self.assertEqual(target.read_bytes(), cached.read_bytes())
        with self.assertRaisesRegex(ValueError, "SHA256"):
            download_verified("https://example.org/data", self.root / "other.json", "0" * 64, cache_file=cached)

    def test_never_promotes_unverified_bytes(self):
        target = self.root / "bad.zip"
        with patch("src.common.downloads.shutil.which", return_value=None), patch(
                "src.common.downloads.urlopen", side_effect=lambda *a, **k: io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(RuntimeError, "Download failed"):
                download_verified("https://example.org/archive", target, "0" * 64)
        self.assertFalse(target.exists())
        self.assertEqual(len(list(self.root.glob("*.failed-*"))), 2)

    def test_wsl_copy_does_not_require_windows_metadata_permissions(self):
        source = self.root / 'source'
        source.mkdir()
        (source / 'code.py').write_bytes(b'print(1)\n')
        (source / 'private.local.json').write_text('{}')
        (source / '__pycache__').mkdir()
        (source / '__pycache__/old.pyc').write_bytes(b'cache')
        with patch('shutil.copystat', side_effect=PermissionError('NTFS metadata')):
            entry.copy_sources(source, self.root / 'copied')
        self.assertEqual((self.root / 'copied/code.py').read_bytes(), b'print(1)\n')
        self.assertFalse((self.root / 'copied/private.local.json').exists())
        self.assertFalse((self.root / 'copied/__pycache__').exists())


class PreflightAndStateTests(unittest.TestCase):
    def complete_status(self, **changes):
        status = dict(status='completed_with_differences', run_count=11, n_per_run=777,
            model_loading=True, inference=True, execution_complete=True, resources_verified=True,
            baseline_reference_match=True, qwen_reference_match=False, strict_reference_match=False,
            execution_policy_passed=True)
        status.update(changes)
        return status

    def test_explicit_model_completion_requires_both_proof_and_zero_exit(self):
        status = self.complete_status()
        self.assertTrue(entry.model_execution_succeeded(status, 0))
        self.assertFalse(entry.model_execution_succeeded(status, 1))
        self.assertFalse(entry.model_execution_succeeded({'status': 'completed_with_differences'}, 0))
        for key in ('execution_complete', 'resources_verified', 'baseline_reference_match', 'execution_policy_passed'):
            with self.subTest(key=key):
                self.assertFalse(entry.model_execution_succeeded(self.complete_status(**{key: False}), 0))
        for changes in ({'run_count': 10}, {'n_per_run': 776}, {'qwen_reference_match': None},
                        {'status': 'failed'}, {'status': 'interrupted'}, {'strict_reference_match': True}):
            self.assertFalse(entry.model_execution_succeeded(self.complete_status(**changes), 0))

    def test_old_driver_returns_specific_error_before_any_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs/reproduction-model-environment.json").write_text(
                json.dumps(dict(required_packages={}, torch_cuda="13.0")), encoding="utf-8")
            class OldTorch:
                __version__ = "2.14.1+cu130"
                class cuda:
                    @staticmethod
                    def is_available():
                        return False
            OldTorch.version = type("Version", (), {"cuda": "13.0"})
            with patch.dict("sys.modules", {"torch": OldTorch}), patch(
                    "src.model_preflight.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")):
                result = inspect_environment(root, {})
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("537.70" in e and "R580" in e for e in result["errors"]))
            self.assertFalse(result["model_loading"])

    def test_complete_but_different_is_not_recorded_as_infrastructure_failure(self):
        self.run_model_wrapper_fixture(dict(status='completed_with_differences', run_count=11,
            maximum_metric_difference=0.001, model_loading=True), 1, 1)

    def test_new_complete_qwen_difference_returns_zero_and_keeps_difference_state(self):
        self.run_model_wrapper_fixture(self.complete_status(maximum_metric_difference=.020592,
            prediction_difference_count=85), 0, 0)

    def test_forged_complete_flag_with_nonzero_process_exit_is_not_accepted(self):
        self.run_model_wrapper_fixture(self.complete_status(), 1, 1)

    def run_model_wrapper_fixture(self, model_status, child_code, expected_code):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source"
            source.mkdir()
            for name in ("src", "configs", "scripts", "artifacts"):
                (source / name).mkdir()
            (source / "configs/reproduction.json").write_text('{"input_sha256":{}}', encoding="utf-8")
            for name in ("lora-lock.json", "evaluation-freeze.json", "reproduction-downloads.json"):
                (source / "artifacts" / name).write_text("{}", encoding="utf-8")
            (source / "reproduce.py").write_text("", encoding="utf-8")
            specification = parent / "paths.json"
            specification.write_text("{}", encoding="utf-8")
            work = parent / "run"
            def execute(command, **kwargs):
                if "src.model_reproduction" in command:
                    output = work / "results/model-reproduction"
                    output.mkdir(parents=True)
                    (output / "status.json").write_text(json.dumps(model_status), encoding="utf-8")
                    return subprocess.CompletedProcess(command, child_code)
                return subprocess.CompletedProcess(command, 0)
            with patch.object(entry, "ROOT", source), patch.object(entry.subprocess, "run", side_effect=execute), patch(
                    "src.model_preflight.inspect_environment", return_value={"status": "passed"}):
                code = entry.main(["--models", "--paths", str(specification), "--work-dir", str(work)])
            result = json.loads((work / "submission-status.json").read_text())
            self.assertEqual(code, expected_code)
            self.assertEqual(result["status"], "completed_with_differences")
            self.assertEqual(result["model_result"]["run_count"], 11)
            self.assertEqual(result['teacher_acceptance'], 'not_recorded')
            self.assertEqual(result['model_result'], model_status)


if __name__ == "__main__":
    unittest.main()
