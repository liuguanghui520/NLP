"""Protect WSL users from repeating the measured mounted-environment timeout."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.model_preflight import inspect_environment, storage_check, windows_mount_path


class StoragePreflightTests(unittest.TestCase):
    def test_detects_custom_drvfs_mount(self):
        with patch('src.model_preflight.subprocess.run', return_value=
                   subprocess.CompletedProcess([], 0, '9p rw,aname=drvfs;path=C:\\,access=client', '')):
            self.assertTrue(windows_mount_path('/custom/windows/env'))

    def test_accepts_native_mount_even_with_mnt_prefix(self):
        with patch('src.model_preflight.subprocess.run', return_value=
                   subprocess.CompletedProcess([], 0, 'ext4 rw,relatime', '')):
            self.assertFalse(windows_mount_path('/mnt/c/native-bind/env'))

    def test_standard_drive_fallback_without_findmnt(self):
        with patch('src.model_preflight.subprocess.run', side_effect=FileNotFoundError):
            # Model Linux-resolved paths even when this test runs on Windows.
            for value, expected in [('/mnt/c/env', True), ('/mnt/d/framework', True),
                                    ('/home/student/env', False)]:
                with patch('src.model_preflight.Path') as paths:
                    paths.return_value.expanduser.return_value.resolve.return_value.as_posix.return_value = value
                    self.assertEqual(windows_mount_path(value), expected)

    def test_checks_symlink_target(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'target'
            target.mkdir()
            link = Path(temp) / 'link'
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest('Symlink creation requires permission on this Windows host')
            with patch('src.model_preflight.subprocess.run', return_value=
                       subprocess.CompletedProcess([], 0, 'drvfs rw', '')) as probe:
                self.assertTrue(windows_mount_path(link))
                self.assertEqual(probe.call_args.args[0][-1], str(target.resolve()))

    def test_mounted_environment_stops_before_imports(self):
        root = Path(__file__).resolve().parents[1]
        with patch('src.model_preflight.is_wsl', return_value=True), \
                patch('src.model_preflight.windows_mount_path', return_value=True), \
                patch('src.model_preflight.importlib.metadata.version') as versions:
            result = inspect_environment(root, {'factory_root': '/mnt/c/factory'})
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['stopped_before_package_imports'])
        self.assertEqual(len(result['errors']), 2)
        self.assertIn('不要直接剪切虚拟环境', result['errors'][0])
        versions.assert_not_called()

    def test_project_on_windows_is_only_warning(self):
        with patch('src.model_preflight.is_wsl', return_value=True), \
                patch('src.model_preflight.windows_mount_path', side_effect=[False, False, True]):
            result = storage_check('/mnt/c/project', {'factory_root': '/home/student/factory'})
        self.assertEqual(result['errors'], [])
        self.assertEqual(len(result['warnings']), 1)

    def test_non_wsl_does_not_reject_host_paths(self):
        with patch('src.model_preflight.is_wsl', return_value=False), \
                patch('src.model_preflight.windows_mount_path') as probe:
            self.assertEqual(storage_check('.', {})['errors'], [])
        probe.assert_not_called()
