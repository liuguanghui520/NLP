"""Checks for hash-locked evidence retrieval without network access."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import submission_reproduce as entry


class SubmissionDownloadTests(unittest.TestCase):
    def test_path_cannot_escape_download_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                entry.safe_path(Path(temporary), '../outside.json')

    def test_hash_mismatch_is_not_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch('src.common.downloads.shutil.which',return_value=None), patch(
                    'src.common.downloads.urlopen',side_effect=lambda *a,**k: io.BytesIO(b'wrong file')):
                with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):
                    entry.fetch(root,'data/file.json','0'*64)
            self.assertFalse((root/'data/file.json').exists())

    def test_verified_file_is_downloaded_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            content=b'{"verified": true}\n'
            expected=hashlib.sha256(content).hexdigest()
            with patch('src.common.downloads.urlopen',return_value=io.BytesIO(content)) as request:
                self.assertEqual(entry.fetch(root,'data/file.json',expected),'data/file.json')
                entry.fetch(root,'data/file.json',expected)
                self.assertEqual(request.call_count,1)

    def test_existing_conflicting_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'file.json').write_bytes(b'changed')
            with patch('src.common.downloads.urlopen') as request:
                with self.assertRaisesRegex(ValueError,'conflicts'):
                    entry.fetch(root,'file.json','0'*64)
                request.assert_not_called()


if __name__=='__main__':
    unittest.main()
