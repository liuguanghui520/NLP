"""No-network tests for release upload integrity and credential containment."""

import hashlib
import io
import unittest
import urllib.request

from scripts.upload_trained_release import Reader, SafeRedirect


class ReleaseUploadTests(unittest.TestCase):
    def test_stream_reader_preserves_bytes_and_digest(self):
        body = b"verified release payload" * 500
        reader = Reader(io.BytesIO(body), "fixture.zip", len(body), lambda *a, **k: None)
        chunks = []
        while True:
            chunk = reader.read(31)
            if not chunk:
                break
            chunks.append(chunk)
        self.assertEqual(b"".join(chunks), body)
        self.assertEqual(reader.count, len(body))
        self.assertEqual(reader.digest.hexdigest(), hashlib.sha256(body).hexdigest())

    def test_redirect_does_not_forward_authorization(self):
        request = urllib.request.Request("https://api.github.com/asset", headers={"Authorization": "Bearer fixture-not-a-token"})
        redirected = SafeRedirect().redirect_request(request, None, 302, "Found", {}, "https://release-assets.githubusercontent.com/asset")
        self.assertIsNone(redirected.get_header("Authorization"))

    def test_redirect_to_http_is_rejected(self):
        request = urllib.request.Request("https://api.github.com/asset")
        with self.assertRaises(ValueError):
            SafeRedirect().redirect_request(request, None, 302, "Found", {}, "http://example.com/asset")


if __name__ == "__main__":
    unittest.main()
