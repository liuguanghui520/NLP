"""Transport/integrity tests only: fixture bytes are not real model weights."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.package_trained_artifacts import require_digest, safe_member, verify_archive, write_archive


class TrainedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weight = self.root / "fixture.safetensors"
        self.weight.write_bytes(b"fixture bytes for archive tests only")
        self.expected = hashlib.sha256(self.weight.read_bytes()).hexdigest()
        self.metadata = dict(artifact_id="fixture-seed42", weight_member="model/fixture.safetensors",
                             weight_sha256=self.expected, seed=42)
        self.payload = {"model/fixture.safetensors": self.weight, "README.md": b"fixture"}

    def test_round_trip_preserves_source_and_checks_every_member(self):
        before = self.weight.read_bytes()
        archive = self.root / "fixture.zip"
        result = write_archive(archive, self.payload, self.metadata)
        checked = verify_archive(archive, result["archive_sha256"])
        self.assertTrue(checked["verified"])
        self.assertEqual(self.weight.read_bytes(), before)
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read("fixture-seed42/manifest.json"))
        self.assertEqual(manifest["files"]["model/fixture.safetensors"]["sha256"], self.expected)

    def test_existing_archive_is_not_overwritten(self):
        archive = self.root / "fixture.zip"
        archive.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            write_archive(archive, self.payload, self.metadata)
        self.assertEqual(archive.read_bytes(), b"keep")

    def test_wrong_checkpoint_identity_stops_packaging(self):
        metadata = dict(self.metadata, weight_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "frozen checkpoint"):
            write_archive(self.root / "wrong.zip", self.payload, metadata)
        self.assertFalse((self.root / "wrong.zip").exists())

    def test_modified_member_is_detected_without_deserialization(self):
        archive = self.root / "fixture.zip"
        write_archive(archive, self.payload, self.metadata)
        broken = self.root / "corrupt.zip"
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(broken, "w") as target:
            for name in source.namelist():
                value = b"changed" if name.endswith("fixture.safetensors") else source.read(name)
                target.writestr(name, value)
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            verify_archive(broken)

    def test_traversal_and_windows_drive_members_are_rejected(self):
        for name in ["../a", "/a", "x/../../a", "C:/a", "x\\..\\a", "x//a"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_member(name)

    def test_source_digest_is_required(self):
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            require_digest(self.weight, "0" * 64)


if __name__ == "__main__":
    unittest.main()
