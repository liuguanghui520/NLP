"""Inspect existing local SwanLab backups without importing SwanLab or models.

Run in the recorded WSL tool environment. Only hashes, field names, counters and
matching sample IDs are saved. No console text, contact values or account secrets.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import zlib

from audit_data_usage import RULES


def sha(data):
    return hashlib.sha256(data).hexdigest()


def records(raw):
    assert struct.unpack("<4sHB", raw[:7]) == (b":SWL", 0xE1D6, 0)
    offset, chunks = 7, []
    while offset < len(raw):
        remaining = 32768 - offset % 32768
        if remaining < 7:
            assert raw[offset:offset + remaining] == b"\0" * remaining
            offset += remaining
            if offset == len(raw):
                break
        assert offset + 7 <= len(raw)
        crc, length, kind = struct.unpack("<IHB", raw[offset:offset + 7])
        assert kind in (1, 2, 3, 4)
        offset += 7
        payload = raw[offset:offset + length]
        assert len(payload) == length
        offset += length
        assert zlib.crc32(payload, zlib.crc32(bytes([kind]))) & 0xFFFFFFFF == crc
        if kind == 1:
            assert not chunks
            yield json.loads(payload)
        elif kind == 2:
            assert not chunks
            chunks.append(payload)
        elif kind == 3:
            assert chunks
            chunks.append(payload)
        else:
            assert chunks
            chunks.append(payload)
            yield json.loads(b"".join(chunks))
            chunks = []
    assert not chunks


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def names(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from names(child)
    elif isinstance(value, list):
        for child in value:
            yield from names(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--swanlab-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    project = Path(__file__).resolve().parents[1]
    samples = []
    for split in ("train", "validation", "test"):
        for line in (project / "data/processed/v1-20261023" / (split + ".jsonl")).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            samples.append((row["sample_id"], row["text"]))
    # Parse YAML with the already installed tool dependency, without importing
    # SwanLab, Transformers, PyTorch, or account/config discovery mechanisms.
    sys.path.insert(0, str(args.swanlab_root.parent))
    import yaml

    output = []
    paths = sorted((args.tool_root / "project-workspaces/qwen17-comparisons").glob("*/T01*/swanlog/*/backup.swanlab"))
    assert len(paths) == 3, f"Expected three final T01 mirrors, found {len(paths)}"
    for path in paths:
        raw = path.read_bytes()
        decoded = list(records(raw))
        types = Counter(str(row.get("model_type", "unknown")) for row in decoded)
        data_keys = {}
        for row in decoded:
            data_keys.setdefault(str(row.get("model_type", "unknown")), set()).update(names(row.get("data", {})))
        texts = [(str(row.get("model_type", "unknown")), "\n".join(strings(row))) for row in decoded]
        aggregate = "\n".join(text for _, text in texts)
        hits = []
        for sample_id, text in samples:
            if len(text) >= 16 and text in aggregate:
                hits.append({"sample_id": sample_id, "kind": "full_visible_text", "record_types": sorted({kind for kind, value in texts if text in value})})
            elif len(text) >= 32 and text[:32] in aggregate:
                hits.append({"sample_id": sample_id, "kind": "first_32_characters_only", "record_types": sorted({kind for kind, value in texts if text[:32] in value})})
        config = path.parent / "files/config.yaml"
        metadata = path.parent / "files/swanlab-metadata.json"
        metadata_value = json.loads(metadata.read_text())
        sensitive_keys = []

        def check_keys(value, prefix=""):
            if isinstance(value, dict):
                for key, child in value.items():
                    field = prefix + str(key)
                    if re.search(r"api_key|secret|password|access_token", str(key), re.I):
                        empty = child in (None, "", False)
                        placeholder = isinstance(child, str) and child.strip() in ("<SWANLAB_API_KEY>", "[SWANLAB_API_KEY]", "${SWANLAB_API_KEY}", "REDACTED", "[REDACTED]", "***")
                        sensitive_keys.append({"field": field, "value_state": "empty" if empty else "placeholder" if placeholder else "nonempty_unclassified"})
                    else:
                        check_keys(child, field + ".")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    check_keys(child, prefix + str(index) + ".")

        config_data = yaml.safe_load(config.read_text())
        check_keys(config_data)
        config_text = json.dumps(config_data, ensure_ascii=False, default=str)
        config_text_hits = [sample_id for sample_id, text in samples if len(text) >= 32 and text[:32] in config_text]
        contact_hits = set()
        for sample_id, text in samples:
            for rule in RULES.values():
                for match in re.finditer(rule, text, re.I):
                    if match.group() in aggregate or match.group() in config_text:
                        contact_hits.add(sample_id)
        metric_keys = sorted({str(row["data"].get("key", "")) for row in decoded if row.get("model_type") in ("Column", "Scalar")})
        output.append({"path": str(path), "sha256": sha(raw), "bytes": len(raw), "crc_and_records_valid": True, "records": len(decoded), "record_types": dict(types), "metric_keys": metric_keys, "data_field_names_by_type": {key: sorted(value) for key, value in data_keys.items()}, "dataset_text_matches": hits, "config_dataset_text_match_ids": config_text_hits, "visible_comment_contact_match_ids": sorted(contact_hits), "sample_text_matching_scope": "Exact full visible text >=16 characters; otherwise exact first 32 characters. No fuzzy matching, truncation reconstruction or cloud-account inspection.", "has_training_example_marker": "training example:" in aggregate, "config_yaml_sha256": sha(config.read_bytes()), "metadata_sha256": sha(metadata.read_bytes()), "metadata_field_names": sorted(metadata_value), "credential_field_presence_only": sensitive_keys})
        assert sha(path.read_bytes()) == sha(raw)
    source_files = [args.tool_root / "src/llamafactory/train/trainer_utils.py", args.tool_root / "src/llamafactory/data/loader.py", args.swanlab_root / "integration/transformers.py", args.swanlab_root / "log/backup/datastore.py"]
    payload = {"checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "scope": "Existing local backup mirrors for selected LoRA seeds42/43/44, installed integration code; no cloud login, upload or deletion.", "sources_sha256": {str(path): sha(path.read_bytes()) for path in source_files}, "script_sha256": sha(Path(__file__).read_bytes()), "backups": output, "training": False, "model_loading": False, "inference": False}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": [{"records": row["records"], "types": row["record_types"], "text_matches": len(row["dataset_text_matches"]), "example_marker": row["has_training_example_marker"]} for row in output]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
