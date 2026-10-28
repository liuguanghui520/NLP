#!/usr/bin/env python3
"""Shared constants and validation helpers for the experiment pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_VERSION = "v1-20261023"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / DATA_VERSION
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-1.7B"
PROMPT_REGISTRY = PROJECT_ROOT / "prompts" / "prompt_registry.json"
RUNS_DIR = PROJECT_ROOT / "runs"

PREDICTION_FIELDS = {
    "run_id",
    "owner",
    "method",
    "config_id",
    "seed",
    "sample_id",
    "split",
    "true_label",
    "raw_output",
    "parsed_label",
    "status",
    "prompt_id",
    "input_chars",
    "input_tokens",
    "truncated",
    "latency_ms",
    "timing_scope",
    "error_message",
    "model_id",
    "checkpoint_id",
}

ALLOWED_STATUSES = {
    "ok",
    "invalid_format",
    "missing_prediction",
    "timeout",
    "oom",
    "runtime_error",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rebuild_run_index(runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    """Rebuild the human- and machine-readable index from run manifests."""
    records: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
        else:
            manifest = {"status": "missing_manifest"}
        records.append(
            {
                "run_id": run_dir.name,
                "owner": manifest.get("owner"),
                "method": manifest.get("method"),
                "purpose": manifest.get("purpose"),
                "status": manifest.get("status"),
                "superseded_by": manifest.get("superseded_by"),
            }
        )

    index = {"generated_at_utc": utc_iso(), "runs": records}
    write_json(runs_dir / "run_index.json", index)
    lines = [
        "# Experiment Run Index",
        "",
        "| run_id | owner | method | purpose | status | superseded_by |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| `{item['run_id']}` | {item['owner'] or ''} | {item['method'] or ''} | "
        f"{item['purpose'] or ''} | {item['status'] or ''} | {item['superseded_by'] or ''} |"
        for item in records
    )
    (runs_dir / "run_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_frozen_split(split: str) -> list[dict[str, Any]]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    rows = load_jsonl(DATA_DIR / f"{split}.jsonl")
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"Sample {row.get('sample_id')} has mismatched split.")
        if row.get("label") not in (0, 1):
            raise ValueError(f"Sample {row.get('sample_id')} has invalid label.")
    return rows


def load_ids(path: Path) -> list[str]:
    payload = load_json(path)
    ids = payload["sample_ids"] if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError(f"Invalid sample ID list: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return ids


def strict_parse_label(raw_output: str | None) -> tuple[int | None, str]:
    if raw_output is None:
        return None, "invalid_format"
    stripped = raw_output.strip()
    if stripped in {"0", "1"}:
        return int(stripped), "ok"
    return None, "invalid_format"


def validate_prediction_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(PREDICTION_FIELDS - row.keys())
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if row.get("true_label") not in (0, 1):
        errors.append("true_label must be 0 or 1")
    parsed = row.get("parsed_label")
    if parsed not in (0, 1, None):
        errors.append("parsed_label must be 0, 1, or null")
    if row.get("status") not in ALLOWED_STATUSES:
        errors.append(f"invalid status: {row.get('status')}")
    if row.get("truncated") not in (True, False):
        errors.append("truncated must be boolean")
    return errors


def ensure_unique_ids(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(rows)
    ids = [row.get("sample_id") for row in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id values in prediction file.")
    return materialized
