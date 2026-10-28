#!/usr/bin/env python3
"""Build the frozen ChnSentiCorp data artifacts reproducibly."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


DATASET_ID = "AiNiklaus/ChnSentiCorp"
DATASET_URL = "https://modelscope.cn/datasets/AiNiklaus/ChnSentiCorp"
VERIFIED_CONTENT_REVISION = "eec2cd0294bbd8711cbbbaab13ec4c2320d7f815"
PROCESSING_VERSION = "v1-20261023"
SPLIT_SEED = 67
TRAIN_SEED = 42
PREFIX_CHARS = 512
INSTRUCTION = "判断评论整体情感，仅输出0表示负面、1表示正面。"
EXPECTED_RAW_ROWS = 7766
EXPECTED_RAW_LABELS = {0: 2444, 1: 5322}
EXPECTED_SPLITS = {
    "train": {0: 1954, 1: 4256},
    "validation": {0: 244, 1: 532},
    "test": {0: 245, 1: 532},
}
EXPECTED_EXCLUSIONS = [(3071, "duplicate_512_prefix"), (3729, "duplicate_512_prefix"), (6375, "empty_text")]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(raw_text: str) -> str:
    return raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()


def json_bytes(value: Any, *, compact: bool = False) -> bytes:
    options = {"ensure_ascii": False, "sort_keys": compact}
    if compact:
        options["separators"] = (",", ":")
    else:
        options["indent"] = 2
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(json_bytes(row, compact=True) for row in rows)


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def category_sample(rows: list[dict[str, Any]], label: int, count: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["label"] == label]
    candidates.sort(key=lambda row: row["source_row"])
    if count > len(candidates):
        raise RuntimeError(f"Cannot sample {count} rows for label {label}; available={len(candidates)}")
    return rng.sample(candidates, count)


def build(raw_csv: Path, output_dir: Path, script_path: Path) -> dict[str, Any]:
    raw_sha = sha256_file(raw_csv)
    if raw_sha != "1d414b9c65caf3a836abdb4efded621968502ad3338e1a9d885ae131c86083f0":
        raise RuntimeError(f"Unexpected raw CSV SHA256: {raw_sha}")

    parsed_rows: list[dict[str, Any]] = []
    with raw_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["label", "review"]:
            raise RuntimeError(f"Unexpected columns: {reader.fieldnames}")
        for source_row, row in enumerate(reader, start=1):
            raw_label = int(row["label"])
            raw_text = row["review"]
            if raw_label not in (0, 1):
                raise RuntimeError(f"Invalid label at source row {source_row}: {raw_label}")
            clean_text = normalized_text(raw_text)
            parsed_rows.append(
                {
                    "sample_id": f"{raw_sha}:{source_row}",
                    "source_sha256": raw_sha,
                    "source_row": source_row,
                    "raw_text": raw_text,
                    "normalized_text": clean_text,
                    "raw_label": raw_label,
                    "label": raw_label,
                    "raw_char_count": len(raw_text),
                    "normalized_char_count": len(clean_text),
                    "line_endings_normalized": raw_text.replace("\r\n", "\n").replace("\r", "\n") != raw_text,
                    "boundary_whitespace_stripped": (
                        raw_text.replace("\r\n", "\n").replace("\r", "\n") != clean_text
                    ),
                }
            )

    if len(parsed_rows) != EXPECTED_RAW_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_RAW_ROWS} raw records, got {len(parsed_rows)}")
    raw_label_counts = Counter(row["raw_label"] for row in parsed_rows)
    if dict(sorted(raw_label_counts.items())) != EXPECTED_RAW_LABELS:
        raise RuntimeError(f"Unexpected raw label counts: {dict(raw_label_counts)}")

    excluded: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    by_exact_text: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in parsed_rows:
        if not row["normalized_text"]:
            excluded.append(
                {
                    **row,
                    "exclusion_reason": "empty_text",
                    "kept_sample_id": None,
                }
            )
            continue
        by_exact_text[row["normalized_text"]].append(row)

    exact_kept: list[dict[str, Any]] = []
    for text, rows in by_exact_text.items():
        rows.sort(key=lambda row: row["source_row"])
        labels = {row["raw_label"] for row in rows}
        if len(labels) > 1:
            conflicts.append(
                {
                    "normalized_text": text,
                    "sample_ids": [row["sample_id"] for row in rows],
                    "source_rows": [row["source_row"] for row in rows],
                    "labels": [row["raw_label"] for row in rows],
                    "reason": "exact_text_label_conflict",
                }
            )
            continue
        kept = rows[0]
        kept["exact_duplicate"] = False
        exact_kept.append(kept)
        for duplicate in rows[1:]:
            excluded.append(
                {
                    **duplicate,
                    "exclusion_reason": "exact_duplicate",
                    "kept_sample_id": kept["sample_id"],
                }
            )

    if conflicts:
        raise RuntimeError("Exact-text label conflicts require human adjudication.")

    by_prefix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exact_kept:
        by_prefix[row["normalized_text"][:PREFIX_CHARS]].append(row)

    final_rows: list[dict[str, Any]] = []
    for prefix, rows in by_prefix.items():
        rows.sort(key=lambda row: row["source_row"])
        kept = rows[0]
        kept["prefix_duplicate"] = False
        kept["text"] = prefix
        kept["text_char_count"] = len(prefix)
        kept["truncated"] = len(kept["normalized_text"]) > PREFIX_CHARS
        kept["empty_text"] = False
        final_rows.append(kept)
        for duplicate in rows[1:]:
            if duplicate["normalized_text"] == kept["normalized_text"]:
                raise RuntimeError("Exact duplicates must be handled before prefix duplicates.")
            excluded.append(
                {
                    **duplicate,
                    "text": prefix,
                    "text_char_count": len(prefix),
                    "truncated": len(duplicate["normalized_text"]) > PREFIX_CHARS,
                    "exact_duplicate": False,
                    "prefix_duplicate": True,
                    "empty_text": False,
                    "exclusion_reason": "duplicate_512_prefix",
                    "kept_sample_id": kept["sample_id"],
                }
            )

    final_rows.sort(key=lambda row: row["source_row"])
    excluded.sort(key=lambda row: row["source_row"])
    exclusion_keys = [(row["source_row"], row["exclusion_reason"]) for row in excluded]
    if exclusion_keys != EXPECTED_EXCLUSIONS:
        raise RuntimeError(f"Unexpected exclusions: {exclusion_keys}")
    if len(final_rows) != 7763:
        raise RuntimeError(f"Expected 7763 final records, got {len(final_rows)}")
    if sum(row["truncated"] for row in final_rows) != 203:
        raise RuntimeError("Unexpected number of records requiring 512-character truncation.")

    labels = [row["label"] for row in final_rows]
    train_indices, temp_indices = train_test_split(
        list(range(len(final_rows))),
        test_size=0.2,
        random_state=SPLIT_SEED,
        stratify=labels,
    )
    temp_labels = [final_rows[index]["label"] for index in temp_indices]
    validation_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=SPLIT_SEED,
        stratify=temp_labels,
    )

    split_members = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split, indices in split_members.items():
        rows = [dict(final_rows[index], split=split) for index in indices]
        rows.sort(key=lambda row: row["source_row"])
        actual = Counter(row["label"] for row in rows)
        if dict(sorted(actual.items())) != EXPECTED_SPLITS[split]:
            raise RuntimeError(f"Unexpected {split} counts: {dict(actual)}")
        split_records[split] = rows

    full_text_sets = {
        split: {row["normalized_text"] for row in rows} for split, rows in split_records.items()
    }
    truncated_text_sets = {split: {row["text"] for row in rows} for split, rows in split_records.items()}
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    cross_split_full_duplicates = sum(
        len(full_text_sets[left] & full_text_sets[right]) for left, right in split_pairs
    )
    cross_split_truncated_duplicates = sum(
        len(truncated_text_sets[left] & truncated_text_sets[right]) for left, right in split_pairs
    )
    if cross_split_full_duplicates or cross_split_truncated_duplicates:
        raise RuntimeError(
            "Cross-split duplicates found: "
            f"full={cross_split_full_duplicates}, truncated={cross_split_truncated_duplicates}"
        )

    train_rows = split_records["train"]
    smoke_rng = random.Random(SPLIT_SEED)
    smoke_negative = category_sample(train_rows, 0, 10, smoke_rng)
    smoke_positive = category_sample(train_rows, 1, 10, smoke_rng)
    smoke_rows = [*smoke_negative, *smoke_positive]

    short_train_rows = [row for row in train_rows if len(row["normalized_text"]) <= 80]
    few_shot_rng = random.Random(SPLIT_SEED)
    few_negative = category_sample(short_train_rows, 0, 2, few_shot_rng)
    few_positive = category_sample(short_train_rows, 1, 2, few_shot_rng)
    few_shot_rows = [
        few_negative[0],
        few_positive[0],
        few_negative[1],
        few_positive[1],
    ]

    test_rows = split_records["test"]
    latency_sample, _ = train_test_split(
        test_rows,
        train_size=200,
        random_state=SPLIT_SEED,
        stratify=[row["label"] for row in test_rows],
    )

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)

    split_files: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        path = output_dir / f"{split}.jsonl"
        write_bytes(path, jsonl_bytes(split_records[split]))
        split_files[split] = path

    write_bytes(output_dir / "excluded.jsonl", jsonl_bytes(excluded))
    write_bytes(output_dir / "conflicts.jsonl", b"")
    write_bytes(
        output_dir / "split_ids.json",
        json_bytes({split: [row["sample_id"] for row in rows] for split, rows in split_records.items()}),
    )
    write_bytes(
        output_dir / "smoke_ids.json",
        json_bytes(
            {
                "version": PROCESSING_VERSION,
                "source_split": "train",
                "seed": SPLIT_SEED,
                "per_class": 10,
                "sample_ids": [row["sample_id"] for row in smoke_rows],
            }
        ),
    )
    write_bytes(
        output_dir / "fewshot_ids.json",
        json_bytes(
            {
                "version": PROCESSING_VERSION,
                "source_split": "train",
                "seed": SPLIT_SEED,
                "max_characters": 80,
                "order": ["negative", "positive", "negative", "positive"],
                "sample_ids": [row["sample_id"] for row in few_shot_rows],
            }
        ),
    )
    write_bytes(
        output_dir / "latency_ids.json",
        json_bytes(
            {
                "version": PROCESSING_VERSION,
                "source_split": "test",
                "seed": SPLIT_SEED,
                "sample_size": 200,
                "sample_ids": [row["sample_id"] for row in latency_sample],
            }
        ),
    )

    alpaca_dir = output_dir / "alpaca"
    for split, rows in split_records.items():
        alpaca_rows = [
            {
                "instruction": INSTRUCTION,
                "input": row["text"],
                "output": str(row["label"]),
            }
            for row in rows
        ]
        index_map = {str(index): row["sample_id"] for index, row in enumerate(rows)}
        write_bytes(alpaca_dir / f"{split}.json", json_bytes(alpaca_rows))
        write_bytes(alpaca_dir / f"{split}_index.json", json_bytes(index_map))

    write_bytes(
        alpaca_dir / "dataset_info.json",
        json_bytes(
            {
                "chnsenticorp_v1_train": {
                    "file_name": "train.json",
                    "formatting": "alpaca",
                    "columns": {"prompt": "instruction", "query": "input", "response": "output"},
                },
                "chnsenticorp_v1_validation": {
                    "file_name": "validation.json",
                    "formatting": "alpaca",
                    "columns": {"prompt": "instruction", "query": "input", "response": "output"},
                },
            }
        ),
    )

    config = {
        "dataset_id": DATASET_ID,
        "processing_version": PROCESSING_VERSION,
        "split_seed": SPLIT_SEED,
        "model_training_seeds": [42, 43, 44],
        "train_fraction": 0.8,
        "validation_test_fraction": 0.5,
        "prefix_characters": PREFIX_CHARS,
        "normalization": ["CRLF_to_LF", "CR_to_LF", "strip_outer_whitespace"],
        "deduplication": ["exact_normalized_text", "first_512_unicode_characters"],
        "stratify": True,
        "shuffle": True,
        "instruction": INSTRUCTION,
    }
    write_bytes(output_dir / "processing_config.json", json_bytes(config))
    write_bytes(
        output_dir / "数据处理运行说明.md",
        (
            "# 数据处理运行说明\n\n"
            "在 Python 3.11 + scikit-learn 1.9.1 环境中执行：\n\n"
            "```bash\n"
            "python scripts/build_frozen_dataset.py build \\\n"
            "  --raw data/raw/ChnSentiCorp/ChnSentiCorp_htl_all.csv \\\n"
            "  --output data/processed/v1-20261023\n"
            "python scripts/build_frozen_dataset.py verify \\\n"
            "  --raw data/raw/ChnSentiCorp/ChnSentiCorp_htl_all.csv \\\n"
            "  --output data/processed/v1-20261023\n"
            "```\n\n"
            "脚本拒绝覆盖已有输出目录。verify 会在临时目录独立重建并逐字节比较。\n"
        ).encode("utf-8"),
    )
    write_bytes(
        output_dir / "validation_report.md",
        (
            "# ChnSentiCorp v1-20261023 核验报告\n\n"
            "状态：**通过**。\n\n"
            "## 来源\n\n"
            f"- 数据集：`{DATASET_ID}`\n"
            f"- 同内容核验 revision：`{VERIFIED_CONTENT_REVISION}`\n"
            f"- 原始文件：`{raw_csv.name}`\n"
            f"- 原始大小：`{raw_csv.stat().st_size}` bytes\n"
            f"- SHA256：`{raw_sha}`\n\n"
            "## 数量与划分\n\n"
            f"- 原始记录：{len(parsed_rows)}（负面 {raw_label_counts[0]} / 正面 {raw_label_counts[1]}）。\n"
            f"- 排除记录：{len(excluded)}；最终记录：{len(final_rows)}。\n"
            f"- 需要 512 字符截断：{sum(row['truncated'] for row in final_rows)}。\n\n"
            "| 划分 | 总数 | 负面 0 | 正面 1 |\n"
            "|---|---:|---:|---:|\n"
            f"| train | {len(split_records['train'])} | "
            f"{sum(row['label'] == 0 for row in split_records['train'])} | "
            f"{sum(row['label'] == 1 for row in split_records['train'])} |\n"
            f"| validation | {len(split_records['validation'])} | "
            f"{sum(row['label'] == 0 for row in split_records['validation'])} | "
            f"{sum(row['label'] == 1 for row in split_records['validation'])} |\n"
            f"| test | {len(split_records['test'])} | "
            f"{sum(row['label'] == 0 for row in split_records['test'])} | "
            f"{sum(row['label'] == 1 for row in split_records['test'])} |\n\n"
            "## 排除记录\n\n"
            + "\n".join(
                f"- source_row={row['source_row']}，reason=`{row['exclusion_reason']}`，"
                f"kept_sample_id={row['kept_sample_id']}。"
                for row in excluded
            )
            + "\n\n## 泄漏检查\n\n"
            f"- 跨划分完整正文重复：{cross_split_full_duplicates}。\n"
            f"- 跨划分 512 字符正文重复：{cross_split_truncated_duplicates}。\n"
            "- 少样本 ID 均属于训练集。\n"
            "- 原始 CSV SHA256 与数据说明一致。\n"
        ).encode("utf-8"),
    )

    stable_artifacts = {
        path.relative_to(output_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "processing_version": PROCESSING_VERSION,
        "source": {
            "download_ref": "master",
            "verified_same_content_revision": VERIFIED_CONTENT_REVISION,
            "raw_file": raw_csv.name,
            "raw_bytes": raw_csv.stat().st_size,
            "raw_sha256": raw_sha,
        },
        "config": config,
        "dependency_versions": {
            "python": "3.11.17",
            "scikit-learn": "1.9.1",
        },
        "script": {
            "path": script_path.name,
            "sha256": sha256_file(script_path),
        },
        "counts": {
            "raw_records": len(parsed_rows),
            "raw_labels": dict(sorted(raw_label_counts.items())),
            "excluded_records": len(excluded),
            "final_records": len(final_rows),
            "truncated_records": sum(row["truncated"] for row in final_rows),
            "split_records": {split: len(rows) for split, rows in split_records.items()},
            "split_labels": {
                split: dict(sorted(Counter(row["label"] for row in rows).items()))
                for split, rows in split_records.items()
            },
        },
        "exclusions": [
            {
                "source_row": row["source_row"],
                "sample_id": row["sample_id"],
                "reason": row["exclusion_reason"],
                "kept_sample_id": row["kept_sample_id"],
            }
            for row in excluded
        ],
        "fixed_subsets": {
            "smoke": {"seed": SPLIT_SEED, "size": len(smoke_rows)},
            "few_shot": {"seed": SPLIT_SEED, "size": len(few_shot_rows)},
            "latency": {"seed": SPLIT_SEED, "size": len(latency_sample)},
        },
        "artifacts_before_manifest": stable_artifacts,
        "validation": {
            "status": "passed",
            "cross_split_full_text_duplicates": cross_split_full_duplicates,
            "cross_split_truncated_text_duplicates": cross_split_truncated_duplicates,
            "conflicts": 0,
        },
        "license_note": (
            "ModelScope README contains template metadata 'Apache License 2.0'. "
            "Independent permission to redistribute review text was not established; "
            "do not publish the text files without further permission review."
        ),
    }
    write_bytes(output_dir / "manifest.json", json_bytes(manifest))

    checksums = {
        path.relative_to(output_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    write_bytes(output_dir / "checksums.json", json_bytes(checksums))
    return manifest


def verify_rebuild(raw_csv: Path, output_dir: Path, script_path: Path) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    with tempfile.TemporaryDirectory(prefix="chnsenticorp-verify-") as temporary:
        rebuilt = Path(temporary) / PROCESSING_VERSION
        build(raw_csv, rebuilt, script_path)
        final_files = {
            path.relative_to(output_dir).as_posix(): path
            for path in output_dir.rglob("*")
            if path.is_file()
        }
        rebuilt_files = {
            path.relative_to(rebuilt).as_posix(): path
            for path in rebuilt.rglob("*")
            if path.is_file()
        }
        if final_files.keys() != rebuilt_files.keys():
            raise RuntimeError(
                f"Artifact names differ: expected={sorted(final_files)}, rebuilt={sorted(rebuilt_files)}"
            )
        mismatches = [
            name
            for name in final_files
            if final_files[name].read_bytes() != rebuilt_files[name].read_bytes()
        ]
        if mismatches:
            raise RuntimeError(f"Artifact byte mismatches: {mismatches}")
        print(f"VERIFY_OK artifacts={len(final_files)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script_path = Path(__file__).resolve()

    if args.command == "build":
        manifest = build(args.raw.resolve(), args.output.resolve(), script_path)
        print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    else:
        verify_rebuild(args.raw.resolve(), args.output.resolve(), script_path)


if __name__ == "__main__":
    main()
