#!/usr/bin/env python3
"""Evaluate one prediction JSONL file with the project-wide metric contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import confusion_matrix

from experiment_contracts import (
    ensure_unique_ids,
    load_frozen_split,
    load_ids,
    load_jsonl,
    validate_prediction_row,
    write_json,
)


def evaluate(
    prediction_path: Path,
    expected_records: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    rows = ensure_unique_ids(load_jsonl(prediction_path))
    expected_by_id = {row["sample_id"]: row for row in expected_records}
    prediction_by_id = {row.get("sample_id"): row for row in rows}
    expected_ids = set(expected_by_id)
    predicted_ids = set(prediction_by_id)
    missing_ids = sorted(expected_ids - predicted_ids)
    extra_ids = sorted(predicted_ids - expected_ids)

    errors: list[str] = []
    for row in rows:
        for error in validate_prediction_row(row):
            errors.append(f"{row.get('sample_id')}: {error}")
        expected = expected_by_id.get(row.get("sample_id"))
        if expected is not None and row.get("true_label") != expected.get("label"):
            errors.append(f"{row.get('sample_id')}: true_label mismatch")

    status_counts = Counter(row.get("status") for row in rows)
    invalid_status_count = sum(count for status, count in status_counts.items() if status != "ok")
    complete = not missing_ids and not extra_ids
    structurally_valid = complete and not errors

    if structurally_valid:
        ordered = [prediction_by_id[sample_id] for sample_id in expected_by_id]
        true_labels = [expected_by_id[row["sample_id"]]["label"] for row in ordered]
        predicted_labels = [row["parsed_label"] if row["status"] == "ok" else -1 for row in ordered]
    else:
        ordered = []
        true_labels = []
        predicted_labels = []

    if structurally_valid:
        valid_mask = [prediction != -1 for prediction in predicted_labels]
        valid_true = [truth for truth, valid in zip(true_labels, valid_mask) if valid]
        valid_pred = [prediction for prediction, valid in zip(predicted_labels, valid_mask) if valid]
        matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, -1])
        accuracy = sum(truth == prediction for truth, prediction in zip(true_labels, predicted_labels)) / len(
            true_labels
        )
        per_class = {}
        class_f1_scores = []
        for index, label in enumerate((0, 1)):
            true_positive = int(matrix[index, index])
            false_negative = int(matrix[index, :].sum()) - true_positive
            false_positive = int(matrix[:, index].sum()) - true_positive
            support = int(matrix[index, :].sum())
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            class_f1_scores.append(f1)
            per_class[str(label)] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        metrics = {
            "accuracy": accuracy,
            "macro_f1": sum(class_f1_scores) / len(class_f1_scores),
            "per_class": per_class,
            "confusion_matrix_labels": ["0", "1", "invalid"],
            "confusion_matrix": matrix.tolist(),
            "valid_prediction_count": len(valid_true),
            "valid_subset_accuracy": (
                sum(truth == prediction for truth, prediction in zip(valid_true, valid_pred)) / len(valid_true)
                if valid_true
                else 0.0
            ),
        }
    else:
        metrics = {
            "accuracy": None,
            "macro_f1": None,
            "per_class": None,
            "confusion_matrix_labels": ["0", "1", "invalid"],
            "confusion_matrix": None,
            "valid_prediction_count": None,
            "valid_subset_accuracy": None,
        }

    result = {
        "prediction_file": str(prediction_path),
        "expected_count": len(expected_records),
        "prediction_count": len(rows),
        "complete": complete,
        "structurally_valid": structurally_valid,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "status_counts": dict(sorted(status_counts.items(), key=lambda item: str(item[0]))),
        "non_ok_status_count": invalid_status_count,
        "invalid_output_count": status_counts.get("invalid_format", 0),
        "invalid_output_rate": (
            status_counts.get("invalid_format", 0) / len(expected_records) if expected_records else 0.0
        ),
        "validation_errors": errors,
        "metrics": metrics,
    }
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = load_frozen_split(args.source_split)
    if args.ids_file:
        selected = set(load_ids(args.ids_file))
        expected = [row for row in expected if row["sample_id"] in selected]
    result = evaluate(args.predictions, expected, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["structurally_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
