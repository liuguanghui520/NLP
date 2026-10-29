#!/usr/bin/env python3
"""Independent standard-library audit of the successful saved-prediction run."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_metrics(records):
    n = len(records)
    correct = sum(
        record.get("status") == "ok"
        and record.get("parsed_label") == record["true_label"]
        for record in records
    )
    invalid = sum(
        record.get("status") != "ok"
        or record.get("parsed_label") not in (0, 1)
        for record in records
    )
    per_class = {}
    f1_values = []
    for label in (0, 1):
        tp = sum(
            record["true_label"] == label
            and record.get("parsed_label") == label
            for record in records
        )
        fp = sum(
            record["true_label"] != label
            and record.get("parsed_label") == label
            for record in records
        )
        fn = sum(
            record["true_label"] == label
            and record.get("parsed_label") != label
            for record in records
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    weighted_f1 = sum(
        per_class[str(label)]["support"] / n * per_class[str(label)]["f1"]
        for label in (0, 1)
    ) if n else 0.0
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": statistics.mean(f1_values),
        "weighted_f1": weighted_f1,
        "invalid_count": invalid,
        "invalid_rate": invalid / n if n else 0.0,
        "per_class": per_class,
    }


def close(actual, expected, tolerance=1e-12):
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: independent_audit.py <successful-work-dir>")
    root = Path(sys.argv[1]).resolve()
    config = load_json(root / "configs/reproduction.json")

    hash_checks = {}
    for relative, expected in config["input_sha256"].items():
        path = root / relative
        actual = sha256(path) if path.is_file() else None
        hash_checks[relative] = {
            "expected": expected,
            "actual": actual,
            "match": actual == expected,
        }

    split_counts = {}
    split_records = {}
    for split in ("train", "validation", "test"):
        records = load_jsonl(root / "data/processed/v1-20261023" / f"{split}.jsonl")
        split_records[split] = records
        split_counts[split] = len(records)

    run_index = load_jsonl(root / "results/run_index.jsonl")
    if len(run_index) != 16:
        raise AssertionError(f"Expected 16 run records, found {len(run_index)}")

    run_audit = []
    metric_failures = []
    by_method = defaultdict(list)
    for item in run_index:
        split = item["split"]
        expected_rows = split_records[split]
        predictions = load_jsonl(root / "results/predictions" / f"{item['run_id']}.jsonl")
        expected_ids = [record["sample_id"] for record in expected_rows]
        actual_ids = [record["sample_id"] for record in predictions]
        truth = {record["sample_id"]: record["label"] for record in expected_rows}
        truth_mismatches = [
            record["sample_id"]
            for record in predictions
            if record["sample_id"] not in truth
            or record["true_label"] != truth[record["sample_id"]]
        ]
        metrics = calculate_metrics(predictions)
        metric_differences = {
            key: metrics[key] - item["metrics"][key]
            for key in ("macro_f1", "accuracy", "weighted_f1", "invalid_rate")
        }
        matches = (
            len(predictions) == len(expected_rows)
            and len(actual_ids) == len(set(actual_ids))
            and set(actual_ids) == set(expected_ids)
            and not truth_mismatches
            and all(close(metrics[key], item["metrics"][key]) for key in metric_differences)
        )
        if not matches:
            metric_failures.append(item["run_id"])
        run = {
            "run_id": item["run_id"],
            "method": item["method"],
            "split": split,
            "seed": item["seed"],
            "prediction_count": len(predictions),
            "expected_count": len(expected_rows),
            "unique_sample_id_count": len(set(actual_ids)),
            "sample_id_set_match": set(actual_ids) == set(expected_ids),
            "truth_mismatch_count": len(truth_mismatches),
            "metrics": metrics,
            "metric_differences": metric_differences,
            "status": "passed" if matches else "failed",
        }
        run_audit.append(run)
        if split == "test":
            by_method[item["method"]].append(run)

    method_summary = {}
    for method in config["method_order"]:
        runs = by_method[method]
        method_summary[method] = {
            "runs": len(runs),
            "seeds": [run["seed"] for run in runs],
            "sample_counts": [run["prediction_count"] for run in runs],
            "macro_f1_mean": statistics.mean(run["metrics"]["macro_f1"] for run in runs),
            "macro_f1_sample_std": (
                statistics.stdev(run["metrics"]["macro_f1"] for run in runs)
                if len(runs) > 1
                else None
            ),
            "accuracy_mean": statistics.mean(run["metrics"]["accuracy"] for run in runs),
            "invalid_counts": [run["metrics"]["invalid_count"] for run in runs],
            "reference": config["expected_test_means"][method],
        }

    output = {
        "status": "passed"
        if all(check["match"] for check in hash_checks.values())
        and split_counts == config["split_counts"]
        and not metric_failures
        and len(run_audit) == 16
        and all(summary["runs"] == len(summary["sample_counts"]) for summary in method_summary.values())
        else "failed",
        "work_dir": str(root),
        "input_hash_count": len(hash_checks),
        "input_hashes_all_match": all(check["match"] for check in hash_checks.values()),
        "hash_checks": hash_checks,
        "split_counts": split_counts,
        "expected_split_counts": config["split_counts"],
        "run_record_count": len(run_index),
        "final_test_run_count": sum(run["split"] == "test" for run in run_audit),
        "final_test_counts": {
            run["run_id"]: run["prediction_count"]
            for run in run_audit
            if run["split"] == "test"
        },
        "metric_failures": metric_failures,
        "runs": run_audit,
        "method_summary": method_summary,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
