#!/usr/bin/env python3
"""Summarize measured per-sample latency and verify the frozen ID set."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from experiment_contracts import load_ids, load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_ids = load_ids(args.ids_file)
    by_id = {row["sample_id"]: row for row in load_jsonl(args.predictions)}
    missing = [sample_id for sample_id in expected_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Missing latency samples: {missing[:10]} (total={len(missing)})")
    selected = [by_id[sample_id] for sample_id in expected_ids]
    latencies = [float(row["latency_ms"]) for row in selected]
    if not all(value >= 0 for value in latencies):
        raise ValueError("Latency values must be non-negative.")

    summary = {
        "sample_count": len(selected),
        "ids_file": str(args.ids_file),
        "timing_scope_values": sorted({row["timing_scope"] for row in selected}),
        "mean_ms": statistics.fmean(latencies),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "throughput_samples_per_second": (
            len(latencies) / (sum(latencies) / 1000) if sum(latencies) else None
        ),
    }
    write_json(args.output, summary)
    print(summary)


if __name__ == "__main__":
    main()
