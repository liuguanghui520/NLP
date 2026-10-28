#!/usr/bin/env python3
"""TF-IDF + logistic-regression smoke test with the frozen data contract."""

from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluate_predictions import evaluate
from experiment_contracts import (
    DATA_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    append_jsonl,
    load_frozen_split,
    load_ids,
    sha256_file,
    utc_iso,
    write_json,
)


RUN_ID = "B_tfidf_lr_c1_md1_preflight_s42_20261024T132100Z"


def main() -> None:
    run_dir = RUNS_DIR / RUN_ID
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "preflight_tfidf_v2"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if checkpoint_dir.exists():
        raise FileExistsError(checkpoint_dir)
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "stdout.log").write_text("", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")

    train_rows = load_frozen_split("train")
    smoke_ids = load_ids(DATA_DIR / "smoke_ids.json")
    smoke_set = set(smoke_ids)
    available_rows = [row for row in train_rows if row["sample_id"] not in smoke_set]
    negative_rows = [row for row in available_rows if row["label"] == 0][:500]
    positive_rows = [row for row in available_rows if row["label"] == 1][:500]
    fitting_rows = sorted(negative_rows + positive_rows, key=lambda row: row["source_row"])
    evaluation_rows = [row for row in train_rows if row["sample_id"] in smoke_set]
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 2),
                    norm="l2",
                    stop_words=None,
                    max_features=None,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    C=1,
                    max_iter=2000,
                    class_weight=None,
                    random_state=42,
                    l1_ratio=0,
                ),
            ),
        ]
    )
    start = time.perf_counter()
    pipeline.fit([row["text"] for row in fitting_rows], [row["label"] for row in fitting_rows])
    fit_seconds = time.perf_counter() - start

    prediction_path = run_dir / "predictions_train.jsonl"
    for row in evaluation_rows:
        start = time.perf_counter()
        predicted = int(pipeline.predict([row["text"]])[0])
        latency_ms = (time.perf_counter() - start) * 1000
        append_jsonl(
            prediction_path,
            {
                "run_id": RUN_ID,
                "owner": "B",
                "method": "tfidf_lr",
                "config_id": "c1_md1",
                "seed": 42,
                "sample_id": row["sample_id"],
                "split": "train",
                "true_label": row["label"],
                "raw_output": str(predicted),
                "parsed_label": predicted,
                "status": "ok",
                "prompt_id": None,
                "input_chars": len(row["text"]),
                "input_tokens": None,
                "truncated": row["truncated"],
                "latency_ms": latency_ms,
                "timing_scope": "batch1_end_to_end",
                "error_message": None,
                "model_id": "tfidf_lr",
                "checkpoint_id": "preflight_tfidf",
                "purpose": "preflight_smoke",
            },
        )

    metrics_path = run_dir / "metrics.json"
    metric_result = evaluate(prediction_path, evaluation_rows, metrics_path)
    model_path = checkpoint_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    restored = joblib.load(model_path)
    restored_prediction = int(restored.predict([evaluation_rows[0]["text"]])[0])
    expected_prediction = int(pipeline.predict([evaluation_rows[0]["text"]])[0])
    if restored_prediction != expected_prediction:
        raise RuntimeError("TF-IDF checkpoint restore mismatch.")

    manifest = {
        "run_id": RUN_ID,
        "owner": "B",
        "method": "tfidf_lr",
        "config_id": "c1_md1",
        "seed": 42,
        "purpose": "preflight_smoke",
        "dataset_id": "AiNiklaus/ChnSentiCorp",
        "dataset_version": "v1-20261023",
        "status": "success" if metric_result["structurally_valid"] else "failed",
        "started_at": utc_iso(),
        "finished_at": utc_iso(),
        "training_rows": len(fitting_rows),
        "evaluation_rows": len(evaluation_rows),
        "fit_seconds": fit_seconds,
        "feature_count": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
        "iterations": int(pipeline.named_steps["classifier"].n_iter_[0]),
        "metrics": metric_result["metrics"],
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": sha256_file(model_path),
        "checkpoint_restore_verified": True,
        "software": {"scikit-learn": version("scikit-learn"), "joblib": version("joblib")},
    }
    write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
