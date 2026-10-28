#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_predictions import evaluate
from experiment_contracts import strict_parse_label


def prediction(sample_id: str, truth: int, raw: str, parsed, status: str) -> dict:
    return {
        "run_id": "test",
        "owner": "C",
        "method": "qwen_zero_shot",
        "config_id": "test",
        "seed": None,
        "sample_id": sample_id,
        "split": "test",
        "true_label": truth,
        "raw_output": raw,
        "parsed_label": parsed,
        "status": status,
        "prompt_id": "test",
        "input_chars": 10,
        "input_tokens": 10,
        "truncated": False,
        "latency_ms": 1.0,
        "timing_scope": "batch1_end_to_end",
        "error_message": None,
        "model_id": "test",
        "checkpoint_id": None,
    }


class EvaluationContractTests(unittest.TestCase):
    def test_strict_parser(self) -> None:
        self.assertEqual(strict_parse_label(" 0\n"), (0, "ok"))
        self.assertEqual(strict_parse_label("1"), (1, "ok"))
        self.assertEqual(strict_parse_label("答案：1"), (None, "invalid_format"))
        self.assertEqual(strict_parse_label(""), (None, "invalid_format"))

    def test_invalid_output_is_in_denominator(self) -> None:
        expected = [
            {"sample_id": "a", "label": 0},
            {"sample_id": "b", "label": 1},
            {"sample_id": "c", "label": 1},
        ]
        rows = [
            prediction("a", 0, "0", 0, "ok"),
            prediction("b", 1, "1", 1, "ok"),
            prediction("c", 1, "不知道", None, "invalid_format"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "predictions.jsonl"
            out = Path(tmp) / "metrics.json"
            pred.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            result = evaluate(pred, expected, out)
        self.assertTrue(result["structurally_valid"])
        self.assertEqual(result["invalid_output_count"], 1)
        self.assertAlmostEqual(result["metrics"]["accuracy"], 2 / 3)
        self.assertEqual(result["metrics"]["confusion_matrix"][1][2], 1)

    def test_missing_prediction_blocks_metrics(self) -> None:
        expected = [{"sample_id": "a", "label": 0}, {"sample_id": "b", "label": 1}]
        rows = [prediction("a", 0, "0", 0, "ok")]
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "predictions.jsonl"
            out = Path(tmp) / "metrics.json"
            pred.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            result = evaluate(pred, expected, out)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["metrics"]["macro_f1"])


if __name__ == "__main__":
    unittest.main()
