#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_contracts import load_frozen_split, sha256_file
from qwen_prompt_runner import (
    frozen_comment,
    load_prompt_mode,
    validate_few_shot_examples,
    validate_run_id,
    validate_test_freeze,
)


class QwenFormalPipelineTests(unittest.TestCase):
    def test_frozen_comment_uses_shared_512_character_prefix(self) -> None:
        row = next(record for record in load_frozen_split("validation") if record["truncated"])
        actual = frozen_comment(row)
        self.assertEqual(actual, row["normalized_text"][:512])
        self.assertLess(len(actual), len(row["normalized_text"]))

    def test_prompt_configs_match_frozen_registry(self) -> None:
        for mode_name in ("qwen_zero_shot", "qwen_few_shot"):
            config, registry, mode = load_prompt_mode(mode_name, "fixed_prompt_v1")
            self.assertEqual(config["prompt_id"], mode["prompt_id"])
            self.assertEqual(config["max_input_tokens"], 2048)
            self.assertFalse(registry["generation_config"]["enable_thinking"])
            self.assertFalse(registry["generation_config"]["do_sample"])
        _, registry, few_mode = load_prompt_mode("qwen_few_shot", "fixed_prompt_v1")
        examples = validate_few_shot_examples(few_mode, registry)
        self.assertEqual([item["label"] for item in examples["examples"]], [0, 1, 0, 1])

    def test_formal_run_id_contract(self) -> None:
        validate_run_id(
            "C_qwen_zero_shot_fixed_prompt_v1_noseed_20261024T000000Z",
            "formal_validation",
        )
        with self.assertRaises(ValueError):
            validate_run_id("C_qwen_zero_shot fixed_prompt", "formal_validation")
        with self.assertRaises(ValueError):
            validate_run_id("C_qwen_zero_shot_s42_20261024T000000Z", "formal_validation")

    def test_test_inference_requires_matching_freeze_file(self) -> None:
        _, _registry, mode = load_prompt_mode("qwen_zero_shot", "fixed_prompt_v1")
        registry_hash = sha256_file(PROJECT_ROOT / "prompts" / "prompt_registry.json")
        with self.assertRaises(ValueError):
            validate_test_freeze(None, "qwen_zero_shot", "fixed_prompt_v1", mode, registry_hash)

        with tempfile.TemporaryDirectory() as temporary:
            freeze_path = Path(temporary) / "freeze.json"
            freeze_path.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "prompt_registry_sha256": registry_hash,
                        "prompt_ids": [mode["prompt_id"]],
                        "config_ids": ["fixed_prompt_v1"],
                        "methods": ["qwen_zero_shot"],
                    }
                ),
                encoding="utf-8",
            )
            freeze = validate_test_freeze(
                freeze_path,
                "qwen_zero_shot",
                "fixed_prompt_v1",
                mode,
                registry_hash,
            )
        self.assertEqual(freeze["status"], "frozen")


if __name__ == "__main__":
    unittest.main()
