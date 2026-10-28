#!/usr/bin/env python3
"""Freeze Qwen zero-shot/few-shot prompts and the selected training examples."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from experiment_contracts import (
    DATA_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    load_frozen_split,
    load_ids,
    sha256_text,
    write_json,
)


TARGET_MARKER = "{{comment}}"
USER_TEMPLATE = (
    "判断评论整体情感，仅输出0表示负面、1表示正面。\n\n"
    "<评论>\n"
    f"{TARGET_MARKER}\n"
    "</评论>"
)


def render_template(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    output_dir = PROJECT_ROOT / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    train_rows = load_frozen_split("train")
    train_by_id = {row["sample_id"]: row for row in train_rows}
    fewshot_ids = load_ids(DATA_DIR / "fewshot_ids.json")
    if len(fewshot_ids) != 4:
        raise ValueError("fewshot_ids.json must contain exactly four IDs.")

    examples = []
    for position, sample_id in enumerate(fewshot_ids, start=1):
        row = train_by_id.get(sample_id)
        if row is None:
            raise ValueError(f"Few-shot sample is not in train split: {sample_id}")
        if len(row["normalized_text"]) > 80:
            raise ValueError(f"Few-shot sample exceeds 80 characters: {sample_id}")
        examples.append(
            {
                "position": position,
                "sample_id": sample_id,
                "label": row["label"],
                "text": row["normalized_text"],
            }
        )
    if [item["label"] for item in examples] != [0, 1, 0, 1]:
        raise ValueError("Few-shot order must be negative, positive, negative, positive.")

    zero_messages = [{"role": "user", "content": USER_TEMPLATE}]
    few_messages: list[dict[str, str]] = []
    for example in examples:
        few_messages.append(
            {
                "role": "user",
                "content": USER_TEMPLATE.replace(TARGET_MARKER, example["text"]),
            }
        )
        few_messages.append({"role": "assistant", "content": str(example["label"])})
    few_messages.append({"role": "user", "content": USER_TEMPLATE})

    generation = {
        "enable_thinking": False,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 8,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }
    registry = {
        "version": "v1-20261023",
        "model_id": "Qwen/Qwen3-1.7B",
        "model_revision": "4855588ea1a12789f2e965e5f52a9e4a24c94b2a",
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": sha256_text(tokenizer.chat_template or ""),
        "target_marker": TARGET_MARKER,
        "generation_config": generation,
        "modes": {
            "qwen_zero_shot": {
                "prompt_id": "qwen3_1p7b_zero_shot_v1",
                "messages_template": zero_messages,
                "rendered_template": render_template(tokenizer, zero_messages),
            },
            "qwen_few_shot": {
                "prompt_id": "qwen3_1p7b_few_shot_4_v1",
                "messages_template": few_messages,
                "rendered_template": render_template(tokenizer, few_messages),
                "example_ids": fewshot_ids,
                "example_order": ["negative", "positive", "negative", "positive"],
            },
        },
    }
    registry["modes"]["qwen_zero_shot"]["template_sha256"] = sha256_text(
        json.dumps(registry["modes"]["qwen_zero_shot"]["messages_template"], ensure_ascii=False, sort_keys=True)
    )
    registry["modes"]["qwen_few_shot"]["template_sha256"] = sha256_text(
        json.dumps(
            {
                "messages": registry["modes"]["qwen_few_shot"]["messages_template"],
                "example_ids": fewshot_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    write_json(output_dir / "prompt_registry.json", registry)
    write_json(
        output_dir / "few_shot_examples.json",
        {
            "version": "v1-20261023",
            "source_split": "train",
            "selection_seed": 67,
            "max_characters": 80,
            "order": ["negative", "positive", "negative", "positive"],
            "examples": examples,
            "prompt_id": registry["modes"]["qwen_few_shot"]["prompt_id"],
            "template_sha256": registry["modes"]["qwen_few_shot"]["template_sha256"],
        },
    )
    print(json.dumps({
        "prompt_registry": str(output_dir / "prompt_registry.json"),
        "few_shot_examples": str(output_dir / "few_shot_examples.json"),
        "few_shot_ids": fewshot_ids,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
