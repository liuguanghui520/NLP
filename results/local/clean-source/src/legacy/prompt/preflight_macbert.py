#!/usr/bin/env python3
"""One-batch MacBERT random-initialization training/save/reload smoke test."""

from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification

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
from evaluate_predictions import evaluate


MODEL_ASSET_DIR = PROJECT_ROOT / "models" / "chinese-macbert-base-config"
RUN_ID = "A_random_macbert_preflight_lr1e-4_wd0_s42_20261024T132200Z"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    run_dir = RUNS_DIR / RUN_ID
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "preflight_macbert_random_init_v2"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if checkpoint_dir.exists():
        raise FileExistsError(checkpoint_dir)
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "stdout.log").write_text("", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ASSET_DIR, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_ASSET_DIR, local_files_only=True)
    config.num_labels = 2
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    model = BertForSequenceClassification(config)
    model.to("cuda")
    model.train()

    train_rows = load_frozen_split("train")
    train_by_id = {row["sample_id"]: row for row in train_rows}
    smoke_ids = load_ids(DATA_DIR / "smoke_ids.json")
    batch_rows = [train_by_id[sample_id] for sample_id in smoke_ids[:4]]
    encoded = tokenizer(
        [row["text"] for row in batch_rows],
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    labels = torch.tensor([row["label"] for row in batch_rows], device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - start

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    config.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    restored = BertForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=2,
        local_files_only=True,
    )
    saved_weights = load_file(checkpoint_dir / "model.safetensors")
    reference = model.state_dict()
    max_difference = max(
        float(
            (
                saved_weights[name].detach().cpu().float()
                - reference[name].detach().cpu().float()
            )
            .abs()
            .max()
        )
        for name in saved_weights
    )
    restored_parameter_count = sum(parameter.numel() for parameter in restored.parameters())
    if max_difference != 0.0:
        raise RuntimeError(f"Checkpoint restore mismatch: max difference {max_difference}")

    checkpoint_sha = sha256_file(checkpoint_dir / "model.safetensors")
    manifest = {
        "run_id": RUN_ID,
        "owner": "A",
        "method": "random_macbert",
        "config_id": "preflight_lr1e-4_wd0",
        "seed": 42,
        "purpose": "preflight_smoke",
        "dataset_id": "AiNiklaus/ChnSentiCorp",
        "dataset_version": "v1-20261023",
        "model_id": "hfl/chinese-macbert-base",
        "model_asset_revision": "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1",
        "pretrained_weights_loaded": False,
        "initialization": "random_from_architecture_config",
        "status": "success",
        "started_at": utc_iso(),
        "finished_at": utc_iso(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "software": {
            "torch": torch.__version__,
            "transformers": version("transformers"),
        },
        "batch_size": len(batch_rows),
        "max_length": 128,
        "training_step_seconds": step_seconds,
        "loss": float(loss.detach().cpu()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "restored_parameter_count": restored_parameter_count,
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
        "checkpoint_path": str(checkpoint_dir),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_restore_max_abs_difference": max_difference,
        "checkpoint_restore_verified": True,
        "smoke_sample_ids": [row["sample_id"] for row in batch_rows],
    }
    write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
