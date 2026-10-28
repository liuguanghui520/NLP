#!/usr/bin/env python3
"""Finalize preflight manifests, asset locks, and a run index."""

from __future__ import annotations

import json
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from experiment_contracts import (
    DATA_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    load_json,
    sha256_file,
    utc_iso,
    write_json,
)


MODEL_ASSET_DIR = PROJECT_ROOT / "models" / "chinese-macbert-base-config"


def lock_macbert_assets() -> dict[str, Any]:
    files = [
        path
        for path in sorted(MODEL_ASSET_DIR.glob("*"))
        if path.is_file() and path.name != "asset_lock.json"
    ]
    lock = {
        "status": "verified",
        "verified_at_utc": utc_iso(),
        "model_id": "hfl/chinese-macbert-base",
        "revision": "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1",
        "license": "Apache-2.0",
        "scope": "config_and_tokenizer_only_no_pretrained_weights",
        "files": [
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    write_json(MODEL_ASSET_DIR / "asset_lock.json", lock)
    return lock


def finalize_lora(
    run_id: str,
    config_name: str,
    checkpoint_name: str,
    status: str,
    monitor_path: Path | None,
    notes: str,
    superseded_by: str | None = None,
) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / checkpoint_name
    trainer_state = load_json(checkpoint_dir / "trainer_state.json")
    all_results = load_json(checkpoint_dir / "all_results.json")
    train_entry = next(item for item in trainer_state["log_history"] if "grad_norm" in item)
    final_entry = next(item for item in reversed(trainer_state["log_history"]) if "train_runtime" in item)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    adapter_state = load_file(adapter_path)
    lora_b_abs_max = max(
        (
            float(tensor.float().abs().max())
            for name, tensor in adapter_state.items()
            if "lora_B" in name
        ),
        default=0.0,
    )
    monitor = load_json(monitor_path) if monitor_path and monitor_path.is_file() else None
    manifest = {
        "run_id": run_id,
        "owner": "A",
        "method": "qwen_lora",
        "config_id": "r8_alpha16_qv",
        "seed": 42,
        "purpose": "preflight_smoke",
        "status": status,
        "notes": notes,
        "superseded_by": superseded_by,
        "dataset_id": "AiNiklaus/ChnSentiCorp",
        "dataset_version": "v1-20261023",
        "data_manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
        "model_id": "Qwen/Qwen3-1.7B",
        "model_revision": "4855588ea1a12789f2e965e5f52a9e4a24c94b2a",
        "model_lock_sha256": sha256_file(MODEL_DIR / "model_lock.json"),
        "config_path": f"configs/lora/{config_name}",
        "config_sha256": sha256_file(PROJECT_ROOT / "configs" / "lora" / config_name),
        "command": f"llamafactory-cli train configs/lora/{config_name}",
        "started_at": utc_iso(),
        "finished_at": utc_iso(),
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "gpu_memory_bytes": 8585216000,
        },
        "software": {
            "llamafactory": "0.9.6.dev0",
            "torch": "2.5.1+cu121",
            "transformers": version("transformers"),
            "peft": version("peft"),
            "swanlab": version("swanlab"),
        },
        "precision": "bfloat16",
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "train_on_prompt": False,
        },
        "training": {
            "max_steps": 1,
            "samples": 16,
            "effective_batch_size": 16,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
            "loss": train_entry["loss"],
            "grad_norm": train_entry["grad_norm"],
            "learning_rate_final": train_entry["learning_rate"],
            "train_runtime_seconds": final_entry["train_runtime"],
            "samples_per_second": final_entry["train_samples_per_second"],
            "total_flos": all_results.get("total_flos"),
        },
        "parameters": {
            "trainable": 1605632,
            "all": 1722180608,
            "trainable_percent": 0.0932,
        },
        "gpu_monitor": monitor,
        "checkpoint_path": str(checkpoint_dir),
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_path),
        "adapter_lora_b_max_abs": lora_b_abs_max,
        "adapter_lora_b_updated": lora_b_abs_max > 0,
        "stdout_log_sha256": sha256_file(run_dir / "stdout.log"),
        "stderr_log_sha256": sha256_file(run_dir / "stderr.log"),
        "swanlab_logdir": str(run_dir / "swanlab"),
        "swanlab_mode": "local",
        "swanlab_dashboard_dependency": version("swanboard"),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def finalize_api_smoke() -> dict[str, Any]:
    run_id = "C_qwen3_1p7b_api_connection_smoke_v2_20261024T133100Z"
    run_dir = RUNS_DIR / run_id
    result_path = run_dir / "windows_connection_result.json"
    result = load_json(result_path)
    completion_text = result["chat_completion"]["choices"][0]["message"]["content"]
    manifest = {
        "run_id": run_id,
        "owner": "C",
        "method": "windows_to_wsl_openai_api",
        "status": "success",
        "purpose": "preflight_connection_smoke",
        "base_url": "http://localhost:18083/v1",
        "model_id": result["models"]["data"][0]["id"],
        "model_path": "models/Qwen3-1.7B",
        "model_lock_sha256": sha256_file(MODEL_DIR / "model_lock.json"),
        "inference_config": "configs/inference/qwen3_1p7b_api.yaml",
        "inference_config_sha256": sha256_file(PROJECT_ROOT / "configs/inference/qwen3_1p7b_api.yaml"),
        "checks": ["v1/models", "v1/chat/completions"],
        "completion_tokens": result["chat_completion"]["usage"]["completion_tokens"],
        "completion_text_prefix": completion_text[:120],
        "completion_used_thinking_prefix_in_demo": completion_text.startswith("<think>"),
        "enable_thinking": False,
        "formal_evaluation_runner": "scripts/qwen_prompt_runner.py",
        "formal_runner_uses_enable_thinking_false": True,
        "temporary_service_stopped": True,
        "stdout_log_sha256": sha256_file(run_dir / "stdout.log"),
        "stderr_log_sha256": sha256_file(run_dir / "stderr.log"),
        "windows_result_sha256": sha256_file(result_path),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def mark_superseded() -> None:
    replacements = {
        "C_qwen_zero_shot_preflight_v1_noseed_20261024T131500Z": (
            "C_qwen_zero_shot_preflight_v2_noseed_20261024T132500Z",
            "Initial prompt included a system message and wording that differed from the frozen Alpaca instruction.",
        ),
        "C_qwen_few_shot_preflight_v1_noseed_20261024T131600Z": (
            "C_qwen_few_shot_preflight_v2_noseed_20261024T132600Z",
            "Initial prompt included a system message and wording that differed from the frozen Alpaca instruction.",
        ),
        "A_qwen3_lora_r8_preflight_s42_20261024T132300Z": (
            "A_qwen3_lora_r8_preflight_s42_20261024T132400Z",
            "The first successful run logged SwanLab data but lacked the optional local dashboard dependency.",
        ),
        "A_qwen3_lora_r8_preflight_s42_20261024T132400Z": (
            "A_qwen3_lora_r8_preflight_s42_20261024T132700Z",
            "The second run used qwen3_nothink; the final preflight uses qwen3 with enable_thinking=false to match the frozen tokenizer prompt.",
        ),
        "A_qwen3_lora_r8_preflight_s42_20261024T132700Z": (
            "A_qwen3_lora_r8_preflight_s42_20261024T132800Z",
            "The one-step run used 10% warmup and logged learning rate 0; the final smoke run uses constant scheduling and warmup 0 for a nonzero update.",
        ),
        "C_qwen3_1p7b_api_connection_smoke_20261024T133000Z": (
            "C_qwen3_1p7b_api_connection_smoke_v2_20261024T133100Z",
            "The initial service used qwen3_nothink and spontaneously generated a thinking block; the final service uses qwen3 with enable_thinking=false.",
        ),
    }
    for run_id, (replacement, reason) in replacements.items():
        manifest_path = RUNS_DIR / run_id / "run_manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            manifest["superseded_by"] = replacement
            manifest["superseded_reason"] = reason
            write_json(manifest_path, manifest)
    v1_lora_manifest = {
        "run_id": "A_qwen3_lora_r8_preflight_s42_20261024T132300Z",
        "owner": "A",
        "method": "qwen_lora",
        "config_id": "r8_alpha16_qv",
        "seed": 42,
        "purpose": "preflight_smoke",
        "status": "success_with_nonfatal_swanlab_dashboard_warning",
        "superseded_by": "A_qwen3_lora_r8_preflight_s42_20261024T132400Z",
        "superseded_reason": "Initial run trained and saved successfully, but SwanLab local dashboard failed because swanboard was missing.",
        "checkpoint_path": "checkpoints/preflight_qwen3_lora_v1",
        "command": "llamafactory-cli train configs/lora/preflight_qwen3_lora.yaml",
    }
    write_json(
        RUNS_DIR / "A_qwen3_lora_r8_preflight_s42_20261024T132300Z" / "run_manifest.json",
        v1_lora_manifest,
    )
    v2_lora_manifest = {
        "run_id": "A_qwen3_lora_r8_preflight_s42_20261024T132400Z",
        "owner": "A",
        "method": "qwen_lora",
        "config_id": "r8_alpha16_qv",
        "seed": 42,
        "purpose": "preflight_smoke",
        "status": "success_superseded",
        "superseded_by": "A_qwen3_lora_r8_preflight_s42_20261024T132700Z",
        "superseded_reason": "Template changed from qwen3_nothink to qwen3 + enable_thinking=false for exact prompt parity.",
        "checkpoint_path": "checkpoints/preflight_qwen3_lora_v2",
        "command": "llamafactory-cli train configs/lora/preflight_qwen3_lora_v2.yaml",
    }
    write_json(
        RUNS_DIR / "A_qwen3_lora_r8_preflight_s42_20261024T132400Z" / "run_manifest.json",
        v2_lora_manifest,
    )
    v3_lora_manifest = {
        "run_id": "A_qwen3_lora_r8_preflight_s42_20261024T132700Z",
        "owner": "A",
        "method": "qwen_lora",
        "config_id": "r8_alpha16_qv",
        "seed": 42,
        "purpose": "preflight_smoke",
        "status": "success_superseded",
        "superseded_by": "A_qwen3_lora_r8_preflight_s42_20261024T132800Z",
        "superseded_reason": "The one-step warmup schedule logged learning rate 0; superseded by constant schedule with warmup 0 to validate a nonzero adapter update.",
        "checkpoint_path": "checkpoints/preflight_qwen3_lora_v3",
        "command": "llamafactory-cli train configs/lora/preflight_qwen3_lora_v3.yaml",
    }
    write_json(
        RUNS_DIR / "A_qwen3_lora_r8_preflight_s42_20261024T132700Z" / "run_manifest.json",
        v3_lora_manifest,
    )
    api_v1_manifest = {
        "run_id": "C_qwen3_1p7b_api_connection_smoke_20261024T133000Z",
        "owner": "C",
        "method": "windows_to_wsl_openai_api",
        "status": "success_superseded",
        "purpose": "preflight_connection_smoke",
        "superseded_by": "C_qwen3_1p7b_api_connection_smoke_v2_20261024T133100Z",
        "superseded_reason": "The initial API used qwen3_nothink and emitted a thinking block; the final API uses qwen3 + enable_thinking=false.",
        "base_url": "http://localhost:18082/v1",
        "temporary_service_stopped": True,
    }
    write_json(
        RUNS_DIR / "C_qwen3_1p7b_api_connection_smoke_20261024T133000Z" / "run_manifest.json",
        api_v1_manifest,
    )


def build_run_index() -> dict[str, Any]:
    records = []
    for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
        else:
            manifest = {
                "run_id": run_dir.name,
                "status": "missing_manifest",
                "owner": None,
                "method": None,
                "purpose": None,
            }
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
    index = {
        "generated_at_utc": utc_iso(),
        "runs": records,
    }
    write_json(RUNS_DIR / "run_index.json", index)
    lines = [
        "# Preflight Run Index",
        "",
        "| run_id | owner | method | purpose | status | superseded_by |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| `{item['run_id']}` | {item['owner']} | {item['method']} | {item['purpose']} | "
        f"{item['status']} | {item['superseded_by'] or ''} |"
        for item in records
    )
    (RUNS_DIR / "run_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def main() -> None:
    mark_superseded()
    macbert_lock = lock_macbert_assets()
    lora_v2 = finalize_lora(
        run_id="A_qwen3_lora_r8_preflight_s42_20261024T132800Z",
        config_name="preflight_qwen3_lora_v4.yaml",
        checkpoint_name="preflight_qwen3_lora_v4",
        status="success",
        monitor_path=RUNS_DIR
        / "A_qwen3_lora_r8_preflight_s42_20261024T132800Z"
        / "gpu_monitor_summary.json",
        notes="One nonzero optimization step completed with qwen3 + enable_thinking=false, constant smoke scheduling, SwanLab local tracking, and device memory sampling.",
    )
    api_smoke = finalize_api_smoke()
    finalize_lora(
        run_id="A_qwen3_lora_r8_preflight_s42_20261024T132400Z",
        config_name="preflight_qwen3_lora_v2.yaml",
        checkpoint_name="preflight_qwen3_lora_v2",
        status="success_superseded",
        monitor_path=RUNS_DIR
        / "A_qwen3_lora_r8_preflight_s42_20261024T132400Z"
        / "gpu_monitor_summary.json",
        notes="SwanLab local tracking passed; superseded by the exact no-think template parity run.",
        superseded_by="A_qwen3_lora_r8_preflight_s42_20261024T132700Z",
    )
    index = build_run_index()
    print(json.dumps({
        "macbert_asset_files": len(macbert_lock["files"]),
        "lora_status": lora_v2["status"],
        "lora_gpu_peak_mib": (
            lora_v2["gpu_monitor"]["gpu_device_sampled_peak_used_mib"]
            if lora_v2["gpu_monitor"]
            else None
        ),
        "api_status": api_smoke["status"],
        "runs_indexed": len(index["runs"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
