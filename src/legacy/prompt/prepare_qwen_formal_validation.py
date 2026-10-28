#!/usr/bin/env python3
"""Validate all static and tokenizer-level prerequisites for formal Qwen inference."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from experiment_contracts import (
    DATA_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    PROMPT_REGISTRY,
    RUNS_DIR,
    load_frozen_split,
    load_json,
    rebuild_run_index,
    sha256_file,
    sha256_text,
    utc_iso,
    utc_stamp,
    write_json,
)
from qwen_prompt_runner import (
    FORMAL_CONFIGS,
    ResourceMonitor,
    Tee,
    frozen_comment,
    load_prompt_mode,
    render_messages,
    software_snapshot,
    validate_few_shot_examples,
)
from transformers import AutoTokenizer


def prompt_token_audit(
    tokenizer,
    mode_name: str,
    mode: dict[str, Any],
    records: list[dict[str, Any]],
    max_tokens: int,
    output_path: Path,
) -> dict[str, Any]:
    marker = "{{comment}}"
    rows: list[dict[str, Any]] = []
    for record in records:
        source_text = frozen_comment(record)
        messages = [
            {
                "role": message["role"],
                "content": message["content"].replace(marker, source_text),
            }
            for message in mode["messages_template"]
        ]
        prompt = render_messages(tokenizer, messages)
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if token_count > max_tokens:
            raise ValueError(
                f"{mode_name} prompt exceeds {max_tokens} tokens for {record['sample_id']}: {token_count}"
            )
        rows.append(
            {
                "mode": mode_name,
                "sample_id": record["sample_id"],
                "label": record["label"],
                "source_chars": len(record["normalized_text"]),
                "shared_input_chars": len(source_text),
                "dataset_prefix_truncated": record["truncated"],
                "input_tokens": token_count,
                "rendered_input_sha256": sha256_text(prompt),
                "token_limit_ok": True,
            }
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tokens = [row["input_tokens"] for row in rows]
    return {
        "sample_count": len(rows),
        "max_input_tokens": max(tokens),
        "mean_input_tokens": sum(tokens) / len(tokens),
        "token_limit": max_tokens,
        "token_limit_truncations": 0,
        "dataset_512_character_truncations": sum(bool(row["dataset_prefix_truncated"]) for row in rows),
        "audit_csv": str(output_path),
    }


def verify_data_hashes() -> dict[str, Any]:
    checksums = load_json(DATA_DIR / "checksums.json")
    checked: dict[str, Any] = {}
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        expected = checksums[name]["sha256"]
        actual = sha256_file(DATA_DIR / name)
        if actual != expected:
            raise ValueError(f"Frozen data hash mismatch for {name}")
        checked[name] = {"expected": expected, "actual": actual}
    return checked


def verify_model_snapshot() -> dict[str, Any]:
    lock = load_json(MODEL_DIR / "model_lock.json")
    checks: list[dict[str, Any]] = []
    for entry in lock["files"]:
        path = MODEL_DIR / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Model file size mismatch: {entry['file']}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Model file hash mismatch: {entry['file']}")
        checks.append({"file": entry["file"], "sha256": actual, "status": "verified"})
    return {
        "file_count": len(checks),
        "snapshot_sha256": lock["snapshot_sha256"],
        "files": checks,
    }


def verify_swanlab(run_dir: Path) -> dict[str, Any]:
    import swanlab

    log_dir = run_dir / "swanlab"
    run = swanlab.init(
        project="NLP-ChnSentiCorp-Sentiment",
        name=f"{run_dir.name}_tracking_check",
        mode="offline",
        log_dir=str(log_dir),
        config={"purpose": "formal_validation_readiness", "inference_started": False},
    )
    run.log({"readiness/prompts_validated": 1, "readiness/inference_started": 0}, step=1)
    swanlab.finish("success")
    swanlab_runs = sorted(log_dir.rglob("*.swanlab")) if log_dir.exists() else []
    return {
        "status": "success",
        "mode": "offline",
        "log_dir": str(log_dir),
        "swanlab_run_files": [str(path) for path in swanlab_runs],
        "swanlab_run_count": len(swanlab_runs),
    }


def main() -> None:
    run_id = f"C_qwen_formal_validation_readiness_noseed_{utc_stamp()}"
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)

    stdout_log = (run_dir / "stdout.log").open("w", encoding="utf-8")
    stderr_log = (run_dir / "stderr.log").open("w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, stdout_log)
    sys.stderr = Tee(original_stderr, stderr_log)
    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "owner": "C",
        "method": "qwen_prompt",
        "config_id": "fixed_prompt_v1",
        "seed": None,
        "purpose": "formal_validation_readiness",
        "status": "running",
        "inference_started": False,
        "test_split_accessed": False,
        "started_at": utc_iso(),
        "finished_at": None,
        "expected_sample_count": 0,
        "completed_sample_count": 0,
    }
    write_json(manifest_path, manifest)
    monitor: ResourceMonitor | None = None
    status = "failed"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        hardware = {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "architecture": torch.cuda.get_device_capability(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "free_before_bytes": torch.cuda.mem_get_info()[0],
            "used_before_bytes": torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0],
        }
        software = software_snapshot()
        monitor = ResourceMonitor(run_dir, interval_seconds=0.5)
        monitor.start()

        validation_records = load_frozen_split("validation")
        if len(validation_records) != 776:
            raise ValueError(f"Unexpected validation count: {len(validation_records)}")
        labels = Counter(row["label"] for row in validation_records)
        if dict(sorted(labels.items())) != {0: 244, 1: 532}:
            raise ValueError(f"Unexpected validation labels: {dict(labels)}")

        config_checks: dict[str, Any] = {}
        modes: dict[str, dict[str, Any]] = {}
        for mode_name in FORMAL_CONFIGS:
            config, registry, mode = load_prompt_mode(mode_name, "fixed_prompt_v1")
            if mode_name == "qwen_few_shot":
                validate_few_shot_examples(mode, registry)
            modes[mode_name] = mode
            config_checks[mode_name] = {
                "config_path": str(FORMAL_CONFIGS[mode_name].relative_to(PROJECT_ROOT)),
                "config_sha256": sha256_file(FORMAL_CONFIGS[mode_name]),
                "prompt_id": mode["prompt_id"],
                "prompt_template_sha256": mode["template_sha256"],
                "max_input_tokens": config["max_input_tokens"],
                "max_new_tokens": config["max_new_tokens"],
            }
            destination = run_dir / f"{mode_name}.yaml"
            destination.write_bytes(FORMAL_CONFIGS[mode_name].read_bytes())

        (run_dir / "prompt_registry.json").write_bytes(PROMPT_REGISTRY.read_bytes())
        (run_dir / "few_shot_examples.json").write_bytes(
            (PROJECT_ROOT / "prompts" / "few_shot_examples.json").read_bytes()
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        registry = load_json(PROMPT_REGISTRY)
        tokenizer_template_hash = sha256_text(tokenizer.chat_template or "")
        if tokenizer_template_hash != registry["chat_template_sha256"]:
            raise ValueError("Tokenizer chat template hash mismatch.")

        prompt_audits = {
            mode_name: prompt_token_audit(
                tokenizer,
                mode_name,
                mode,
                validation_records,
                config_checks[mode_name]["max_input_tokens"],
                run_dir / f"prompt_token_audit_{mode_name}.csv",
            )
            for mode_name, mode in modes.items()
        }
        data_verification = verify_data_hashes()
        model_verification = verify_model_snapshot()
        swanlab_check = verify_swanlab(run_dir)

        dry_run = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_qwen_formal_validation.py"),
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        (run_dir / "formal_command_plan.txt").write_text(dry_run.stdout, encoding="utf-8")

        formal_validation_runs = [
            path.name
            for path in RUNS_DIR.iterdir()
            if path.is_dir()
            and any(
                path.name.startswith(f"C_{mode_name}_fixed_prompt_v1_noseed_")
                for mode_name in FORMAL_CONFIGS
            )
        ]

        resource_summary = monitor.stop()
        monitor = None
        write_json(run_dir / "resource_summary.json", resource_summary)
        readiness = {
            "status": "ready",
            "inference_started": False,
            "test_split_accessed": False,
            "validation_sample_count": len(validation_records),
            "validation_label_counts": {str(key): value for key, value in sorted(labels.items())},
            "prompt_configs": config_checks,
            "prompt_token_audits": prompt_audits,
            "data_verification": data_verification,
            "model_verification": model_verification,
            "swanlab": swanlab_check,
            "formal_validation_runs_before_start": formal_validation_runs,
            "formal_command_plan": dry_run.stdout,
            "resource_summary": resource_summary,
            "hardware": hardware,
            "software": software,
        }
        write_json(run_dir / "readiness.json", readiness)
        manifest.update(
            {
                "status": "success",
                "finished_at": utc_iso(),
                "hardware": hardware,
                "software": software,
                "precision": "bfloat16",
                "data_sha256": {
                    name: values["actual"] for name, values in data_verification.items()
                },
                "prompt_registry_sha256": sha256_file(PROMPT_REGISTRY),
                "model_snapshot_sha256": model_verification["snapshot_sha256"],
                "readiness_file": str(run_dir / "readiness.json"),
                "resource_summary_file": str(run_dir / "resource_summary.json"),
                "swanlab": swanlab_check,
                "prompt_token_audits": prompt_audits,
                "formal_validation_runs_before_start": formal_validation_runs,
            }
        )
        status = "success"
        print(json.dumps(readiness, ensure_ascii=False, indent=2))
    except Exception as exc:
        if monitor is not None:
            try:
                manifest["resource_summary"] = monitor.stop()
            except Exception as monitor_exc:  # noqa: BLE001
                print(f"resource monitor shutdown error: {monitor_exc}", file=sys.stderr)
            monitor = None
        manifest.update(
            {
                "status": "failed",
                "finished_at": utc_iso(),
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_log.close()
        stderr_log.close()
        manifest["artifact_sha256"] = {
            path.name: sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "run_manifest.json"
        }
        write_json(manifest_path, manifest)
        rebuild_run_index()
    if status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
