#!/usr/bin/env python3
"""Run audited Qwen3 zero-shot or few-shot inference on a frozen data split."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil
import swanlab
import torch
import yaml
from evaluate_predictions import evaluate
from experiment_contracts import (
    DATA_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    PROMPT_REGISTRY,
    RUNS_DIR,
    append_jsonl,
    load_frozen_split,
    load_ids,
    load_json,
    rebuild_run_index,
    sha256_file,
    sha256_text,
    strict_parse_label,
    utc_iso,
    utc_stamp,
    write_json,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

SWANLAB_PROJECT = "NLP-ChnSentiCorp-Sentiment"
FORMAL_CONFIGS = {
    "qwen_zero_shot": PROJECT_ROOT / "configs" / "prompt" / "qwen_zero_shot.yaml",
    "qwen_few_shot": PROJECT_ROOT / "configs" / "prompt" / "qwen_few_shot.yaml",
}
REQUIRED_GENERATION = {
    "enable_thinking": False,
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 8,
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class ResourceMonitor:
    """Sample device memory/utilization and host RSS while the run executes."""

    def __init__(self, output_dir: Path, interval_seconds: float = 0.5):
        self.csv_path = output_dir / "gpu_samples.csv"
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._process = psutil.Process()

    def _sample(self) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "elapsed_seconds": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_total_mib": None,
            "gpu_utilization_percent": None,
            "gpu_power_watts": None,
            "process_rss_bytes": self._process.memory_info().rss,
        }
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            values = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
            if len(values) >= 4:
                sample["gpu_memory_used_mib"] = _optional_float(values[0])
                sample["gpu_memory_total_mib"] = _optional_float(values[1])
                sample["gpu_utilization_percent"] = _optional_float(values[2])
                sample["gpu_power_watts"] = _optional_float(values[3])
        return sample

    def _loop(self) -> None:
        started = time.perf_counter()
        fieldnames = [
            "elapsed_seconds",
            "gpu_memory_used_mib",
            "gpu_memory_total_mib",
            "gpu_utilization_percent",
            "gpu_power_watts",
            "process_rss_bytes",
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            while not self._stop.is_set():
                sample = self._sample()
                sample["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                self._samples.append(sample)
                writer.writerow(sample)
                handle.flush()
                self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        device_samples = [
            item["gpu_memory_used_mib"]
            for item in self._samples
            if item["gpu_memory_used_mib"] is not None
        ]
        utilization_samples = [
            item["gpu_utilization_percent"]
            for item in self._samples
            if item["gpu_utilization_percent"] is not None
        ]
        rss_samples = [item["process_rss_bytes"] for item in self._samples]
        return {
            "sample_count": len(self._samples),
            "sample_interval_seconds": self.interval_seconds,
            "gpu_device_sampled_peak_used_mib": max(device_samples) if device_samples else None,
            "gpu_device_sampled_peak_utilization_percent": (
                max(utilization_samples) if utilization_samples else None
            ),
            "peak_process_rss_bytes": max(rss_samples) if rss_samples else None,
            "gpu_samples_csv": str(self.csv_path),
        }


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return payload


def _template_sha256(mode_name: str, mode: dict[str, Any]) -> str:
    if mode_name == "qwen_zero_shot":
        payload: Any = mode["messages_template"]
    elif mode_name == "qwen_few_shot":
        payload = {
            "messages": mode["messages_template"],
            "example_ids": mode["example_ids"],
        }
    else:
        raise ValueError(f"Unknown Qwen mode: {mode_name}")
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def load_prompt_mode(mode_name: str, config_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = FORMAL_CONFIGS[mode_name]
    config = _read_yaml(config_path)
    registry = load_json(PROMPT_REGISTRY)
    mode = registry["modes"][mode_name]

    required = {
        "method": mode_name,
        "config_id": config_id,
        "prompt_id": mode["prompt_id"],
        "model_id": registry["model_id"],
        "model_revision": registry["model_revision"],
        "formal_source_split": "validation",
        "max_input_tokens": 2048,
        "max_new_tokens": 8,
        "enable_thinking": False,
        "do_sample": False,
        "num_beams": 1,
        "batch_size": 1,
        "text_rule": "frozen_512_unicode_prefix",
    }
    mismatches = [
        f"{key}: config={config.get(key)!r}, expected={expected!r}"
        for key, expected in required.items()
        if config.get(key) != expected
    ]
    if mismatches:
        raise ValueError("Prompt config contract mismatch: " + "; ".join(mismatches))
    if registry["generation_config"] != {
        **REQUIRED_GENERATION,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }:
        raise ValueError("Frozen prompt registry generation contract has changed.")

    actual_template_sha = _template_sha256(mode_name, mode)
    if mode.get("template_sha256") != actual_template_sha:
        raise ValueError("Frozen prompt template hash does not match its registered content.")
    return config, registry, mode


def validate_few_shot_examples(mode: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / "prompts" / "few_shot_examples.json"
    payload = load_json(path)
    example_ids = [item["sample_id"] for item in payload["examples"]]
    if example_ids != mode["example_ids"]:
        raise ValueError("Few-shot example order differs from the frozen prompt registry.")
    if [item["label"] for item in payload["examples"]] != [0, 1, 0, 1]:
        raise ValueError("Few-shot labels must be negative, positive, negative, positive.")
    if payload["template_sha256"] != mode["template_sha256"]:
        raise ValueError("Few-shot examples file hash contract differs from prompt registry.")
    if registry["version"] != payload["version"]:
        raise ValueError("Few-shot examples and prompt registry versions differ.")

    train_by_id = {row["sample_id"]: row for row in load_frozen_split("train")}
    for item in payload["examples"]:
        row = train_by_id.get(item["sample_id"])
        if row is None:
            raise ValueError(f"Few-shot sample is not in train: {item['sample_id']}")
        if row["normalized_text"] != item["text"] or row["label"] != item["label"]:
            raise ValueError(f"Few-shot frozen content differs for {item['sample_id']}")
    return payload


def validate_test_freeze(
    path: Path | None,
    mode_name: str,
    config_id: str,
    mode: dict[str, Any],
    registry_sha256: str,
) -> dict[str, Any]:
    if path is None:
        raise ValueError("Test inference is blocked until an explicit project freeze file is supplied.")
    freeze = load_json(path)
    if freeze.get("status") != "frozen":
        raise ValueError("Test freeze file status must be 'frozen'.")
    checks = {
        "prompt_registry_sha256": registry_sha256,
    }
    for key, expected in checks.items():
        if freeze.get(key) != expected:
            raise ValueError(f"Test freeze mismatch for {key}.")
    if mode["prompt_id"] not in freeze.get("prompt_ids", []):
        raise ValueError("Prompt is not approved by the test freeze file.")
    if config_id not in freeze.get("config_ids", []):
        raise ValueError("Config is not approved by the test freeze file.")
    if mode_name not in freeze.get("methods", []):
        raise ValueError("Method is not approved by the test freeze file.")
    return freeze


def render_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_with_token_limit(tokenizer, mode: dict[str, Any], text: str, max_tokens: int) -> tuple[str, str, int]:
    marker = "{{comment}}"
    low, high = 0, len(text)
    best: tuple[str, str, int] | None = None
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle]
        messages = [
            {
                "role": message["role"],
                "content": message["content"].replace(marker, candidate),
            }
            for message in mode["messages_template"]
        ]
        prompt = render_messages(tokenizer, messages)
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if token_count <= max_tokens:
            best = (prompt, candidate, token_count)
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise RuntimeError("Task instruction and examples exceed the max input token limit.")
    return best


def frozen_comment(record: dict[str, Any]) -> str:
    text = record.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError(f"Missing frozen 512-character text for {record.get('sample_id')}")
    if len(text) != record.get("text_char_count"):
        raise ValueError(f"Frozen text length mismatch for {record.get('sample_id')}")
    if text != record["normalized_text"][:512]:
        raise ValueError(f"Frozen text is not the first 512 characters for {record.get('sample_id')}")
    return text


def validate_run_id(run_id: str, purpose: str) -> None:
    if not re.fullmatch(r"C_[A-Za-z0-9._-]+", run_id):
        raise ValueError("run_id must start with 'C_' and contain no spaces or path separators.")
    if purpose == "formal_validation" and "_noseed_" not in run_id:
        raise ValueError("Formal prompt-inference run_id must contain '_noseed_'.")


def software_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "llamafactory": _package_version("llamafactory"),
        "swanlab": _package_version("swanlab"),
        "swanboard": _package_version("swanboard"),
        "optuna": _package_version("optuna"),
        "scikit-learn": _package_version("scikit-learn"),
        "psutil": _package_version("psutil"),
    }


def environment_text(software: dict[str, Any], hardware: dict[str, Any]) -> str:
    packages = "\n".join(f"{name}: {value}" for name, value in sorted(software.items()))
    nvidia = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=False,
    )
    gpu_snapshot = nvidia.stdout.strip() if nvidia.returncode == 0 else "nvidia-smi unavailable"
    return (
        f"captured_at_utc: {utc_iso()}\n"
        f"project_root: {PROJECT_ROOT}\n"
        f"cwd: {Path.cwd()}\n\n"
        f"[software]\n{packages}\n\n"
        f"[hardware]\n{json.dumps(hardware, ensure_ascii=False, indent=2)}\n\n"
        f"[nvidia-smi]\n{gpu_snapshot}\n"
    )


def build_manifest(
    args: argparse.Namespace,
    run_id: str,
    purpose: str,
    config_path: Path,
    registry: dict[str, Any],
    mode: dict[str, Any],
    records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    hardware: dict[str, Any],
    software: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    model_lock = load_json(MODEL_DIR / "model_lock.json")
    data_manifest = load_json(DATA_DIR / "manifest.json")
    return {
        "run_id": run_id,
        "owner": "C",
        "method": args.mode,
        "config_id": args.config_id,
        "seed": None,
        "purpose": purpose,
        "dataset_id": data_manifest["dataset_id"],
        "dataset_version": data_manifest["processing_version"],
        "data_sha256": {
            name: sha256_file(DATA_DIR / name)
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")
        },
        "model_id": registry["model_id"],
        "model_revision": registry["model_revision"],
        "model_snapshot_sha256": model_lock["snapshot_sha256"],
        "model_lock_sha256": sha256_file(MODEL_DIR / "model_lock.json"),
        "prompt_id": mode["prompt_id"],
        "prompt_template_sha256": mode["template_sha256"],
        "prompt_registry_sha256": sha256_file(PROMPT_REGISTRY),
        "prompt_config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "prompt_config_sha256": sha256_file(config_path),
        "few_shot_examples_sha256": (
            sha256_file(PROJECT_ROOT / "prompts" / "few_shot_examples.json")
            if args.mode == "qwen_few_shot"
            else None
        ),
        "generation_config": registry["generation_config"],
        "input_rule": "frozen 512-Unicode-character prefix, then Qwen 2048-token guard",
        "git_commit": _git_commit(),
        "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "parent_run_id": args.parent_run_id,
        "retry_reason": args.retry_reason,
        "hardware": hardware,
        "software": software,
        "precision": "bfloat16",
        "expected_sample_count": len(records),
        "validation_dataset_count": len(validation_records),
        "completed_sample_count": 0,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "code_sha256": {
            "qwen_prompt_runner.py": sha256_file(Path(__file__).resolve()),
            "experiment_contracts.py": sha256_file(PROJECT_ROOT / "scripts" / "experiment_contracts.py"),
            "evaluate_predictions.py": sha256_file(PROJECT_ROOT / "scripts" / "evaluate_predictions.py"),
        },
        "swanlab": {
            "project": SWANLAB_PROJECT,
            "mode": "offline",
            "experiment_name": run_id,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["qwen_zero_shot", "qwen_few_shot"], required=True)
    parser.add_argument("--source-split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--config-id")
    parser.add_argument("--parent-run-id")
    parser.add_argument("--retry-reason")
    parser.add_argument("--test-freeze-file", type=Path)
    parser.add_argument("--resource-interval-seconds", type=float, default=0.5)
    args = parser.parse_args()

    config_id = args.config_id or "fixed_prompt_v1"
    args.config_id = config_id
    subset_requested = args.ids_file is not None or args.limit is not None
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.source_split == "test" and subset_requested:
        parser.error("Subset requests on the test split are forbidden.")
    if args.source_split == "train" and not subset_requested:
        parser.error("Full-train inference is not a formal route; provide a frozen subset with --ids-file.")
    purpose = "preflight_smoke" if subset_requested else (
        "formal_validation" if args.source_split == "validation" else "formal_test"
    )
    run_id = args.run_id or (
        f"C_{args.mode}_{config_id}_{'preflight' if subset_requested else args.source_split}"
        f"_noseed_{utc_stamp()}"
    )
    validate_run_id(run_id, purpose)

    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    stdout_log = (run_dir / "stdout.log").open("w", encoding="utf-8")
    stderr_log = (run_dir / "stderr.log").open("w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, stdout_log)
    sys.stderr = Tee(original_stderr, stderr_log)

    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, Any] | None = None
    swan_run = None
    monitor: ResourceMonitor | None = None
    status = "runtime_error"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        started_at = utc_iso()
        hardware = {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "architecture": torch.cuda.get_device_capability(0),
        }
        software = software_snapshot()
        config, registry, mode = load_prompt_mode(args.mode, config_id)
        if args.mode == "qwen_few_shot":
            validate_few_shot_examples(mode, registry)

        registry_sha256 = sha256_file(PROMPT_REGISTRY)
        test_freeze = None
        if args.source_split == "test":
            test_freeze = validate_test_freeze(
                args.test_freeze_file,
                args.mode,
                config_id,
                mode,
                registry_sha256,
            )
        elif args.test_freeze_file is not None:
            raise ValueError("--test-freeze-file may only be used with --source-split test.")

        validation_records = load_frozen_split("validation")
        records = load_frozen_split(args.source_split)
        if args.ids_file:
            selected_ids = load_ids(args.ids_file)
            selected = set(selected_ids)
            records = [row for row in records if row["sample_id"] in selected]
            if len(records) != len(selected_ids):
                missing = sorted(selected - {row["sample_id"] for row in records})
                raise ValueError(f"Selected IDs are not all in {args.source_split}: {missing[:10]}")
        if args.limit is not None:
            records = records[: args.limit]
        if not records:
            raise ValueError("No records selected.")
        if purpose == "formal_validation":
            if len(records) != len(validation_records):
                raise ValueError("Formal validation run must contain every validation sample.")
            if {row["sample_id"] for row in records} != {
                row["sample_id"] for row in validation_records
            }:
                raise ValueError("Formal validation sample IDs differ from the frozen validation set.")

        config_snapshot = run_dir / "config.yaml"
        shutil.copyfile(FORMAL_CONFIGS[args.mode], config_snapshot)
        shutil.copyfile(PROMPT_REGISTRY, run_dir / "prompt_registry.json")
        if args.mode == "qwen_few_shot":
            shutil.copyfile(PROJECT_ROOT / "prompts" / "few_shot_examples.json", run_dir / "few_shot_examples.json")
        (run_dir / "environment.txt").write_text(
            environment_text(software, hardware),
            encoding="utf-8",
        )
        write_json(run_dir / "checkpoints.json", [])
        if test_freeze is not None:
            shutil.copyfile(args.test_freeze_file, run_dir / "test_freeze.json")

        manifest = build_manifest(
            args,
            run_id,
            purpose,
            FORMAL_CONFIGS[args.mode],
            registry,
            mode,
            records,
            validation_records,
            hardware,
            software,
            started_at,
        )
        manifest["artifacts"] = {
            "config": str(config_snapshot),
            "environment": str(run_dir / "environment.txt"),
            "prompt_registry": str(run_dir / "prompt_registry.json"),
            "checkpoints": str(run_dir / "checkpoints.json"),
        }
        write_json(manifest_path, manifest)

        swan_run = swanlab.init(
            project=SWANLAB_PROJECT,
            name=run_id,
            mode="offline",
            log_dir=str(run_dir / "swanlab"),
            config={
                "run_id": run_id,
                "method": args.mode,
                "config_id": config_id,
                "source_split": args.source_split,
                "prompt_id": mode["prompt_id"],
                "prompt_template_sha256": mode["template_sha256"],
                "model_revision": registry["model_revision"],
                "max_input_tokens": config["max_input_tokens"],
                "max_new_tokens": config["max_new_tokens"],
                "precision": "bfloat16",
            },
        )
        manifest["swanlab"]["status"] = "initialized"
        manifest["swanlab"]["log_dir"] = str(run_dir / "swanlab")

        monitor = ResourceMonitor(run_dir, args.resource_interval_seconds)
        monitor.start()
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        tokenizer_template_sha = sha256_text(tokenizer.chat_template or "")
        if tokenizer_template_sha != registry["chat_template_sha256"]:
            raise ValueError("Tokenizer chat template differs from the frozen prompt registry.")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.to("cuda")
        model.eval()
        torch.cuda.synchronize()
        manifest["model_load_seconds"] = time.perf_counter() - load_start
        manifest["parameters_total"] = sum(parameter.numel() for parameter in model.parameters())
        manifest["parameters_trainable"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

        torch.cuda.reset_peak_memory_stats()
        prediction_path = run_dir / f"predictions_{args.source_split}.jsonl"
        metrics_path = run_dir / f"metrics_{args.source_split}.json"
        for index, record in enumerate(records, start=1):
            source_text = frozen_comment(record)
            prompt, visible_text, input_tokens = render_with_token_limit(
                tokenizer,
                mode,
                source_text,
                config["max_input_tokens"],
            )
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to("cuda")
            attention_mask = encoded["attention_mask"].to("cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=config["do_sample"],
                    num_beams=config["num_beams"],
                    max_new_tokens=config["max_new_tokens"],
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            generation_ms = (time.perf_counter() - start) * 1000
            generated = output[0, input_ids.shape[1] :]
            raw_output = tokenizer.decode(generated, skip_special_tokens=True)
            parsed, parse_status = strict_parse_label(raw_output)
            token_truncated = len(visible_text) < len(source_text)
            row = {
                "run_id": run_id,
                "owner": "C",
                "method": args.mode,
                "config_id": config_id,
                "seed": None,
                "sample_id": record["sample_id"],
                "split": args.source_split,
                "true_label": record["label"],
                "raw_output": raw_output,
                "parsed_label": parsed,
                "status": parse_status,
                "prompt_id": mode["prompt_id"],
                "input_chars": len(visible_text),
                "input_tokens": input_tokens,
                "truncated": bool(record["truncated"] or token_truncated),
                "latency_ms": generation_ms,
                "timing_scope": "batch1_generation_only",
                "error_message": None,
                "model_id": registry["model_id"],
                "checkpoint_id": None,
                "purpose": purpose,
                "rendered_input": prompt,
                "rendered_input_sha256": sha256_text(prompt),
                "source_chars": len(record["normalized_text"]),
                "shared_input_chars": len(source_text),
                "prefix_truncated": bool(record["truncated"]),
                "token_truncated": token_truncated,
            }
            append_jsonl(prediction_path, row)
            manifest["completed_sample_count"] = index
            if index % 25 == 0:
                write_json(manifest_path, manifest)
            if index % 10 == 0 or index == len(records):
                progress = {"progress/completed": index, "progress/fraction": index / len(records)}
                swan_run.log(progress, step=index)
                print(f"completed {index}/{len(records)}", flush=True)

        metric_result = evaluate(prediction_path, records, metrics_path)
        torch.cuda.synchronize()
        manifest["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        manifest["torch_memory_peak_scope"] = "after_model_load_before_inference"
        resource_summary = monitor.stop()
        monitor = None
        write_json(run_dir / "resource_summary.json", resource_summary)
        manifest["resource_summary"] = resource_summary
        manifest["predictions_file"] = str(prediction_path)
        manifest["metrics_file"] = str(metrics_path)
        manifest["resource_summary_file"] = str(run_dir / "resource_summary.json")
        manifest["completed_sample_count"] = len(records)
        manifest["finished_at"] = utc_iso()
        manifest["status"] = "success" if metric_result["structurally_valid"] else "failed"
        status = manifest["status"]
        final_scalars = {
            "validation/accuracy": metric_result["metrics"]["accuracy"],
            "validation/macro_f1": metric_result["metrics"]["macro_f1"],
            "validation/invalid_output_rate": metric_result["invalid_output_rate"],
            "resource/gpu_peak_allocated_bytes": manifest["gpu_peak_allocated_bytes"],
            "resource/gpu_peak_reserved_bytes": manifest["gpu_peak_reserved_bytes"],
            "resource/gpu_device_sampled_peak_used_mib": resource_summary[
                "gpu_device_sampled_peak_used_mib"
            ],
            "resource/peak_process_rss_bytes": resource_summary["peak_process_rss_bytes"],
        }
        swan_run.log(
            {key: value for key, value in final_scalars.items() if value is not None},
            step=len(records),
        )
        manifest["swanlab"]["status"] = "success"
        swanlab.finish("success")
        manifest["swanlab"]["run_files"] = [
            str(path.relative_to(run_dir))
            for path in sorted((run_dir / "swanlab").rglob("*.swanlab"))
        ]
        swan_run = None
        write_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "purpose": purpose,
                    "status": status,
                    "completed": len(records),
                    "metrics": metric_result["metrics"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except BaseException as exc:
        failure_status = (
            "interrupted"
            if isinstance(exc, KeyboardInterrupt)
            else "oom"
            if isinstance(exc, torch.cuda.OutOfMemoryError)
            else "runtime_error"
        )
        if monitor is not None:
            try:
                manifest = manifest or {}
                manifest["resource_summary"] = monitor.stop()
            except Exception as monitor_exc:  # noqa: BLE001
                print(f"resource monitor shutdown error: {monitor_exc}", file=sys.stderr)
            monitor = None
        manifest = manifest or {
            "run_id": run_id,
            "owner": "C",
            "method": args.mode,
            "config_id": config_id,
            "purpose": purpose,
        }
        manifest["status"] = failure_status
        manifest["finished_at"] = utc_iso()
        manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
        if swan_run is not None:
            try:
                manifest.setdefault("swanlab", {})["status"] = "crashed"
                manifest["swanlab"]["status"] = (
                    "aborted" if failure_status == "interrupted" else "crashed"
                )
                swanlab.finish(
                    "aborted" if failure_status == "interrupted" else "crashed",
                    error=str(exc),
                )
            except Exception as swan_exc:  # noqa: BLE001
                manifest.setdefault("swanlab", {})["failure_reason"] = str(swan_exc)
            swan_run = None
        write_json(manifest_path, manifest)
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_log.close()
        stderr_log.close()
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            artifacts = {
                path.name: sha256_file(path)
                for path in sorted(run_dir.iterdir())
                if path.is_file() and path.name != "run_manifest.json"
            }
            manifest["artifact_sha256"] = artifacts
            write_json(manifest_path, manifest)
            rebuild_run_index()
    if status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
