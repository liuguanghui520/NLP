#!/usr/bin/env python3
"""Build a sanitized, self-describing snapshot of the current project deliverables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_PREFIX = ".delivery_stage_"
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
]
FORBIDDEN_SUFFIXES = {
    ".safetensors",
    ".pt",
    ".pth",
    ".bin",
    ".joblib",
    ".pyc",
}
FORBIDDEN_NAME_PARTS = {
    "/train.jsonl",
    "/validation.jsonl",
    "/test.jsonl",
    "/excluded.jsonl",
    "/conflicts.jsonl",
    "/alpaca/",
    "/models/Qwen3-1.7B/model-",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(
    source: Path,
    destination: Path,
    *,
    excluded_directory_names: set[str] | None = None,
) -> None:
    excluded = excluded_directory_names or set()
    for path in sorted(source.rglob("*")):
        if any(part in excluded for part in path.parts):
            continue
        if not path.is_file():
            continue
        copy_file(path, destination / path.relative_to(source))


def sanitize_prediction_file(source: Path, destination: Path) -> dict[str, Any]:
    original_sha = sha256_file(source)
    removed_keys: set[str] = set()
    row_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("rendered_input", "raw_text", "normalized_text", "text"):
                if key in row:
                    row.pop(key)
                    removed_keys.add(key)
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_count += 1
    return {
        "source_name": source.name,
        "package_name": destination.name,
        "rows": row_count,
        "removed_fields": sorted(removed_keys),
        "original_sha256": original_sha,
        "packaged_sha256": sha256_file(destination),
        "reason": "Review-body fields are omitted because redistribution rights are not established.",
    }


def sanitize_registry_payload(registry: dict[str, Any], original_sha: str) -> dict[str, Any]:
    few = registry["modes"]["qwen_few_shot"]
    messages = few["messages_template"]
    for index, message in enumerate(messages[:-1]):
        if message["role"] == "user":
            original_text_sha = hashlib.sha256(message["content"].encode("utf-8")).hexdigest()
            messages[index]["content"] = (
                "[REDACTED_FROZEN_FEW_SHOT_EXAMPLE]"
                f"\noriginal_content_sha256={original_text_sha}"
            )
    few["rendered_template"] = (
        "[REDACTED: rendered few-shot prompt contains training review text; "
        f"original_registry_sha256={original_sha}]"
    )
    registry["package_sanitization"] = {
        "status": "sanitized",
        "original_registry_sha256": original_sha,
        "redacted": ["qwen_few_shot.messages_template example bodies", "qwen_few_shot.rendered_template"],
    }
    return registry


def sanitize_examples_payload(examples: dict[str, Any], original_sha: str) -> dict[str, Any]:
    for example in examples["examples"]:
        example["original_text_sha256"] = hashlib.sha256(example.pop("text").encode("utf-8")).hexdigest()
        example["text"] = "[REDACTED_TRAINING_REVIEW_TEXT]"
    examples["package_sanitization"] = {
        "status": "sanitized",
        "original_examples_sha256": original_sha,
    }
    return examples


def copy_runs(destination: Path) -> list[dict[str, Any]]:
    redactions: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in (PROJECT_ROOT / "runs").iterdir() if path.is_dir()):
        target_run = destination / run_dir.name
        run_redactions: list[dict[str, Any]] = []
        for path in sorted(run_dir.rglob("*")):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            relative = path.relative_to(run_dir)
            target = target_run / relative
            if path.name.startswith("predictions_") and path.suffix == ".jsonl":
                run_redactions.append(sanitize_prediction_file(path, target))
            elif path.name == "prompt_registry.json":
                original_sha = sha256_file(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                write_json(target, sanitize_registry_payload(payload, original_sha))
                run_redactions.append(
                    {
                        "source_name": path.name,
                        "package_name": path.name,
                        "original_sha256": original_sha,
                        "packaged_sha256": sha256_file(target),
                        "reason": "Few-shot example review bodies and rendered prompt were redacted.",
                    }
                )
            elif path.name == "few_shot_examples.json":
                original_sha = sha256_file(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                write_json(target, sanitize_examples_payload(payload, original_sha))
                run_redactions.append(
                    {
                        "source_name": path.name,
                        "package_name": path.name,
                        "original_sha256": original_sha,
                        "packaged_sha256": sha256_file(target),
                        "reason": "Few-shot example review bodies were redacted.",
                    }
                )
            else:
                copy_file(path, target)
        if run_redactions:
            payload = {
                "warning": "Prediction files in this package are sanitized copies.",
                "local_original_retained": True,
                "files": run_redactions,
            }
            write_json(target_run / "PACKAGE_REDACTION.json", payload)
            redactions.extend(
                {"run_id": run_dir.name, **item}
                for item in run_redactions
            )
    copy_file(PROJECT_ROOT / "runs" / "run_index.json", destination / "run_index.json")
    copy_file(PROJECT_ROOT / "runs" / "run_index.md", destination / "run_index.md")
    return redactions


def sanitize_prompts(destination: Path) -> dict[str, Any]:
    registry_path = PROJECT_ROOT / "prompts" / "prompt_registry.json"
    examples_path = PROJECT_ROOT / "prompts" / "few_shot_examples.json"
    original_registry_sha = sha256_file(registry_path)
    original_examples_sha = sha256_file(examples_path)
    registry = sanitize_registry_payload(
        json.loads(registry_path.read_text(encoding="utf-8")),
        original_registry_sha,
    )
    examples = sanitize_examples_payload(
        json.loads(examples_path.read_text(encoding="utf-8")),
        original_examples_sha,
    )
    write_json(destination / "prompt_registry.json", registry)
    write_json(destination / "few_shot_examples.json", examples)
    note = (
        "# Prompt file sanitization\n\n"
        "The package preserves prompt templates, example IDs, labels, order and hashes.\n\n"
        "Full few-shot review bodies and the rendered few-shot prompt are omitted because "
        "redistribution rights for the source review text are not established. The complete "
        "local originals remain under `D:\\NLP\\prompts` for audit only.\n"
    )
    (destination / "PROMPT_REDACTION.md").write_text(note, encoding="utf-8")
    return {
        "original_registry_sha256": original_registry_sha,
        "original_examples_sha256": original_examples_sha,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_data_summary(destination: Path) -> None:
    source = PROJECT_ROOT / "data" / "processed" / "v1-20261023"
    files = [
        "manifest.json",
        "checksums.json",
        "split_ids.json",
        "smoke_ids.json",
        "fewshot_ids.json",
        "latency_ids.json",
        "processing_config.json",
        "validation_report.md",
        "数据处理运行说明.md",
    ]
    for name in files:
        copy_file(source / name, destination / name)
    copy_file(source / "alpaca" / "dataset_info.json", destination / "dataset_info.json")
    copy_file(
        PROJECT_ROOT / "data" / "raw" / "ChnSentiCorp" / "README.md",
        destination / "SOURCE_README.md",
    )


def copy_model_locks(destination: Path) -> None:
    pairs = [
        ("models/Qwen3-1.7B/model_lock.json", "Qwen3-1.7B_model_lock.json"),
        ("models/Qwen3-1.7B/MODEL_LOCK.md", "Qwen3-1.7B_MODEL_LOCK.md"),
        ("models/Qwen3-1.7B/LICENSE", "Qwen3-1.7B_LICENSE"),
        ("models/Qwen3-1.7B/config.json", "Qwen3-1.7B_config.json"),
        ("models/Qwen3-1.7B/generation_config.json", "Qwen3-1.7B_generation_config.json"),
        ("models/chinese-macbert-base-config/asset_lock.json", "MacBERT_asset_lock.json"),
        ("models/chinese-macbert-base-config/config.json", "MacBERT_config.json"),
    ]
    for source, target in pairs:
        copy_file(PROJECT_ROOT / source, destination / target)
    (destination / "MODEL_ASSETS.md").write_text(
        "# Model assets\n\n"
        "- Qwen/Qwen3-1.7B: local weights are not included; see the model lock and external source.\n"
        "- hfl/chinese-macbert-base: only architecture config and Tokenizer were used; no pretrained weights.\n"
        "- Large model files and checkpoints remain in the local workspace and must be shared by link.\n",
        encoding="utf-8",
    )


def build_checkpoint_inventory(destination: Path) -> None:
    rows = []
    root = PROJECT_ROOT / "checkpoints"
    for checkpoint in sorted(path for path in root.iterdir() if path.is_dir()):
        files = [path for path in checkpoint.rglob("*") if path.is_file()]
        rows.append(
            {
                "checkpoint": checkpoint.name,
                "files": len(files),
                "bytes": sum(path.stat().st_size for path in files),
                "contains_adapter": (checkpoint / "adapter_model.safetensors").is_file(),
                "contains_model": (checkpoint / "model.safetensors").is_file(),
                "contains_pipeline": (checkpoint / "pipeline.joblib").is_file(),
                "preflight_only": checkpoint.name.startswith("preflight_"),
            }
        )
    write_json(destination / "checkpoint_inventory.json", {"checkpoints": rows})
    lines = [
        "# Checkpoint inventory",
        "",
        "No checkpoint binaries are included in this package.",
        "",
        "| checkpoint | files | bytes | adapter | model | pipeline | status |",
        "|---|---:|---:|---|---|---|---|",
    ]
    lines.extend(
        f"| {item['checkpoint']} | {item['files']} | {item['bytes']} | "
        f"{item['contains_adapter']} | {item['contains_model']} | {item['contains_pipeline']} | "
        f"{'preflight only' if item['preflight_only'] else 'formal'} |"
        for item in rows
    )
    (destination / "CHECKPOINT_INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def formal_run_status() -> dict[str, list[str]]:
    status: dict[str, list[str]] = {
        "tfidf_lr": [],
        "random_macbert": [],
        "qwen_zero_shot": [],
        "qwen_few_shot": [],
        "qwen_lora": [],
    }
    for run_dir in sorted(path for path in (PROJECT_ROOT / "runs").iterdir() if path.is_dir()):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        method = manifest.get("method")
        purpose = manifest.get("purpose")
        if method in status and purpose in {"formal_validation", "formal_test"}:
            status[method].append(run_dir.name)
    return status


def write_status_documents(destination: Path, run_status: dict[str, list[str]]) -> None:
    method_labels = {
        "tfidf_lr": "TF-IDF + Logistic Regression",
        "random_macbert": "Randomly initialized MacBERT",
        "qwen_zero_shot": "Qwen zero-shot",
        "qwen_few_shot": "Qwen few-shot",
        "qwen_lora": "Qwen LoRA",
    }
    rows = [
        f"| {method_labels[key]} | {len(values)} | {', '.join(values) if values else '缺失'} |"
        for key, values in run_status.items()
    ]
    readme = f"""# NLP 当前阶段成果包

生成日期：2026-10-28（北京时间）  
状态：**阶段性汇总，不是最终课程提交包**

## 当前正式运行

| 路线 | 正式运行数 | run_id |
|---|---:|---|
{chr(10).join(rows)}

当前已完成的正式实验只有 Qwen 零样本和少样本验证集推理。本目录中未找到正式
TF-IDF、随机初始化 MacBERT、Qwen LoRA、最终测试集或正式200条延迟结果。

## 包内内容

- `code/`：脚本、测试、配置、提示格式和环境锁。
- `documentation/`：课程、实验设计、数据说明、交接材料和现有结果文档。
- `data_manifest/`：冻结数据清单、哈希、ID 和验收说明，不含评论正文。
- `model_locks/`：模型版本、哈希、架构配置和许可说明，不含权重。
- `run_evidence/`：全部运行清单、指标、日志、SwanLab及脱敏预测。
- `checkpoint_inventory`：检查点清单；检查点二进制不进入压缩包。
- `prompts_sanitized/`：提示模板、示例ID和哈希；示例评论正文已脱敏。

## 重要限制

- 测试集未运行；只有全部路线配置冻结后才能使用。
- SwanLab、预检、失败和重试记录均保留，但预检指标不是正式结论。
- 运行清单中的产物哈希指向本地原始预测文件；包内预测删除了 `rendered_input`
  评论正文，因此使用清单验证时请以本地原始文件为准。
- 包内不含原始语料、完整冻结JSONL、Alpaca正文、模型权重或检查点。
- 权威数据仍位于 `D:\\NLP\\data`，模型位于 `D:\\NLP\\models`，检查点位于
  `D:\\NLP\\checkpoints`。

详见 `MISSING_ITEMS.md` 和 `package_manifest.json`。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    missing = """# 当前缺失材料

## 阻断最终报告

1. 正式 TF-IDF 六组验证集搜索及选优。
2. 正式随机初始化 MacBERT 四组搜索、早停和三种子复验。
3. 正式 Qwen LoRA r=8/r=16 验证、最佳适配器和三种子复验。
4. Optuna 正式 trials 与各路线最终配置冻结文件。
5. 配置冻结后的正式测试集预测和指标。
6. 固定200条测试样本的正式延迟测量。
7. 全路线统一总对比表、三种子均值/标准差和样本级配对分析。
8. 正式图表、至少三类错误归因、成本与局限性分析。

## 课程交付

1. 项目根 README 和一键复现命令。
2. 8—12页实验报告PDF和报告源文件。
3. 300字以内的项目摘要。
4. 每个成员的贡献声明。
5. AI使用记录A、B、C；当前仅有C，且C中仍需更新旧阶段说明。
6. 不超过2分钟的演示视频或外部视频链接。
7. 最终单ZIP（≤50MB）及提交前检查。

## 数据与模型外链

最终提交前需为 Qwen模型、正式检查点/适配器、原始数据来源提供可访问链接和SHA256。
"""
    (destination / "MISSING_ITEMS.md").write_text(missing, encoding="utf-8")


def inspect_staging(staging_root: Path) -> list[str]:
    errors: list[str] = []
    example_texts = [
        item["text"]
        for item in json.loads(
            (PROJECT_ROOT / "prompts" / "few_shot_examples.json").read_text(encoding="utf-8")
        )["examples"]
    ]
    for path in staging_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(staging_root).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden suffix: {relative}")
        if any(part in "/" + relative for part in FORBIDDEN_NAME_PARTS):
            errors.append(f"forbidden content path: {relative}")
        if path.suffix.lower() in {
            ".md",
            ".py",
            ".json",
            ".jsonl",
            ".yaml",
            ".yml",
            ".txt",
            ".log",
            ".csv",
            ".sh",
            ".lock",
        }:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SENSITIVE_PATTERNS:
                if pattern.search(text):
                    errors.append(f"possible secret pattern: {relative}")
            if path.suffix.lower() == ".jsonl" and '"rendered_input"' in text:
                errors.append(f"rendered input field present: {relative}")
            if any(example_text in text for example_text in example_texts):
                errors.append(f"known review text present: {relative}")
    return errors


def build_archive(staging_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(staging_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging_root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "NLP_阶段成果_20261028.zip",
    )
    args = parser.parse_args()
    archive_path = args.output.resolve()
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive_path}")
    if archive_path.parent != PROJECT_ROOT:
        raise ValueError("The archive must be written directly under the project root.")

    staging_token = tempfile.mkdtemp(prefix=STAGE_PREFIX, dir=PROJECT_ROOT)
    staging_base = Path(staging_token).resolve()
    if staging_base.parent != PROJECT_ROOT or not staging_base.name.startswith(STAGE_PREFIX):
        raise RuntimeError(f"Unsafe staging directory: {staging_base}")
    package_name = f"NLP_阶段成果_{datetime.now().astimezone():%Y%m%d}"
    staging_root = staging_base / package_name
    staging_root.mkdir(parents=True)
    try:
        copy_tree(
            PROJECT_ROOT / "scripts",
            staging_root / "code" / "scripts",
            excluded_directory_names={"__pycache__"},
        )
        copy_tree(
            PROJECT_ROOT / "tests",
            staging_root / "code" / "tests",
            excluded_directory_names={"__pycache__"},
        )
        copy_tree(PROJECT_ROOT / "configs", staging_root / "code" / "configs")
        copy_tree(PROJECT_ROOT / "environment", staging_root / "code" / "environment")
        copy_tree(PROJECT_ROOT / "benchmark", staging_root / "code" / "benchmark")
        copy_tree(PROJECT_ROOT / "records", staging_root / "code" / "records")
        prompt_hashes = sanitize_prompts(staging_root / "prompts_sanitized")
        copy_data_summary(staging_root / "data_manifest")
        copy_model_locks(staging_root / "model_locks")
        build_checkpoint_inventory(staging_root)
        redactions = copy_runs(staging_root / "run_evidence")
        copy_tree(PROJECT_ROOT / "optuna", staging_root / "optuna")
        copy_tree(PROJECT_ROOT / "文件", staging_root / "documentation" / "authoritative_documents")
        for name in (
            "新对话交接_正式训练启动.md",
            "训练前验收报告.md",
            "Qwen正式验证运行准备报告.md",
            "Qwen正式验证结果摘要.md",
        ):
            copy_file(PROJECT_ROOT / name, staging_root / "documentation" / name)
        run_status = formal_run_status()
        write_status_documents(staging_root, run_status)
        scope = {
            "package_type": "current_stage_delivery_snapshot",
            "not_final_course_submission": True,
            "generated_at": datetime.now().astimezone().isoformat(),
            "formal_runs": run_status,
            "test_split_used": False,
            "formal_latency_available": False,
            "included_full_model_weights": False,
            "included_full_review_text": False,
            "prompt_hashes": prompt_hashes,
            "prediction_redactions": redactions,
        }
        write_json(staging_root / "DELIVERY_SCOPE.json", scope)
        manifest_files = {
            path.relative_to(staging_root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging_root.rglob("*"))
            if path.is_file()
        }
        write_json(
            staging_root / "package_manifest.json",
            {
                "package_name": package_name,
                "generated_at": scope["generated_at"],
                "file_count": len(manifest_files),
                "files": manifest_files,
            },
        )
        errors = inspect_staging(staging_root)
        if errors:
            raise RuntimeError("Package safety checks failed: " + "; ".join(errors[:20]))
        build_archive(staging_root, archive_path)
        if archive_path.stat().st_size > 50 * 1024 * 1024:
            raise RuntimeError(f"Archive exceeds 50MB: {archive_path.stat().st_size} bytes")
        print(
            json.dumps(
                {
                    "archive": str(archive_path),
                    "bytes": archive_path.stat().st_size,
                    "sha256": sha256_file(archive_path),
                    "file_count_before_manifest": len(manifest_files),
                    "formal_runs": run_status,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if staging_base.parent == PROJECT_ROOT and staging_base.name.startswith(STAGE_PREFIX):
            shutil.rmtree(staging_base)


if __name__ == "__main__":
    main()
