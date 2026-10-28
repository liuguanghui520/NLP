"""Package frozen trained artifacts as bytes; never import or execute model code."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
BASELINE_FIELDS = {
    "config_id", "analyzer", "ngram_range", "C", "min_df", "sublinear_tf",
    "max_features", "max_iter", "solver", "class_weight", "jieba_HMM", "seed",
    "method", "stage", "lr", "epochs", "maxlen", "freeze_lower", "weighted",
    "batch", "eval_batch", "device", "precision", "weight_decay", "warmup_ratio",
    "grad_clip", "accumulation", "length_group_mega", "optimizer", "scheduler",
    "total_parameters", "trainable_parameters", "data_manifest_sha256",
}
LORA_FIELDS = {
    "stage", "do_train", "finetuning_type", "template", "enable_thinking",
    "trust_remote_code", "dataset", "cutoff_len", "train_on_prompt", "packing",
    "preprocessing_num_workers", "dataloader_num_workers", "lora_rank", "lora_alpha",
    "lora_dropout", "lora_target", "learning_rate", "weight_decay", "optim",
    "lr_scheduler_type", "warmup_ratio", "num_train_epochs", "seed", "data_seed",
    "per_device_train_batch_size", "gradient_accumulation_steps", "max_grad_norm",
    "gradient_checkpointing", "bf16", "save_strategy", "logging_steps",
    "overwrite_output_dir", "report_to", "use_swanlab", "swanlab_mode",
    "swanlab_project", "swanlab_run_name", "plot_loss",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require_digest(path, expected):
    if not Path(path).is_file() or sha256(path) != expected:
        raise ValueError(f"Artifact identity mismatch: {path}")


def safe_member(name):
    """Use portable relative ZIP members; reject traversal, drives and backslashes."""
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    if not path.parts or ":" in path.parts[0] or str(path) != name:
        raise ValueError(f"Non-portable archive member: {name}")
    return path


def payload_fingerprint(value):
    if isinstance(value, Path):
        return dict(bytes=value.stat().st_size, sha256=sha256(value))
    return dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())


def verify_archive(path, expected_archive_sha256=None):
    """Stream every member, checking CRC and SHA256 without extracting or loading."""
    if expected_archive_sha256 is not None:
        require_digest(path, expected_archive_sha256)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive members")
        for item in archive.infolist():
            safe_member(item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Archive symlinks are not allowed")
        candidates = [name for name in names if name.count("/") == 1 and name.endswith("/manifest.json")]
        if len(candidates) != 1:
            raise ValueError("Expected one package-root manifest")
        manifest = json.loads(archive.read(candidates[0]))
        prefix = manifest["artifact_id"] + "/"
        safe_member(manifest["artifact_id"])
        expected_names = {prefix + name for name in manifest["files"]}
        expected_names.update({prefix + "manifest.json", prefix + "SHA256SUMS"})
        if set(names) != expected_names:
            raise ValueError("Archive members disagree with manifest")
        for name, expected in manifest["files"].items():
            safe_member(name)
            digest = hashlib.sha256()
            count = 0
            with archive.open(prefix + name) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    count += len(block)
            if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Archive member integrity mismatch: {name}")
        checksums = {}
        for line in archive.read(prefix + "SHA256SUMS").decode("utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if name in checksums:
                raise ValueError("Repeated checksum line")
            checksums[name] = digest
        expected_checksums = {name: value["sha256"] for name, value in manifest["files"].items()}
        expected_checksums["manifest.json"] = hashlib.sha256(archive.read(prefix + "manifest.json")).hexdigest()
        if checksums != expected_checksums:
            raise ValueError("Checksum list disagrees with package manifest")
    return dict(artifact_id=manifest["artifact_id"], members=len(names),
                weight_sha256=manifest["weight_sha256"], verified=True)


def write_archive(path, payload, metadata):
    """Write a new ZIP, verify its exact bytes, then promote the partial file."""
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {path}")
    artifact_id = metadata["artifact_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", artifact_id):
        raise ValueError("Use a portable artifact ID")
    fingerprints = {}
    for name, value in payload.items():
        safe_member(name)
        if name in ("manifest.json", "SHA256SUMS"):
            raise ValueError("Reserved package metadata filename")
        fingerprints[name] = payload_fingerprint(value)
    weight = metadata["weight_member"]
    if fingerprints[weight]["sha256"] != metadata["weight_sha256"]:
        raise ValueError("Planned weight differs from the frozen checkpoint")
    manifest = dict(metadata, files=fingerprints)
    content = dict(payload, **{"manifest.json": encode(manifest)})
    sums = {name: value["sha256"] for name, value in fingerprints.items()}
    sums["manifest.json"] = hashlib.sha256(content["manifest.json"]).hexdigest()
    content["SHA256SUMS"] = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name, value in sorted(content.items()):
            target = artifact_id + "/" + name
            if isinstance(value, Path):
                archive.write(value, target, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
            else:
                archive.writestr(target, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
    verified = verify_archive(partial)
    for name, value in payload.items():
        if isinstance(value, Path) and payload_fingerprint(value) != fingerprints[name]:
            raise ValueError(f"Source changed during packaging: {value}")
    partial.rename(path)
    return dict(verified, archive=path.name, archive_bytes=path.stat().st_size,
                archive_sha256=sha256(path), model_bytes=sum(v["bytes"] for n, v in fingerprints.items() if n.startswith("model/")),
                source_files={str(value): fingerprints[name] for name, value in payload.items() if isinstance(value, Path)})


def file_members(directory, names):
    """Select only declared inference files, keeping optimizer states outside."""
    result = {}
    directory = Path(directory).resolve()
    for name in names:
        path = directory / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError(f"Missing or non-local inference file: {path}")
        result["model/" + name] = path
    return result


def plans(root, paths):
    """Resolve the nine validation-selected artifacts and check frozen identities."""
    contract = load(root / "configs/reproduction.json")
    for name, expected in contract["input_sha256"].items():
        require_digest(root / name, expected)
    index = [json.loads(line) for line in (root / "results/run_index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    tests = {(r["method"], r["seed"]): r for r in index if r["split"] == "test"}
    base_identity = load(root / "artifacts/lora-lock.json")
    result = []
    for method, key, config_id in [("tfidf", "baseline_results", "char_1_3"), ("bert", "bert_results", "len512")]:
        directory = Path(paths[key])
        if load(directory / "frozen_winner.json")["config"]["config_id"] != config_id:
            raise ValueError(f"Frozen winner changed: {method}")
        for seed in SEEDS:
            run_id = f"{method}-{config_id}-seed{seed}"
            run = directory / run_id
            selection = load(run / "selection.json")
            source = Path(selection["model"])
            config = load(run / "config.json")
            if config["seed"] != seed or config["config_id"] != config_id:
                raise ValueError(f"Source training config mismatch: {run_id}")
            expected = tests[(method, seed)]["checkpoint_id"]
            weight_name = "pipeline.joblib" if method == "tfidf" else "model.safetensors"
            model_dir = source.parent if method == "tfidf" else source
            files = [weight_name] if method == "tfidf" else [weight_name, "config.json", "tokenizer.json", "tokenizer_config.json"]
            payload = file_members(model_dir, files)
            require_digest(model_dir / weight_name, expected)
            if selection["model_sha256"] != expected:
                raise ValueError(f"Selected checkpoint disagrees with test identity: {run_id}")
            training = {k: v for k, v in config.items() if k in BASELINE_FIELDS}
            training["package_versions"] = config.get("environment", {}).get("versions", {})
            training["training_source_sha256"] = config.get("environment", {}).get("source_sha256")
            metadata = dict(artifact_id=run_id, method=method, config_id=config_id, seed=seed,
                            source_run_id=run_id, artifact_kind="sklearn_pipeline" if method == "tfidf" else "full_encoder_classifier",
                            weight_member="model/" + weight_name, weight_sha256=expected,
                            selection=dict(split="validation", primary="macro_f1", epoch=selection["epoch"], macro_f1=selection["macro_f1"]),
                            training_config_source_sha256=sha256(run / "config.json"), requires_base_model=False)
            result.append(dict(metadata=metadata, payload=payload, training=training, test=tests[(method, seed)]["metrics"]))
    for seed in SEEDS:
        directory = Path(paths["lora_adapters"][str(seed)])
        run = directory.parents[1]
        frozen = next(row for row in base_identity["adapters"] if row["seed"] == seed)
        expected = tests[("qwen_lora", seed)]["checkpoint_id"]
        require_digest(directory / "adapter_model.safetensors", expected)
        require_digest(directory / "adapter_config.json", frozen["adapter_config_sha256"])
        if expected != frozen["adapter_sha256"]:
            raise ValueError("LoRA checkpoint disagrees with frozen adapter lock")
        raw = load(run / "train.json")
        if raw["seed"] != seed or raw["lora_rank"] != 8 or raw["lora_target"] != "q_proj,k_proj,v_proj,o_proj":
            raise ValueError("LoRA source config differs from T01")
        selected = load(run / "best_checkpoint.json")
        if Path(selected["checkpoint"]).resolve() != directory.resolve():
            raise ValueError("LoRA adapter is not the validation-selected checkpoint")
        payload = file_members(directory, ["adapter_model.safetensors", "adapter_config.json"])
        license_file = Path(paths["qwen_model"]) / "LICENSE"
        require_digest(license_file, base_identity["base_model_files"]["LICENSE"])
        payload["licenses/Qwen-LICENSE"] = license_file
        training = {k: v for k, v in raw.items() if k in LORA_FIELDS}
        training["model_repository"] = "Qwen/Qwen3-1.7B"
        training["package_versions"] = load(root / "reports/audits/environment.json")["packages"]
        metadata = dict(artifact_id=f"qwen-lora-T01-seed{seed}", method="qwen_lora", config_id="T01", seed=seed,
                        source_run_id=tests[("qwen_lora", seed)]["run_id"], artifact_kind="peft_lora_adapter",
                        weight_member="model/adapter_model.safetensors", weight_sha256=expected,
                        adapter_config_sha256=frozen["adapter_config_sha256"],
                        selection=dict(split="validation", primary="macro_f1", epoch=selected["epoch"], global_step=selected["global_step"], macro_f1=selected["macro_f1"]),
                        training_config_source_sha256=sha256(run / "train.json"), requires_base_model=True,
                        base_model=dict(repository="Qwen/Qwen3-1.7B", files=base_identity["base_model_files"]),
                        inference=load(root / "configs/methods/lora-frozen.json")["inference"])
        result.append(dict(metadata=metadata, payload=payload, training=training, test=tests[("qwen_lora", seed)]["metrics"]))
    if len(result) != 9:
        raise ValueError("Expected nine trained artifacts")
    return result, contract


def package_readme(metadata):
    base = "需要另外取得Qwen3-1.7B基座；本包只含普通LoRA适配器。" if metadata["requires_base_model"] else "本包包含该路线保存的完整推理产物。"
    return (f"# {metadata['artifact_id']}\n\n"
            f"方案：{metadata['method']}；配置：{metadata['config_id']}；训练种子：{metadata['seed']}。\n\n"
            f"{base}\n\n"
            "- model/：推理产物。\n- metadata/training-config.json：实际训练参数，历史机器路径从本说明性副本中移除。\n"
            "- metadata/test-metrics.json：原冻结测试记录，不是本次重新预测。\n"
            "- manifest.json与SHA256SUMS：逐文件大小、身份与校验和。\n\n"
            "标签0为负面，1为正面；按验证Macro-F1选择检查点，未按测试挑选种子。\n\n"
            "提取到自己的目录后，显式配置模型和基座位置；原模型配置文件中的历史路径保持原字节，不能直接当作新机器地址。\n"
            "LoRA使用冻结Qwen基座的分词器、qwen3模板与关闭思考设置；检查点附带的训练状态或额外tokenizer不在适配器包内。\n"
            "模型运行依赖以仓库requirements.integration.lock与环境E01记录为准；本包不包含Python环境、优化器状态或数据集。\n\n"
            "来源：Qwen https://modelscope.cn/models/Qwen/Qwen3-1.7B ；MacBERT https://huggingface.co/hfl/chinese-macbert-base 。\n"
            "Qwen适配器包附基座现存LICENSE；MacBERT源目录没有独立LICENSE文件，未补造许可文件。\n\n"
            "本次仅按字节读取、打包和流式校验；没有加载模型、重新训练、重新预测或上传。\n").encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-id", help="Create a new nine-artifact batch")
    mode.add_argument("--verify-archive", type=Path, help="Verify one archive without source models")
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.local.json")
    args = parser.parse_args(argv)
    if args.verify_archive:
        print(json.dumps(verify_archive(args.verify_archive), ensure_ascii=True))
        return 0
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("Use a portable unique run ID")
    output = ROOT / "artifacts/packages" / args.run_id
    evidence = ROOT / "reports/packaging" / args.run_id
    if output.exists() or evidence.exists():
        parser.error("Use a new run ID; existing packages and evidence are not overwritten")
    output.mkdir(parents=True)
    evidence.mkdir(parents=True)
    started = time.perf_counter()
    status = dict(status="preflight", run_id=args.run_id, artifacts=[],
                  created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  command=sys.argv, python=platform.python_version(), script_sha256=sha256(Path(__file__)),
                  training=False, model_loading=False, inference=False, uploaded=False)
    status["compression"] = dict(format="zip", method="deflate", level=1, zlib=zlib.ZLIB_RUNTIME_VERSION)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    status["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True)
    status["working_tree_dirty"] = bool(dirty.stdout.strip())
    (evidence / "status.json").write_bytes(encode(status))
    try:
        entries, contract = plans(ROOT, load(args.paths))
        model_bytes = sum(value.stat().st_size for item in entries for name, value in item["payload"].items() if name.startswith("model/"))
        if shutil.disk_usage(output).free < model_bytes + 64 * 1024 * 1024:
            raise OSError("Insufficient free space for new archives")
        status.update(status="packaging", model_payload_bytes=model_bytes)
        print(f"Verified nine artifact identities; model bytes={model_bytes}", flush=True)
        for pos, entry in enumerate(entries, 1):
            metadata = entry["metadata"]
            metadata.update(labels={"0": "negative", "1": "positive"}, data_version="v1-20261023",
                            data_split_seed=67, shared_max_characters=512,
                            data_file_sha256={name: digest for name, digest in contract["input_sha256"].items() if name.endswith(("/train.jsonl", "/validation.jsonl", "/test.jsonl"))})
            payload = dict(entry["payload"])
            payload["metadata/training-config.json"] = encode(entry["training"])
            payload["metadata/test-metrics.json"] = encode(entry["test"])
            payload["README.md"] = package_readme(metadata)
            name = metadata["artifact_id"]
            print(f"[{pos}/9] Packaging {name}", flush=True)
            item_started = time.perf_counter()
            packed = write_archive(output / f"{name}.zip", payload, metadata)
            packed.update(method=metadata["method"], seed=metadata["seed"],
                          selection=metadata["selection"], requires_base_model=metadata["requires_base_model"],
                          elapsed_seconds=time.perf_counter() - item_started)
            status["artifacts"].append(packed)
            (evidence / "status.json").write_bytes(encode(status))
            print(f"[{pos}/9] Verified {name}: {packed['archive_bytes']} bytes", flush=True)
        status.update(status="completed", elapsed_seconds=time.perf_counter() - started,
                      archive_total_bytes=sum(row["archive_bytes"] for row in status["artifacts"]))
        sums = "".join(f"{row['archive_sha256']}  {row['archive']}\n" for row in status["artifacts"])
        (output / "SHA256SUMS").write_text(sums, encoding="utf-8")
        (evidence / "SHA256SUMS").write_text(sums, encoding="utf-8")
        (output / "catalog.json").write_bytes(encode(status))
        (evidence / "catalog.json").write_bytes(encode(status))
        print(f"Ready: {output}", flush=True)
        result = 0
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      elapsed_seconds=time.perf_counter() - started, error=str(error), traceback=traceback.format_exc())
        (evidence / "error.log").write_text(status["traceback"], encoding="utf-8")
        print(status["traceback"], file=sys.stderr, flush=True)
        result = 130 if isinstance(error, KeyboardInterrupt) else 1
    (evidence / "status.json").write_bytes(encode(status))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
