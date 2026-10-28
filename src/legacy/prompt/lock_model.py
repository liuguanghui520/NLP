#!/usr/bin/env python3
"""Create a deterministic integrity lock for the local Qwen model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open


MODEL_ID = "Qwen/Qwen3-1.7B"
SOURCE_URL = "https://www.modelscope.cn/models/Qwen/Qwen3-1.7B"
REVISION = "4855588ea1a12789f2e965e5f52a9e4a24c94b2a"
LICENSE = "Apache-2.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()

    config_path = model_dir / "config.json"
    generation_path = model_dir / "generation_config.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    index_path = model_dir / "model.safetensors.index.json"
    required = [config_path, generation_path, tokenizer_config_path, index_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required model files: {missing}")

    config = load_json(config_path)
    generation_config = load_json(generation_path)
    tokenizer_config = load_json(tokenizer_config_path)
    index = load_json(index_path)
    weight_files = sorted(set(index["weight_map"].values()))
    for name in weight_files:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"Missing indexed weight shard: {name}")

    tensor_count = 0
    parameter_count = 0
    weight_shards: list[dict[str, Any]] = []
    for name in weight_files:
        path = model_dir / name
        shard_tensors = 0
        shard_parameters = 0
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                count = 1
                for dim in shape:
                    count *= dim
                shard_tensors += 1
                shard_parameters += count
        tensor_count += shard_tensors
        parameter_count += shard_parameters
        weight_shards.append(
            {
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "tensors": shard_tensors,
                "parameters": shard_parameters,
            }
        )

    ignored_outputs = {"model_lock.json", "MODEL_LOCK.md"}
    files = [
        path
        for path in sorted(model_dir.rglob("*"))
        if path.is_file() and path.relative_to(model_dir).as_posix() not in ignored_outputs
    ]
    file_entries = [
        {
            "file": path.relative_to(model_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    snapshot_manifest = "".join(
        f"{entry['file']}\0{entry['bytes']}\0{entry['sha256']}\n" for entry in file_entries
    ).encode("utf-8")
    snapshot_sha256 = hashlib.sha256(snapshot_manifest).hexdigest()

    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise RuntimeError("Tokenizer chat_template is missing.")
    chat_template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()

    lock = {
        "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "source_url": SOURCE_URL,
        "download_ref": REVISION,
        "resolved_revision": REVISION,
        "revision_type": "ModelScope git commit",
        "license": LICENSE,
        "local_path": "models/Qwen3-1.7B",
        "file_count": len(file_entries),
        "total_bytes": sum(entry["bytes"] for entry in file_entries),
        "snapshot_sha256": snapshot_sha256,
        "model_config": {
            "architecture": config.get("architectures"),
            "model_type": config.get("model_type"),
            "torch_dtype": config.get("torch_dtype"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "vocab_size": config.get("vocab_size"),
        },
        "weights": {
            "index_metadata_total_size": index.get("metadata", {}).get("total_size"),
            "tensor_count": tensor_count,
            "parameter_count": parameter_count,
            "shards": weight_shards,
        },
        "tokenizer": {
            "chat_template_present": True,
            "chat_template_sha256": chat_template_sha256,
            "eos_token": tokenizer_config.get("eos_token"),
            "pad_token": tokenizer_config.get("pad_token"),
        },
        "repository_generation_config": generation_config,
        "required_experiment_overrides": {
            "enable_thinking": False,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 8,
            "reason": "零样本和少样本实验要求关闭思考并使用贪心解码。",
        },
        "files": file_entries,
    }

    json_path = model_dir / "model_lock.json"
    json_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Qwen3-1.7B 模型锁定",
        "",
        f"- 状态：`{lock['status']}`",
        f"- 模型：`{MODEL_ID}`",
        f"- ModelScope revision：`{REVISION}`",
        f"- 许可：`{LICENSE}`",
        f"- 文件数：`{len(file_entries)}`",
        f"- 总大小：`{lock['total_bytes']}` bytes",
        f"- 参数数：`{parameter_count}`",
        f"- Snapshot SHA256：`{snapshot_sha256}`",
        f"- Chat Template SHA256：`{chat_template_sha256}`",
        "",
        "## 权重分片",
        "",
        "| 文件 | 大小 | SHA256 |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| `{entry['file']}` | {entry['bytes']} | `{entry['sha256']}` |"
        for entry in weight_shards
    )
    lines.extend(
        [
            "",
            "## 正式实验必须覆盖的生成参数",
            "",
            "- `enable_thinking=false`",
            "- `do_sample=false`",
            "- `num_beams=1`",
            "- `max_new_tokens=8`",
        ]
    )
    (model_dir / "MODEL_LOCK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": lock["status"],
        "revision": REVISION,
        "file_count": len(file_entries),
        "total_bytes": lock["total_bytes"],
        "snapshot_sha256": snapshot_sha256,
        "parameter_count": parameter_count,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
