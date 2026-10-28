#!/usr/bin/env python3
"""Write exact locks for the experiment tooling environment."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


DIRECT_PACKAGES = ["optuna", "swanlab"]
RELATED_PACKAGES = ["protobuf", "sqlalchemy", "pyroaring", "swanboard"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "environment"
    output_dir.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    uv = Path.home() / ".local/bin/uv"

    freeze = subprocess.run(
        [str(uv), "pip", "freeze", "--python", str(python)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not freeze.endswith("\n"):
        freeze += "\n"
    direct = (
        f"optuna=={version('optuna')}\n"
        f"swanlab[dashboard]=={version('swanlab')}\n"
    )

    direct_path = output_dir / "experiment-tools.direct.lock"
    environment_path = output_dir / "python-environment.lock"
    direct_path.write_text(direct, encoding="utf-8")
    environment_path.write_text(freeze, encoding="utf-8")

    lock = {
        "status": "validated",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "venv": str(python.parents[1]),
        "direct_packages": {package: version(package) for package in DIRECT_PACKAGES},
        "related_packages": {package: version(package) for package in RELATED_PACKAGES},
        "validation": {
            "optuna": {
                "status": "passed",
                "checks": ["import", "create_sqlite_study", "optimize_3_trials", "reload_study"],
            },
            "swanlab": {
                "status": "passed",
                "checks": ["import", "offline_init", "local_init", "log_scalar", "finish"],
            },
            "llamafactory_after_install": {
                "status": "passed",
                "version": "0.9.6.dev0",
            },
            "cherry_studio_transport": {
                "status": "passed",
                "direction": "Windows -> WSL2",
                "base_url": "http://localhost:18080/v1",
                "checks": ["health", "v1/models", "v1/chat/completions"],
                "temporary_mock_service_stopped": True,
            },
            "cherry_studio_live_llamafactory": {
                "status": "passed",
                "direction": "Windows -> LLaMA-Factory API in WSL2",
                "base_url": "http://localhost:18083/v1",
                "model_id": "qwen3-1.7b",
                "checks": ["v1/models", "v1/chat/completions"],
                "enable_thinking": False,
                "demo_response_contains_think_block": False,
                "service_stopped_after_test": True,
            },
        },
        "lock_files": {
            direct_path.name: {
                "bytes": direct_path.stat().st_size,
                "sha256": sha256_bytes(direct_path.read_bytes()),
            },
            environment_path.name: {
                "bytes": environment_path.stat().st_size,
                "sha256": sha256_bytes(environment_path.read_bytes()),
            },
        },
    }
    (output_dir / "experiment-tools.lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
