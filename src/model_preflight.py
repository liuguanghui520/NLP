"""Read-only model runtime checks; no weights, data or predictions are loaded."""

import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def inspect_environment(root, specification):
    contract = json.loads((Path(root) / "configs/reproduction-model-environment.json").read_text(encoding="utf-8"))
    result = dict(status="failed", python=sys.version, executable=sys.executable,
                  platform=platform.platform(), errors=[], packages={}, model_loading=False,
                  inference=False, training=False)
    errors = result["errors"]
    if sys.version_info[:2] != (3, 11):
        errors.append("The locked model environment requires Python 3.11; the stdlib audit supports 3.11+")
    for name, expected in contract["required_packages"].items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        result["packages"][name] = version
        if version is None or version.split("+")[0] != expected:
            errors.append(f"{name}: expected {expected}, found {version}; install requirements.integration.lock")
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        result["nvidia_smi"] = dict(returncode=probe.returncode, stdout=probe.stdout.strip(), stderr=probe.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        result["nvidia_smi"] = dict(error=str(error))
    try:
        import torch
        result.update(torch_build=torch.__version__, torch_cuda=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        if torch.version.cuda != contract["torch_cuda"]:
            errors.append(f"Expected PyTorch CUDA {contract['torch_cuda']} build, found {torch.version.cuda}; no automatic substitution")
        if not result["cuda_available"]:
            errors.append("CUDA unavailable. For cu130 upgrade the Windows NVIDIA driver to a CUDA 13-compatible R580+ driver and restart; driver 537.70/CUDA 12.2 is insufficient")
        elif not torch.cuda.is_bf16_supported():
            errors.append("This CUDA device/runtime does not support BF16; no FP16/CPU/QLoRA substitution")
        else:
            value = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16)
            result["bf16_matmul"] = (value @ value).float().cpu().tolist()
            torch.cuda.synchronize()
            del value
    except Exception as error:
        errors.append(f"CUDA/BF16 execution failed: {error}")
    factory_value = specification.get("factory_root")
    factory = Path(factory_value).expanduser().resolve() if factory_value else None
    qwen_value = specification.get("qwen_model")
    if not qwen_value or not Path(qwen_value).expanduser().is_dir():
        errors.append("qwen_model must point to your local Qwen3-1.7B directory")
    if factory is None or not (factory / "src/llamafactory").is_dir():
        errors.append("factory_root must point to your LLaMA-Factory checkout")
    else:
        try:
            commit = subprocess.check_output(["git", "-C", str(factory), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
            result["factory_commit"] = commit
            if commit != contract["factory_commit"]:
                errors.append(f"LLaMA-Factory commit mismatch: expected {contract['factory_commit']}, found {commit}")
            changed = subprocess.check_output(["git", "-C", str(factory), "diff", "HEAD", "--", "src/llamafactory", "pyproject.toml"], text=True)
            if changed:
                errors.append("LLaMA-Factory tracked runtime source has local changes; use the pinned clean source")
            script = ("import sys; sys.path.insert(0,sys.argv[1]); "
                      "from llamafactory.chat import ChatModel; print('ChatModel import passed')")
            probe = subprocess.run([sys.executable, "-c", script, str(factory / "src")],
                                   capture_output=True, text=True, timeout=120)
            result["factory_import"] = dict(returncode=probe.returncode, stdout=probe.stdout, stderr=probe.stderr)
            if probe.returncode:
                errors.append("LLaMA-Factory ChatModel import failed; see factory_import in preflight.json")
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"LLaMA-Factory verification failed: {error}")
    try:
        probe = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=60)
        result["pip_check"] = dict(returncode=probe.returncode, stdout=probe.stdout, stderr=probe.stderr)
        if probe.returncode:
            errors.append("pip check failed; see preflight.json")
    except (OSError, subprocess.SubprocessError) as error:
        errors.append(f"pip check unavailable: {error}")
    result["status"] = "passed" if not errors else "failed"
    return result
