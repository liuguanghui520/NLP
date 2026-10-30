"""Read-only model runtime checks; no weights, data or predictions are loaded."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


def is_wsl():
    return platform.system() == "Linux" and (
        "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME")))


def windows_mount_path(value):
    """Inspect resolved mounts, including symlinks and custom DrvFS mount points."""
    path = Path(value).expanduser().resolve()
    try:
        probe = subprocess.run(["findmnt", "--noheadings", "--output", "FSTYPE,OPTIONS",
                                "--target", str(path)], capture_output=True, text=True, timeout=5)
        if probe.returncode == 0 and probe.stdout.strip():
            kind, *options = probe.stdout.strip().split(maxsplit=1)
            return kind.lower() == "drvfs" or (
                kind.lower() in {"9p", "virtiofs"} and "drvfs" in " ".join(options).lower())
    except (OSError, subprocess.TimeoutExpired):
        pass
    # Standard WSL drive paths remain detectable when findmnt is unavailable.
    return bool(re.match(r"^/mnt/[a-z](?:/|$)", path.as_posix(), flags=re.IGNORECASE))


def storage_check(root, specification):
    paths = {"virtual_environment": str(Path(sys.prefix).resolve()),
             "factory_root": str(Path(specification["factory_root"]).expanduser().resolve())
             if specification.get("factory_root") else None,
             "project_root": str(Path(root).resolve())}
    result = dict(wsl=is_wsl(), paths=paths, windows_mounts=[], errors=[], warnings=[])
    if not result["wsl"]:
        return result
    for name, value in paths.items():
        if value and windows_mount_path(value):
            result["windows_mounts"].append(name)
            if name == "project_root":
                result["warnings"].append(
                    "提交包位于Windows挂载目录，可继续；若读取缓慢，可复制源码到WSL的/home目录。")
            else:
                result["errors"].append(
                    f"{name}位于Windows挂载目录：{value}。这可能导致导入超时。"
                    "请保留旧环境，在WSL原生/home目录重新创建虚拟环境、重新获取固定版本框架并安装冻结依赖；"
                    "不要直接剪切虚拟环境。激活新环境并更新路径JSON后重新预检，详见README的挂载目录超时处理。")
    return result


def inspect_environment(root, specification):
    contract = json.loads((Path(root) / "configs/reproduction-model-environment.json").read_text(encoding="utf-8"))
    result = dict(status="failed", python=sys.version, executable=sys.executable,
                  platform=platform.platform(), errors=[], packages={}, model_loading=False,
                  inference=False, training=False)
    errors = result["errors"]
    result["storage"] = storage_check(root, specification)
    result["warnings"] = result["storage"]["warnings"]
    errors.extend(result["storage"]["errors"])
    if errors:
        # Avoid importing thousands of package files from a known slow mount.
        result["stopped_before_package_imports"] = True
        return result
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
        except subprocess.TimeoutExpired as error:
            result["factory_import"] = dict(status="timeout", timeout_seconds=error.timeout)
            errors.append(
                "LLaMA-Factory检查超时。确认当前Python环境和factory_root均在WSL原生/home目录；"
                "保留日志，按README重建环境并更新路径JSON后重试。若已在原生目录，"
                "继续排查内存、磁盘及导入问题，不直接认定为挂载目录故障或放宽超时。")
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"LLaMA-Factory verification failed: {error}")
    try:
        probe = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=60)
        result["pip_check"] = dict(returncode=probe.returncode, stdout=probe.stdout, stderr=probe.stderr)
        if probe.returncode:
            errors.append("pip check failed; see preflight.json")
    except subprocess.TimeoutExpired as error:
        result["pip_check"] = dict(status="timeout", timeout_seconds=error.timeout)
        errors.append("pip check超时；检查虚拟环境是否位于Windows挂载目录，按README在WSL原生目录重建；"
                      "若路径正确，保留日志并继续排查依赖和磁盘问题。")
    except (OSError, subprocess.SubprocessError) as error:
        errors.append(f"pip check unavailable: {error}")
    result["status"] = "passed" if not errors else "failed"
    return result
