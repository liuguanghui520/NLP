# Repository Guidelines

## Project Structure & Module Organization

This repository supports a Chinese sentiment-analysis course project comparing prompting and fine-tuning. It contains engineering utilities and records, rather than the upstream training framework.

- `open-llamafactory.cmd`: opens an interactive WSL shell, activates the environment, and enables ModelScope.
- `start-llamafactory.ps1`: launches the local WebUI on port 7860.
- `verify_llamafactory.py` and `verify_tools.py`: environment smoke checks.
- `logs/tools-install/`: dependency snapshots and installation/verification evidence.
- Root Markdown documents: setup instructions, source records, and project handoff. Consult the latest dated corrections in `项目交接-本机.md`.

Upstream source lives at `/home/lgh/LLaMA-Factory`; the virtual environment is `/home/lgh/.venvs/llamafactory`. Model weights and datasets reside outside this repository under `/home/lgh/models/` and `/home/lgh/datasets/`.

## Build, Test, and Development Commands

No repository build step is configured. From Windows CMD:

```cmd
open-llamafactory.cmd
```

For the WebUI, run `.\start-llamafactory.ps1` in PowerShell, then visit `http://127.0.0.1:7860`.

Run either smoke check from Windows using the WSL interpreter:

```cmd
wsl -d Ubuntu -u lgh -- /home/lgh/.venvs/llamafactory/bin/python /mnt/c/Users/lgh/Documents/ChatGPT/NLP/verify_llamafactory.py
```

Replace the final filename with `verify_tools.py` to check dependency compatibility, SwanLab, and Optuna. Training uses `llamafactory-cli train <config.yaml>` inside the activated shell; no project training YAML is currently present.

## Coding Style & Naming Conventions

Use four-space indentation and `snake_case` for Python. Keep platform launchers small and quote paths. Use UTF-8 for documentation and explicit encodings for text I/O. Follow existing `verify_*.py` naming for smoke checks. No formatter or linter is configured.

## Testing Guidelines

Checks use Python assertions; no pytest suite or coverage threshold exists. CUDA checks require the NVIDIA GPU. The framework check briefly binds port 7860. The tools check overwrites verification snapshots/logs and adds offline runs and SQLite studies; preserve previous evidence before rerunning. Report these as software checks, never model-quality results.

## Commit & Pull Request Guidelines

There are no commits yet, so no historical convention exists. Use concise imperative subjects, such as `docs: clarify WSL startup`. PRs should describe changes, validation, environment assumptions, and evidence files; include screenshots for UI changes. Exclude credentials, model weights, and bulk datasets.

## Experiment & Configuration Boundaries

Preserve existing dependency versions and record intentional changes. Do not silently replace Qwen3-1.7B, ordinary LoRA, or dataset splits. Formal experimental design and result analysis belong to the student. Record AI assistance, use unique run IDs, retain failures, and keep test data out of tuning.
