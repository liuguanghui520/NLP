#!/usr/bin/env python3
"""Run the fixed zero-shot and few-shot Qwen validation experiments in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiment_contracts import RUNS_DIR, utc_stamp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "qwen_prompt_runner.py"
CONFIG_ID = "fixed_prompt_v1"
MODES = ("qwen_zero_shot", "qwen_few_shot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["all", *MODES],
        default=["all"],
        help="Run both prompt modes serially by default.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    modes = MODES if "all" in args.modes else tuple(dict.fromkeys(args.modes))

    stamp = utc_stamp()
    commands: list[list[str]] = []
    for mode in modes:
        run_id = f"C_{mode}_{CONFIG_ID}_noseed_{stamp}"
        run_dir = RUNS_DIR / run_id
        if run_dir.exists():
            raise FileExistsError(run_dir)
        commands.append(
            [
                sys.executable,
                str(RUNNER),
                "--mode",
                mode,
                "--source-split",
                "validation",
                "--run-id",
                run_id,
                "--config-id",
                CONFIG_ID,
            ]
        )

    for command in commands:
        print("EXEC " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(
        f"FORMAL_QWEN_VALIDATION_{'PLANNED' if args.dry_run else 'COMPLETED'} "
        f"runs={len(commands)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
