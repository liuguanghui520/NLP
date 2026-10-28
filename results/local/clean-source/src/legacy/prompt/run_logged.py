#!/usr/bin/env python3
"""Run a finite command while preserving separate stdout/stderr logs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path


def pump(source, target, log) -> None:
    for chunk in iter(lambda: source.readline(), ""):
        if not chunk:
            break
        target.write(chunk)
        target.flush()
        log.write(chunk)
        log.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command")
    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)

    with args.stdout.open("w", encoding="utf-8") as stdout_log, args.stderr.open("w", encoding="utf-8") as stderr_log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        stdout_thread = threading.Thread(
            target=pump,
            args=(process.stdout, sys.stdout, stdout_log),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pump,
            args=(process.stderr, sys.stderr, stderr_log),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
