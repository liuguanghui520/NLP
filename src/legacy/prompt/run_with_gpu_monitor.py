#!/usr/bin/env python3
"""Run a finite command with stdout/stderr and GPU device-memory sampling."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def pump(source, target, log) -> None:
    for chunk in iter(lambda: source.readline(), ""):
        if not chunk:
            break
        target.write(chunk)
        target.flush()
        log.write(chunk)
        log.flush()


def sample_gpu(process: subprocess.Popen, csv_path: Path, interval: float) -> list[int]:
    samples: list[int] = []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["elapsed_seconds", "memory_used_mib", "memory_total_mib", "gpu_util_pct", "power_w"])
        started = time.perf_counter()
        while process.poll() is None:
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
                used = int(float(values[0]))
                samples.append(used)
                writer.writerow([round(time.perf_counter() - started, 3), *values])
                handle.flush()
            time.sleep(interval)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--monitor-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command")

    for path in (args.stdout, args.stderr, args.monitor_csv, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)

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
        samples = sample_gpu(process, args.monitor_csv, args.interval)
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()

    summary = {
        "command": command,
        "return_code": return_code,
        "sample_count": len(samples),
        "gpu_device_sampled_peak_used_mib": max(samples) if samples else None,
        "monitor_csv": str(args.monitor_csv),
        "stdout_log": str(args.stdout),
        "stderr_log": str(args.stderr),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
