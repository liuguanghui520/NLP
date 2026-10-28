#!/usr/bin/env python3
"""Refresh per-run artifact hashes without changing run outcomes or predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_contracts import (
    RUNS_DIR,
    load_json,
    rebuild_run_index,
    sha256_file,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    refreshed = 0
    for run_dir in sorted(path for path in args.runs_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = load_json(manifest_path)
        if "artifact_sha256" not in manifest:
            continue
        manifest["artifact_sha256"] = {
            path.name: sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "run_manifest.json"
        }
        write_json(manifest_path, manifest)
        refreshed += 1
    index = rebuild_run_index(args.runs_root)
    print(f"artifact_hashes_refreshed runs={refreshed} index_entries={len(index['runs'])}")


if __name__ == "__main__":
    main()
