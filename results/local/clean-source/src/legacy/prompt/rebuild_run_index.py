#!/usr/bin/env python3
"""Rebuild the experiment run index from all saved manifests."""

from __future__ import annotations

from experiment_contracts import rebuild_run_index


def main() -> None:
    index = rebuild_run_index()
    print(f"run_index_rebuilt runs={len(index['runs'])}")


if __name__ == "__main__":
    main()
