#!/usr/bin/env python3
"""Initialize persistent Optuna studies without adding experiment trials."""

from __future__ import annotations

import json
from pathlib import Path

import optuna

from experiment_contracts import PROJECT_ROOT, write_json


STUDIES = {
    "tfidf_lr": "maximize",
    "random_macbert": "maximize",
    "qwen_lora": "maximize",
}


def main() -> None:
    storage_dir = PROJECT_ROOT / "optuna"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_dir / 'official_studies.db'}"
    created = []
    for name, direction in STUDIES.items():
        study = optuna.create_study(
            study_name=name,
            storage=storage,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=True,
        )
        created.append(
            {
                "study_name": study.study_name,
                "direction": study.direction.name,
                "existing_trials": len(study.trials),
            }
        )
    write_json(
        storage_dir / "studies.json",
        {
            "storage": "optuna/official_studies.db",
            "optuna_version": optuna.__version__,
            "sampler_seed": 42,
            "studies": created,
        },
    )
    print(json.dumps(created, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
