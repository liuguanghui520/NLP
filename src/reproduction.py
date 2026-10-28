"""Read-only verification of frozen evidence, with an optional analysis replay.

The default path uses Python's standard library only. It never reads personal
model paths, loads weights, trains, or appends to the canonical run index.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from uuid import uuid4

from src.common.io import digest, load, rows, save
from src.evaluation.metrics import evaluate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/reproduction.json"
METRICS = ("macro_f1", "accuracy", "weighted_f1", "invalid_rate")


def checked_path(root, relative):
    """Reject absolute paths and paths escaping the checkout, including symlinks."""
    name = Path(relative)
    path = (root / name).resolve()
    if name.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Evidence path must stay inside the checkout: {relative}")
    return path


def verify_data(root, config):
    """Check immutable bytes, split membership, labels, and model-visible text."""
    directory = checked_path(root, config["data_directory"])
    manifest = load(checked_path(root, config["data_manifest"]))
    for name, expected in manifest["files"].items():
        path = checked_path(directory, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen data hash mismatch: {name}")
    ids = load(directory / "split_ids.json")
    datasets, all_ids, prefixes = {}, set(), set()
    for split, expected_count in config["split_counts"].items():
        dataset = rows(directory / f"{split}.jsonl")
        current_ids = [row["sample_id"] for row in dataset]
        if len(dataset) != expected_count or current_ids != ids[split]:
            raise ValueError(f"Frozen split count or ID order mismatch: {split}")
        if len(set(current_ids)) != len(current_ids) or all_ids.intersection(current_ids):
            raise ValueError(f"Repeated or overlapping sample IDs: {split}")
        for row in dataset:
            if type(row["label"]) is not int or row["label"] not in (0, 1):
                raise ValueError(f"Invalid label: {row['sample_id']}")
            if row["split"] != split:
                raise ValueError(f"Wrong split field: {row['sample_id']}")
            text = row["text"]
            if not text or len(text) > config["max_characters"] or text in prefixes:
                raise ValueError(f"Empty, overlength or repeated model input: {row['sample_id']}")
            prefixes.add(text)
        all_ids.update(current_ids)
        datasets[split] = dataset
    return datasets


def audit(root, config):
    """Recompute every frozen run, then average the three training seeds."""
    for name, expected in config["input_sha256"].items():
        path = checked_path(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen evidence hash mismatch: {name}")
    datasets = verify_data(root, config)
    index = rows(checked_path(root, config["run_index"]))
    index_by_id = {item["run_id"]: item for item in index}
    if len(index_by_id) != len(index) or set(index_by_id) != set(config["run_ids"]):
        raise ValueError("Frozen run IDs are missing, repeated or changed")
    values, verified = defaultdict(list), []
    max_difference = 0.0
    for run_id in config["run_ids"]:
        item = index_by_id[run_id]
        predictions = rows(root / "results/predictions" / f"{run_id}.jsonl")
        result = evaluate(predictions, datasets[item["split"]])
        if not result["complete"]:
            raise ValueError(f"Incomplete prediction evidence: {run_id}: {result['errors']}")
        actual = result["metrics"]
        for key in METRICS:
            difference = abs(actual[key] - item["metrics"][key])
            max_difference = max(max_difference, difference)
            if difference > config["metric_tolerance"]:
                raise ValueError(f"Metric mismatch: {run_id}: {key}")
        verified.append(dict(run_id=run_id, split=item["split"], seed=item["seed"], **actual))
        if item["split"] == "test":
            values[item["method"]].append(dict(seed=item["seed"], **actual))
    summary = []
    for method in config["method_order"]:
        runs = values[method]
        expected_seeds = config["training_seeds"] if method in config["trained_methods"] else [None]
        if Counter(row["seed"] for row in runs) != Counter(expected_seeds):
            raise ValueError(f"Training seed coverage mismatch: {method}")
        item = dict(method=method, n=config["split_counts"]["test"], seeds=[r["seed"] for r in runs])
        for key in METRICS:
            scores = [run[key] for run in runs]
            item[key] = dict(mean=statistics.mean(scores), sample_std=statistics.stdev(scores) if len(scores) > 1 else None)
            if not math.isclose(item[key]["mean"], config["expected_test_means"][method][key], rel_tol=0, abs_tol=config["metric_tolerance"]):
                raise ValueError(f"Main result mismatch: {method}: {key}")
        summary.append(item)
    return dict(runs=verified, methods=summary, max_metric_difference=max_difference,
                split_counts={split: len(data) for split, data in datasets.items()})


def environment(root):
    """Record the actual interpreter; do not infer a new-machine acceptance."""
    result = dict(python=platform.python_version(), executable=sys.executable,
                  platform=platform.platform(), site_disabled=bool(sys.flags.no_site))
    if shutil.which("git"):
        command = ["git", "-C", str(root), "rev-parse", "--show-toplevel"]
        git = subprocess.run(command, capture_output=True, text=True)
        if git.returncode == 0 and Path(git.stdout.strip()).resolve() == root.resolve():
            result["git_commit"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            result["working_tree_dirty"] = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    return result


def check_analysis_environment(root, config):
    """Fail with installation instructions instead of silently upgrading packages."""
    expected = load(root / config["analysis_environment"])
    actual = {}
    errors = []
    if platform.python_version() != expected["python"]:
        errors.append(f"Python {platform.python_version()} != {expected['python']}")
    for name, version in expected["packages"].items():
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
        if actual[name] != version:
            errors.append(f"{name}: {actual[name]} != {version}")
    if not shutil.which("git"):
        errors.append("Git is required by the existing full report generator")
    if errors:
        raise ValueError("Analysis environment mismatch. Use the pinned analysis environment; no packages were changed. " + "; ".join(errors))
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    if not available.intersection({"Microsoft YaHei", "Noto Sans CJK SC", "SimHei"}):
        raise ValueError("Install a supported Chinese font before generating figures: Microsoft YaHei, Noto Sans CJK SC or SimHei")
    return actual


def software_checks(root, output):
    """Run software checks with a writable, isolated temporary directory."""
    previous = tempfile.tempdir
    directory = output / "test-temp"
    directory.mkdir()
    tempfile.tempdir = str(directory)
    try:
        with (output / "software-checks.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                suite = unittest.defaultTestLoader.discover(str(root / "tests"))
                result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise RuntimeError("Software checks failed; see software-checks.log")
        return dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors))
    finally:
        tempfile.tempdir = previous


def analysis_replay(root, output, config, methods):
    """Replay the saved-data report under a new ID; never invoke a model."""
    packages = check_analysis_environment(root, config)
    checks = software_checks(root, output)
    run_id = "repro-" + uuid4().hex[:12]
    command = [sys.executable, str(root / "scripts/build_comparison_report.py"),
               "--config", config["analysis_config"], "--run-id", run_id]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # An independent source copy must not inherit its parent checkout's Git ID.
    env["GIT_CEILING_DIRECTORIES"] = str(root.parent)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(output / "mpl-cache")
    save(output / "analysis-command.json", command)
    with (output / "analysis-stdout.log").open("w", encoding="utf-8") as stdout:
        with (output / "analysis-stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
    if process.returncode:
        raise RuntimeError("Analysis replay failed; see analysis-stderr.log")
    directory = root / "results/analysis" / run_id
    status = load(directory / "status.json")
    figures = load(directory / "figure-index.json")
    if status["status"] != "completed" or len(figures) != config["expected_figure_pairs"]:
        raise ValueError("Analysis did not complete all expected figures")
    generated = {row["method"]: row for row in load(directory / "methods.json")}
    for item in methods:
        for key in METRICS:
            if abs(generated[item["method"]][key]["mean"] - item[key]["mean"]) > config["metric_tolerance"]:
                raise ValueError("Generated report disagrees with independently audited metrics")
    return dict(run_id=run_id, packages=packages, software_checks=checks,
                figure_pairs=len(figures), status_path=str(directory / "status.json"))


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="store_true", help="Also replay 22 figure pairs and software checks in the pinned environment")
    parser.add_argument("--output", help="New directory below results/reproduction; defaults to a unique ID")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required for the standard-library audit")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = checked_path(root, args.output or f"results/reproduction/{run_id}")
    boundary = (root / "results/reproduction").resolve()
    if not output.is_relative_to(boundary) or output == boundary:
        parser.error("Output must be a new directory below results/reproduction")
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    status = dict(status="running", mode="offline_analysis" if args.analysis else "saved_predictions_audit",
                  command=sys.argv if argv is None else argv, environment=environment(root),
                  training=False, model_loading=False, inference=False,
                  new_machine_model_reproduction="not_executed", teacher_acceptance="not_recorded")
    status["entry_source_sha256"] = {
        name: digest(root / name)
        for name in ["reproduce.py", "src/reproduction.py", CONFIG]
        if (root / name).is_file()
    }
    save(output / "status.json", status)
    try:
        config = load(root / CONFIG)
        status["seed_registry"] = config["seed_registry"]
        print("Checking frozen data, predictions and seed coverage...", flush=True)
        result = audit(root, config)
        save(output / "metrics.json", result)
        save(output / "config.json", config)
        save(output / "input-sha256.json", config["input_sha256"])
        lines = ["# 离线复现核查", "", "本结果重算已有预测，没有加载模型，不等于新机或教师验收。", "",
                 "|方法|测试N|Macro-F1|Accuracy|种子|", "|---|---|---|---|---|"]
        for item in result["methods"]:
            lines.append(f"|{item['method']}|{item['n']}|{item['macro_f1']['mean']:.8f}|{item['accuracy']['mean']:.8f}|{item['seeds']}|")
        (output / "结果核查.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if args.analysis:
            print("Replaying software checks and existing-data figures...", flush=True)
            status["analysis"] = analysis_replay(root, output, config, result["methods"])
        # Detect unintended writes to frozen evidence during any stage.
        for name, expected in config["input_sha256"].items():
            if digest(root / name) != expected:
                raise ValueError(f"Frozen evidence changed during verification: {name}")
        status.update(status="passed", audited_runs=len(result["runs"]),
                      max_metric_difference=result["max_metric_difference"])
        print(f"Offline verification passed. Evidence: {output}", flush=True)
        return_code = 0
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=str(error), traceback=traceback.format_exc())
        (output / "error.log").write_text(status["traceback"], encoding="utf-8")
        print(f"Verification failed; original evidence preserved. See {output / 'error.log'}", file=sys.stderr)
        return_code = 130 if isinstance(error, KeyboardInterrupt) else 1
    status.update(elapsed_seconds=time.perf_counter() - started,
                  completed_at=datetime.now(timezone.utc).isoformat())
    status["output_sha256"] = {
        p.relative_to(output).as_posix(): digest(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "status.json"
        and not {"mpl-cache", "test-temp"}.intersection(p.relative_to(output).parts)
    }
    save(output / "status.json", status)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
