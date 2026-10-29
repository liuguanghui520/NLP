"""Audit hash-locked evidence; --models additionally performs real inference.

The default audit needs Python 3.11+ standard library only. No training is run.
The ZIP remains dataset-free. Each execution gets a fresh working directory.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote
from uuid import uuid4

from src.common.downloads import download_verified, safe_path, sha256


ROOT = Path(__file__).resolve().parent
REVISION = "106cdbed8db84057dba6b9c980d3a2f914856dd0"
BASE_URL = f"https://raw.githubusercontent.com/liuguanghui520/NLP/{REVISION}/"


def copy_sources(source, target):
    """Copy file bytes without chmod/utime across Windows and WSL mounts."""
    target.mkdir(parents=True, exist_ok=False)
    for item in source.rglob('*'):
        if '__pycache__' in item.parts or item.suffix == '.pyc' or item.name.endswith('.local.json'):
            continue
        destination = target / item.relative_to(source)
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, destination)


def fetch(root, relative, expected, *, cache=None, prefix=""):
    """Never accept an HTTP success response without a matching SHA256."""
    target = safe_path(root, relative)
    if target.exists() and sha256(target) != expected:
        raise ValueError(f"Existing input conflicts with frozen evidence: {relative}")
    download_verified(BASE_URL + quote(relative, safe="/"), target, expected,
                      cache_file=safe_path(cache, relative) if cache else None,
                      prefix=prefix, log=root / "download-events.jsonl")
    return relative


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", help="New, empty directory outside the extracted submission")
    parser.add_argument("--models", action="store_true", help="Run all 11 real-model test predictions after the saved-result audit")
    parser.add_argument("--paths", type=Path, help="Your Qwen, LLaMA-Factory and optional extracted-asset paths for --models")
    parser.add_argument("--preflight", action="store_true", help="Check the model environment without downloading inputs or loading models")
    parser.add_argument("--input-cache", type=Path, help="Existing checkout/work directory containing the 47 hash-locked inputs")
    parser.add_argument("--download-prefix", default="", help="Optional HTTPS mirror prefix ending in /; SHA256 verification remains mandatory")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11+ is required")
    if (args.models or args.preflight) and (args.paths is None or not args.paths.is_file()):
        parser.error("--models/--preflight requires an existing --paths JSON; see configs/reproduce-models.example.json")
    unique = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    work = Path(args.work_dir).resolve() if args.work_dir else Path(tempfile.gettempdir()) / f"nlp-course-{unique}"
    if work.is_relative_to(ROOT) or ROOT.is_relative_to(work):
        parser.error("Working directory must be separate from the extracted submission")
    if work.exists():
        parser.error(f"Refusing to overwrite existing directory: {work}")
    work.mkdir(parents=True)
    receipt = {"status": "running", "source_revision": REVISION,
               "working_directory": str(work), "model_loading": False,
               "inference": args.models, "training": False,
               "teacher_acceptance": "not_recorded"}
    print(f"Evidence and new audit output: {work}", flush=True)
    try:
        for directory in ("src", "configs"):
            copy_sources(ROOT / directory, work / directory)
        shutil.copyfile(ROOT / "reproduce.py", work / "reproduce.py")
        if args.models:
            copy_sources(ROOT / 'scripts', work / 'scripts')
            (work/'artifacts').mkdir()
            for name in ('lora-lock.json','evaluation-freeze.json','reproduction-downloads.json'):
                shutil.copyfile(ROOT/'artifacts'/name,work/'artifacts'/name)
        (work/'submission-source.json').write_text(json.dumps(dict(source_revision=REVISION),indent=2),encoding='utf-8')
        if args.models or args.preflight:
            from src.model_preflight import inspect_environment
            specification = json.loads(args.paths.read_text(encoding="utf-8-sig"))
            check = inspect_environment(ROOT, specification)
            (work / "preflight.json").write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            receipt["preflight"] = check
            if check["status"] != "passed":
                raise RuntimeError("Model environment preflight failed: " + "; ".join(check["errors"]) + "; see preflight.json")
            if args.preflight:
                receipt.update(status="passed", mode="preflight_only", inference=False)
                (work / "submission-status.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print("Model environment preflight passed; no model loaded and no inference performed.", flush=True)
                return 0
        config = json.loads((work / "configs/reproduction.json").read_text(encoding="utf-8"))
        items = list(config["input_sha256"].items())
        receipt["downloaded_or_verified"] = []
        for relative, expected in items:
            receipt["downloaded_or_verified"].append(fetch(
                work, relative, expected, cache=args.input_cache, prefix=args.download_prefix))
        command = [sys.executable, "-S", str(work / "reproduce.py")]
        receipt["command"] = command
        with (work / "stdout.log").open("w", encoding="utf-8") as out:
            with (work / "stderr.log").open("w", encoding="utf-8") as err:
                result = subprocess.run(command, cwd=work, stdout=out, stderr=err)
        if result.returncode:
            raise RuntimeError(f"Offline audit failed with exit code {result.returncode}; see stderr.log")
        if args.models:
            model_command=[sys.executable,'-m','src.model_reproduction','--specification',str(args.paths.resolve())]
            if args.download_prefix:
                model_command.extend(['--download-prefix', args.download_prefix])
            receipt['model_command']=model_command
            print('Saved results verified; starting 11 real-model test runs...',flush=True)
            with (work/'model-stdout.log').open('w',encoding='utf-8') as out:
                with (work/'model-stderr.log').open('w',encoding='utf-8') as err:
                    result=subprocess.run(model_command,cwd=work,stdout=out,stderr=err)
            model_status_path = work / 'results/model-reproduction/status.json'
            model_status = json.loads(model_status_path.read_text(encoding='utf-8')) if model_status_path.exists() else {}
            receipt['model_result'] = model_status
            receipt['model_loading'] = model_status.get('model_loading', False)
            if model_status.get('status') == 'completed_with_differences':
                receipt.update(status='completed_with_differences', downloaded_input_count=len(items))
                (work / 'submission-status.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                print('All model runs completed with metric differences; see results/model-reproduction/metrics.json', flush=True)
                return 1
            if result.returncode or model_status.get('status') != 'passed':
                raise RuntimeError('Real-model reproduction failed: ' + model_status.get('error', 'see model-stderr.log'))
        receipt.update(status="passed", downloaded_input_count=len(items))
        print("Real-model reproduction passed." if args.models else "Saved-prediction audit passed. This is not new model inference.", flush=True)
        code = 0
    except (Exception, KeyboardInterrupt) as error:
        receipt.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        print(f"Failed: {error}. Evidence retained: {work}", file=sys.stderr)
        code = 1
    (work / "submission-status.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
