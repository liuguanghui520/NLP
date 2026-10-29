"""Download hash-locked evidence outside the hand-in ZIP, then audit it.

Python 3.11+ standard library only. No model is loaded and no training is run.
The ZIP remains dataset-free. Each execution gets a fresh working directory.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
REVISION = "106cdbed8db84057dba6b9c980d3a2f914856dd0"
BASE_URL = f"https://raw.githubusercontent.com/liuguanghui520/NLP/{REVISION}/"


def safe_path(root, relative):
    name = Path(relative)
    if name.is_absolute() or name.drive or name.anchor or ".." in name.parts:
        raise ValueError(f"Path outside working directory: {relative}")
    return root.resolve().joinpath(*name.parts)


def fetch(root, relative, expected):
    """Never accept an HTTP success response without a matching SHA256."""
    target = safe_path(root, relative)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        raw_url = BASE_URL + quote(relative, safe="/")
        transports = (
            ("ghfast_raw_mirror", "https://ghfast.top/" + raw_url),
            ("github_raw", raw_url),
        )
        curl = "curl.exe" if sys.platform == "win32" else "curl"
        failures = []
        for transport, url in transports:
            print(f"download_transport={transport} path={relative}", file=sys.stderr, flush=True)
            result = subprocess.run(
                [
                    curl, "-fsSL", "--retry", "2", "--retry-all-errors",
                    "--connect-timeout", "20", "--max-time", "120",
                    url,
                ],
                capture_output=True,
            )
            if result.returncode:
                failures.append(
                    f"{transport} exit {result.returncode}: "
                    f"{result.stderr.decode('utf-8', errors='replace').strip()}"
                )
                continue
            content = result.stdout
            if hashlib.sha256(content).hexdigest() == expected:
                break
            failures.append(f"{transport} returned a SHA256 mismatch")
        else:
            raise RuntimeError(
                f"Download transport failed for {relative}: {'; '.join(failures)}"
            )
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Downloaded evidence hash mismatch: {relative}")
        target.write_bytes(content)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Packaged file conflicts with frozen evidence: {relative}")
    return relative


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", help="New, empty directory outside the extracted submission")
    parser.add_argument("--models", action="store_true", help="Run all 11 real-model test predictions after the saved-result audit")
    parser.add_argument("--paths", type=Path, help="Your Qwen, LLaMA-Factory and optional extracted-asset paths for --models")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11+ is required")
    if args.models and (args.paths is None or not args.paths.is_file()):
        parser.error("--models requires an existing --paths JSON; see configs/reproduce-models.example.json")
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
            shutil.copytree(ROOT / directory, work / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "paths.local.json"))
        shutil.copy2(ROOT / "reproduce.py", work / "reproduce.py")
        if args.models:
            shutil.copytree(ROOT / 'scripts', work / 'scripts', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            (work/'artifacts').mkdir()
            for name in ('lora-lock.json','evaluation-freeze.json','reproduction-downloads.json'):
                shutil.copy2(ROOT/'artifacts'/name,work/'artifacts'/name)
        (work/'submission-source.json').write_text(json.dumps(dict(source_revision=REVISION),indent=2),encoding='utf-8')
        config = json.loads((work / "configs/reproduction.json").read_text(encoding="utf-8"))
        items = list(config["input_sha256"].items())
        with ThreadPoolExecutor(max_workers=1) as pool:
            receipt["downloaded_or_verified"] = list(pool.map(lambda item: fetch(work, *item), items))
        command = [sys.executable, "-S", str(work / "reproduce.py")]
        receipt["command"] = command
        with (work / "stdout.log").open("w", encoding="utf-8") as out:
            with (work / "stderr.log").open("w", encoding="utf-8") as err:
                result = subprocess.run(command, cwd=work, stdout=out, stderr=err)
        if result.returncode:
            raise RuntimeError(f"Offline audit failed with exit code {result.returncode}; see stderr.log")
        if args.models:
            model_command=[sys.executable,'-m','src.model_reproduction','--specification',str(args.paths.resolve())]
            receipt['model_command']=model_command
            print('Saved results verified; starting 11 real-model test runs...',flush=True)
            with (work/'model-stdout.log').open('w',encoding='utf-8') as out:
                with (work/'model-stderr.log').open('w',encoding='utf-8') as err:
                    result=subprocess.run(model_command,cwd=work,stdout=out,stderr=err)
            if result.returncode:
                raise RuntimeError('Real-model reproduction failed or differs from reference; see model logs and results/model-reproduction/status.json')
        receipt.update(status="passed", downloaded_input_count=len(items))
        print("Real-model reproduction passed." if args.models else "Saved-prediction audit passed. This is not new model inference.", flush=True)
        code = 0
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        print(f"Failed: {error}. Evidence retained: {work}", file=sys.stderr)
        code = 1
    (work / "submission-status.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
