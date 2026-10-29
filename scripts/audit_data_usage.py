"""Read-only data/contact and annotation-scope audit; never load a model.

The output contains masked excerpts and sample IDs, never matched contact values.
This is a candidate scan, not a complete personal-information certification.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import subprocess


RULES = {
    "mobile": r"(?<![0-9])(?:\+?86[- ]?)?1[3-9][0-9]{9}(?![0-9])",
    "landline": r"(?<!\d)0\d{2,3}[-－ ]?\d{7,8}(?!\d)",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "identity18_candidate": r"(?<![0-9A-Za-z])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?![0-9A-Za-z])",
    "explicit_account": r"(?:微信|weixin|wechat|QQ|身份证|护照)[号碼码]?\s*[:：=]\s*[A-Za-z0-9_-]{5,}",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scan_data(root):
    patterns = {key: re.compile(value, re.I) for key, value in RULES.items()}
    inputs, summaries, findings = {}, {}, []
    for split in ("train", "validation", "test"):
        path = root / "data/processed/v1-20261023" / (split + ".jsonl")
        inputs[path.relative_to(root).as_posix()] = digest(path)
        rows = read_rows(path)
        counts, affected = Counter(), set()
        for row in rows:
            text = row["raw_text"]
            spans = [(key, match.start(), match.end()) for key, pattern in patterns.items() for match in pattern.finditer(text)]
            # Store no raw context, which could also contain linked names or addresses.
            for key, start, end in spans:
                counts[key] += 1
                affected.add(row["sample_id"])
                findings.append({
                    "sample_id": row["sample_id"], "source_row": row["source_row"],
                    "split": split, "rule": key, "raw_start": start, "raw_end": end,
                    "masked_match": "[联系方式候选已隐藏]",
                    "matched_span_in_model_visible_prefix": text[start:end] in row["text"],
                    "review_status": "candidate_only",
                })
        summaries[split] = {"scanned_comments": len(rows), "candidate_comments": len(affected), "candidate_matches": sum(counts.values()), "matches_by_rule": dict(counts)}
    return {"rules": RULES, "scope": "All 7763 retained comments: raw_text; model-visible membership recorded. Excluded source rows, Git history and all derived copies not scanned by this step.", "limitations": ["Pattern detection cannot certify absence of names, addresses, obfuscated identifiers or re-identification.", "Numbers are candidates; context review distinguishes contact numbers, public business numbers and numeric-entity artifacts.", "No evaluated text, labels, split IDs or predictions modified.", "No original contact value, linked name, address or raw context is saved in findings."], "input_sha256": inputs, "splits": summaries, "findings": findings}


def audit_annotations(root):
    directory = root / "results/error-review/review-formal-20261028-01"
    rows = read_rows(directory / "samples-all402.jsonl")
    pending = read_rows(directory / "needs-review.jsonl")
    approval = read_json(root / "reports/analysis/case-confirmed-20261029-01/status.json")
    notes = read_json(root / "configs/analysis/case-limitations-20261029-v2.json")
    assert len(rows) == len({row["sample_id"] for row in rows}) == 402
    origins = Counter(row["annotation_origin"] for row in rows)
    assert origins == {"human_confirmed_in_design_chat": 20, "ai_assisted_under_confirmed_rules": 382}
    assert len({row["sample_id"] for row in pending}) == 124
    assert approval["student_interpretation_approved"] and approval["cases"] == len(notes["cases"]) == 9
    ids = {row["sample_id"] for row in rows}
    for sample_id, case in notes["cases"].items():
        assert sample_id in ids and case["approval"]["approved"]
    return {"error_union_comments": len(rows), "origins": dict(origins), "pending_unique_comments": 124, "pending_set_not_changed": True, "confirmed_cases": list(notes["cases"]), "report_selection": approval["report_selection"], "proposed_report_scope": "Nine previously student-confirmed examples; five descriptive body groups, two appendix/limitation examples. AI-assisted category distributions remain exploratory, not human-adjudicated prevalence.", "source_labels_unchanged": True, "new_manual_approval_claimed": False, "input_sha256": {p.relative_to(root).as_posix(): digest(p) for p in [directory / "samples-all402.jsonl", directory / "needs-review.jsonl", root / "reports/analysis/case-confirmed-20261029-01/status.json", root / "configs/analysis/case-limitations-20261029-v2.json"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    contacts = scan_data(root)
    annotations = audit_annotations(root)
    write_json(output / "contact-candidates.json", contacts)
    write_json(output / "annotation-scope.json", annotations)
    assert all(digest(root / name) == value for name, value in contacts["input_sha256"].items())
    write_json(output / "status.json", {"status": "audit_completed_findings_require_context", "started_at": started, "completed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "script_sha256": digest(Path(__file__)), "input_data_unchanged": True, "training": False, "model_loading": False, "inference": False, "outputs": {p.name: digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"output": str(output), "splits": contacts["splits"], "confirmed_cases": 9, "pending": 124}, ensure_ascii=False))


if __name__ == "__main__":
    main()
