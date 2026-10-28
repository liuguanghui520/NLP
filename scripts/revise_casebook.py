"""Create a separate evidence-preserving case-analysis revision; no metrics/model run."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.case_analysis import DEFAULT_CASE_NOTES, enrich_case, render_casebook, render_report_selection


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/analysis/comparison-formal-20261028-03/casebook.json")
    parser.add_argument("--notes", default=DEFAULT_CASE_NOTES)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, notes_path = (ROOT / args.source).resolve(), (ROOT / args.notes).resolve()
    output = (ROOT / args.output).resolve()
    for path in [source, notes_path, output]:
        path.relative_to(ROOT)
    if output.exists():
        raise FileExistsError("Use a new revision directory; never overwrite archived delivery")
    paths = [source, notes_path, Path(__file__), ROOT / "scripts/case_analysis.py", ROOT / "scripts/formal_comparison.py"]
    inputs = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    old_cases = json.loads(source.read_text(encoding="utf-8"))
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    cases = [enrich_case(case, notes) for case in old_cases]
    if len({c["improvement"] for c in cases}) != len(cases):
        raise ValueError("Repeated case recommendation summaries")
    for old, new in zip(old_cases, cases):
        for key in ["sample", "all_runs", "error_attributions", "selection"]:
            if old[key] != new[key]:
                raise ValueError(f"Frozen case evidence changed: {key}")
    output.mkdir(parents=True)
    (output / "casebook.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "真实错误案例与局限.md").write_text(render_casebook(cases), encoding="utf-8")
    approved = all(c['analysis']['student_interpretation_approved'] for c in cases)
    if notes.get('report_selection'):
        (output / "正文案例与局限定稿.md").write_text(render_report_selection(cases, notes), encoding="utf-8")
    command = "python scripts/revise_casebook.py --source " + args.source + " --notes " + args.notes + " --output reports/analysis/<new-case-revision-id>"
    (output / "README.md").write_text(
        "# 案例局限与建议修订\n\n"
        "本版替代formal-03案例章节中重复的解释，不替代其数值、图表或标注。9个案例逐一包含观察、假设、局限、建议、验收和成本风险。\n\n"
        + ("案例解释已获本人逐例确认，正文5组、局限或附录2例安排已确认。优先阅读正文案例与局限定稿.md。确认范围不包含改进实测或整份报告验收。\n\n" if approved else "案例解释尚待本人确认。\n\n") +
        "- 阅读：真实错误案例与局限.md。\n- 结构化内容：casebook.json。\n- 输入输出及事实不变校验：status.json。\n\n"
        "原formal-03的casebook、文档和交付哈希保留不变；源码本次有意修订，历史代码以提交c1bab6f及已有source-snapshot为准。\n\n"
        "正式生成器也已读取同一份版本化解释文件，后续新批次不再恢复统一模板。\n\n复跑仅此修订（全新目录，不加载模型）：\n\n```text\n" + command + "\n```\n\n"
        "AI使用：Codex读取既有证据、提出逐例解释、转录本人对解释和分组的确认。未新增20条以外的人工情感标注，未运行改进实验；具体AI模型版本未知。\n",
        encoding="utf-8")
    outputs = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    assert all(digest(ROOT / rel) == sha for rel, sha in inputs.items())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    status = dict(status="completed", revision_id=output.name,
        created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), command=sys.argv,
        git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        input_sha256=inputs, output_sha256=outputs, cases=len(cases),
        case_interpretations_unique=len({c["improvement"] for c in cases}),
        frozen_sample_prediction_attribution_unchanged=True,
        training=False, model_loading=False, inference=False,
        test_feedback_used=True, recommendations_executed=False, student_interpretation_approved=approved,
        approval_scope="nine case interpretations and report selection only; not full report or causal validation",
        approval_record=notes.get('approval_record'), report_selection=notes.get('report_selection'))
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(output), cases=len(cases), frozen_evidence_unchanged=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
