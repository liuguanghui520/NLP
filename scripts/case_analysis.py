"""Evidence-linked case interpretation and rendering, without model execution."""

from copy import deepcopy


DEFAULT_CASE_NOTES = "configs/analysis/case-limitations-20261029-v2.json"


def enrich_case(case, notes):
    result = deepcopy(case)
    sample = result["sample"]
    sample_id = sample["sample_id"]
    if sample_id not in notes["cases"]:
        raise ValueError(f"Case interpretation has not been reviewed: {sample_id}")
    analysis = deepcopy(notes["cases"][sample_id])
    for quote in analysis["evidence_quotes"]:
        if not quote or quote not in sample["normalized_text"]:
            raise ValueError(f"Case evidence is absent from the source text: {sample_id}")
    for field in ["observations", "hypotheses", "limitations", "recommendations"]:
        if not analysis.get(field):
            raise ValueError(f"Missing case interpretation section: {field}")
    for recommendation in analysis["recommendations"]:
        if not all(recommendation.get(k) for k in ["action", "validation", "success_criterion", "cost_and_risk"]):
            raise ValueError("A recommendation needs an action, validation, success criterion and cost/risk")
    runs = result["all_runs"]
    if len({p["run_id"] for p in runs}) != len(runs):
        raise ValueError("Duplicate case run identity")
    if any(p["sample_id"] != sample_id or p["true_label"] != sample["source_label"] for p in runs):
        raise ValueError("Case run truth/identity mismatch")
    actual_errors = {p["run_id"] for p in runs if p["status"] != "ok" or p["parsed_label"] != sample["source_label"]}
    recorded_errors = [p["run_id"] for p in result["error_attributions"]]
    if len(recorded_errors) != len(set(recorded_errors)) or set(recorded_errors) != actual_errors:
        raise ValueError("Case attribution set differs from official errors")
    analysis.update(
        interpretation_origin="AI_assisted_case_interpretation",
        student_interpretation_approved=bool(analysis.get("approval", {}).get("approved", False)),
        interpretation_version=notes["version"],
        verified_facts={
            "run_count": len(runs),
            "official_wrong_runs": sorted(actual_errors),
            "normalized_char_count": len(sample["normalized_text"]),
            "char_truncated": sample["char_truncated"],
            "evidence_in_common_prefix": [q in sample["visible_prefix"] for q in analysis["evidence_quotes"]],
        },
        recommendations_executed=False,
        test_feedback_used=True,
    )
    result["analysis"] = analysis
    # Keep the legacy summary field for consumers while making its contents case-specific.
    result["improvement"] = "；".join(r["action"] for r in analysis["recommendations"])
    return result


def render_casebook(cases):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "# 真实错误案例：逐例分析、局限与改进建议",
        "",
        "本版保留已确认的9个案例、原标签、本人判断、预测与T1—T6归类，仅修订解释。"
        "案例来自定向错例试标，不代表总体错误分布；解释由AI整理并按逐例审核记录注明确认状态。",
        "",
        "## 阅读口径",
        "",
        "观察事实来自冻结预测；可能解释不等于已验证机制。下述建议均未执行，不代表已取得增益。"
        "本轮没有训练、加载模型或新增推理，未改测试真值或评分政策。",
        "",
        "后续若验证建议，先在训练/验证材料制定方案；需要真实模型运行时另行确认。"
        "这些建议已经使用测试反馈，不能把原777条测试集再称作独立的改进验证集。"
        "原测试上的探索性复验须单独编号；独立增益需在事先冻结、未参与建议形成的数据上验证。",
        "",
    ]
    for i, case in enumerate(cases, 1):
        sample, analysis = case["sample"], case["analysis"]
        lines += [f"## 案例{i}：{analysis['title']}", "", f"完整ID：`{sample['sample_id']}`。",
            "", f"源标签={sample['source_label']}；本人判断={sample['assessed_label']}；"
            f"原标注来源={sample['annotation_origin']}；复核状态={sample['review_status']}。",
            "", "案例解释确认：" + ("本人已确认（仅限本例解释与建议，未验证改进效果）" if analysis["student_interpretation_approved"] else "待本人确认"),
            "", "**规范化全文**", "", sample["normalized_text"], "", "**共同可见前缀**", "", sample["visible_prefix"], "",
            "|方法／种子|原始输出|解析／状态|原标签下的归类|", "|---|---|---|---|"]
        categories = {e["run_id"]: e["error_type_v1"] for e in case["error_attributions"]}
        for run in case["all_runs"]:
            values = [f"{run['method']}/{run['seed']}", run["raw_output"],
                f"{run['parsed_label']}/{run['status']}", categories.get(run["run_id"], "正确")]
            lines.append("|" + "|".join(cell(v) for v in values) + "|")
        lines += ["", "**直接证据**", "", "；".join(f"“{q}”" for q in analysis["evidence_quotes"]) + "。",
            "", "**观察到的失败**", "", *[f"- {x}" for x in analysis["observations"]],
            "", "**可能解释，尚未验证**", "", *[f"- {x}" for x in analysis["hypotheses"]],
            "", "**本例的局限与不能推出的结论**", "", *[f"- {x}" for x in analysis["limitations"]],
            "", "**有针对性的改进与验收方式**", ""]
        for j, recommendation in enumerate(analysis["recommendations"], 1):
            lines += [f"{j}. **建议：**{recommendation['action']}", "",
                f"   **如何验证：**{recommendation['validation']}", "",
                f"   **验收关注：**{recommendation['success_criterion']}", "",
                f"   **成本与副作用：**{recommendation['cost_and_risk']}", ""]
        lines += ["**目前可以写入报告的结论**", "", analysis["reportable_conclusion"], ""]
    lines += ["## 跨案例优先级与共同局限", "",
        "1. **先厘清任务和标签。** 对标签分歧案例647先复核评价标准；对6197区分住客与酒店，对6865/6957区分评价对象。复核不会自动修改冻结测试标签。", "",
        "2. **分开修复格式与语义。** 5773支持优先设计输出协议对照；5546的合法反向标签则需要语义与指令诊断。格式约束不能直接解释或保证修复合法标签误判。", "",
        "3. **再处理混合评价与训练波动。** 6353/4086需要对称的正负混合评价检查；7015需要保留种子分歧。不能以多次相同TF-IDF预测作为独立重复，也不能按测试选一个更好的种子。", "",
        "4. **论文结论以现有证据为限。** 这些案例支持描述性情境分析，不支持内部机制因果、跨领域泛化或改进已生效的结论。402条中382条为AI辅助标注，T6和标签分歧保留；9个案例的选择也不是随机总体抽样。", "",
        "5. **所有改进有成本和风险。** 对象/说话者标记、提示、特征、集成等不能同时改变后归因单一因素。新增训练、推理与人工复核分别记录成本；没有测量就不填收益、耗时或金额。", ""]
    return "\n".join(lines)


def render_report_selection(cases, notes):
    by_number = {i: c for i, c in enumerate(cases, 1)}
    lines = ["# 正文案例与局限定稿", "",
        "本人于2026-10-29在设计对话中逐例确认解释，并确认正文5组、局限或附录2例的安排。"
        "本文件是报告素材定稿，不表示改进建议已经实测、124条待复核已裁定或整份课程报告已验收。", ""]
    for section in notes.get("report_selection", []):
        lines += ["## " + section["placement"] + "：" + section["title"], "",
            "原案例编号：" + "、".join(map(str, section["case_numbers"])) + "。", ""]
        for number in section["case_numbers"]:
            c = by_number[number]
            lines += [f"**案例{number}，ID：`{c['sample']['sample_id']}`**", "",
                "关键片段：" + "；".join("“" + q + "”" for q in c["analysis"]["evidence_quotes"]) + "。", "",
                *["- " + observation for observation in c["analysis"]["observations"]], ""]
        lines += ["**已确认的解释与局限：**" + section["approved_summary"], "",
            "**建议：**" + section["approved_recommendation"], ""]
    lines += ["## 使用边界", "",
        "所有建议使用了测试反馈，仅作为未来改进方向。真实模型推理或训练补测仍需单独确认；"
        "检验改进时应同时报告收益、退化、非法率和成本，不覆盖原冻结结果。", "",
        "案例7的解释获确认不等于源标签已改；原标签正面、本人初判负面的分歧继续保留。"
        "20条本人确认与382条AI辅助标注的来源保持不变。", ""]
    return "\n".join(lines)
