# 案例局限与建议修订

本版替代formal-03案例章节中重复的解释，不替代其数值、图表或标注。9个案例逐一包含观察、假设、局限、建议、验收和成本风险。

案例解释已获本人逐例确认，正文5组、局限或附录2例安排已确认。优先阅读正文案例与局限定稿.md。确认范围不包含改进实测或整份报告验收。

- 阅读：真实错误案例与局限.md。
- 结构化内容：casebook.json。
- 输入输出及事实不变校验：status.json。

原formal-03的casebook、文档和交付哈希保留不变；源码本次有意修订，历史代码以提交c1bab6f及已有source-snapshot为准。

正式生成器也已读取同一份版本化解释文件，后续新批次不再恢复统一模板。

复跑仅此修订（全新目录，不加载模型）：

```text
python scripts/revise_casebook.py --source results/analysis/comparison-formal-20261028-03/casebook.json --notes configs/analysis/case-limitations-20261029-v2.json --output reports/analysis/<new-case-revision-id>
```

AI使用：Codex读取既有证据、提出逐例解释、转录本人对解释和分组的确认。未新增20条以外的人工情感标注，未运行改进实验；具体AI模型版本未知。
