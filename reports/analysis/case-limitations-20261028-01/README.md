# 案例局限与建议修订

本版替代formal-03案例章节中重复的解释，不替代其数值、图表或标注。9个案例逐一包含观察、假设、局限、建议、验收和成本风险；新增解释待本人确认。

- 阅读：真实错误案例与局限.md。
- 结构化内容：casebook.json。
- 输入输出及事实不变校验：status.json。

原formal-03的casebook、文档和交付哈希保留不变；源码本次有意修订，历史代码以提交c1bab6f及已有source-snapshot为准。

正式生成器也已读取同一份版本化解释文件，后续新批次不再恢复统一模板。

复跑仅此修订（全新目录，不加载模型）：

```text
python scripts/revise_casebook.py --source results/analysis/comparison-formal-20261028-03/casebook.json --notes configs/analysis/case-limitations-20261028-v1.json --output reports/analysis/<new-case-revision-id>
```

AI使用：Codex读取既有案例证据、逐例提出解释和可验证建议；未运行建议中的诊断或改进实验，未新增人工确认，具体AI模型版本未知。原始判断与建议中的不确定性均保留。
