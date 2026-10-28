# 正式对比交付索引

复跑命令（仅离线，目录必须未存在）：

```powershell
python scripts/build_comparison_report.py --config configs/analysis/comparison-formal-v1.json --run-id <全新批次ID>
```

配置固定bootstrap种子67、5000次、batch250、linear分位。使用现有依赖，无模型目录依赖。标注重放见results/error-review/review-formal-20261028-01/README.md；无需再次调用AI。

- 对比分析观察稿.md：主表、参数、区间、成本和局限（解释待本人确认）。
- 真实错误案例与局限.md / casebook.json：全文、前缀、所有模型与种子输出。
- 待复核清单.md：集中124条，另存人工裁定，不修改正式真值。
- 分类别指标表.md、参数总表.md、种子敏感性.md：明细。
- 图表总览.html：PNG/PDF与数据来源；10组正文候选，其余附录。
- 结论证据台账.json、AI使用记录.md、验收清单.md。
- results/analysis/repro-bdb6aed978b3/status.json：环境、命令、输入输出SHA256及Git状态。

图表审查清单.md和软件检查记录用于工程验收；不替代学生结果解释。
