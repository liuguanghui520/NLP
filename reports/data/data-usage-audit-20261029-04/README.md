# 错例范围与数据使用核查证据

日期：2026-10-29。当前最终核查批次，正文见[收尾说明](../../../docs/报告素材/错例范围与数据使用核查-20261029.md)。

- `annotation-scope.json`：402条来源、124条待复核及九例原确认记录的只读核查。
- `contact-candidates.json`：7763条扫描、9条命中、11处候选；不含原号码、姓名或上下文正文。
- `contact-context-review.json`：9条命中的AI上下文核对，以及本人选择不额外脱敏的记录。
- `modelscope-dataset-card.md`、`upstream-origin-summary.md`、`upstream-evidence.json`：许可标记、来源和CSV哈希；保留未完成的追加网络查证。
- `swanlab-audit.json`：三个正式LoRA本地备份，字段、计数及哈希；不包含原日志、账户凭据或评论正文。
- `status.json`：本地扫描执行状态及输出哈希；`closeout-manifest.json`包含后续补充证据的完整清单。

本批次前01—03为脚本调整过程中的中间核查，不作最终报告证据；例如早期非空密钥字段判断未区分占位符。当前已按实际YAML结构区分空值、占位符与其他非空值。工具证据来自只读WSL访问，不重新训练或预测，也没有改写历史标签及预测。
