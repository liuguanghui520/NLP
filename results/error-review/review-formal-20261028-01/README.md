# 正式错例辅助标注

20条来源于交接第5节本人确认，382条为本次AI逐条辅助判断；二者不是双人独立标注。T1—T6是描述性情境，不是模型内部原因。

全部402条均有判断或明确待复核，原始777条标签和11份预测未改。T5为格式事实；分歧与不确定语义归T6。全文证据超出共同前缀的运行另列可见性复核，token可见文本未知时不猜测。

六类材料：pilot-human-confirmed.jsonl、ai-remaining382.jsonl、samples-all402.jsonl、run-error-attributions.jsonl、needs-review.jsonl、annotation-audit.json。

复跑（必须新目录）：`python scripts/build_formal_annotations.py --output results/error-review/<new-id>`。决策TSV来自会话逐条阅读，复跑是校验和转录，不会重新发起AI标注。
