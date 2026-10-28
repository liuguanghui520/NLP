# LoRA学习率策略确认清单

日期：2026-10-26。以下六套由用户明确给出，AI仅转录、纠正框架字段拼写并核查，不增加研究问题或候选值；尚未启动训练。

| ID | 峰值learning_rate | warmup_ratio | lr_scheduler_type | 相对基准改变 |
|---|---:|---:|---|---|
| LR00 | 0.0001 | 0.1 | linear | 基准 |
| LR01 | 0.0003 | 0.1 | linear | 峰值学习率 |
| LR02 | 0.00005 | 0.1 | linear | 峰值学习率 |
| LR03 | 0.0001 | 0.2 | linear | 预热比例 |
| LR04 | 0.0001 | 0.05 | linear | 预热比例 |
| LR05 | 0.0001 | 0.1 | cosine | 调度方式 |

所有组固定rank8、alpha16、q_proj/v_proj、dropout0.05。其余原训练配置保持一致：Qwen3-1.7B、普通LoRA、BF16、3轮、seed42、微批量1/累积16、AdamW、weight_decay0、冻结train6210/validation776、同一模板与分类评估口径。各组从同一原始基座独立训练，与LR00比较，不从上一组适配器继续训练。

用户写的liner按框架枚举linear记录；warmup-ratio按warmup_ratio记录。六套均属于学习率策略因素组，但本次实际清单中每个新候选只改变一个具体字段，不混改三项。

LR00对应已完成基准qwen17-full-20261025-231706-ea8173；总共6种唯一配置，其中5个新候选。执行前核对基准与新运行的模型/数据/指标口径和版本，避免把已完成基准重复计数。首轮种子复验、rank/模块/dropout其他候选和最终总预算仍单独确定。

机器可读清单：configs/lora_learning_rate_policies.json。原基准配置：configs/lora_reference.json。配置选取与研究问题由学生本人负责；本文件不将技术转录作为学生独立研究理由。
