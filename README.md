# 中文评论情感分析：统一工程入口

## 单一仓库与发布方式（2026-10-28更新）

唯一正式工作目录为NLP仓库根目录，已保留原发布工作副本的提交历史。此前`.publication/data-processing-v1`仅为迁移前来源，不再用于开发或后续提交。

根据本人本次明确要求，项目源码、配置、已有数据、预测、错误清单和运行日志统一公开，不再维护公开包/私有包两套项目。账户凭据、可重建缓存、迁移回滚副本不上传。模型基座及工具目录中的大权重仍通过资源清单定位，不因本次目录统一自动搬入Git。

仓库中的本机路径配置用于追溯，新机须填写自己的资源位置。历史文档中“不含正文”“仅本地”“未获发布授权”等描述保留为当时状态，本节为当前发布口径。数据许可证据的历史限制继续披露，不能因选择公开就宣称已完成许可核验。详见[单目录统一记录](docs/单目录统一与发布记录.md)。

文档入口：[项目最新进度表](docs/项目最新进度表.md) · [分类文档导航](docs/README.md)。当前工程结果已生成，正式对比解释、人工错例归因、资源外链、新机真实复现及最终交付仍需完成；自动生成图表不等于分析完成。

本项目接入TF-IDF、预训练MacBERT全参微调、Qwen零样本、Qwen少样本和普通生成式SFT LoRA。融合以既有模型与证据为基础，不重新划分数据或重复训练。当前最终状态见 `artifacts/final-acceptance.json`，各路线成绩由预测独立重算，不能把历史验证成绩当测试成绩。

## 配置与数据

复制 `configs/paths.example.json` 为 `configs/paths.local.json`，按本人资源位置填写。Linux运行时使用Linux路径；缺失资源直接报错。本机配置不得包含账户密钥。资源校验和见 `data/manifests/`、`artifacts/` 和 `results/imported/`。外部工具工作区中的大权重仍需单独获取，其外链尚未上传核验；仓库内已有数据、日志及预测按本次要求一并公开。

冻结数据：train6210、validation776、test777，划分种子67，0=负面、1=正面。BERT/TF-IDF按验证Macro-F1选配置与最佳轮次；训练种子42/43/44全部保留。普通LoRA沿用T01，绝不替换为分类头LoRA或QLoRA。提示包与LoRA原提示格式保留，按完整方案比较。

## 已实现命令

在本项目根目录、使用已配置的Python环境运行：

```text
python -m unittest discover -s tests -v
python -m src.cli check
python -m src.cli import
python -m src.cli evaluate
python -m src.cli summarize
python -m src.cli errors
python -m src.cli plot
```

`import`核验已有基线和提示/LoRA验证产物并生成派生预测，原件保持不变。`evaluate`检查完整ID、顺序、真值及原始输出/解析/状态一致性。Qwen仅strip后精确0/1合法；非法计错，运行失败阻止发布完整主成绩。`errors`生成待人工填写的错例表及共同失败、修正、退化记录；配对检验为精确McNemar及Holm校正，不显著不等于等效。

已有模型真实推理（每次使用新输出目录）：

```text
python -m src.cli predict --method qwen_zero_shot --smoke --output results/local/my-smoke
python -m src.cli predict --method qwen_zero_shot --split test --output results/local/my-zero-test
python -m src.cli predict --method qwen_few_shot --split test --output results/local/my-few-test
python -m src.cli predict --method qwen_lora --seed 42 --split test --output results/local/my-lora42-test
python -m src.cli benchmark --method bert --seed 42 --output results/local/my-bert42-benchmark
python -m src.cli benchmark --method tfidf --seed 42 --output results/local/my-tfidf42-benchmark
```

`predict`完成全划分预测，并执行固定20条预热、200条串行batch1正式延迟。`--smoke`只处理4条验证样本与微型计时，不产生正式成绩。支持的method为bert、tfidf、qwen_zero_shot、qwen_few_shot、qwen_lora；三个训练种子须分别执行。基线主成绩沿用已核验test-isolated产物，额外benchmark用于推理资源补测。

已验收的基线重新训练入口（不是默认融合步骤）：

```text
python -m src.cli train --method bert --smoke --output /your/workspace/new-bert-smoke
python -m src.cli train --method tfidf --output /your/workspace/new-tfidf-search
```

LoRA重新训练仍用已归档的工具端编排，见src/legacy/lora及接收协议；不承诺完整续训CLI。训练状态重载检查只验收epoch边界状态和重载预测。历史入口归档于src/legacy，统一入口为src/cli.py。

## 结果与证据

- `results/run_index.jsonl`：各路线正式验证/测试运行及来源哈希。
- `results/metrics/`、`results/summaries/`：重算指标、种子汇总、验证对照及训练成本。
- `results/benchmarks/`：计时协议及推理资源摘要。
- `reports/audits/`、`reports/figures/`：审计、曲线、混淆矩阵及比较图。
- `results/predictions/`、`results/errors/`、`results/local/`：逐样本预测、错例和派生产物，本次统一公开。
- `results/imported/`：成员交付和源码接收索引；接收日期不冒充原开发日期。

训练loss按路线分图，不比较不同loss原始数值。历史设备结果分组；CPU与GPU执行位置均标明。基线曾使用测试集，本次不以测试分数改参数、换种子或重新划分。

本机真实模型推理与新机独立复现分别验收。人工错例归因、课程报告、演示、外链和新机检查的实际缺项见 `docs/integration/复现与限制.md`。旧README保留在 `docs/history/README-before-integration.md`；旧设计文档中的随机初始化BERT等历史设想不覆盖本轮冻结协议。
