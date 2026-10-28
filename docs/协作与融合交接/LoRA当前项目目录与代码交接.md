# LoRA当前项目目录与代码交接

整理日期：2026-10-28（北京时间）。

本文依据本轮对项目代码、配置、结果索引与说明文档的只读查看整理，描述当前结构，不代表后续统一目录设计已落地。本次仅保存文档，没有运行训练、评估、迁移文件或执行Git操作。工具目录中的全部实际文件本轮未逐一复验。

## 1. 整体组织

当前LoRA内容分为两部分：

- **项目仓库：**保存训练编排代码、配置定义、数据、结果索引、汇总报告及说明文档。
- **个人LLaMA-Factory工作区：**保存实际训练配置、入口源码副本、检查点/适配器、训练日志、逐轮验证预测、SwanLab记录及资源采样。

GitHub协作不要求成员使用同一安装地址。本文以`<project_root>`表示个人克隆后的项目位置，以`<个人LLaMA-Factory目录>`表示个人工具地址，不登记个人绝对路径。

## 2. 项目仓库中的当前目录

以下目录树仅列出LoRA及其直接依赖的主要内容，不是整个仓库清单。

```text
<project_root>/
├── scripts/
│   ├── profile_qwen17.py
│   ├── run_qwen17_comparisons.py
│   ├── summarize_qwen17_comparisons.py
│   ├── prepare_qwen17_seed_repeats.py
│   ├── audit_qwen17_seed_repeats.py
│   ├── freeze_qwen17_lora.py
│   ├── record_run_resources.py
│   ├── evaluate_qwen06.py
│   ├── prepare_qwen06_experience.py
│   ├── tune_qwen06.py
│   └── test_evaluate_qwen06.py
├── configs/
│   ├── lora_reference.json
│   ├── lora_learning_rate_policies.json
│   └── lora_comparison_10.json
├── data/processed/v1-20261023/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   ├── manifest.json
│   ├── split_ids.json
│   ├── smoke_ids.json
│   ├── fewshot_ids.json
│   ├── latency_ids.json
│   └── alpaca/
├── results/
│   ├── qwen17-profile/
│   ├── qwen17-comparisons/
│   ├── qwen17-freeze/
│   ├── qwen06-validation/
│   └── qwen06-optuna/
├── reports/
│   ├── qwen17-full-20261025-231706-ea8173核验.md
│   ├── qwen17-comparisons-20261026-012117-9e8e51/
│   └── qwen17-comparisons-20261026-200544-750ecb/
└── 根目录说明文档
```

当前主结构为`scripts/`、`configs/`、`data/`、`results/`、`reports/`和根目录说明文档。此前目录设计中的`docs/`与`paths/`本轮查看时尚未出现在项目根目录，不能把规划结构写为已实现。

## 3. 主要脚本职责与依赖

| 文件 | 当前职责 |
|---|---|
| `scripts/profile_qwen17.py` | Qwen3-1.7B配置检查、一步训练冒烟、完整LoRA训练、每轮分类验证、记录最佳检查点及训练资源 |
| `scripts/run_qwen17_comparisons.py` | 准备/运行参数对照批次，串行调用训练入口，更新各配置状态并收集摘要 |
| `scripts/summarize_qwen17_comparisons.py` | 核验参数对照的逐轮预测完整性，重算指标，汇总配置结果及成本 |
| `scripts/prepare_qwen17_seed_repeats.py` | 准备验证集选优配置T01的种子43、44复跑，保留中断批次关联 |
| `scripts/audit_qwen17_seed_repeats.py` | 核验三种子的配置、预测、检查点与资源，汇总单次成绩、均值和样本标准差 |
| `scripts/freeze_qwen17_lora.py` | 检查模型/适配器、配置和环境校验信息，生成最终LoRA评估协议及本机路径manifest；自身不执行测试推理 |
| `scripts/record_run_resources.py` | 包装子进程，记录墙钟时间、进程树RSS、设备显存/利用率/功率采样及采样能量估计 |
| `scripts/evaluate_qwen06.py` | 原始模型/LoRA验证推理；还提供被1.7B代码复用的任务说明、SHA256、验证数据读取和指标计算函数 |
| `scripts/prepare_qwen06_experience.py` | 准备0.6B、20条训练样本的工程冒烟配置和启动入口，不属于正式1.7B结果 |
| `scripts/tune_qwen06.py` | 0.6B小规模Optuna工程练习，不等于正式1.7B参数搜索 |
| `scripts/test_evaluate_qwen06.py` | 对分类指标和非法输出计分规则进行单元检查 |

### 3.1 当前依赖关系

```text
冻结数据＋LoRA配置
        ↓
profile_qwen17.py ← evaluate_qwen06.py中的共用函数
        ↓
run_qwen17_comparisons.py（批次编排）
        ↓
summarize_qwen17_comparisons.py（对照核验）
        ↓
prepare_qwen17_seed_repeats.py（准备43/44复跑）
        ↓
批次运行入口执行复跑
        ↓
audit_qwen17_seed_repeats.py（三种子核验）
        ↓
freeze_qwen17_lora.py（最终协议与产物映射）
```

`record_run_resources.py`为训练子进程提供资源记录。文件名包含`qwen06`不代表其中全部函数仅服务0.6B；后续代码融合时需明确共用模块依赖，不能直接删除这些文件。

## 4. 配置与数据

### 4.1 配置定义

- `configs/lora_reference.json`：基准配置。
- `configs/lora_learning_rate_policies.json`：学习率相关策略配置。
- `configs/lora_comparison_10.json`：基准B00及9组新增配置定义。

对照配置ID包括：

| ID | 对照内容 |
|---|---|
| B00 | 基准，复用已完成基准训练 |
| L01、L02 | 学习率变化 |
| L03、L04 | warmup比例变化 |
| L05 | 学习率调度器变化 |
| R01 | LoRA秩及对应alpha变化 |
| D01、D02 | dropout变化 |
| T01 | 注意力注入模块扩展为q/k/v/o |

配置定义文件中部分状态描述写于运行前，实际完成状态应结合对应`batch.json`、核验报告和运行证据判断，不能只读配置文件中的旧状态。

### 4.2 数据依赖

使用`data/processed/v1-20261023/`冻结数据。当前冻结协议记录：训练6,210条、验证776条、测试777条。

LoRA读取Alpaca训练数据，每轮分类验证读取通用`validation.jsonl`。数据文件与manifest均记录SHA256。测试文件存在不代表已执行测试评估。

## 5. 关键运行与仓库索引

| 用途 | 运行编号 | 项目内相对位置 |
|---|---|---|
| 1.7B一步冒烟 | `qwen17-smoke-20261025-230917-2065ab` | `results/qwen17-profile/` |
| B00完整基准训练 | `qwen17-full-20261025-231706-ea8173` | `results/qwen17-profile/` |
| 9组新增参数对照 | `qwen17-comparisons-20261026-012117-9e8e51` | `results/qwen17-comparisons/` |
| 中断的种子复跑批次 | `qwen17-comparisons-20261026-143111-673346` | `results/qwen17-comparisons/` |
| 成功的43/44复跑批次 | `qwen17-comparisons-20261026-200544-750ecb` | `results/qwen17-comparisons/` |
| 最终LoRA协议冻结 | `qwen17-lora-freeze-20261026-221453-a2c83b` | `results/qwen17-freeze/` |

基准与参数对照的完成情况见对应核验报告；成功复跑批次的本地`batch.json`记录状态为`completed`。中断批次须保留，不当作成功实验或额外独立种子重复计入最终汇总。

项目`results/`主要保存配置、索引和摘要，不代表完整检查点及所有逐轮预测都已复制进主仓库。

## 6. 个人工具工作区及训练产物

本节只保留当前索引中的工具内相对位置，不写个人绝对安装地址：

```text
<个人LLaMA-Factory目录>/
└── project-workspaces/
    ├── qwen17-profile/
    │   └── <run_id>/
    └── qwen17-comparisons/
        └── <batch_id>/
            └── <配置ID>/
```

配置ID示例：`L01`、`R01`、`T01`、`T01-S43`、`T01-S44`。

| 文件或目录 | 内容 |
|---|---|
| `train.json` | 本次实际训练配置；主项目中可能另存为`training_config.json` |
| `adapter/checkpoint-*` | 实际LoRA检查点及相关保存状态 |
| `validation-step-*/predictions.jsonl` | 每轮逐样本预测，可供后续错例筛选 |
| `validation-step-*/metrics.json` | 对应轮次分类指标 |
| `validation_history.json` | 各轮分类验证记录 |
| `best_checkpoint.json` | 验证集选出的最佳检查点 |
| `parameters.json` | 总参数与可训练参数 |
| `timing.json` | Trainer区间及分类验证计时 |
| `resource-record/console.log` | 合并的stdout/stderr |
| `resource-record/resources.csv` | 资源采样序列 |
| `resource-record/status.json` | 资源记录状态与摘要 |
| `swanlog/` | SwanLab本地记录 |
| 批次级`batch.json` | 各配置状态、工具路径、最佳适配器与校验和 |

具体文件集合以对应运行类型、代码及实际产物为准，冒烟运行不等于完整训练，不要求其具有完整三轮验证记录。

协作者需要在自己的本地配置中登记工具位置。当前代码和部分历史manifest仍含原执行机器绝对路径，尚未全面完成可移植路径改造；不能仅替换本文占位符就认定可在新机运行。

## 7. 最终冻结结果的入口

冻结目录：

```text
results/qwen17-freeze/qwen17-lora-freeze-20261026-221453-a2c83b/
├── protocol.json
└── local_manifest.json
```

- `protocol.json`：共享的训练、推理、数据哈希及评价口径。
- `local_manifest.json`：本机基座与三个适配器的路径、校验和、依赖和源码记录；属于本地信息，不公开提交。

冻结配置为T01：Qwen3-1.7B普通LoRA、rank8、alpha16、q/k/v/o、dropout0.05、学习率1e-4、warmup0.1、linear、3轮；种子42、43、44分别保留，对应最佳检查点均记录为`checkpoint-1167`。

推理规则为关闭思考、无示例、BF16、贪心、最多8个新token，输出strip后严格为0/1才合法。最终三个种子均须评估，不能根据测试分数选择一个种子。

冻结说明记录尚未执行最终测试集评估；已有验证分数不能改称测试成绩。本轮未加载模型或执行推理，也不以该记录证明此后其他会话绝无新增执行。

## 8. 当前主要说明文档

| 文档 | 用途 |
|---|---|
| `1.7B完整训练与成本记录说明.md` | 基准训练流程、时间和资源口径，包含早期准备状态及后续补充 |
| `九套LoRA批量训练运行说明.md` | 参数对照批次、启动及记录规则；部分状态写于执行前 |
| `LoRA十套配置清单.md` | 对照配置说明 |
| `LoRA最终评估冻结记录.md` | 最终T01配置、三种子适配器及评估边界 |
| `reports/qwen17-full-20261025-231706-ea8173核验.md` | 基准真实运行核验 |
| 两个成功对照/复跑批次下的`audit-*/核验汇总.md` | 验证预测重算、参数对照及三种子核验 |
| `项目目录与路径索引.md` | 面向协作的目标目录及本地路径配置设计，不能当作全部已实现 |

历史说明中“尚未执行”“prepared”等状态应按日期和后续核验记录理解。交接时以具体运行的最新状态、产物和校验信息为证据，不仅依据说明文档的早期段落。

## 9. 后续融合时需要保留和处理的内容

- 保留训练原始证据、配置、版本、逐轮预测、最佳检查点与失败/中断批次关联。
- 把共用指标、哈希和验证数据读取逻辑从0.6B命名脚本中明确出来，避免其他成员误删依赖。
- 统一配置和路径读取，消除作者机器硬编码；本地路径信息不进入公开仓库。
- 统一各路线逐样本预测字段，再实现跨方案合并与错例导出；目前还没有完整错例归因报告模块。
- 现有验证生成耗时不能直接替代已设计的正式延迟基准，需要统一计时边界、预热和固定样本清单。
- 分开归档0.6B工程练习、1.7B冒烟、参数搜索、种子复跑及最终测试，避免混算成绩和成本。
- 原始模型下载revision未知的事实继续披露；本地权重SHA256锁定不等于已经补齐上游revision。
- 当前1.7B代码使用SwanLab云模式，早期设计写默认离线；代码融合需明确实际采用方式，不将两者混写为同一状态。

本文仅用于当前LoRA模块盘点与执行交接，不代表其他成员的TF-IDF、BERT或提示实验已融合，也不新增实验参数或运行任务。
