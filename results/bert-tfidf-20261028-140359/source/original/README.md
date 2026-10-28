# 中文酒店评论情感分析 · 提示 vs 微调 —— 工作总览与复现指南

> 这份文档一份两用：
> - **第 1–4 章**给人和组员看：我们做了什么、产生什么效果、数据怎么比。
> - **第 5–7 章**给另一个 AI 智能体看：照着做，能复现出**逐位一致**的结果；做完用 `tools/verify_repro.py` 一键判定。
>
> 所有数字都能追溯到 `期望结果/` 里的 CSV，**没有一处是手写进去的**。校验器比对的就是这些文件本身。

---

## 0. 三行速览

| | |
|---|---|
| **任务** | 方向 C「提示 vs 微调」对比：同一份中文酒店评论情感二分类数据上，比较**传统方法 / 小模型全参微调 / 提示 / LoRA** 四条路线 |
| **本文档覆盖的两条路线** | ① 非大模型基线：TF-IDF + LogisticRegression（4 组 n-gram 对比）<br>② 小模型全参微调：MacBERT-base 全参微调（8 组超参 + 6 点数据量曲线 + 1 组分组方式对照） |
| **主线结论** | MacBERT 全参比 TF-IDF 高 **3.86 pp**（McNemar p=0.0006，**显著**）；但 8 组超参之间**差异全部不显著**（最大 0.51 pp，p=0.5966）；数据量拐点在 **100~500 条**（n=100 差 12.23 pp 显著，n≥500 全不显著） |

---

## 1. 我们做了什么工作

### 1.1 组内分工与本文档的边界

课程要求每条路线内部跑多个变体 → 各选**一个最优方案** → 代表之间互相比。本组三人分头做三个方向（提示 / 微调 / …），最终各交一个代表方案互比。

**本文档是"baseline 侧"**，由 Jasper 负责，包含两条臂：

| 臂 | 变体数 | 产出 | 代表方案（按验证集选） |
|---|---|---|---|
| 传统方法基线 | 4 组 n-gram | `results_TFIDF/` | `char_1_3`（字符 1–3 gram + LR） |
| 小模型全参微调 | 8 组超参 + 6 点数据量曲线 + 1 组分组对照 | `results_BERT_FULL/` | `lr_1e5`（MacBERT-base，学习率 1e-5） |

> **提示臂与 LoRA 臂由另外两位同学负责**，他们的脚本（`src/models/qwen_prompt.py`、`src/models/qwen_lora.py`）也在包里，但产物未包含 —— 因此本文档的主对比表里那两行是 `待补`，不是"没做"。两条臂的交接要求见 `docs/交接清单_提示与LoRA臂.md`（在项目内，未随包分发）。

### 1.2 数据：一个已经冻结、不再变动的底座

这一层是所有路线能横向比较的前提，所以做得比模型更严格。

| 项目 | 值 |
|---|---|
| 来源 | [ModelScope AiNiklaus/ChnSentiCorp](https://modelscope.cn/datasets/AiNiklaus/ChnSentiCorp)，原始文件 `ChnSentiCorp_htl_all.csv` |
| SHA256（硬校验） | `1d414b9c65caf3a836abdb4efded621968502ad3338e1a9d885ae131c86083f0` |
| 原始记录 | 7,766 条（负面 2,444 / 正面 5,322），二分类、不平衡 2.18× |
| 清洗 | 严格 UTF-8-SIG + CSV 解析；CRLF/CR 归一为 LF；去首尾空白；空文本排除 1 条；全文精确重复 0 条、标签冲突 0 条；512 字符截断前缀碰撞 2 条（保留源序号最小者） |
| **最终记录** | **7,763** 条（数量守恒 7766 = 7763 + 1 + 2） |
| 划分 | 两次分层 `train_test_split`，`random_state=67`；**train 6,210 / validation 776 / test 777**；测试集负 245 / 正 532 |
| 固定子集 | 冒烟 20 条（种子 67）、少样本 4 条（种子 67）、延迟 200 条（种子 67） |
| 种子分工 | 划分种子 **67** ≠ 模型训练种子 **42** |

**关键决定：划分归属直接冻结成 ID 清单。**
`冻结资产/data/processed/v1-20261023/split_ids.json` 存着全部 7,763 条 `sample_id` 的归属。复现时**不重新抽样**——sklearn 不同版本对 `StratifiedShuffleSplit` 的实现有过调整，重抽可能让 `sample_id` 归属漂移，一漂移后面所有模型的数字全部作废。

`sample_id` 格式 `<源文件SHA256>:<源记录序号>`，四条路线共用同一个主键，这是能做**逐条配对显著性检验**（McNemar）的前提。

### 1.3 传统方法基线：TF-IDF + LogisticRegression

4 组变体，只改"怎么切词 / 切几 gram"，其余全部固定（C=4.0, min_df=2, sublinear_tf, max_features=200000）：

| 配置 | analyzer | n-gram | 分词器 |
|---|---|---|---|
| `char_2_4` | char | (2,4) | 无 |
| `char_1_3` | char | (1,3) | 无 |
| `char_2_5` | char | (2,5) | 无 |
| `word_1_2` | word | (1,2) | jieba |

这一组直接回应课程原理点 1「分词/子词与词表选择对效果的影响」：中文没有空格，字符 n-gram 和词级 n-gram 差多少。

### 1.4 小模型全参微调：MacBERT-base 全参

**8 组超参（单变量为主）**，训练种子固定 42：

| 配置 | 学习率 | epoch | 冻结下层 | 类别加权 | maxlen | 改动点 |
|---|---|---|---|---|---|---|
| `main` | 2e-5 | 3 | 0 | 否 | 256 | 基准 |
| `lr_1e5` | **1e-5** | 3 | 0 | 否 | 256 | 学习率 ↓ |
| `lr_5e5` | 5e-5 | 3 | 0 | 否 | 256 | 学习率 ↑ |
| `ep5` | 2e-5 | **5** | 0 | 否 | 256 | 轮数 ↑ |
| `len128` | 2e-5 | 3 | 0 | 否 | **128** | 截断长度 ↓ |
| `len512` | 2e-5 | 3 | 0 | 否 | **512** | 截断长度 ↑ |
| `weighted` | 2e-5 | 3 | 0 | **是** | 256 | 类别不平衡 |
| `partial` | **3e-5** | 3 | **6 层** | 否 | 256 | ⚠️ 唯一双变量组合 |

> `partial` 同时改了学习率和冻结层数，是个**双变量**组合，它的差值只能当"组合效果"读，不能归因到某一个变量。报告里对此有专门说明。

另外三组实验：

- **数据量曲线**：训练集取 100 / 500 / 1000 / 2000 / 3000 / 6210（分层抽样，种子 42），看"要多少数据才够"。
- **分组方式对照**：长度分桶 batch vs 随机 batch（这是工程提速手段，必须验证它没偷走精度）。
- **延迟测量**：冻结的 200 条，batch=1，预热 20 次，统计 mean/P50/P95/max。

### 1.5 选代表规则：按验证集，不按测试集

**全项目四条路线统一**：每个变体都先在验证集（776 条，与测试集 777 条完全独立）上评一次，**取验证集准确率最高的那个当代表**，测试集只在最终比较时用一次。

| 配置 | 验证集 acc | 测试集 acc | 是否代表组 |
|---|---|---|---|
| **`lr_1e5`** | **0.9162** | 0.9228 | ✅ **代表组** |
| `ep5` | 0.9137 | 0.9215 | |
| `main` | 0.9124 | 0.9189 | |
| `len512` | 0.9124 | **0.9279** | ❌ 测试集最高，但验证集只 0.9124 |
| `partial` | 0.9111 | 0.9189 | |
| `len128` | 0.9085 | 0.9266 | |
| `lr_5e5` | 0.9059 | 0.9176 | |
| `weighted` | 0.9059 | 0.9241 | |

> **为什么不用测试集选**：用测试集挑代表 = 拿最终考试成绩挑谁上场，测出来的高分会被这个选择本身抬高，后面跟别的路线比就不公平了。`len512` 正是活例子 —— 测试集 0.9279 比代表组高 0.51 pp，但验证集低 0.39 pp；按测试集选它，等于用 777 条考卷反复挑模型。
>
> 这条规则对 TF-IDF 臂（按验证集选 `char_1_3`）、LoRA 臂（按秩的验证集分数选）、提示臂（按变体验证集分数选）同样成立。

**另外并存一个基准组**：报告的"单变量超参敏感性对比"以 `main` 为基准，"路线间对比"以代表组 `lr_1e5` 为基准 —— 两个基准各管一件事，报告 §5.3(2) 有专门的说明表。

---

## 2. 产生什么效果

### 2.1 四条路线主对比（表 1）

| 路线 | 代表方案 | 测试集 acc | macro F1 | weighted F1 | 平均延迟 | 训练耗时 |
|---|---|---|---|---|---|---|
| 传统方法基线 | TF-IDF `char_1_3` + LR | 0.8842 | 0.8610 | 0.8820 | **0.748 ms** | 5.9 s |
| 小模型全参微调 | MacBERT-base 全参 `lr_1e5` | **0.9228** | **0.9108** | **0.9229** | 103.519 ms | 8,325 s |
| 提示 | 待同学补齐 | — | — | — | — | — |
| LoRA | 待同学补齐 | — | — | — | — | — |

> 延迟一列：基线两臂是同机 CPU 实测；提示/LoRA 两臂跑在 Kaggle T4 上，**跨硬件不可横比**。报告 §7.1 加了"延迟测量环境"列，可横比的只有 TF-IDF vs MacBERT（同机 CPU）和 提示 vs LoRA（同 T4）。

### 2.2 四条能站得住的结论

**① 微调带来的提升是真的，不是噪声。**
TF-IDF 0.8842 → MacBERT 0.9228，**Δ = +3.86 pp**，配对 McNemar 检验 b=21 / c=51、**p = 0.000632**，bootstrap Δacc 95% CI = **[−6.05, −1.80] pp**（负号表示 TF-IDF 更差），一致率 90.73%。样本外、可复算、显著。

**② 但"调参"调不出显著差别。**
8 组超参两两配对检验 **7 组全部不显著**，最大差距也只有 0.51 pp（`len512` vs `lr_1e5`，p = 0.5966，CI = [−0.90, +1.93] pp）。这说明在这个任务上，MacBERT 的收益来自"用了预训练表示并做全参微调"这件事本身，而不是学习率/轮数/截断长度怎么调 —— 剩下这点差别落在评测噪声里。

**③ 数据量拐点在 100~500 条之间。**
与全量（6,210 条）逐点配对：

| 规模点 | Δ Accuracy | McNemar p | bootstrap Δacc 95% CI | 结论 |
|---|---|---|---|---|
| n=100 | **−12.23 pp** | **<0.0001** | [−15.06, −9.40] pp | **显著** |
| n=500 | −0.26 pp | 0.8774 | [−1.93, +1.29] pp | 不显著 |
| n=1000 | +0.00 pp | 0.8774 | [−1.67, +1.67] pp | 不显著 |
| n=2000 | +0.39 pp | 0.7283 | [−1.03, +1.80] pp | 不显著 |
| n=3000 | −0.39 pp | 0.7111 | [−1.80, +1.03] pp | 不显著 |

几百条就已经够微调出接近全量的效果，再加数据换不来**可测**的提升（注意 n=1000 那一点甚至和全量**逐位相同**，0.9228 —— 巧合，但确实如此）。

**④ 精度、延迟、训练成本要一起看。**
TF-IDF 0.748 ms/条 vs MacBERT 103.519 ms/条 —— **差 138 倍**；再算上 8,325 s（约 2.3 小时）的全参训练成本。如果场景允许 0.7 ms 的延迟，传统方法在这个任务上是极有竞争力的选择；如果允许 2.3 小时训练换 3.86 pp，那微调更值。

### 2.3 错误样例（3 条）

报告 §6.1 要做**语义归因**（为什么错），这一节的**选例和数据**由脚本自动产出，**归因与结论由作者自己写**：

| sample_id | 真实 | 预测 | 置信度 | token 数 |
|---|---|---|---|---|
| `…:5510` | negative | positive | 0.9996 | 24 |
| `…:6568` | negative | positive | 0.9994 | 31 |
| `…:6925` | negative | positive | 0.9994 | 53 |

三条**全部是"负面被预测成正面"**，且模型置信度都在 0.999 以上 —— 模型很自信地错了，这是最有分析价值的错误类型（典型如"性价比还不错，门面很不起眼"这类**转折句**和**长评论里后半段的转折**）。完整的正文见 `期望结果/results_BERT_FULL/error_samples.csv`（全项目只有这 3 条正文进包）。

### 2.4 类别级错误分布

`期望结果/results_BERT_FULL/errors_by_class.csv` 给出逐条错误（负 216 / 正 29 / 共 245 条错误），`acc_by_length.csv` 给出按 token 长度的分桶准确率。混淆矩阵（代表组）：

|  | 预测 negative | 预测 positive |
|---|---|---|
| 真实 negative | 216 | 29 |
| 真实 positive | 31 | 501 |

准确率 0.9228 = (216 + 501) / 777。

---

## 3. 数据对比（明细）

### 3.1 MacBERT 8 组超参

| 配置 | lr | epoch | 冻结层 | 类别加权 | maxlen | 可训练参数 | 测试 acc | macro F1 | weighted F1 | 训练耗时 |
|---|---|---|---|---|---|---|---|---|---|---|
| `main` | 2e-5 | 3 | 0 | 否 | 256 | 102,269,186 (100%) | 0.9189 | 0.9066 | 0.9191 | 6,770 s |
| `partial` | 3e-5 | 3 | 6 | 否 | 256 | 43,119,362 (42.2%) | 0.9189 | 0.9054 | 0.9186 | 4,687 s |
| `weighted` | 2e-5 | 3 | 0 | 是 | 256 | 102,269,186 (100%) | 0.9241 | 0.9125 | 0.9243 | 6,788 s |
| `lr_5e5` | 5e-5 | 3 | 0 | 否 | 256 | 102,269,186 (100%) | 0.9176 | 0.9048 | 0.9177 | 6,993 s |
| `len128` | 2e-5 | 3 | 0 | 否 | 128 | 102,269,186 (100%) | 0.9266 | 0.9151 | 0.9267 | 5,650 s |
| `len512` | 2e-5 | 3 | 0 | 否 | 512 | 102,269,186 (100%) | **0.9279** | **0.9169** | **0.9281** | 12,150 s |
| **`lr_1e5`** | 1e-5 | 3 | 0 | 否 | 256 | 102,269,186 (100%) | 0.9228 | 0.9108 | 0.9229 | 8,325 s |
| `ep5` | 2e-5 | 5 | 0 | 否 | 256 | 102,269,186 (100%) | 0.9215 | 0.9096 | 0.9217 | 13,456 s |

> 耗时是本机 CPU（12 线程 R5 5600H）的实测，换机器会变 —— 校验器对 `seconds` 只提示不判失败。

### 3.2 数据量曲线

| n_train | 测试 acc | macro F1 | weighted F1 |
|---|---|---|---|
| 100 | 0.8005 | 0.7225 | 0.7768 |
| 500 | 0.9202 | 0.9082 | 0.9205 |
| 1,000 | 0.9228 | 0.9119 | 0.9233 |
| 2,000 | 0.9266 | 0.9153 | 0.9268 |
| 3,000 | 0.9189 | 0.9070 | 0.9193 |
| 6,210 | 0.9189 | 0.9066 | 0.9191 |

### 3.3 分组方式对照（长度分桶 vs 随机 batch）

| 分组方式 | n_train | 测试 acc | macro F1 | weighted F1 | 耗时 |
|---|---|---|---|---|---|
| 长度分桶 | 1,000 | 0.9228 | 0.9119 | 0.9233 | **1,585 s** |
| 随机 | 1,000 | 0.9112 | 0.8998 | 0.9123 | 2,517 s |

配对检验：Δ = +1.16 pp，McNemar b=14 / c=5，**p = 0.0636（不显著）**，bootstrap Δacc 95% CI = [+0.13, +2.32] pp，一致率 97.55%。

**如实表述**：分桶把训练时间压掉 37%（1,585 s vs 2,517 s），McNemar 检验没有达到 0.05 显著性，但 bootstrap 区间没有跨 0。这两条证据只够支持"**分桶没有牺牲精度**"，**不够**支持"**分桶更好**"。报告按这个口径写的。

### 3.4 延迟与成本

| 变体 | n | mean (ms) | P50 (ms) | P95 (ms) | max (ms) | batch | 预热 | 测量环境 |
|---|---|---|---|---|---|---|---|---|
| MacBERT 全参 | 200 | 103.519 | 81.034 | 207.391 | 284.492 | 1 | 20 | Windows (torch 2.10.0+cpu) |
| TF-IDF + LR | 200 | 0.748 | 0.632 | 1.184 | 5.648 | 1 | 20 | WSL (sklearn 1.9.1) |

> ⚠️ **必须在机器空闲时测**。训练进程占满 CPU 时测出来的数字虚高一倍多（本项目踩过：219.8 ms/条 vs 空闲时 103.5 ms/条，那一轮已作废）。
>
> ⚠️ **跨硬件不可横比**。提示/LoRA 在 Kaggle T4、基线两臂在本机 CPU。

### 3.5 配对显著性检验总表（21 组）

| 类别 | 组数 | 结论 |
|---|---|---|
| 基线 vs 微调 | 8 | **全部显著**（TF-IDF 输给 MacBERT，Δ 最大 −4.38 pp，p 均 < 0.01） |
| 超参对比 | 7 | **全部不显著**（最大 \|Δ\| = 0.51 pp，最小 p = 0.5966） |
| 分组方式对照 | 1 | 不显著（p = 0.0636） |
| 数据量曲线 | 5 | 仅 n=100 显著（Δ = −12.23 pp，p < 0.0001），其余 4 点不显著 |

四组两两不重叠、互不重复计数。完整数据（含 b/c、CI、一致率）见 `期望结果/paired_tests.csv`。

---

## 4. 包里有什么 / 不有什么

```
最终作业_复现包/
├── README.md                     ← 本文档
├── 项目报告_终版.md               ← 完整报告（正式提交版）
├── 项目报告.md                   ← 报告模板（fill_report.py 的输入）
├── 项目摘要.md / 项目摘要_模板.md
├── 环境/
│   └── requirements.txt          ← 依赖与版本
├── 脚本/                          ← 复现需要的全部代码（见第 5 章，先整份拷到工作目录）
│   ├── train_tfidf.md  train_bert.md  train_prompt.md  train_lora.md
│   ├── _extract_md_code.py  run_bert_local.bat
│   ├── src/{data,configs,eval,models,scripts}/…
│   └── tools/…                    ← 含 preflight_check.py（2 分钟预检）、verify_repro.py（判定）
├── 冻结资产/
│   └── data/processed/v1-20261023/
│       ├── split_ids.json        ← 7,763 条 ID 的划分归属（复现划分的唯一真源）
│       ├── fewshot_ids.json  smoke_ids.json  latency_ids.json
│       ├── statistics.json       ← 划分统计（校验器拿它对账）
│       └── status.json  label_map.txt
└── 期望结果/                      ← 我们跑出来的结果，逐位比对的目标
    ├── results_TFIDF/            （4 组 n-gram + 逐条预测 + 混淆矩阵图）
    ├── results_BERT_FULL/        （8 组超参 + 曲线 + 分组对照 + 延迟 + 逐条预测 + 训练日志 + 图）
    └── paired_tests.csv / .md / paired_tests_key.json
```

**包里刻意不放**（重要）：

| 不放的东西 | 原因 |
|---|---|
| 模型权重（`macbert_base/` 400 MB、`bert_FULL.pt` 390 MB） | 体积大且非源码；复现者自行下载（脚本会自动下，见 5.4） |
| **评论正文**（train/validation/test.jsonl、excluded.jsonl、原始 CSV） | 原仓库只有模板式 Apache-2.0 标记，**未找到能核实评论正文再分发权利的独立依据**，因此 `redistribution_verified=false`：正文不进任何提交包，只在 Kaggle session 内以 `Add Input` 方式使用 |
| 虚拟环境、缓存、录屏、任何 API Key | — |
| 提示臂 / LoRA 臂的产物 | 由另外两位同学产出，尚未交付；脚本已就绪 |

全项目只有 **3 条**评论正文进包：`期望结果/results_BERT_FULL/error_samples.csv`（报告 §6.1 的失败样例）。本文档 §2.3 与 `项目报告_终版.md` 的 §6.1 引用了其中 2 条做原文展示，**再无其他正文** —— 用原始数据每条评论的前 12 字做指纹回扫全包，命中的就是这三个文件。

**原始数据怎么办**：复现者按 5.3 的地址自行下载原始 CSV，`rebuild_data.py` 会做 **SHA256 硬校验** —— 拿到的不是同一份数据会直接报错退出，不会悄悄跑歪。

---

## 5. 复现指南

> **写给 AI 智能体**：这一章是操作手册。目标是让你（或任何一个智能体）跑完之后，`tools/verify_repro.py` 输出"全部一致"。
>
> **铁律三条**：
> 1. **不要改任何脚本里的常量**。种子、超参、路径都是冻结的；改了就对不上。
> 2. **不要用测试集选模型/选代表**。代表一律按验证集选（`src/eval/representative.py` 是唯一真源）。
> 3. **遇到"数字不一样"先查数据，再查代码，最后才怀疑随机性**。排查顺序见 5.10。

### 5.0 准备一个干净工作目录

```bash
# 1) 建工作目录
mkdir work && cd work

# 2) 把「脚本/」目录下的**所有内容**复制到 work/ 根下
#    （Windows: xcopy /E /I "..\脚本\*" work
#      Linux/macOS: cp -r ../脚本/. ./）
#    复制后 work/ 下应该有：src/  tools/  train_*.md  run_bert_local.bat  _extract_md_code.py

# 3) 把「期望结果/」和「冻结资产/」也复制到 work/ 下（校验器默认就找这两个目录）
cp -r ../期望结果 ./
cp -r ../冻结资产 ./
```

复制完成后 `work/` 应该是这样：

```
work/
├── src/  tools/  train_*.md  run_bert_local.bat  _extract_md_code.py
├── 期望结果/
├── 冻结资产/
└── data/                      ← 5.3 步会生成
```

### 5.1 环境

| 用途 | 版本 | 说明 |
|---|---|---|
| Python | 3.11.x | 3.14 跑 torch 性能很差，别用 |
| torch | 2.10.0+cpu | CPU 版即可；有 GPU 更快 |
| transformers | 5.20.2 | **必须 ≥ 4.51**（才有 `Qwen3ForSequenceClassification`） |
| scikit-learn | 1.9.1 | 建议一致（但划分不再依赖它，见 5.3） |
| pandas / numpy / matplotlib | ≥2.0 / ≥1.24 / ≥3.7 | matplotlib 需要有中文字体（msyh / simhei） |
| jieba | ≥0.42 | TF-IDF 的词级对照用；缺了会跳过该组，其余照跑 |

```bash
pip install -r src/requirements.txt
```

**本机特殊情况**：如果 Windows 的「智能应用控制」拦截了 torch / sklearn 的 C 扩展（`WinError 4551`），把 sklearn 相关的步骤放到 WSL 里跑，torch 相关的步骤用独立 venv 跑。本项目就是这么分开的（sklearn/jieba/matplotlib 在 WSL，torch/transformers 在 `D:\nlp_venv`）。

**Kaggle**：T4 ×2 镜像里以上依赖全可用，而且 BERT 部分 45 分钟能跑完（CPU 要 26 小时）。

### 5.2 目录与随机种子检查

所有脚本都用 `seed=42`（模型训练）和 `67`（划分/子集抽样）。确认环境变量没被人改过：

```bash
env | grep -i -E "seed|pythonutf8"   # 应该没有输出
export PYTHONUTF8=1                  # 建议设上，避免中文 print 崩在 GBK 控制台上
export PYTHONIOENCODING=utf-8
```

### 5.3 步骤 1：准备数据

> ⚠️ **从这一步开始，所有命令都在 `work/` 根目录下执行**（先 `cd work`）。脚本按"当前工作目录"定位数据和结果目录，cwd 不对会写到别处 —— `src/models/tfidf_baseline.py` 现在会对此报警。

**下载原始数据**

- 地址：<https://modelscope.cn/datasets/AiNiklaus/ChnSentiCorp>
- 文件：`ChnSentiCorp_htl_all.csv`（2,904,858 字节）
- SHA256 必须是 `1d414b9c65caf3a836abdb4efded621968502ad3338e1a9d885ae131c86083f0`

放到：

```
work/data/ChnSentiCorp/ChnSentiCorp_htl_all.csv
```

**重建冻结划分**

```bash
python src/data/rebuild_data.py \
    --source data/ChnSentiCorp/ChnSentiCorp_htl_all.csv \
    --frozen-dir 冻结资产/data/processed/v1-20261023 \
    --out data/processed/v1-20261023
```

预期输出（**逐位一致才算对**）：

```
源文件 SHA256: 1d414b9c65caf3a836abdb4efded621968502ad3338e1a9d885ae131c86083f0
  ✓ 与冻结版一致 (2,904,858 bytes)
原始记录 7766  非法 0
  排除明细: {'empty_text': 1, 'duplicate_truncated_prefix': 2}
最终有效 7763  (期望 7763)
       train:  6210  负面0= 1954  正面1= 4256
   validation:   776  负面0=  244  正面1=  532
         test:   777  负面0=  245  正面1=  532
  ✅ 全部一致 —— 划分与冻结版逐位相同
```

> 这个脚本**不依赖 sklearn**（纯 Python 解析 + 直接读冻结的 `split_ids.json`），所以无论装哪个版本，结果都一样。
> `src/data/process_data.py` 是更早的版本（自己用 sklearn 抽样），留作对照，不要用它重建。

### 5.4 步骤 2：TF-IDF 基线（几十秒）

```bash
python src/models/tfidf_baseline.py
```

- 自动找到 `data/processed/v1-20261023/`，结果写到 `src/results/results_TFIDF/`（目录不存在会自动建）
- 产出：`tfidf_ablation.csv`（4 组）、`predictions.csv`（777 条逐条）、`errors_by_class.csv`、`tfidf_report.txt`、`tfidf_compare.png`
- 预期：`char_1_3` 的 `val_acc=0.9021, test_acc=0.8842, macro_f1=0.861, weighted_f1=0.882`，`n_features=130996`

> ⚠️ **这一步需要 scikit-learn**。本项目的 Windows 系统 Python 装了 sklearn 但 C 扩展被「智能应用控制」拦掉（`WinError 4551`），所以实际是在 WSL 里跑的（`/home/rl/nlp/venv/bin/python`，sklearn 1.9.1）。如果你的环境能正常 `import sklearn`，直接跑就行；跑不了就把这一步放 WSL（其余 torch 步骤留在本地 venv）。

### 5.4b 步骤 2.5：预检（2 分钟，省掉 26 小时白跑）

完整复现要跑 26 小时训练。**先花 2 分钟做预检**，把「环境 / 数据 / 路径」这三件事一次验完：

```bash
python tools/preflight_check.py
```

它会重建数据、逐文件比对 SHA256、跑一遍结果校验，然后告诉你能不能进入正式训练。**训练之前跑一次，训练之后再跑一次**。

它分两个阶段，不会把「还没训练」误判成失败：

| 什么时候跑 | 它做什么 | 期望输出 |
|---|---|---|
| **训练之前** | 重建数据 + SHA256 比对；发现训练产物还没生成，就跳过结果比对 | `结论：✅ 预检通过（数据与环境就绪），可以进入正式训练。` |
| **训练之后** | 重建数据 + SHA256 比对 + 结果逐位比对 | `结论：✅ 预检通过 —— 结果与期望结果逐位一致。` |

### 5.5 步骤 3：MacBERT 全参微调（CPU 约 26 小时 / Kaggle 约 45 分钟）

**Windows 本机**：

```bat
run_bert_local.bat
```

这个 bat 会：检查有没有别的训练在跑 → 从 `train_bert.md` 生成 `_bert_run.py` → 跑训练。**它是可断点续跑的**：中途 Ctrl+C，再跑一次会跳过已完成的配置继续。

**Linux / Kaggle**：把 `train_bert.md` 里 `exec(r"""...""")` 那段代码整段贴进一个 cell 跑（脚本会自动检测 GPU/CPU）。

**先跑冒烟验证（3 分钟）**：设 `BERT_SMOKE=1`，只跑极小规模确认代码通。正式跑时 `BERT_SMOKE=0`（bat 默认就是 0）。

**跑完应该得到**（`src/results/results_BERT_FULL/`）：

| 文件 | 内容 |
|---|---|
| `ablation_results.csv` | 8 组超参汇总 |
| `predictions_{config}.csv` | **8 个文件**，每组 777 条逐条预测 |
| `val_log.csv` / `train_log.csv` | 逐 epoch 验证集准确率 / 逐 step 损失 |
| `data_curve.csv` | 6 点数据量曲线 |
| `predictions_curve_n{100,500,1000,2000,3000}.csv` | 5 个曲线点的逐条预测 |
| `acc_by_length_{config}.csv` | 按 token 长度的分桶准确率 |
| `run_log.txt` | 完整运行日志 |

> **曲线点为什么要单独补跑**：早期曲线点沿用了 `main` 这个名字，逐条预测写进了 `predictions_main.csv`，主结果还原后那批预测丢失了 —— 于是曲线只能做趋势解读，做不了逐点配对检验。所以用 `BERT_CURVE_FORCE=1` 把 5 个点重跑一遍，结果**另存** `data_curve_repred.csv`，不覆盖 `data_curve.csv`；两份文件逐位对比本身就是"同一份代码、同一颗种子能跑出同样结果"的证据。
>
> ```bat
> set BERT_CURVE_FORCE=1
> run_bert_local.bat
> ```
> 断点续跑粒度 = 一个规模点，中断不丢已完成点。

### 5.6 步骤 4：选代表组

```bash
python tools/switch_representative.py           # 先 dry-run 看候选
python tools/switch_representative.py --apply   # 确认后写入
```

它按验证集排名挑代表，然后把 `predictions.csv` / `acc_by_length.csv` / `errors_by_class.csv` / `error_samples.csv` / `bert_report.txt` 全切到代表组，并写 `representative.json`。

**预期**：代表组 = `lr_1e5`（val 0.9162），切换后 `predictions.csv` 的准确率 = 0.9228。

> ⚠️ 这一步是**纯重算**，不重训、不改任何预测结果，只是把"以哪个配置为代表"的产物指针切过去。

### 5.7 步骤 5：配对显著性检验

```bash
python src/eval/paired_tests.py --root .
```

产出 `src/results/paired_tests.csv`（21 组）、`paired_tests.md`、`paired_tests_key.json`。

> `paired_tests_key.json` 里的 `PT_*` 键是报告占位符的数据源 —— 报告里所有"显著/不显著"的判断句都是从这个文件自动填的，不手工写。

### 5.8 步骤 6：延迟测量

```bash
python src/eval/latency_baseline.py
```

**必须在机器空闲时跑**（训练进程占满 CPU 时数字虚高一倍多）。

产出 `latency_baseline.csv`（MacBERT 全参 + TF-IDF+LR 两行）。若只要其中一行：`--skip-tfidf` / `--skip-bert`。

### 5.9 步骤 7：出报告（可选）

```bash
python src/scripts/fill_report.py     # 项目报告.md → 项目报告_终版.md，摘要同理
python tools/make_bert_plots.py       # loss 曲线 + 混淆矩阵（用代表组）
python tools/md2pdf.py                # Markdown → PDF（需要 Chrome/Edge）
```

> `md2pdf.py` 依赖本机 Chrome/Edge 的 headless 打印；没有 Chrome 也能出 PDF，只是排版差些。**这一步不影响复现判定** —— 核心交付是数字。

### 5.10 步骤 8：校验（判定复现成败）

```bash
python tools/verify_repro.py --results src/results --expect 期望结果 \
                             --data data/processed/v1-20261023
```

或者跑打包好的预检（它会先重建数据、再调这个校验器）：

```bash
python tools/preflight_check.py
```

判定分三档：

| 档 | 查什么 | 不一致怎么办 |
|---|---|---|
| **【严格】** | 6 个汇总 CSV 的指标列 + **全部 17 份逐条预测（16 份 BERT + 1 份 TF-IDF，每份 777 条，按 `sample_id` 对齐）** | **复现失败**，按 5.11 排查 |
| **【提示】** | `seconds`（训练耗时）、`mean_ms`（延迟）、bootstrap CI | 只提示，不判失败 —— 跟 CPU 型号、负载、numpy 小版本有关，本来就不该期待一致 |
| **【结构】** | 数据划分（SHA256 / 记录数 / 排除规则 / 长度分布 / 6,210-776-777 三划分条数） | **复现失败**，通常是数据没重建对 |

全过时的输出：

```
  结论：✅ 全部 N 项一致 —— 复现成功
```

退出码 `0` = 成功，`1` = 有不一致。

### 5.11 复现失败怎么查

按这个顺序查，从最常见的原因开始：

| 症状 | 最可能的原因 | 怎么办 |
|---|---|---|
| **结果写到了奇怪的位置** | **不在项目根目录跑** —— 脚本按 cwd 定位数据和结果目录 | `cd work` 再跑；`tfidf_baseline.py` 会打印 `⚠ 数据不在 … 下` 提醒你 |
| `ModuleNotFoundError: No module named 'sklearn'` | 当前解释器没装 sklearn | 换装了 sklearn 的解释器，或把这一步放 WSL（本项目就是这么分的） |
| 逐条预测大面积不一致，但汇总 acc 差不多 | **数据划分漂移**（重建时没用 `split_ids.json`，或源文件不是那一份） | 跑 5.3 的脚本，它会逐字段打印和冻结版的比对 |
| `sample_id` 对不上 / 报"缺行" | 同上，或 `predictions.csv` 来自不同版本的 `test.jsonl` | 删掉 `src/results/` 整个重跑 |
| 某一组 acc 差 1~2 pp | **环境差异**（transformers 版本、tokenizer 版本） | 核对 `环境/requirements.txt`；transformers 锁 5.20.2 |
| 延迟 / 耗时对不上 | **正常**，硬件不同 | 忽略，校验器对这两列只提示不判失败 |
| bootstrap CI 差一点 | **正常**，numpy 小版本的浮点差异 | 忽略 |
| 训练直接崩，报 `UnicodeEncodeError` | Windows 控制台 GBK | 设 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8` |
| 训练崩，报 `WinError 4551` | Windows「智能应用控制」拦了 torch 的 DLL | 换独立 venv，或把该步骤放 WSL |
| `import torch` 报 torchvision 版本冲突 | transformers 5.x 硬依赖 torchvision | `pip install torchvision==0.25.0+cpu`（配合 torch 2.10） |
| 下载模型超时 | huggingface.co 直连不通 | 用 `https://hf-mirror.com` 镜像（脚本已默认，且有 curl 兜底） |
| 跑第二次报"另一个训练在跑" | 上一次进程没退 | `tools/check_no_other_run.ps1` 是守门脚本，确认没有残留的 `_bert_run.py` 进程 |

### 5.12 期望结果速查表

校验时对照这张表最快：

| 文件 | 关键值 |
|---|---|
| `results_TFIDF/tfidf_ablation.csv` | `char_1_3`: val 0.9021 / test 0.8842 / mf1 0.861 / wf1 0.882 / n_feat 130996<br>`char_2_4`: 0.9008 / 0.8816 / 0.8553 / 0.8781<br>`char_2_5`: 0.8982 / 0.8816 / 0.8557 / 0.8783<br>`word_1_2`: 0.8943 / 0.8970 / 0.8783 / 0.8959 |
| `results_BERT_FULL/ablation_results.csv` | `lr_1e5`: 0.9228 / 0.9108 / 0.9229 / 102,269,186<br>`main`: 0.9189 / 0.9066 / 0.9191<br>`partial`: 0.9189 / 0.9054 / 0.9186<br>`weighted`: 0.9241 / 0.9125 / 0.9243<br>`lr_5e5`: 0.9176 / 0.9048 / 0.9177<br>`len128`: 0.9266 / 0.9151 / 0.9267<br>`len512`: 0.9279 / 0.9169 / 0.9281<br>`ep5`: 0.9215 / 0.9096 / 0.9217 |
| `results_BERT_FULL/data_curve.csv` | 100→0.8005/0.7225/0.7768；500→0.9202/0.9082/0.9205；1000→0.9228/0.9119/0.9233；2000→0.9266/0.9153/0.9268；3000→0.9189/0.9070/0.9193；6210→0.9189/0.9066/0.9191 |
| `results_BERT_FULL/batching_control.csv` | len_group: 0.9228/0.9119/0.9233/1585s；random: 0.9112/0.8998/0.9123/2517s |
| `results_BERT_FULL/latency_baseline.csv` | MacBERT: mean 103.519 / P50 81.034 / P95 207.391 / max 284.492（n=200）<br>TF-IDF+LR: mean 0.748 / P50 0.632 / P95 1.184 / max 5.648（n=200） |
| `paired_tests.csv` | 21 组；`significant_95` 为 True 的只有：基线vs微调 8 组 + 数据量曲线 n=100 |
| `冻结资产/.../statistics.json` | 7,766 → 7,763；train 6,210 / val 776 / test 777；测试集负 245 / 正 532 |

---

## 6. 已知坑位速查

1. **GBK 控制台**：Windows 下打印 `✓`/`⚠` 会 `UnicodeEncodeError`。所有脚本都加了 `sys.stdout.reconfigure(encoding='utf-8', errors='replace')`；`.bat` 里也设了 `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`。**`.bat` 文件里不要加中文注释** —— cmd 按 GBK 解析注释会崩。
2. **不能同时跑两个训练**：`run_bert_local.bat` 开头调 `tools/check_no_other_run.ps1` 守门。
3. **原子写 CSV**：训练中途崩溃不能留下半截 CSV，所以所有结果文件都是先写临时文件再 rename。
4. **日志污染**：训练日志曾被别的进程写坏过一次；`tools/fix_log_pollution.py` 是修复工具，污染版留证在 `资料/_archive/logs/*.polluted.bak`。
5. **代表组切换是纯重算**：`switch_representative.py` 不重训。
6. **延迟必须空闲时测**：训练占满 CPU 时虚高一倍多（219.8 → 103.5 ms）。
7. **跨硬件延迟不可横比**：本机 CPU vs Kaggle T4。
8. **划分不重新抽样**：永远读 `split_ids.json`。

---

## 7. 边界与合规

- **数据再分发**：`redistribution_verified = false`。原仓库只有模板式 Apache-2.0 标记，未找到能核实评论正文再分发权利的独立依据。因此评论正文**不进任何提交包**，原始数据只在 Kaggle session 内以 `Add Input → Upload` 方式使用，**不建公开 Kaggle Dataset**。
- **包内唯一的正文**：3 条失败样例（`error_samples.csv`），用于报告 §6.1 的错误归因分析。
- **权重与数据**：MacBERT 权重由脚本自动从 `hfl/chinese-macbert-base` 下载（`hf-mirror.com` 镜像优先），不入包。
- **AI 辅助**：实验设计、数据处理、脚本编写与结果核验由 AI 智能体协助完成；**实验结论的语义归因与文字由作者本人撰写**。详见报告 §7.4 的 AI 使用声明。
- **本仓库内的产物归属**：代码由 AI 辅助生成并逐条跑通验证，所有数字来自实测 CSV，无手工填写。

---

## 8. 一句话给下一个智能体

> 从 `脚本/` 出发，**先重建数据（别抽样，用冻结的 `split_ids.json`）**，再跑 TF-IDF（秒级）和 MacBERT（CPU 一天 / T4 45 分钟），然后 `switch_representative.py --apply` 选代表、`paired_tests.py` 做检验、`verify_repro.py` 判定。**任何数字对不上，先查数据，再查代码，最后才怀疑随机性。**
