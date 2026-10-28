# 中文评论情感分析：复现与证据入口

本项目比较TF-IDF＋逻辑回归、预训练MacBERT全参数微调、Qwen3-1.7B零样本、固定四样本和普通LoRA T01。统一数据、种子与评分口径，保留原始预测、日志和失败记录。

**当前完成的是离线复现准备。默认命令重算已有预测，不加载模型。真实模型新机复现和教师验收单独记录。** “48小时补交”是复现失败后的补救期限，不能代替资源和环境准备。

## 下载后，一条命令核查主结果

安装Python 3.11及以上（已测3.13.9），取得本分支：

~~~text
git clone --branch codex/integration --single-branch https://github.com/liuguanghui520/NLP.git
cd NLP
python reproduce.py
~~~

也可下载[分支ZIP](https://github.com/liuguanghui520/NLP/archive/refs/heads/codex/integration.zip)再解压运行。Windows支持命令行调用reproduce.cmd。默认核查无需pip安装、CUDA、模型、作者的WSL目录或个人路径配置。

每次自动建立全新 results/reproduction/<批次>/，成功时退出码0且status.json为passed。它检查冻结数据哈希、划分与标签、16份验证／测试记录、11份测试及种子覆盖，并独立重算主指标；不改原始预测或运行索引。

| 方法 | 测试Macro-F1 | Accuracy | 运行数 |
|---|---:|---:|---:|
| TF-IDF＋逻辑回归 | 86.10% | 88.42% | 3 |
| MacBERT全参数微调 | 91.52% | 92.62% | 3 |
| Qwen零样本 | 62.56% | 59.59% | 1 |
| Qwen固定四样本 | 76.10% | 76.58% | 1 |
| Qwen普通LoRA | 91.12% | 92.28% | 3 |

每运行测试777条；训练路线先逐运行计算再平均。完整精确值与1e-12验收容差在 configs/reproduction.json。结果目录保存配置、输入和入口源码哈希、实际解释器、指标、状态与错误日志。旧输出目录被拒绝覆盖。

可以用禁用第三方site初始化的命令进一步核查：

~~~text
python -S reproduce.py --output results/reproduction/my-stdlib-audit
~~~

## 重建完整离线图表

使用Python3.13.9创建独立分析环境，不升级已有模型环境：

~~~text
python -m venv .venv-analysis
.venv-analysis\Scripts\python -m pip install -r requirements.analysis.lock
.venv-analysis\Scripts\python reproduce.py --analysis
~~~

Linux将解释器路径改为.venv-analysis/bin/python。完整模式还需Git及中文字体Microsoft YaHei、Noto Sans CJK SC或SimHei。版本不匹配会明确报错；默认轻量核查不要求这些依赖。

完整模式生成新ID的22组PNG/PDF、参数、统计和错误材料，同时运行软件检查；仍然不训练或推理。日志和新分析位置写入本次status.json。requirements.analysis.lock对应离线分析；requirements.integration.lock是WSL历史模型环境快照，不能混用。

## 真实模型复现：资源齐备后另验收

真实预测需要Qwen基座、三个BERT最佳模型、三个TF-IDF Pipeline及三个普通LoRA适配器。当前全部训练产物的可下载链接与新机加载流程尚未验收，不能只下载公开基座便认定主结果可重新预测。

代码、配置和小型证据随仓库下载；大权重可通过GitHub Release分发，需补实际下载URL、大小、SHA256及解压结构。资源身份见[模型资源说明](artifacts/README.md)、artifacts/lora-lock.json及基线运行记录；历史说明中的发布状态以本README和[最新复现说明](docs/reproduction/README.md)为准。

高级入口 src/cli.py 支持check、import、predict、evaluate、benchmark、summarize等；这些历史命令依赖外部工具资源，不能替代上面的无模型一键核查。真实预测在资源齐备的独立副本进行，复制 configs/paths.example.json，填写自己的路径，按实际环境和资源清单验收。重新训练不属于默认复现操作。

## 数据、种子与指标

冻结版本v1-20261023：训练6210、验证776、测试777；0=负面、1=正面。所有路线共享前512个Unicode字符，再按各自token预算处理。验证Macro-F1选优，Accuracy为辅助；非法生成不删除，计错并构成真实类别FN。

数据划分、少样本、预热与延迟子集种子67；训练42／43／44；bootstrap67；图中代表运行42。提示为固定四条示例、关闭思考、贪心解码，最多8个新token。三种子按均值和样本SD报告，不按测试挑种子。完整种子索引见 configs/reproduction.json。

已有数据、预测和日志按本人2026-10-28的授权统一公开。来源许可核验的历史限制继续披露于数据清单和报告；账户密钥、个人路径配置及可重建缓存不属于交付文件。模型大权重保留在个人工具工作区，另行打包时核对身份。

## 如何判断“复现通过”

| 层次 | 验收内容 |
|---|---|
| 已有结果核查 | 一键命令完成，16份记录与11份测试指标、种子和哈希通过 |
| 完整离线分析 | 在锁定分析环境完成软件检查和22组图，主值与核查结果一致 |
| 新机真实模型复现 | 下载真实产物，在新环境加载并重新预测，保存命令、日志与结果差值 |
| 教师／AI验收 | 保存其实际机器、代码版本、执行层次及通过／失败记录 |

前两项不能替代后两项。若教师／AI报错，保存首次反馈时间、完整错误和环境，在48小时内提供修复提交或环境说明，再用新目录复验；不覆盖失败，不更换数据或种子凑结果。当前实际状态见[交付验收记录](docs/reproduction/交付验收记录-20261029.md)。

GitHub自动核查配置位于.github/workflows/reproduction.yml，只运行无模型核查；是否通过以实际Actions记录为准，不等同于教师验收。

## 代码与证据在哪里

| 位置 | 内容 |
|---|---|
| reproduce.py、src/reproduction.py | 可读的一键核查、环境检查、离线分析与状态记录 |
| configs/reproduction.json | 冻结数据、主值、16份运行、种子及输入哈希 |
| src/evaluation/metrics.py | 唯一指标定义、严格0/1解析及完整性检查 |
| src/adapters/、src/legacy/ | 三方代码接入与原训练代码归档 |
| tests/ | 指标、来源保护、参数图与复现失败边界的软件检查 |
| data/manifests/、results/predictions/ | 冻结数据清单与逐样本预测 |
| results/evidence/、logs/ | 原始训练日志与工程检查 |
| reports/analysis/、reports/figures/ | 正式比较、案例、模型／参数确认结论及图表 |

阅读材料：[完整复现指南](docs/reproduction/README.md)、[环境E01](docs/environment/实验运行环境与延迟口径.md)、[模型结论](reports/analysis/conclusions-confirmed-20261029-01/模型方案对比结论.md)、[参数结论](reports/analysis/conclusions-confirmed-20261029-01/参数对比结论.md)、[九例确认稿](reports/analysis/case-confirmed-20261029-01/正文案例与局限定稿.md)、[图表总览](reports/analysis/comparison-formal-20261028-03/图表总览.html)。

课程报告素材见[证据汇总](docs/报告素材/九案例至技术选型结论与证据汇总-20261029.md)，当前[报告草稿v3](docs/报告初稿/20261029-v3/项目报告.md)及[三人贡献声明](docs/报告初稿/20261029-v3/README.md)仍保留待补与待审核标记。

本轮AI用于工程入口、依赖和文档整理及被授权的离线验证。已确认的学生分析保留，382条AI辅助标注与20条本人试标来源分开，124条待复核不改正式标签。旧README原字节保存在 docs/history/README-before-reproduction-20261029.md，便于追溯历史哈希。
