# 复现交付与验收

本说明区分“核对已有结果”“真实模型预测”“重新训练”。本轮本人选择先完成交付准备和离线复现，没有授权本轮启动模型预测或训练。教师／AI最终验收仍需记录其实际执行范围。

## 1. 从GitHub取得文件

仓库：[liuguanghui520/NLP](https://github.com/liuguanghui520/NLP/tree/codex/integration)，当前工作分支为`codex/integration`。

```text
git clone --branch codex/integration --single-branch https://github.com/liuguanghui520/NLP.git
cd NLP
python reproduce.py
```

也可下载[融合分支ZIP](https://github.com/liuguanghui520/NLP/archive/refs/heads/codex/integration.zip)，解压到自己的目录后执行相同命令。分支会继续变化；每次验收应保存具体提交号或源码清单，不能只写“下载最新版”。

默认命令只用Python标准库，支持Python3.11及以上，已测解释器3.13.9。不需要CUDA、模型权重、作者的WSL目录或`configs/paths.local.json`。运行目录可以包含中文和空格。Windows也可在命令行执行`reproduce.cmd`。

## 2. 默认一键核查做什么

1. 核对数据清单、正文、Alpaca、划分及子集的SHA256。
2. 检查6210／776／777条、ID顺序、标签定义、跨集ID和模型可见前缀重复。
3. 核对冻结的16份验证／测试预测，并调用统一评估函数独立重算指标。
4. 对11份最终测试结果，检查训练种子42／43／44和提示单次设置，重算各方法均值与样本SD。
5. 比较主结果，容差为1e-12；不按测试换配置、种子或标签。
6. 写入全新`results/reproduction/<批次>/`，保留命令、实际Python、代码哈希、配置、输入哈希、指标、结果表、状态或错误日志。

| 方法 | 测试N／每运行 | Macro-F1均值 | Accuracy均值 | 种子 |
|---|---:|---:|---:|---|
| TF-IDF＋逻辑回归 | 777 | 0.86104046 | 0.88416988 | 42／43／44 |
| MacBERT全参 | 777 | 0.91518235 | 0.92621193 | 42／43／44 |
| Qwen零样本 | 777 | 0.62558894 | 0.59588160 | 确定性单次 |
| Qwen四样本 | 777 | 0.76104902 | 0.76576577 | 确定性单次 |
| 普通LoRA T01 | 777 | 0.91116020 | 0.92277992 | 42／43／44 |

精确值以`configs/reproduction.json`及输出`metrics.json`为准，表中只展示舍入值。只有输出`status.json`的`status=passed`且退出码0，才表示本次指定层次核查通过；非空错误日志需要处理。已有输出目录会被拒绝，不自动覆盖。

可进一步验证默认入口没有依赖第三方包：

```text
python -S reproduce.py --output results/reproduction/my-stdlib-audit
```

`-S`禁用site初始化；每次换新目录。输出中明确保留`model_loading=false`、`inference=false`、`new_machine_model_reproduction=not_executed`，不将已有预测重算写成真实模型复现。

## 3. 完整离线分析：图表和软件检查

这一步仍不加载模型。使用独立分析环境，版本以本轮实际运行环境锁定；不改变原有模型环境。

```text
python -m venv .venv-analysis
.venv-analysis\Scripts\python -m pip install -r requirements.analysis.lock
.venv-analysis\Scripts\python reproduce.py --analysis
```

创建venv时应使用Python3.13.9。Linux对应解释器为`.venv-analysis/bin/python`。上述安装命令是新机操作指南，是否完成全新安装以验收记录为准，不能从本机已有依赖推定新机安装成功。

本轮已在同一台Windows机器的新venv中完成锁定包安装、`pip check`和独立源码目录的离线分析。首次创建环境遇到临时目录权限不足时，保留失败记录，重试仅将当前进程TEMP／TMP指向本次项目可写临时目录；不修改系统全局设置。详情见[验收记录](交付验收记录-20261029.md)。

还需Git和中文字体：Microsoft YaHei、Noto Sans CJK SC、SimHei三者至少一种。缺包、版本不符或字体不足时入口报错，不自动升级环境。默认轻量核查不要求这些绘图依赖。

完整模式先核查数据／指标，再运行软件检查，以新分析ID生成22组PNG/PDF、参数表、配对统计和错误材料，最后核对新报告与独立重算的主指标。命令、stdout/stderr、软件测试日志和新分析目录均记录在`status.json`中。字体和平台可能影响图像字节，结果一致性以数据、指标、配置和输出完整性判断，不要求跨平台PNG哈希完全相同。

## 4. 依赖、配置与种子

| 层次 | Python／依赖 | 配置与边界 |
|---|---|---|
| 默认核查 | Python≥3.11；标准库 | 不导入PyTorch、Transformers、scikit-learn或绘图库 |
| 完整离线分析 | 已测Python3.13.9；`requirements.analysis.lock` | NumPy／SciPy／Matplotlib及依赖精确锁定，见`configs/reproduction-analysis-environment.json` |
| 历史模型运行 | WSL Python3.11.17；`requirements.integration.lock`及环境E01 | 该锁为已测模型环境快照，不能直接当成Windows分析环境的安装清单；LLaMA-Factory及CUDA构建还需工具环境证据 |

数据划分、少样本抽样、预热子集、延迟子集均为67；训练种子42／43／44；bootstrap为67；图中代表运行42。提示关闭思考、贪心解码、最多8个新token。种子索引保存在`configs/reproduction.json`，不重划数据。

## 5. 真实模型复现的资源准备

真实预测需要原始基座、3个BERT最佳模型、3个TF-IDF Pipeline、3个普通LoRA T01适配器和对应tokenizer／配置。资源身份清单见`artifacts/lora-lock.json`、`qwen-prompt-model-lock.json`、基线审计与逐运行检查点哈希；[下载状态清单](../../artifacts/reproduction-downloads.json)明确区分已随仓库发布与尚未发布的产物。

9个已训练产物已发布至[GitHub Release](https://github.com/liuguanghui520/NLP/releases/tag/trained-artifacts-20261029-01)，服务端SHA256、字节大小与匿名下载可达性均已核对。完整列表及下载／校验步骤见[训练产物下载说明](训练产物下载说明-20261029.md)。下载完成不等于模型已加载或完成真实预测；Qwen原始基座仍需另取，原始基座也不能替代已训练产物。

`configs/paths.example.json`供各成员填写自己的位置，`configs/paths.local.json`为个人运行配置。高级历史CLI依赖外部工具工作区；本轮默认入口已移除此类前置依赖。真实预测应在资源齐备的独立工作副本执行，使用全新run_id，并核对样本数、模型／数据／提示哈希、逐样本输出和指标。重新训练另行授权，不属于默认验收。

## 6. 教师／AI验收与48小时修复

“跑不通”条款要求的是教师／AI按README完成实际执行。48小时是失败后的补救窗口，不是提前省略依赖、资源或安装验证的理由。

验收记录至少保存：执行人／工具、日期、机器与系统、代码提交或源码哈希、下载来源、环境安装记录、完整命令、退出码、stdout/stderr、样本数量、指标与主结果差值。分别填写：

| 验收项 | 能证明什么 | 当前不能替代什么 |
|---|---|---|
| 默认核查通过 | 仓库内数据、预测、指标和种子可核查 | 不证明模型加载和真实预测 |
| 独立源码目录离线通过 | 不依赖作者绝对目录完成指定离线工作 | 不证明全新依赖安装或另一台机器 |
| 新机真实模型预测通过 | 按资源、环境与命令重新预测 | 不等于重新训练逐样本完全一致 |
| 教师／AI实际验收 | 以其实际执行层次与记录为准 | 未执行的层次不能打勾 |

如果教师／AI报错：保存原失败输出和首次反馈时间，定位下载、环境、路径、代码或输入身份问题；在48小时内提交环境说明或修复提交，使用新输出目录重验，保留失败与重试关联。不得为“跑通”更换数据、标签、基座或挑种子，也不覆盖原失败记录。当前尚无教师首次失败时间，不虚填倒计时或通过签字。

## 7. 代码阅读与AI记录

`reproduce.py`是入口，`src/reproduction.py`按路径边界、数据检查、指标重算、环境检查、软件测试、离线图表及状态记录分函数；指标定义仍只使用`src/evaluation/metrics.py`。参数、种子、预期主值和依赖外置为配置，便于阅读与核查。

`tests/test_reproduction.py`覆盖缺种子、数据篡改、跨集重复、已有目录拒绝覆盖、失败留档及无需个人模型路径等交付风险。旧训练框架源码保留归档，不声称本轮已重构全部历史代码。

本轮AI辅助：编写可读入口和验收检查、整理依赖与README、执行被授权的离线核查。没有启动模型推理或训练；真实模型、新机和教师验收状态如实保留。最新实际执行记录见同目录的交付验收记录。
