# 中文评论情感分析：复现与证据入口

本项目比较TF-IDF＋逻辑回归、预训练MacBERT全参数微调、Qwen3-1.7B零样本、固定四样本和普通LoRA T01。统一数据、种子与评分口径，保留原始预测、日志和失败记录。

**已完成离线复现与9个训练产物的公开发布。默认命令重算已有预测，不加载模型。真实模型新机复现和教师验收单独记录。** “48小时补交”是复现失败后的补救期限，不能代替资源和环境准备。

## 正式复现边界

完整模型复现只推荐已核验的原环境（WSL2、Python3.11.17、torch2.14.1+cu130、原冻结依赖、支持BF16的GPU）；实测RTX4070 Laptop 8GB及宿主机16GiB，不能视作通用最低配置。普通CPU的默认命令只独立重算已有预测。其他CUDA／PyTorch兼容尝试没有达到一致性要求，不作为正式推荐。主结果以4070／4060两机实测为基准：TF-IDF、MacBERT逐样本一致，Qwen按硬件分别报告成绩与偏差；不把跨硬件逐token一致作为全部实验结论成立的前提。Qwen为贪心推理，计算栈的浮点运算与舍入差异可能改变高分token及后续输出；固定种子不能消除该差异。具体限制见[提交包README](docs/最终提交/20261029/README.md)。

## 下载后，一条命令核查主结果

2026-10-30回传更新：RTX4060 Laptop已完成11次全量预测，TF-IDF及MacBERT逐样本一致；Qwen零样本72条、四样本7条、LoRA合计6条输出不同，按硬件分别报告。验证诊断中输入token、权重、buffer一致，已观察到首步得分及token排序变化；4060同机重复一致，受控后端未消除跨机分歧。主报告已将跨机差异与BF16局限纳入主结果；FP32仅有小规模试测，不替换正式成绩。见[定位与诊断](docs/reproduction/4060复现差异定位与诊断-20261030.md)。

WSL安装注意：虚拟环境和LLaMA-Factory放在`/home/用户名/`等原生目录。预检会提前拒绝位于Windows挂载目录的环境或框架，并给出重建说明；提交包源码可留在Windows目录。若曾出现导入超时，按[提交包README的处理步骤](docs/最终提交/20261029/README.md#Windows挂载目录超时处理)重建环境，不直接移动虚拟环境。

安装Python 3.11及以上（已测3.13.9），取得本分支：

~~~text
git clone --branch main --single-branch https://github.com/liuguanghui520/NLP.git
cd NLP
python reproduce.py
~~~

也可下载[分支ZIP](https://github.com/liuguanghui520/NLP/archive/refs/heads/main.zip)再解压运行。Windows支持命令行调用reproduce.cmd。默认核查无需pip安装、CUDA、模型、作者的WSL目录或个人路径配置。

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

完整模式生成新ID的22组PNG/PDF、参数、统计和错误材料，同时运行软件检查；仍然不训练或推理。日志和新分析位置写入本次status.json。requirements.analysis.lock对应离线分析；requirements.integration.lock现已补为原机158个包的完整冻结清单，适用于Python3.11的WSL/Linux模型环境，不能与Windows绘图环境混用。

## 真实模型复现：资源齐备后另验收

真实预测需要Qwen基座、三个BERT最佳模型、三个TF-IDF Pipeline及三个普通LoRA适配器。9个训练产物现已发布至[GitHub Release](https://github.com/liuguanghui520/NLP/releases/tag/trained-artifacts-20261029-01)，压缩包共约1.19GB，GitHub服务端SHA256与本地一致，匿名下载可达。Qwen基座另从官方来源获取。2026年10月29日回传表明已有预测重算通过，指标差0；真实模型模式因复现机537.70驱动不支持cu130而中止，完成0/11次。修复说明见[复现失败定位与修复](docs/reproduction/复现失败定位与修复-20261029.md)。

代码、配置和小型证据随仓库下载；训练产物按[下载说明](docs/reproduction/训练产物下载说明-20261029.md)获取，[机器可读清单](artifacts/reproduction-downloads.json)列出9个ZIP的URL、大小和SHA256。模型包含相对目录及配置，不包含Qwen基座和Python环境。资源身份见[模型资源说明](artifacts/README.md)、artifacts/lora-lock.json及基线记录；历史说明以本README和最新复现说明为准。

修复版已在原机完成11次真实预测，各777条，逐种子及主表指标差0；完整仓库65项、随包48项检查通过。新机仍须更新驱动后重跑。验证证据见[修复说明](docs/reproduction/复现失败定位与修复-20261029.md)。

原模型环境完整冻结为158个包，另固定LLaMA-Factory commit。首次搭建按[提交包README的模型环境步骤](docs/最终提交/20261029/README.md)安装。CUDA13构建需要兼容驱动；完整入口先预检再下载。仓库根目录也可执行：

```text
python submission_reproduce.py --preflight --paths configs/reproduce-models.local.json
python submission_reproduce.py --models --paths configs/reproduce-models.local.json --input-cache .
```

路径JSON由`configs/reproduce-models.example.json`复制并填写各人的地址。两个命令分别建立新目录；前一条只检查环境，后一条完整执行11次预测。输入缓存只读并逐文件校验，避免重复下载。

高级入口 src/cli.py 支持check、import、predict、evaluate、benchmark、summarize等；这些历史命令依赖外部工具资源，不能替代上面的无模型一键核查。真实预测在资源齐备的独立副本进行，复制 configs/paths.example.json，填写自己的路径，按实际环境和资源清单验收。重新训练不属于默认复现操作。

## 数据、种子与指标

冻结版本v1-20261023：训练6210、验证776、测试777；0=负面、1=正面。所有路线共享前512个Unicode字符，再按各自token预算处理。验证Macro-F1选优，Accuracy为辅助；非法生成不删除，计错并构成真实类别FN。

数据划分、少样本、预热与延迟子集种子67；训练42／43／44；bootstrap67；图中代表运行42。提示为固定四条示例、关闭思考、贪心解码，最多8个新token。三种子按均值和样本SD报告，不按测试挑种子。完整种子索引见 configs/reproduction.json。

已有数据、预测和日志按2026-10-28的授权统一公开。来源许可核验的历史限制继续披露于数据清单和报告；账户密钥、个人路径配置及可重建缓存不属于交付文件。模型大权重保留在个人工具工作区，另行打包时核对身份。

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

## 正式提交材料

最新版ZIP见[课程提交版Release](https://github.com/liuguanghui520/NLP/releases/tag/course-submission-20261029-04)；报告摘要和三份实名贡献声明已同步扩写，独立项目摘要300字。训练产物仍使用原固定Release链接。

报告、PDF与独立摘要使用同一份300字文本。GitHub Release镜像文件名为`20245726_liuguanghui_chinese_sentiment.zip`；内容及SHA256与下方中文命名的课程ZIP相同。下载镜像后按Release说明改为课程要求的中文名称提交。

- [最终项目报告及三份贡献声明](docs/最终提交/20261029/README.md)，提交日期2026年10月29日。
- [报告PDF](output/pdf/报告-20261030.pdf)：正文12页，附录10页；图片随PDF内嵌。
- [课程提交ZIP](deliverables/20245726_刘广徽_中文评论情感分析.zip)：根目录仅报告、摘要、README、src及三份个人声明。权重、录屏、数据集不入包。
- 解压提交ZIP后执行`python src/run.py`重算主表；配置模型环境与资源后执行`python src/run.py --models --paths src/project/configs/reproduce-models.local.json`重新预测11个测试运行。
- 新机回传：已有预测核查通过，模型预测因旧驱动失败；修复版本增加预检、完整依赖、缓存及下载校验。原机复验与组员新机验收分别留证。

交互演示使用Cherry Studio连接本地服务，说明见[演示材料](docs/demo/README.md)。[演示录屏](https://pan.baidu.com/s/10l-SBqKrzkCp4hh3KQ8XAA?pwd=668w)，提取码`668w`，111.317秒。

当前协作与交付入口为main，保留融合分支和原有提交历史；历史过程文档不进入课程ZIP。
