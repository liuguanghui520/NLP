# 中文酒店评论情感分析：提交包与复现

课程方向C：提示与微调对比。成员：刘广徽20245726（40%）、杨函齐20246005（30%）、李嘉谊林20245816（30%）。比例由组长提供，职责与三份个人声明一致。

这是任务书对齐后的候提交包。三名成员仍需完成个人声明、整稿核对和真实模型新机验收。未填写的确认项不视为已经签署；当前材料不声称教师验收通过。

## 1. 解压后的一条命令

准备Python **3.11及以上**及可访问GitHub的网络，进入解压目录执行：

```text
python submission_reproduce.py
```

默认核查只使用Python标准库，不需要安装第三方依赖、GPU或模型。入口将必要数据和已有预测下载到系统临时目录下的全新`nlp-course-<唯一ID>`工作目录，再运行统一评估。控制台会打印实际位置。下载源固定为Git提交`106cdbed8db84057dba6b9c980d3a2f914856dd0`，47个输入逐一检查SHA256，配置保存在`configs/reproduction.json`。当前ZIP包含最新代码和报告，固定提交只用来提供已经发布的冻结实验输入，不代表本轮修改已推送。

需要指定工作位置时，选择**解压目录之外、尚不存在**的目录：

```text
python submission_reproduce.py --work-dir C:/NLP-reproduce/check-001
```

不能覆盖旧目录；网络失败、哈希不一致或指标不一致会返回非零退出码并保留`submission-status.json`及已有下载。修复后换全新目录重跑。网络代理使用操作系统或Python标准代理配置，不在代码中填写账户或密钥。默认操作会在本机下载数据证据；这些文件不在提交ZIP中。

## 2. 预期输出与主表

下载工作目录中保存`stdout.log`、`stderr.log`、`submission-status.json`，其下`results/reproduction/<唯一ID>/`保存状态、逐运行指标、主表、配置、输入哈希及实际环境。

通过条件：外层`submission-status.json`为`passed`、内层`status.json`为`passed`、16份运行完整、11份测试每份777条、指标最大差为0。退出码0才表示本次离线核查完成。

| 方案 | 测试Macro-F1 | Accuracy | 训练种子 |
|---|---:|---:|---|
| TF-IDF＋LR | 86.10% | 88.42% | 42、43、44 |
| MacBERT全参 | 91.52% | 92.62% | 42、43、44 |
| Qwen零样本 | 62.56% | 59.59% | 确定性单次 |
| Qwen固定四样本 | 76.10% | 76.58% | 确定性单次 |
| Qwen普通LoRA T01 | 91.12% | 92.28% | 42、43、44 |

训练路线先逐种子计算指标，再平均。多数类参考Macro-F1为40.64%、Accuracy为68.47%。标签0负面、1正面；非法生成算错，不从分母删除。精确值与容差`1e-12`见配置，不要求舍入后字符串逐字相同。

数据6210／776／777；字符上限512；数据划分、少样本、冒烟、延迟子集和bootstrap种子67，训练种子42／43／44。Qwen关闭思考、贪心、最多8个新token。

## 3. 环境与不同复现层次

| 层次 | 环境／命令 | 能验证的范围 |
|---|---|---|
| 本包默认入口 | Python≥3.11标准库；上面的一条命令 | 已有预测、数据身份、主指标及种子独立重算 |
| 完整离线图表 | 完整仓库；Python3.13.9，`requirements.analysis.lock`；`python reproduce.py --analysis` | 22组图、统计与软件检查；需要完整的历史日志和证据目录 |
| 真实模型预测 | WSL2 Ubuntu22.04.5、Python3.11.17、环境E01，见`docs/environment/`及`requirements.integration.lock` | 加载权重重新预测；当前新机全量验收待完成 |

当前默认入口不重新训练、不加载模型、不验证模型输出重现。教师／AI如验收真实模型主结果，需要执行下一节，不能用离线重算替代。锁定文件分别对应分析和模型环境，不能把WSL CUDA快照直接当作Windows安装清单。没有自动升级依赖或改为QLoRA。

## 4. 资源链接、完整图表与真实预测

- [源码、冻结数据、预测与原始日志](https://github.com/liuguanghui520/NLP/tree/106cdbed8db84057dba6b9c980d3a2f914856dd0)：固定Git提交。
- [该版本完整仓库下载](https://github.com/liuguanghui520/NLP/archive/106cdbed8db84057dba6b9c980d3a2f914856dd0.zip)：在本机解压后可按仓库README运行完整分析；它是外部资源，不作为第二份课程提交ZIP上传。
- [9份已训练产物](https://github.com/liuguanghui520/NLP/releases/tag/trained-artifacts-20261029-01)：TF-IDF、MacBERT、LoRA各三个种子。下载清单、SHA256与解压说明在`artifacts/reproduction-downloads.json`和`docs/reproduction/训练产物下载说明-20261029.md`。
- [Qwen3-1.7B基座](https://huggingface.co/Qwen/Qwen3-1.7B)、[MacBERT来源](https://huggingface.co/hfl/chinese-macbert-base)：完整真实预测使用已训练MacBERT；不能拿未微调基座替代。
- [原始数据来源](https://modelscope.cn/datasets/AiNiklaus/ChnSentiCorp)：本项目使用酒店CSV并重新分层划分，不是论文原划分。来源与清洗、许可核查见报告。

真实模型环境安装及获取资源后，将`configs/paths.example.json`复制成`configs/paths.local.json`，填写**使用者自己的**模型、数据、适配器及LLaMA-Factory工具位置。上游工具不并入本包，不要求使用作者的个人目录。以完整仓库的独立工作副本执行：

```text
python -m src.cli predict --method qwen_lora --seed 42 --split test --output results/local/new-machine-lora42-001
```

每运行应输出777条预测、指标、配置与状态。该命令只覆盖一个模型种子，完整主表需核对三种子训练路线与两份提示结果，使用新运行ID保存实际差异。具体入口与当前资源检查限制见`docs/reproduction/README.md`。全部真实模型预测涉及GPU运行，本次只准备说明与离线核查，没有代填通过结果。

## 5. 包内结构

```text
报告.pdf                                  # 正文12页，附录另计
项目摘要.md                              # 275字，无图表
README.md                                # 本说明
贡献声明_20245726_刘广徽.md
贡献声明_20246005_杨函齐.md
贡献声明_20245816_李嘉谊林.md
src/                                     # 全部融合源码与原路线代码
scripts/                                 # 训练、分析、审计与打包等入口
configs/                                 # 冻结参数、种子、路径示例
tests/                                   # 软件检查
reproduce.py / submission_reproduce.py
requirements.analysis.lock / requirements.integration.lock
docs/                                    # 环境、资源、复现与演示说明
artifacts/                               # 仅资源身份／下载JSON，不含权重
report-source/                           # 可编辑报告和架构图
任务书逐项核对.md / 提交前待办.md / 提交清单.json
```

不含数据集、逐样本正文预测、模型权重、checkpoint、视频、虚拟环境、密钥、个人路径配置或`.git`。完整原始运行证据由外链获取；报告只引用必要的案例片段。包内所有项目源码可读，上游LLaMA-Factory作为外部框架按其来源安装。

## 6. 演示与最终提交

[演示录屏](https://pan.baidu.com/s/12nyNQJliIKabozmBmfuR-g?pwd=9vxp)，提取码`9vxp`，文件`20245726_刘广徽_中文评论情感分析_演示.mp4`，111.317秒。已人工确认分享可访问；最终检查姓名学号、真实运行及难例画面，保留分享有效期。元数据与SHA256见`artifacts/demo-video.json`。

仅将`20245726_刘广徽_中文评论情感分析.zip`这一份ZIP上传课程收集表，视频填写网盘链接；≤50MB，截止2026-10-30 24:00：[金山文档收集表](https://f.kdocs.cn/g/5htZQGSM)。不要把训练产物的9个ZIP当作课程提交包上传。

若教师／AI不能按README复现，保存首次报错、环境、完整命令、退出码和日志，在48小时内补交环境说明或修复，以新目录重验；不覆盖失败，不换数据或挑种子。补交后仍跑不通，第③项0分、总分不超过25分。当前未收到教师首次失败记录，不预填验收通过或修复截止时间。
