# 中文评论情感分析：源码、报告与复现

提交日期：2026年10月29日。组长刘广徽20245726；杨函齐20246005、李嘉谊林20245816。代码主分支为[main](https://github.com/liuguanghui520/NLP)。最新版提交ZIP见[课程提交版Release](https://github.com/liuguanghui520/NLP/releases/tag/course-submission-20261029-01)。

## 1. 解压后的一条命令

准备Python **3.11及以上**与可访问GitHub的网络，在解压目录运行：

```text
python src/run.py
```

Windows也可执行`src\一键复现.cmd`。默认入口只使用Python标准库，无需GPU或第三方依赖；自动下载47份冻结输入、校验SHA256、检查数据划分及种子，调用统一指标代码，重新计算五种方案的测试主表。**该模式重算已有预测；重新加载模型预测使用第3节的命令。**

数据和已有预测不在ZIP内，下载来源锁定为[固定提交106cdbe](https://github.com/liuguanghui520/NLP/tree/106cdbed8db84057dba6b9c980d3a2f914856dd0)，配置为`src/project/configs/reproduction.json`。固定历史输入与代码主分支main各有用途，不因后续提交改变实验数据身份。

每次写入独立工作目录，默认在系统临时目录`nlp-course-<唯一ID>`，控制台打印实际地址。需要指定保存位置时，选择解压目录之外、尚不存在的目录：

```text
python src/run.py --work-dir C:/NLP-reproduce/check-001
```

Linux例如`--work-dir /home/yourname/nlp-check-001`。本地不存在的目录由程序创建；不覆盖旧结果。网络失败、文件哈希不一致或指标不一致时返回非零退出码，保留状态与错误。重试使用新目录，不更改数据、标签、模型或种子。

## 2. 预期输出、主要结果与种子

工作目录保存：

| 文件 | 含义 |
|---|---|
| `submission-status.json` | 下载、命令、整体状态 |
| `stdout.log`、`stderr.log` | 指标核查输出与错误 |
| `results/reproduction/<唯一ID>/status.json` | 运行数、实际Python、输入与代码哈希、最大指标差 |
| 同目录`metrics.json`、`结果核查.md` | 各运行和五方案主表、种子均值与样本SD |

验收条件：外层及内层状态均为`passed`，16份正式记录完整，11份最终测试各777条，主指标差为0，退出码0。容差`1e-12`；以下表格是舍入值，精确参考值在配置中。

| 方法 | Macro-F1 | Accuracy | 训练种子 |
|---|---:|---:|---|
| TF-IDF＋逻辑回归 | 86.10% | 88.42% | 42、43、44 |
| MacBERT全参微调 | 91.52% | 92.62% | 42、43、44 |
| Qwen3零样本 | 62.56% | 59.59% | 确定性单次 |
| Qwen3固定四样本 | 76.10% | 76.58% | 确定性单次 |
| Qwen3普通LoRA T01 | 91.12% | 92.28% | 42、43、44 |

训练方案先逐种子计算再平均；多数类参考Macro-F1为40.64%、Accuracy为68.47%。标签0负面、1正面；生成结果去首尾空白后只有完整0/1合法，非法输出保留分母并计错。训练／验证／测试6210／776／777，先取512 Unicode字符。数据划分、少样本、冒烟、延迟子集、bootstrap种子67，训练种子42／43／44。

## 3. 加载真实权重：一条命令重跑完整主表

该模式使用已训练产物重新预测，不重新训练；串行执行TF-IDF、MacBERT、LoRA各三个种子以及零／少样本，共11个完整测试运行。每个运行777条，再执行20条预热和固定200条延迟测量。

先配置以下环境和外部资源：

1. 已测模型环境：WSL2 Ubuntu22.04.5、Python3.11.17；PyTorch2.14.1、Transformers5.8.0、PEFT0.18.1、scikit-learn1.9.1、LLaMA-Factory0.9.6.dev0。完整快照为`src/project/requirements.integration.lock`。CUDA构建与硬件说明见[环境E01](https://github.com/liuguanghui520/NLP/blob/main/docs/environment/实验运行环境与延迟口径.md)。CPU核查使用第1节；完整模型模式要求支持BF16的CUDA设备，原运行显卡为RTX4070 Laptop 8GB，不自动改精度或换模型。
2. 获取[Qwen3-1.7B基座](https://modelscope.cn/models/Qwen/Qwen3-1.7B)，固定文件身份见`src/project/artifacts/lora-lock.json`。模型会逐文件验SHA256；MacBERT使用发布的已训练分类器，不使用未微调基座替代。
3. 安装或使用已兼容的[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，工具不塞入本项目。使用者填写自己的路径。
4. 将`src/project/configs/reproduce-models.example.json`复制为`reproduce-models.local.json`，填入本机Qwen目录及LLaMA-Factory目录。不要把示例中的`null`保留为真实路径。

```json
{
  "qwen_model": "/home/yourname/models/Qwen3-1.7B",
  "factory_root": "/home/yourname/LLaMA-Factory",
  "trained_artifacts_root": "/home/yourname/nlp-trained-artifacts"
}
```

`trained_artifacts_root`应包含9个解压目录，如`bert-len512-seed42/model/`、`tfidf-char_1_3-seed42/model/`及`qwen-lora-T01-seed42/model/`。缺失的已训练产物会从[GitHub Release](https://github.com/liuguanghui520/NLP/releases/tag/trained-artifacts-20261029-01)下载并校验，9个压缩产物合计约1.19GB；Qwen基座约4.08GB，另行获取。所有下载都发生在使用者本机，不属于提交ZIP。

在上述模型环境中运行：

```text
python src/run.py --models --paths src/project/configs/reproduce-models.local.json
```

这条命令先核对冻结主表，再准备各人的本地路径和选择元数据，依次加载已训练模型、重新生成11份测试预测。原模型文件不修改；路径重建只写入新工作目录。Qwen保持BF16、关闭思考、贪心、最多8个新token；LoRA仍为普通LoRA、T01、q/k/v/o、r8、alpha16，推理不放示例。

额外输出：`model-stdout.log`、`model-stderr.log`、`results/model-reproduction/metrics.json`和`status.json`。各模型目录`results/local/reproduced-.../`保存原始输出、777条预测、指标、配置、延迟与资源。`passed`表示完成且与冻结主表一致；`completed_with_differences`表示预测已完整执行但指标不同，程序保留差值并返回1，不能改参考值掩盖差异。环境、生成与数值精度问题按日志排查。

## 4. 完整代码与依赖

`src/project/src/`为统一CLI、模型接入、指标、分析、演示和三条原路线源码；`scripts/`为数据处理、训练及分析工具；`configs/`为冻结参数与路径示例；`tests/`为软件测试。上游框架、模型、数据均为外部依赖。

默认模式仅标准库。完整图表的已测分析环境为Python3.13.9及`requirements.analysis.lock`，与WSL模型环境分开；不要混装或自动升级原依赖。若需要完整历史图表复算，取得[完整仓库](https://github.com/liuguanghui520/NLP)，按根README在冻结证据齐全时运行`python reproduce.py --analysis`；默认轻量入口不依赖绘图环境。

## 5. 包内结构与外链

```text
报告.pdf
项目摘要.md
README.md
src/
  run.py
  一键复现.cmd
  提交清单.json
  project/                 # 全部工程源码、配置、依赖及测试
贡献声明_20245726_刘广徽.md
贡献声明_20246005_杨函齐.md
贡献声明_20245816_李嘉谊林.md
```

仅提交这一个ZIP，命名`20245726_刘广徽_中文评论情感分析.zip`，小于50MB；不含数据集、模型权重、checkpoint、视频、虚拟环境、密钥、个人运行配置或过程交接文档。

- [数据来源：ModelScope AiNiklaus/ChnSentiCorp](https://modelscope.cn/datasets/AiNiklaus/ChnSentiCorp)。酒店CSV重新分层划分，非论文原划分；许可与使用范围见报告第⑦节。
- [已训练权重和适配器：GitHub Release](https://github.com/liuguanghui520/NLP/releases/tag/trained-artifacts-20261029-01)。资源链接与SHA256也保存在`src/project/artifacts/reproduction-downloads.json`。
- [演示录屏：百度网盘](https://pan.baidu.com/s/12nyNQJliIKabozmBmfuR-g?pwd=9vxp)，提取码`9vxp`。文件`20245726_刘广徽_中文评论情感分析_演示.mp4`，111.317秒，Cherry Studio真实运行与难例对照。
- [完整代码、原始预测、日志及图表](https://github.com/liuguanghui520/NLP)。main为正式协作入口，固定输入仍按历史提交和哈希锁定。

报告摘要与独立项目摘要完全一致，均为300字，涵盖问题、实验协议、效果、成本及错误分析；三份贡献声明列明具体工作、核查步骤、40／20／15小时及对应姓名文字署名。

新机复现已完成，完成信息由组长于2026年10月29日确认。再次复现时请保留实际机器、日期、命令、退出码、输出与指标差异。若教师或AI复现报错，保留首次失败记录，48小时内补交环境说明或修复，并以全新目录重验；不覆盖失败、不换数据、不挑选种子。
