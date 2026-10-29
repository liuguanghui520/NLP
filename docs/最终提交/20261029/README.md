# 中文评论情感分析：源码、报告与复现

提交日期：2026年10月29日。组长刘广徽20245726；杨函齐20246005、李嘉谊林20245816。代码主分支为[main](https://github.com/liuguanghui520/NLP)。本包为[复现修订版](https://github.com/liuguanghui520/NLP/releases/tag/course-submission-20261029-04)，包含环境、下载及路径兼容修复；以包内`src/提交清单.json`的文件身份为准。[原课程提交版Release](https://github.com/liuguanghui520/NLP/releases/tag/course-submission-20261029-01)保留原始发布记录。

## 正式复现的限制与验收边界

| 项目 | 本提交版的具体要求或已验证范围 |
|---|---|
| 普通CPU核查 | Python3.11以上，可运行`python src/run.py`独立重算已有预测主表；不加载模型，不代表完成真实模型重预测 |
| 完整真实模型复现 | 只推荐已经通过严格核查的原环境路径：WSL2 Ubuntu22.04.5、Python3.11.17、PyTorch2.14.1+cu130、Transformers5.8.0、PEFT0.18.1，以及158项冻结依赖和固定Factory commit |
| 已验证硬件 | RTX4070 Laptop、8GB显存、16GiB宿主机内存；单卡串行、batch1。该配置是实际验证范围，不是已证明的最低配置，也不保证任意CPU／显卡可完成全部模型 |
| 驱动与精度 | 需支持CUDA13运行时及GPU BF16；按预检实算确认。BERT用GPU FP32，Qwen用GPU BF16；不自动改CPU、FP16、量化或更小基座。系统安装Toolkit不能替代兼容驱动 |
| 资源与网络 | 需取得原Qwen基座约4.08GB及9份训练产物ZIP合计约1.19GB，并另留解压、依赖安装和输出空间；具体文件以SHA256确认。提交ZIP不含这些大文件，首次下载需要网络 |
| 软件与种子 | 同PyTorch版本或同种子均不能单独证明一致；须同时核查构建、完整依赖、模板、精度、数据和权重，并以实际预测验收 |
| 尚未通过的兼容方案 | torch2.5.1／CUDA12.1的Qwen结果未达到原主结果一致标准；同PyTorch版本的CUDA12.6候选尚未完成严格验证。它们不进入正式推荐，也不写为复现成功；CPU部分模型或少量样本检查不能替代11次全量测试 |
| 严格验收 | 11次运行各777条；对应种子及主表指标差不超过1e-12，逐样本原始输出和解析标签无差异，才记`passed`。指标相同但预测不同也不通过；不改原分数或参考预测迁就新环境 |

**计算栈差异与贪心推理的影响。** Qwen采用贪心解码（`do_sample=False`）：每一步选择得分最高的token。即使模型权重、输入和随机种子相同，PyTorch、CUDA、cuDNN等计算栈的变化，也可能改变浮点运算顺序、精度与舍入，进而改变候选token得分。候选分数接近时，最高分token可能改变；自回归生成又会把该token作为后续输入，使后续输出、非法输出率及最终指标发生变化。固定种子控制随机数，不能消除这类浮点计算差异。因此，版本号或种子相同都不能替代实际全量验证，发现差异仍按未通过一致性验收处理。原理依据见[PyTorch数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)。

已有原机严格重预测证据为11次×777条、指标差0、逐样本输出差0。另一台机器的回传只完成已有预测核查，真实模型尝试因旧驱动在加载前停止；原机通过不替代教师或组员的新机验收。`completed_with_differences`表示运行完成但主结果未通过一致性验收，不能当作成功提交。

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

1. 模型环境使用WSL2/Linux、Python **3.11**。原机为3.11.17、PyTorch **2.14.1+cu130**、Transformers5.8.0、PEFT0.18.1、scikit-learn1.9.1。`src/project/requirements.integration.lock`现包含原机158个依赖的完整冻结清单，包括`datasets==4.0.0`、`accelerate==1.11.0`、`trl==0.24.0`；框架单独安装。原清单仅有10项，已修复。完整模型模式要求CUDA与BF16，显卡8GB；不自动改精度、模型或依赖版本。
2. 获取[Qwen3-1.7B基座](https://modelscope.cn/models/Qwen/Qwen3-1.7B)，固定文件身份见`src/project/artifacts/lora-lock.json`。模型会逐文件验SHA256；MacBERT使用发布的已训练分类器，不使用未微调基座替代。
3. [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)固定为commit **c683e59edd8f893abc5e1403184a1d0644817d2a**。工具不塞入本项目，填写自己的目录；版本号同为0.9.6.dev0不代表源码相同。预检会核对commit、源码修改及`ChatModel`实际导入。
4. 将`src/project/configs/reproduce-models.example.json`复制为`reproduce-models.local.json`，填入本机Qwen目录及LLaMA-Factory目录。不要把示例中的`null`保留为真实路径。

```json
{
  "qwen_model": "/home/yourname/models/Qwen3-1.7B",
  "factory_root": "/home/yourname/LLaMA-Factory",
  "trained_artifacts_root": "/home/yourname/nlp-trained-artifacts"
}
```

`trained_artifacts_root`应包含9个解压目录，如`bert-len512-seed42/model/`、`tfidf-char_1_3-seed42/model/`及`qwen-lora-T01-seed42/model/`。缺失的已训练产物会从[GitHub Release](https://github.com/liuguanghui520/NLP/releases/tag/trained-artifacts-20261029-01)下载并校验，9个压缩产物合计约1.19GB；Qwen基座约4.08GB，另行获取。所有下载都发生在使用者本机，不属于提交ZIP。

**驱动前提：** PyTorch的cu130构建需要支持CUDA13的NVIDIA驱动。CUDA13.x的兼容下限为R580系列；实际以CUDA和BF16小矩阵运算通过为准，参见[NVIDIA兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。回传机器的驱动537.70／CUDA12.2不能运行本构建。先从NVIDIA或整机厂商安装与显卡匹配的CUDA13兼容正式驱动并重启，再启动WSL；安装CUDA Toolkit不能替代驱动升级。不要求与原机驱动版本完全相同。

在WSL中从解压目录安装到**新的独立环境**。以下使用已安装的uv管理Python，`$HOME`会展开为各人的用户目录：

```bash
uv python install 3.11.17
uv venv --seed --python 3.11.17 "$HOME/.venvs/nlp-course-fix"
git clone https://github.com/hiyouga/LLaMA-Factory.git "$HOME/LLaMA-Factory-nlp"
git -C "$HOME/LLaMA-Factory-nlp" checkout c683e59edd8f893abc5e1403184a1d0644817d2a
uv pip install --python "$HOME/.venvs/nlp-course-fix/bin/python" -r src/project/requirements.integration.lock -e "$HOME/LLaMA-Factory-nlp"
source "$HOME/.venvs/nlp-course-fix/bin/activate"
python -m pip check
```

已存在的环境或工具目录不要覆盖，换一个新目录名。安装源可换成能够访问的PyPI镜像，保留安装命令与日志，固定版本不变。如果没有uv，按[uv官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)准备；已有Python3.11也可用`python3.11 -m venv`创建环境，再在该环境中用pip安装同一冻结清单和固定框架源码。

填好前述路径JSON后，先执行快速检查，无需下载数据或加载权重：

```text
python src/run.py --preflight --paths src/project/configs/reproduce-models.local.json
```

控制台打印新工作目录，`preflight.json`逐项记录驱动、PyTorch CUDA构建、BF16运算、包版本、框架commit、导入和`pip check`。失败返回1并直接列出缺项；通过后再运行下方完整命令。完整命令也会自动执行同样的前置检查。

在上述模型环境中运行：

```text
python src/run.py --models --paths src/project/configs/reproduce-models.local.json
```

这条命令先检查环境、核对冻结主表，再准备各人的本地路径和选择元数据，依次加载已训练模型、重新生成11份测试预测。原模型文件不修改；路径重建只写入新工作目录。Qwen保持BF16、关闭思考、贪心、最多8个新token；LoRA仍为普通LoRA、T01、q/k/v/o、r8、alpha16，推理不放示例。

网络不稳定时，可复用已经下载成功的工作目录或完整仓库：在命令末尾增加`--input-cache /你的已有目录`，程序逐个校验47份输入后复制到新工作目录，不修改缓存。权重ZIP可直接放在`trained_artifacts_root`中。下载失败会自动重试；如主动选择镜像，可加`--download-prefix https://你选择的镜像域名/`，仍须匹配原始SHA256。无需访问GitHub Contents API，不消耗其匿名限流额度。

下载和损坏文件记录在`download-events.jsonl`；不完整或哈希错误的文件以`.failed-<唯一ID>`保留，绝不作为权重加载。训练产物必须通过整包SHA256、ZIP CRC及成员SHA256，再从临时目录完成解压。禁止把多次Range请求随意拼接后仅按文件大小验收。

额外输出：`model-stdout.log`、`model-stderr.log`、`results/model-reproduction/metrics.json`、`prediction-differences.jsonl`和`status.json`。各模型目录`results/local/reproduced-.../`保存原始输出、777条预测、指标、配置、延迟与资源。逐种子与方案均值均对照冻结参考，逐样本原始输出和解析标签也须一致；`passed`表示两项均通过，`completed_with_differences`表示11次预测完整执行但指标不同；两层状态会保留同一状态和实际差值，返回1。`failed`表示流程失败，`interrupted`表示中断；错误信息说明具体原因，不与指标差异混淆。

## 4. 完整代码与依赖

`src/project/src/`为统一CLI、模型接入、指标、分析、演示和三条原路线源码；`scripts/`为数据处理、训练及分析工具；`configs/`为冻结参数与路径示例；`tests/`为软件测试。上游框架、模型、数据均为外部依赖。

在已安装依赖的模型环境中，可于`src/project/`运行`python -m unittest discover -s tests -v`。精简包保留可独立构造测试输入的工程检查；依赖整份历史错例／资源证据的检查和发布工具检查留在完整仓库，避免数据不入包却要求本地读取的矛盾。完整仓库本轮66项软件检查通过；这与模型效果及新机验收分别记录。

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

验证状态：原机已完成11次真实模型预测，各777条，主指标与逐样本输出差异均为0。回传新机只完成已有预测核查；其模型流程因旧驱动在加载前停止，完成0/11次，不记作完整模型通过。教师／组员新机的实际环境、命令、退出码和结果差异须按本README留证。

若教师或AI复现报错，保留首次失败记录，48小时内补交环境说明或修复，并以全新目录重验；不覆盖失败、不换数据、不挑选种子。原机验证与组员新机验证分别记录。
