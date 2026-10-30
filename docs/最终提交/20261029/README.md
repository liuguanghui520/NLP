# 中文评论情感分析：源码、报告与复现

提交日期：2026年10月30日。组长刘广徽20245726；杨函齐20246005、李嘉谊林20245816。代码主分支为[main](https://github.com/liuguanghui520/NLP)。本包于2026年10月30日在[已发布复现修订版](https://github.com/liuguanghui520/NLP/releases/tag/course-submission-20261029-04)基础上增加WSL挂载路径预检、重建指引和数值运行记录；以包内`src/提交清单.json`的文件身份为准，已发布版本保留原始附件。[原课程提交版Release](https://github.com/liuguanghui520/NLP/releases/tag/course-submission-20261029-01)保留原始发布记录。

## 复现环境与两机结果基准

| 项目 | 本提交版的具体要求或已验证范围 |
|---|---|
| 普通CPU核查 | Python3.11以上，可运行`python src/run.py`独立重算已有预测主表；不加载模型，不代表完成真实模型重预测 |
| 完整真实模型复现 | 只推荐已经通过严格核查的原环境路径：WSL2 Ubuntu22.04.5、Python3.11.17、PyTorch2.14.1+cu130、Transformers5.8.0、PEFT0.18.1，以及158项冻结依赖和固定Factory commit |
| 已验证硬件 | RTX4070 Laptop（宿主机16GiB）及RTX4060 Laptop（宿主机约32GiB），均为8GB显存、单卡串行batch1，完成11次全量预测；两机Qwen成绩不同并分别报告。该配置不是已证明的最低配置 |
| 驱动与精度 | 需支持CUDA13运行时及GPU BF16；按预检实算确认。BERT用GPU FP32，Qwen用GPU BF16；不自动改CPU、FP16、量化或更小基座。系统安装Toolkit不能替代兼容驱动 |
| 资源与网络 | 需取得原Qwen基座约4.08GB及9份训练产物ZIP合计约1.19GB，并另留解压、依赖安装和输出空间；具体文件以SHA256确认。提交ZIP不含这些大文件，首次下载需要网络 |
| 软件与种子 | 同PyTorch版本或同种子均不能单独证明一致；须同时核查构建、完整依赖、模板、精度、数据和权重，并以实际预测验收 |
| 尚未通过的兼容方案 | torch2.5.1／CUDA12.1的Qwen结果未达到原主结果一致标准；同PyTorch版本的CUDA12.6候选尚未完成严格验证。它们不进入正式推荐，也不写为复现成功；CPU部分模型或少量样本检查不能替代11次全量测试 |
| 结果核对 | 先核查资源身份和11次各777条完整性；TF-IDF、MacBERT对照两机一致结果。Qwen以两机实测成绩和差异为基准，允许如实报告硬件相关波动，不要求跨硬件逐token相同；两机区间不是通用容差上界 |

**计算栈差异与贪心推理的影响。** Qwen采用贪心解码（`do_sample=False`）：每一步选择得分最高的token。即使模型权重、输入和随机种子相同，PyTorch、CUDA、cuDNN等计算栈的变化，也可能改变浮点运算顺序、精度与舍入，进而改变候选token得分。候选分数接近时，最高分token可能改变；自回归生成又会把该token作为后续输入，使后续输出、非法输出率及最终指标发生变化。固定种子控制随机数，不能消除这类浮点计算差异。因此，版本号或种子相同都不能替代实际全量验证，发现差异时报告其相对两机基准的变化，不直接等同于程序无法运行。原理依据见[PyTorch数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)。

4070独立环境严格重预测为11次×777条、指标差0、逐样本输出差0。2026年10月30日最新4060回传已完整完成11次，但有85条输出差异、61条标签差异，最大指标差0.020592；环境与运行通过，严格结果一致性未通过。旧驱动阻断为历史失败记录，不代表最新状态。`completed_with_differences`表示完整运行后的历史精确对照存在差异；历史脚本因此返回1；本版采用下述完整执行判定，历史记录保持不变。若11次均完整且资源校验通过，应保留各项成绩和差异表，按本报告两机基准解释，不误报为环境失败或模型未运行。`failed`、OOM、超时或缺预测仍须排查。

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

默认已有预测核查条件：外层及内层状态均为`passed`，16份正式记录完整，11份最终测试各777条，主指标差为0，退出码0，容差`1e-12`。该入口重算4070原记录；下表左列为其预期值，右列为4060真实模型复验。加载模型后的Qwen结果按两机基准解释，不套用已有预测重算的零差异条件。

| 方法 | Macro-F1：4070／4060 | Accuracy：4070／4060 | 训练种子 |
|---|---|---|---|
|TF-IDF＋逻辑回归|86.104%／86.104%|88.417%／88.417%|42、43、44|
|MacBERT全参微调|91.518%／91.518%|92.621%／92.621%|42、43、44|
|Qwen3零样本|62.559%／64.380%|59.588%／61.647%|确定性单次|
|Qwen3固定四样本|76.105%／75.927%|76.577%／76.448%|确定性单次|
|Qwen3普通LoRA T01|91.116%／90.925%|92.278%／92.106%|42、43、44|

训练方案先逐种子计算再平均；多数类参考Macro-F1为40.64%、Accuracy为68.47%。标签0负面、1正面；生成结果去首尾空白后只有完整0/1合法，非法输出保留分母并计错。训练／验证／测试6210／776／777，先取512 Unicode字符。数据划分、少样本、冒烟、延迟子集、bootstrap种子67，训练种子42／43／44。

## 3. 加载真实权重：一条命令重跑完整主表

该模式使用已训练产物重新预测，不重新训练；串行执行TF-IDF、MacBERT、LoRA各三个种子以及零／少样本，共11个完整测试运行。每个运行777条，再执行20条预热和固定200条延迟测量。

先配置以下环境和外部资源：

1. 模型环境使用WSL2/Linux、Python **3.11**。原机为3.11.17、PyTorch **2.14.1+cu130**、Transformers5.8.0、PEFT0.18.1、scikit-learn1.9.1。`src/project/requirements.integration.lock`现包含原机158个依赖的完整冻结清单，包括`datasets==4.0.0`、`accelerate==1.11.0`、`trl==0.24.0`；框架单独安装。原清单仅有10项，已修复。完整模型模式要求CUDA与BF16，显卡8GB；不自动改精度、模型或依赖版本。
2. 获取[Qwen3-1.7B基座](https://modelscope.cn/models/Qwen/Qwen3-1.7B)，固定ModelScope revision为`4855588ea1a12789f2e965e5f52a9e4a24c94b2a`，下载时选择该版本；文件身份还须通过`src/project/artifacts/lora-lock.json`逐项核对。模型会逐文件验SHA256；MacBERT使用发布的已训练分类器，不使用未微调基座替代。
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

在WSL中从解压目录安装到**新的独立环境**。**虚拟环境和LLaMA-Factory必须放在WSL原生文件系统，例如`/home/用户名/`，不要放在`/mnt/c/`、`/mnt/d/`等Windows挂载目录。** 本次独立复现在Windows挂载目录出现框架导入120秒、`pip check`60秒超时；使用相同依赖在WSL原生目录重建后通过。保留原超时条件，不通过延长等待掩盖问题。提交包本身可以留在Windows目录。

以下使用已安装的uv管理Python，`$HOME`应为WSL原生用户目录（通常为`/home/用户名`），先运行`echo "$HOME"`确认。若它被自定义到了Windows挂载目录，须改用自己的WSL原生目录作为下面的安装位置：

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

### Windows挂载目录超时处理

若已经在Windows挂载目录安装过环境或框架，保留旧目录和失败日志，**在WSL原生目录新建环境并重新安装，不直接剪切或复制虚拟环境**。使用上面的安装命令；目录已存在时换新名称，并同步替换各条命令中的目录。框架重新检出同一个固定commit，不更换版本。

激活新环境，确认实际Python位置：

```bash
python -c "import sys; print('环境：', sys.prefix); print('解释器：', sys.executable)"
```

虚拟环境应位于`/home/用户名/.venvs/...`等原生目录。将`reproduce-models.local.json`里的`factory_root`更新为新框架目录，例如`/home/用户名/LLaMA-Factory-nlp`；其他有效模型路径可以保留。然后重新运行下方预检。

预检会记录实际环境、框架和项目路径；在WSL中检测到虚拟环境或框架位于Windows挂载目录（含可识别的自定义挂载点和符号链接目标）时，提前停止并给出重建提示，避免继续等待导入。项目源码位于Windows目录仅记录提醒，不阻止运行。框架导入仍限120秒、`pip check`仍限60秒；原生目录也超时则保留`preflight.json`并排查内存、磁盘和依赖，不把所有超时都归因于挂载路径。

路径预检修订仅进行了工程检查，没有重新运行模型；已有4070独立复现证据中的11次预测、每次777条及逐样本一致结果保持原记录。目录选择依据见[微软WSL文件存储建议](https://learn.microsoft.com/en-us/windows/wsl/filesystems)。

### 环境通过但结果有差异

2026年10月30日另一台RTX4060 Laptop主机在相同PyTorch2.14.1+cu130、158项依赖、固定框架及合格驱动下完成11次×777条预测，挂载目录检查也通过，但有85条原始输出或标签差异，其中61条改变标签。TF-IDF和MacBERT一致；Qwen零样本72条、少样本7条、LoRA合计6条不同。五次Qwen运行的3885条输入渲染哈希与参考全部一致，最大指标差约0.020592。

因此，升级驱动和预检通过解决运行前提，不能单独证明跨显卡结果一致。此类运行状态为`completed_with_differences`，属于本实验观察到的跨硬件现象；它与环境加载失败、预测不完整分开解释。不要盲目重装相同版本或覆盖参考预测；将硬件相关差异与两机实测基准并列报告。需保留差异，并比较GPU、attention计算后端、矩阵精度开关、数值缓冲区及逐步token得分；当前证据尚不能将差异唯一归因于某个具体内核。

新预测的`config.json`会记录`numerical_runtime`，包括GPU能力与SM数、CUDA／cuDNN、精度开关、确定性及SDPA开关、实际attention实现和缓冲区指纹。此记录只读取设置，不改变正式推理。配套的小规模验证集诊断为独立工具，原计算路径与受控计算路径分开留证；其结果不替代11次完整测试。

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

额外输出：`model-stdout.log`、`model-stderr.log`、`results/model-reproduction/metrics.json`、`prediction-differences.jsonl`和`status.json`。各模型目录`results/local/reproduced-.../`保存原始输出、777条预测、指标、配置、延迟与资源。逐种子、均值及逐样本仍对照4070冻结参考以生成差异证据；`passed`表示精确对照匹配，`completed_with_differences`表示11次完整而输出有差异；本版资源通过且固定基线一致时均返回0，差异仍单独报告。对TF-IDF、MacBERT核对一致性，对Qwen结合两机基准报告变化；这两种状态不能与未完成流程混为一谈。`failed`表示流程失败，`interrupted`表示中断；错误信息说明具体原因，不与指标差异混淆。

## 4. 完整代码与依赖

`src/project/src/`为统一CLI、模型接入、指标、分析、演示和三条原路线源码；`scripts/`为数据处理、训练及分析工具；`configs/`为冻结参数与路径示例；`tests/`为软件测试。上游框架、模型、数据均为外部依赖。

在已安装依赖的模型环境中，可于`src/project/`运行`python -m unittest discover -s tests -v`。精简包保留可独立构造测试输入的工程检查；依赖整份历史错例／资源证据的检查和发布工具检查留在完整仓库，避免数据不入包却要求本地读取的矛盾。完整仓库本轮88项软件检查通过，随包71项检查通过；这与模型效果及新机验收分别记录。

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
- [演示录屏：百度网盘](https://pan.baidu.com/s/10l-SBqKrzkCp4hh3KQ8XAA?pwd=668w)，提取码`668w`。文件`演示视频.mp4`，111.317秒，Cherry Studio真实运行与难例对照。
- [完整代码、原始预测、日志及图表](https://github.com/liuguanghui520/NLP)。main为正式协作入口，固定输入仍按历史提交和哈希锁定。

报告摘要与独立项目摘要完全一致，均为300字，涵盖实验协议、效果、成本及跨机稳定性；三份贡献声明列明具体工作、核查步骤、40／20／15小时及对应姓名文字署名。

验证状态：4070与4060均已完成11次真实模型预测、每次777条。TF-IDF与MacBERT逐样本一致；Qwen有85条跨机输出差异，其中61条改变标签，纳入报告主结果。新版流程判定以真实回传8547条预测完成离线重放核查；没有重新跑模型，教师验收尚未记录。

若教师或AI复现报错，保留首次失败记录，48小时内补交环境说明或修复，并以全新目录重验；不覆盖失败、不换数据、不挑选种子。原机验证与组员新机验证分别记录。

## 跨机诊断与报告解释（2026-10-30）

| 方案 | 4070原Macro-F1 | 4060复验Macro-F1 | 差值（百分点） | 输出差异数 | 本次跨机表现 |
|---|---:|---:|---:|---:|---|
|TF-IDF|86.104%|86.104%|+0.000|0|逐样本一致|
|MacBERT|91.518%|91.518%|+0.000|0|逐样本一致|
|Qwen零样本|62.559%|64.380%|+1.821|72|环境差异明显|
|Qwen四样本|76.105%|75.927%|-0.178|7|有差异，幅度较小|
|Qwen LoRA|91.116%|90.925%|-0.191|6|有差异，幅度较小|

表1b　原4070结果与4060完整复验。差异按“运行×样本”计数，TF-IDF、MacBERT及LoRA各含3个种子，其余各1次；两机均为11次×777条。85条输出差异中61条改变解析标签。成本仍采用环境E01，不混合两机成绩。

两机排名均为MacBERT、LoRA、TF-IDF、四样本、零样本。TF-IDF和MacBERT的指标及逐样本预测一致；零样本Macro-F1变化+1.821个百分点、Accuracy变化+2.059个百分点，跨机波动最明显；四样本和LoRA的Macro-F1分别变化−0.178、−0.191个百分点，幅度较小，但不等于逐样本一致或已证明统计等效。因此主结论同时考虑分类效果、成本和本次跨机稳定性，保留零样本及其局限。

两机固定16条验证输入token、权重及数值buffer一致，原BF16路径16条首步得分指纹全部不同，1条在首token发生分歧；4060两轮重复的全部逐步得分一致。受控Flash／Math仍各有1条输出不同，不作为正式修复。主报告已将跨机结果与BF16精度局限纳入主结论，具体GPU内核及FP32全量改善效果尚未验证。保留零样本与原参考成绩，主结论同时报告效果、成本和两机稳定性。

### 复现结果应如何解释

以两机实际测试数据为基准，MacBERT的Macro-F1均为91.518%，TF-IDF均为86.104%，两条路线逐样本预测一致。Qwen零样本为62.559%／64.380%，跨机差1.821个百分点；四样本为76.105%／75.927%，差0.178个百分点；LoRA为91.116%／90.925%，差0.191个百分点。两机均保持MacBERT、LoRA、TF-IDF、四样本、零样本的效果排序。因此MacBERT兼顾本次分类效果、成本和跨机稳定性，TF-IDF提供稳定轻量参照；LoRA与四样本有小幅硬件相关波动，零样本波动更明显。这里的两机区间是已观测基准，不是所有硬件的误差上界。

安装、资源准备、预检和完整预测命令保持原样。本版1e-12／逐样本对照继续提供精确差异证据。11次完整、资源校验通过且固定基线一致时，Qwen差异不阻止返回0；新增状态字段区分工程完成与数值一致。历史回传的退出码1不改写。更换硬件后的新成绩应独立留档，不覆盖参考。

双机逐运行指标、均值、样本SD及诊断汇总随包保存在`src/project/artifacts/cross-machine-study-20261030.json`，便于核对报告数字；该文件不包含数据集或模型权重。

### 执行状态与复现证据

新版入口将完整执行和数值精确一致分开：资源校验通过、11次各777条完整且TF-IDF／MacBERT指标和逐样本一致时返回0；Qwen差异继续保留为completed_with_differences，并记录execution_complete=true、baseline_reference_match=true、qwen_reference_match=false、strict_reference_match=false。缺预测、资源失败或固定基线异常仍返回非零。历史回传退出码1不改写；返回0表示工程执行与基线核对通过，不代表教师已经验收。

88项完整仓库与71项随包检查用于软件行为验证；AI独立复现的两机真实预测及本轮真实输出的离线状态重放分别留证。`teacher_acceptance=not_recorded`，没有把软件返回0写成教师已经验收。
