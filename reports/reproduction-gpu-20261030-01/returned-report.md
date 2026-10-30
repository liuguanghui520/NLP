# 本机独立复现报告

报告日期：2026-10-30（Asia/Shanghai）  
提交包：`20245726_刘广徽_中文评论情感分析.zip`  
提交包 SHA256：`129a6f6ec4ae54ab6eb7b1066dda47b22160697bf39aa1c9addc0e4699f2b5f8`

## 结论摘要

| 验收项 | 结论 | 核心证据 |
|---|---|---|
| 环境搭建 | **通过** | 正式预检 `passed`；Python 3.11.17、PyTorch 2.14.1+cu130、158/158 锁文件依赖匹配、CUDA/BF16、固定框架 commit 和 `pip check` 均通过 |
| 已有预测的指标核查 | **通过** | 47 份输入哈希通过；16 份正式记录重算；11 个测试运行共 8547 条；最大指标差为 `0.0` |
| 实际模型预测是否完整完成 | **是** | 11/11 次运行完成，每次 777 条，共 8547 条新测试预测；没有训练、缺样本或中断 |
| 主结果是否满足 README 严格一致性 | **否** | 85 条逐样本差异；最大指标差 `0.020592020592020588`，远大于 `1e-12`；框架状态为 `completed_with_differences`，退出码 1 |
| 是否修改提交源码/参考成绩/锁定文件 | **否** | 原 ZIP 未改；包内 135 个清单文件在执行前后哈希均匹配。仅按 README 新增了本地路径 JSON，见第 8 节 |

最终状态必须区分“推理完整”与“复现一致”：本机已经完整执行真实权重推理，但没有达到提交 README 的严格通过标准，不能记作完整模型复现通过。

## 1. 材料、隔离与路径

- 原始 ZIP 未复制、修改或重打包，计算得到 SHA256 `129a6f6ec4ae54ab6eb7b1066dda47b22160697bf39aa1c9addc0e4699f2b5f8`。
- 在全新 Windows 目录解压后先阅读 README；没有读取或复用本机原项目中的代码、环境、模型、数据、缓存、预测或日志。
- 源码另复制到 WSL 原生 ext4：`/home/yilin/nlp-course-repro-20261030/submission`。
- 独立 Python：`/home/yilin/nlp-repro-cache-20261030/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11`。
- 独立 uv 缓存：`/home/yilin/nlp-repro-cache-20261030/uv-cache`。
- 虚拟环境：`/home/yilin/.venvs/nlp-course-repro-20261030-02`。
- LLaMA-Factory：`/home/yilin/LLaMA-Factory-nlp-20261030`。
- Qwen 基座：`/home/yilin/nlp-models/Qwen3-1.7B`。
- 训练产物：`/home/yilin/nlp-trained-artifacts-20261030`。
- 正式模型工作目录：`/home/yilin/nlp-course-repro-20261030/model-reproduction-001`。

虚拟环境、LLaMA-Factory、模型和正式输出均在 WSL 原生 `/home/yilin`，挂载检查结果为 ext4，`windows_mounts` 为空。

## 2. 本机环境

| 项目 | 实际配置 |
|---|---|
| 操作系统 | Windows 11 家庭中文版，10.0.26200，64 位 |
| WSL | WSL2；Ubuntu 22.04.1 LTS；内核 `6.6.87.2-microsoft-standard-WSL2` |
| CPU | Intel Core i9-13900H；Windows 报告 14 核/20 线程 |
| 内存 | 宿主机 34,070,388,736 字节；实验时 WSL 约 15 GiB，另有 4 GiB swap |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，compute capability 8.9 |
| NVIDIA 驱动 | Windows KMD `621.01`，CUDA UMD `13.5`；WSL `nvidia-smi` 报告 NVIDIA-SMI `620.67.01` |
| Python | 3.11.17，Clang 23.1.3 构建 |
| PyTorch | `2.14.1+cu130`，`torch.version.cuda=13.0` |
| CUDA/BF16 | `cuda_available=true`；BF16 支持为真；BF16 2×2 实算结果为全 2.0 |
| 关键依赖 | Transformers 5.8.0、PEFT 0.18.1、scikit-learn 1.9.1、NumPy 2.4.6、Datasets 4.0.0、Accelerate 1.11.0、TRL 0.24.0 |
| 完整依赖 | `requirements.integration.lock` 的 158/158 项版本完全匹配；另有 `wheel 0.48.0` 和 `llamafactory 0.9.6.dev0` |
| LLaMA-Factory | commit `c683e59edd8f893abc5e1403184a1d0644817d2a`，工作树干净 |
| uv | 0.13.6；官方安装脚本 SHA256 `f7a56e7afaec49a3bcbc3ac3cdbe917e7c6ce30e0d815308a12e531aa5ef23b7` |

驱动 `621.01` 已高于 README 所述 CUDA 13 兼容下限 R580，故没有升级驱动或重启。CUDA 13 兼容说明来源：<https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>。

正式预检命令：

```bash
python src/run.py --preflight --paths src/project/configs/reproduce-models.local.json
```

结果：`passed`，退出码 0，用时 10.958 秒。记录见 `evidence/preflight/preflight.json` 和 `evidence/runs/official-model-preflight-001/`。

依赖安装后 `pip check` 返回 `No broken requirements found.`；包内可独立运行的 56 项软件测试全部通过。完整实际冻结清单见 `evidence/verification/pip-freeze-actual.txt`，158 项匹配报告见 `evidence/verification/dependency-match-final.json`。

## 3. 外部资源与校验

| 资源 | 来源/版本 | 校验结果 |
|---|---|---|
| Qwen3-1.7B | README 指定的 ModelScope `Qwen/Qwen3-1.7B`，`revision=master`，下载日期 2026-10-30 | 13/13 个锁定文件 SHA256 匹配；两个权重分片为 3,441,185,608 与 622,329,984 字节 |
| 训练产物 | GitHub Release `trained-artifacts-20261029-01`，9 个 ZIP | 9/9 ZIP 的文件大小和 SHA256 匹配；总字节 1,186,034,812 |
| LLaMA-Factory | GitHub `hiyouga/LLaMA-Factory` | checkout 后 `HEAD` 与 README 固定 commit 完全一致 |
| 冻结输入与已有预测 | 包内配置指定的 GitHub 固定 revision `106cdbed8db84057dba6b9c980d3a2f914856dd0` | 47/47 个输入按 `reproduction.json` 的 SHA256 校验 |

Qwen 的 13 个逐文件哈希见 `evidence/verification/qwen-model-hashes.json`。9 个训练产物的原始 GitHub URL、大小和 SHA256 锁定在 `evidence/model/inputs/artifacts/reproduction-downloads.json`；实际传输记录见 `evidence/downloads/trained-artifact-mirror-events.jsonl`。

GitHub 直连在本机 WSL 上出现连接超时。按 README 允许的可选镜像机制，改用 `https://ghfast.top/` 传输；原始 GitHub URL、文件大小和包内 SHA256 均不变，下载后逐字节校验通过。镜像只改变传输路径，没有改变模型、数据、参考预测或验收标准。

训练产物哈希如下：

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `bert-len512-seed42.zip` | 381052561 | `bbb9001a10d1e7603ae43311dd9827fc1e092b8194268f70c448a0aa3c448268` |
| `bert-len512-seed43.zip` | 381053504 | `5cf7f3769fe9ea13c4bce2f8c52f41241050c3784eb113f1103064eb44a8b576` |
| `bert-len512-seed44.zip` | 381051263 | `a56bde23114291070a46e88c4a0eafbcfd185af0017c809bce35176192cac078` |
| `tfidf-char_1_3-seed42.zip` | 2353995 | `a50738066234a3e8c77d4429c34e84cb2aea3667d4bae64f39af6d3b268b1d7b` |
| `tfidf-char_1_3-seed43.zip` | 2353999 | `b514f7a3100fb54c430adc4d4d7c1e0dd69fc6e9a6d725437704f8eaf52cf2d5` |
| `tfidf-char_1_3-seed44.zip` | 2353999 | `feed61cb825a45d81a5dc29f5ea286a11e9b7220bf84547ebdc2d83e5fa8d897` |
| `qwen-lora-T01-seed42.zip` | 11937864 | `7cd36afc5cb73c0fe97c6778b7025550d8bb1dcd428afcd887bb5d8d25ca88eb` |
| `qwen-lora-T01-seed43.zip` | 11938781 | `aa6c031ab1c141e4d9fe893e430ab5afe473e3077611789473b6396f90b15b61` |
| `qwen-lora-T01-seed44.zip` | 11938846 | `c055494976535c8a0850ac18b8ace695ed9dfcc825157b29da60a9bc2a203cae` |

## 4. 已有预测的独立指标核查

第一次直接 GitHub 下载在 `requirements.analysis.lock` 处连续超时并保留失败目录；之后在失败目录中仅补齐缺失文件，逐份按固定 SHA256 校验，再用该目录作为输入缓存启动全新审计目录。第一次失败记录未被覆盖。

成功审计使用新工作目录 `/home/yilin/nlp-course-repro-20261030/check-saved-predictions-002`：

- 外层和内层状态均为 `passed`，退出码 0。
- 47 份固定输入复制后再次校验。
- 16 份正式记录完整；其中 11 个测试运行，每个 777 条，共 8547 条。
- 最大指标差 `0.0`。
- 数据划分为 6210/776/777；样本 ID、唯一性、真值、标签合法性及模型输入检查均通过。

该结果是已有预测与统一指标代码的一致性核查，不包含任何新模型推理，不能单独代表真实模型复现通过。

证据：`evidence/saved-audit/`、`evidence/verification/independent-reference/`。

## 5. 真实模型推理执行

实际正式命令为：

```bash
python src/run.py --models \
  --paths src/project/configs/reproduce-models.local.json \
  --work-dir /home/yilin/nlp-course-repro-20261030/model-reproduction-001 \
  --input-cache /home/yilin/nlp-course-repro-20261030/check-saved-predictions-002
```

其中 `--input-cache` 是 README 提供的网络不稳定恢复选项；47 份输入仍被重新逐项哈希验证。没有复用任何预测，没有读取本机原项目。

- 开始：2026-10-30 02:21:57.830 +08:00
- 结束：2026-10-30 02:36:33.152 +08:00
- 墙钟用时：875.454 秒
- 进程退出码：1
- 内层与外层状态：`completed_with_differences`
- 推理：11/11 完成；训练：未执行；每次 777 条；总计 8547 条
- 所有运行的 `status.json` 均为 `completed`

每项运行都加载了已发布的既有权重。TF-IDF 为 CPU FP32，MacBERT 为 GPU FP32，Qwen 零/少样本与 LoRA 使用 GPU BF16。LoRA 实际配置为普通 LoRA、T01、q/k/v/o、r=8、alpha=16，与权重锁文件一致。

## 6. 指标比较

下表由新生成的 11 份 `predictions.jsonl` 独立重算，不使用冻结指标代替计算。训练方法先逐种子计算再取均值；Qwen 零/少样本为确定性单次。

| 方法 | 新 Macro-F1 / 参考 | 差值（百分点） | 新 Accuracy / 参考 | 差值（百分点） | 非法率 新/参考 | 逐样本差异 | 严格结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| TF-IDF + LR | 0.861040458 / 0.861040458 | 0.0000 | 0.884169884 / 0.884169884 | 0.0000 | 0 / 0 | 0 | 通过 |
| MacBERT 全参 | 0.915182348 / 0.915182348 | 0.0000 | 0.926211926 / 0.926211926 | 0.0000 | 0 / 0 | 0 | 通过 |
| Qwen3 零样本 | 0.643801891 / 0.625588940 | +1.8213 | 0.616473616 / 0.595881596 | +2.0592 | 0.187902188 / 0.203346203 | 72 | 未通过 |
| Qwen3 固定四样本 | 0.759273799 / 0.761049018 | -0.1775 | 0.764478764 / 0.765765766 | -0.1287 | 0.001287001 / 0.002574003 | 7 | 未通过 |
| Qwen3 普通 LoRA T01 | 0.909253251 / 0.911160198 | -0.1907 | 0.921063921 / 0.922779923 | -0.1716 | 0 / 0 | 6 | 未通过 |

完整四指标对比和精确差值见 `evidence/verification/report-tables/metric-comparison.csv`。最大绝对指标差是零样本 Accuracy 的 `0.020592020592020588`。README 门槛是 `1e-12`，不能因分数接近、排名相同或仅部分样本通过而放宽。

## 7. 逐样本核验

独立核验直接读取预测 JSONL 和测试真值，按“去首尾空白后完整为 0 或 1”的严格规则重新解析；非法输出保留在 777 条分母中并计错。

- 11 次运行，每次 777 条，共 8547 条。
- 每次样本数均为 777、唯一 ID 均为 777；顺序、ID 集合、真值、原始输出类型、存储解析标签与严格重解析结果全部匹配。
- 没有缺失样本、额外样本、重复 ID 或真值错位。
- Qwen 零样本非法输出 146/777，非法率 `18.7902%`；参考为 158/777、`20.3346%`。
- Qwen 少样本非法输出 1/777，非法率 `0.1287%`；参考为 2/777、`0.2574%`。
- TF-IDF、MacBERT、LoRA 的新预测非法率均为 0。
- 共 85 条逐样本差异：零样本 72 条、少样本 7 条、LoRA 合计 6 条。
- 85 条差异中 61 条改变了严格解析标签，24 条只改变原始文本而解析标签不变。

逐样本完整差异见 `evidence/verification/independent-model/sample-differences.jsonl`；框架自身生成的差异清单见 `evidence/model/aggregate/prediction-differences.jsonl`。两者数量均为 85。

## 8. 延迟与资源记录

每个运行均执行 20 条预热和固定 200 条延迟测量，batch=1；逐条延迟、benchmark 和资源峰值均在 `evidence/model/local-runs/<run_id>/`。

| 方法 | 200 条 p50 范围 | p95 范围 | GPU 峰值分配/保留 | 结束 RSS 范围 |
|---|---:|---:|---:|---:|
| TF-IDF ×3 | 0.458–0.514 ms | 0.820–0.918 ms | 不适用（CPU） | 0.999–1.007 GB |
| MacBERT ×3 | 5.466–5.562 ms | 13.705–13.845 ms | 437/493 MB | 1.444–1.446 GB |
| Qwen LoRA ×3 | 54.839–70.551 ms | 78.112–98.952 ms | 3536/5455 MB | 2.102–2.152 GB |
| Qwen 零样本 | 92.225 ms | 347.663 ms | 3529/3569 MB | 1.886 GB |
| Qwen 四样本 | 169.618 ms | 250.664 ms | 3575/3674 MB | 1.866 GB |

完整字段见 `evidence/verification/runtime-001/runtime-summary.csv` 和 `.json`。

## 9. 修改、失败与重试

### 9.1 提交材料是否修改

- 原 ZIP 没有修改，SHA256 在执行前后均为 `129a6f6ec4ae54ab6eb7b1066dda47b22160697bf39aa1c9addc0e4699f2b5f8`。
- 包内清单中的 135 个文件全部同时与 ZIP 内容和清单 SHA256 匹配。
- 解压源码中唯一新增文件是 README 要求的新机本地路径 `src/project/configs/reproduce-models.local.json`，其 SHA256 为 `cc201f215179809d6692023a4d84ba9ff93b7e34c60e4b47d93e287ec1e8fb05`。
- 没有修改源码、模型权重、分词器、提示、数据、参考预测、参考成绩、锁定文件或提交清单。

修改前后结果：

| 状态 | 执行结果 |
|---|---|
| 未添加本地路径前 | 示例配置三个路径为 `null`，没有用无效路径执行模型；包内 135 个清单文件已验证与 ZIP 一致 |
| 添加本地路径后 | 正式预检退出码 0、`passed`；完整模型运行退出码 1、`completed_with_differences`，11×777 条预测完整 |
| 完整推理结束后复核 | 原始 ZIP 哈希不变；135 个清单文件仍全部匹配；解压树仍只有上述一个额外本地配置文件 |

提交完整性证据：`evidence/verification/submission-integrity.json` 与 `submission-integrity-postrun.json`。

### 9.2 网络失败与恢复

1. GitHub 直连静态内容与 Release CDN 多次 TCP 超时。
2. 第一次已有预测审计失败，退出码 1；失败工作目录及 29 条下载事件保留。
3. 第一次训练产物直连下载失败，未把残缺文件当作权重。
4. 改用 README 允许的 HTTPS 镜像前缀 `https://ghfast.top/`，逐个文件校验大小和 SHA256 后原子改名。
5. 新目录重跑已有预测审计通过；完整模型运行再次验证全部权重与输入。

失败现场见 `evidence/failures/`、`evidence/downloads/trained-artifact-download-events.jsonl` 及 `evidence/runs/` 中对应命令目录。

### 9.3 差异原因

数据、Qwen 13 个基础文件、9 个训练产物、提示身份、精度、种子、关键依赖版本和 LLaMA-Factory commit 均已锁定并通过校验。差异集中在 Qwen 三条生成方案；TF-IDF 和 MacBERT 逐样本完全一致。

本地 GPU 是 RTX 4060 Laptop，而 README 的已验证硬件是 RTX 4070 Laptop；驱动也未与原作者核对到完全相同版本。README 明确说明计算栈可能改变贪心生成的 token。上述为可能影响因素，但**没有证据可把本次差异唯一归因于某一项**，因此不作已确认根因结论。按用户要求，没有改精度、降级 PyTorch、换基座、量化、改提示或借助重跑挑选通过结果。

### 9.4 工具层故障

- 最初把多行 Bash 脚本直接传给 `wsl.exe -lc` 时，Windows PowerShell 与 WSL 的参数转发破坏了 Bash 变量，导致两次只读网络探测没有访问目标 URL。随后改为先把原始脚本写入文件、再由 WSL 执行，调试重试通过；该问题没有修改输入或产生模型结果。
- 第一次创建的 Python 3.11.17 虚环境目录未实际用于安装依赖或推理，因为安装器报告命中共用 Python 安装；随后用专用 `UV_CACHE_DIR` 和 `UV_PYTHON_INSTALL_DIR` 重新安装解释器并创建 `nlp-course-repro-20261030-02`，正式实验只使用后者。
- 后续几次省略包装器的只读状态查询也遇到同一内联变量问题，没有写入实验目录；正式状态均由脚本文件或绝对路径重新读取。最终交付校验命令曾把原始 ZIP 路径误指向复现目录一次，修正后得到同一原始 SHA256，交付 ZIP 本身未受影响。

## 10. 五项最终回答

1. **环境搭建是否成功：是。** 正式预检通过，环境、框架、依赖、CUDA 和 BF16 均满足本机路径与版本要求。
2. **已有预测的指标核查是否通过：是。** 16 份记录、11 个测试运行、8547 条测试预测均完整；最大指标差 0。
3. **实际模型预测是否完整完成：是。** 11/11 次真实权重推理完成，每次 777 条，共 8547 条；未训练。
4. **主结果是否满足 README 一致性要求：否。** 85 条逐样本差异，最大指标差 0.020592020592020588，超过 1e-12；状态为 `completed_with_differences`。
5. **是否修改过提交材料：否。** 原 ZIP 与 135 个清单文件哈希不变；只新增 README 要求的本地路径 JSON。修改前后执行结果如第 9 节。

## 11. 证据索引

| 证据 | 路径 |
|---|---|
| 机器可读五项结论 | `evidence/verification/verdict.json` |
| 原始 ZIP 与解压源码完整性 | `evidence/verification/submission-integrity-postrun.json` |
| 平台、CPU、内存、GPU、WSL | `evidence/platform/platform.json`、`nvidia-smi-q.txt` |
| 最终环境与冻结依赖 | `evidence/runs/final-environment-capture-001/`、`evidence/verification/pip-freeze-actual.txt`、`dependency-match-final.json` |
| 正式预检 | `evidence/preflight/preflight.json` |
| 已有预测成功核查 | `evidence/saved-audit/` |
| 已有预测首次失败现场 | `evidence/failures/saved-audit-direct-github/` |
| Qwen 哈希 | `evidence/verification/qwen-model-hashes.json` |
| 权重与输入下载事件 | `evidence/downloads/`、`evidence/model/aggregate/download-events.jsonl` |
| 正式模型总体状态与指标 | `evidence/model/submission-status.json`、`evidence/model/aggregate/status.json`、`metrics.json` |
| 每运行配置、预测、延迟、资源 | `evidence/model/local-runs/` |
| 冻结参考预测 | `evidence/model/reference-predictions/` |
| 框架逐样本差异 | `evidence/model/aggregate/prediction-differences.jsonl` |
| 独立指标与逐样本复核 | `evidence/verification/independent-model/` |
| 指标对比表 | `evidence/verification/report-tables/metric-comparison.csv` |
| 结构合法性表 | `evidence/verification/report-tables/prediction-validity.csv` |
| 全部关键命令、起止时间、退出码、stdout/stderr | `evidence/command_index.jsonl`、`evidence/runs/` |

证据包不包含虚拟环境、模型权重、数据集、下载 ZIP 或下载缓存，也不包含凭据。
