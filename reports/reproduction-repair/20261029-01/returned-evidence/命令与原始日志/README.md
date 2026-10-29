# 命令与原始日志索引

所有时间和路径均为本机实际执行记录。失败尝试没有覆盖，后续尝试均使用新编号。

## 第一层：已有预测重算

首个成功命令：

```text
python src/run.py --work-dir "D:\NLPP\reproduction-runs\L1-attempt-009-20261029T1629"
```

工作目录：`D:\NLPP\复现证据\独立工作副本\提交包复现副本`。

运行前对 `submission_reproduce.py` 只做了工程兼容修复，见根目录的 `问题与修复/问题清单.md` 与补丁。模型、数据、标签、预测文件、指标算法、种子和实验协议均未修改。

| 尝试 | 结果 | 退出码 | 失败位置 | 证据 |
|---|---|---:|---|---|
| 001 | 失败 | 1 | 原始 raw 下载 `WinError 10054` | `03_第一层已有预测重算/attempt-001-*`；本地目录 `D:\NLPP\reproduction-runs\L1-attempt-001-20261029T1542` |
| 002 | 失败 | 1 | Windows `safe_path` 边界判断 | `attempt-002-*`；本地目录 `L1-attempt-002-20261029T1544` |
| 003 | 失败 | 1 | raw 下载 `WinError 10054` | `attempt-003-*`；本地目录 `L1-attempt-003-20261029T1550` |
| 004 | 失败 | 1 | raw 下载 `WinError 10054` | `attempt-004-*`；本地目录 `L1-attempt-004-20261029T1553` |
| 005 | 失败 | 1 | API 下载 `IncompleteRead(642468/651110)` | `attempt-005-*`；本地目录 `L1-attempt-005-20261029T1600` |
| 006 | 失败 | 1 | 内容 API 次级限流 403，下载 37/47 后停止 | `attempt-006-*`；本地目录 `L1-attempt-006-20261029T1617` |
| 007 | 失败 | 1 | 内容 API 次级限流 403，首个文件即停止 | `attempt-007-*`；本地目录 `L1-attempt-007-20261029T1621` |
| 008 | 失败 | 1 | 固定提交镜像一份文件 504/连接超时 | `attempt-008-*`；本地目录 `L1-attempt-008-20261029T1623` |
| 009 | 通过 | 0 | 无 | `attempt-009-command.txt`、`attempt-009-runner-stdout.log`、`attempt-009-runner-stderr.log`、`attempt-009-exit.txt`、`成功运行-L1-attempt-009/` |

成功目录是唯一进入指标重算的第 9 次。外层 `submission-status.json` 为 `passed`，47 份输入均经过锁定 SHA256 校验；内层 `结果核查.md` 的五个方案 Macro-F1 精确差值均为 0。

另有一次固定提交 Git 下载探测被人工中断：下载超过 3 分钟仍只有 0 字节临时 pack。记录见 `01_文件身份/fixed-commit-download-*`。该下载没有被当作成功证据。

## 第二层：真实模型重新预测

执行环境：

```text
/home/yilin/.venvs/nlp-reproduction-20261029-a02/bin/python
```

实际命令：

```text
python src/run.py \
  --models \
  --paths src/project/configs/reproduce-models.local.json \
  --work-dir "/mnt/d/NLPP/reproduction-runs/L2-attempt-001-20261029T1648"
```

执行脚本与命令记录：`04_第二层模型重新预测/model-run-attempt-001.sh`、`model-run-attempt-001-command.txt`。

- 进程退出码：1（`model-run-attempt-001-exit.txt`）。
- 内嵌第一层审计：通过。
- 11 个真实模型运行：0 个启动。
- 实际预测样本：0。
- 失败阶段：`src/model_reproduction.py` 在导入 torch 后、加载任何模型前的 CUDA/BF16 前置检查。
- 外层状态：`失败运行-L2-attempt-001/submission-status.json`。
- 模型日志：`失败运行-L2-attempt-001/model-stdout.log`、`model-stderr.log`。
- 内层状态：`失败运行-L2-attempt-001/results-model-reproduction/status.json`。
- 完整本地运行目录：同一个 `完整本地运行目录.txt` 所指的包外目录，内含下载输入、预测和所有运行材料。

外层包装器把该子进程失败统一报告成“failed or differs”；必须以内层状态和模型 stderr 区分。本次内层 `completed=[]`，错误为 CUDA/BF16 前置条件不满足，不是指标差异。

## 环境创建与依赖

- 第一次：官方 PyPI 端点持续 `SYN_SENT`，约 12 分钟无安装结果后人工中断；证据在 `02_环境创建与依赖/requirements-install*` 与 `问题与修复/requirements-install-attempt-001-interruption.txt`。
- 第二次：新建 Python 3.11.17 venv，通过清华 PyPI 镜像安装 73 个包，实际耗时 29 分 12 秒，安装退出码 0；证据在 `02_环境创建与依赖/requirements-install-attempt-002.*`。
- 第二次 `uv venv` 自身成功并输出 Python 3.11.17；外层包装器因内部 `$?` 捕获为空而误报退出码 2。原始 stdout、stderr、退出码均未覆盖；实际依赖安装是后续独立脚本，退出码 0。
- 版本冻结：`文件身份与环境/pip-freeze.stdout.log`。
- 一致性检查：`文件身份与环境/pip-check.stdout.log`，`All installed packages are compatible`。

## 模型资源下载

- Qwen 基座：本机已有固定文件，13/13 SHA256 匹配，日志在 `文件身份与环境/qwen-base-file-hashes.log`。
- TF-IDF 三份训练产物：`tfidf-char_1_3-seed42.zip`、`seed43.zip`、`seed44.zip` 均通过官方锁定 SHA256。日志在 `04_第二层模型重新预测/trained-artifact-*`。
- MacBERT seed42：镜像原包拼接后字节数正确但整包 SHA256 不匹配，ZIP 的 model 成员解压 CRC/zlib 失败；因此从未传给模型。失败状态在 `segmented-downloads/bert-len512-seed42.zip.integration-attempt-001.status.json`。余下训练产物没有在 CUDA 阻断后继续批量下载。

## 原始模型运行计时与资源记录

因为没有模型启动，`results/local/reproduced-*` 没有生成，也不存在可用于本机的延迟或 GPU 资源记录。不能把参考机型延迟视为本机延迟，也没有伪造截图或空结果。
