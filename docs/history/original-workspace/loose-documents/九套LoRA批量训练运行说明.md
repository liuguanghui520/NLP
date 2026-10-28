# 九套LoRA批量训练运行说明

日期：2026-10-26。按用户最终要求，由用户手动运行，不由AI启动。

## 已准备批次

批次qwen17-comparisons-20261026-012117-9e8e51，状态prepared。只执行L01、L02、L03、L04、L05、R01、D01、D02、T01，顺序串行；已完成基准B00不重新训练。每套都从原始Qwen3-1.7B加载，不承接上一套适配器。

9份实际train.json与冻结代码副本已生成在/home/lgh/LLaMA-Factory/project-workspaces/qwen17-comparisons/上述批次/；项目索引在results/qwen17-comparisons/上述批次/batch.json。

## 启动命令

先在WebUI卸载模型、停止0.6B API，保持电源连接及电脑唤醒。新开Windows PowerShell运行：

```powershell
wsl -d Ubuntu -u lgh -- /home/lgh/.venvs/llamafactory/bin/python -u /mnt/c/Users/lgh/Documents/ChatGPT/NLP/scripts/run_qwen17_comparisons.py --run-prepared /home/lgh/LLaMA-Factory/project-workspaces/qwen17-comparisons/qwen17-comparisons-20261026-012117-9e8e51
```

保持终端开启。当前已有空闲WebUI可保留，但不要在网页同时开始训练或加载另一个模型。脚本检查启动前设备显存不超过2048MiB；若超限则停止并保留原进程，由用户先卸载模型。

此命令只能启动该prepared批次一次；已开始/失败/中断批次不会直接覆盖重跑。需要重试时先查看日志，为失败组另建run_id并记录关联，不能直接重复整批以覆盖证据。

## SwanLab与本地记录

项目https://swanlab.cn/@liuguanghui520/nlp-qwen17，每套实验名为批次ID加配置ID，例如qwen17-comparisons-20261026-012117-9e8e51-L01。

- use_swanlab=True、report_to=none、swanlab_mode=cloud，沿用现有登录；不在配置写入云端API Key。
- 每套记录真实训练loss、学习率、配置；每轮保存检查点后生成776条验证预测，上传classification/macro_f1、classification/accuracy、classification/invalid_rate、classification/validation_seconds。
- 复用已完成基准归档的原训练/分类验证入口，不改变已确认的数据和评价方式。资源记录器额外保留UTC/perf_counter两套时间及差异字段。
- 每套resource-record/console.log为完整合并stdout/stderr日志，resources.csv为采样数据，status.json为资源状态；各轮validation-step-*/保存原始预测和指标。
- 每套训练完成后核对3轮、每轮776个唯一ID，重算Macro-F1，再登记最佳checkpoint及其适配器SHA256。
- 运行索引batch.json逐组更新pending/running/completed/failed状态，并复制配置、验证历史、最佳检查点、计时和资源摘要到项目结果目录。
- 普通单组训练失败保留记录并继续下一套，不降级模型/量化/批量；Ctrl+C中断时停止批次，保留已完成及未执行状态。

SwanLab不是唯一证据来源，云端上传问题应查看本地日志。含正文的预测文件按既定许可边界留本地。

## 查看进度

在另一个Windows PowerShell读取本地索引：

```powershell
$batch = Get-Content -LiteralPath 'C:\Users\lgh\Documents\ChatGPT\NLP\results\qwen17-comparisons\qwen17-comparisons-20261026-012117-9e8e51\batch.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$batch.status
$batch.jobs | Format-Table id,state,validation_macro_f1
```

最终batch.status为completed且9项均completed，才表示九套完整通过；部分失败则incomplete，不把失败目标值补成零分。

## 已执行的检查

检查模型文件SHA256、数据/manifest哈希、依赖版本和上游提交与已完成基准一致；10套唯一性、改动因素组、alpha=2r、训练参数接口解析通过。准备目录仅写入文件，没有加载训练权重或执行新增训练。

实现入口：scripts/run_qwen17_comparisons.py。无标志运行会新建批次并立即执行，因此本次使用明确的--run-prepared命令。--check-only只核对；--prepare-only只生成新批次。不要无意间重复生成并执行第二个批次。

AI负责工程实现、检查、冻结文件和操作说明；参数组合由用户给定，运行由用户手动启动，结果解释由学生完成。未执行Git提交或推送。
