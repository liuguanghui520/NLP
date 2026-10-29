# 预测差异

第二层没有生成任何新模型预测：`results/model-reproduction/status.json` 的 `completed` 为空，运行在外层 `model_reproduction.py` 的 CUDA/BF16 前置检查处停止。因此：

- 新旧模型预测差异样本数为 0（不是“比较后无差异”，而是没有可比的新预测）。
- 不输出伪造的样本 ID、解析标签或原始输出。
- 计划中的 11 × 777 = 8,547 条本机真实模型预测，实际为 0 条。

第一层下载的 16 份既有预测文件全部与锁定 SHA256 一致，因此第一层没有预测内容差异。`第一层预测差异.csv` 只有表头，用于明确空集合。

要生成真正的预测差异，必须先在满足 PyTorch cu130/BF16 的 NVIDIA 驱动与硬件上运行完整第二层，再按 `sample_id` 对齐新预测和冻结参考，输出 `true_label`、`raw_output`、`parsed_label` 以及差异类型。
