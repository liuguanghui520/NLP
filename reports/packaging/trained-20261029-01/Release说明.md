# 中文酒店评论情感分析：已训练产物

本版本提供种子42、43、44的9个已训练产物：

- MacBERT全参数微调最佳模型3份，含分词器与配置。
- 字符1—3 gram TF-IDF＋逻辑回归Pipeline 3份。
- Qwen3-1.7B普通LoRA T01适配器3份：r8、alpha16、q/k/v/o。

9个ZIP合计1,186,034,812字节（约1.19GB，十进制），已在WSL生成并在Windows流式复核逐文件SHA256和ZIP CRC。附件README.md提供目录说明，catalog.json保存身份、大小和SHA256，SHA256SUMS可用于核对下载。

LoRA适配器仍需要另取与仓库`artifacts/lora-lock.json`一致的Qwen3-1.7B基座，不能作为完整独立模型加载。各包不包含训练优化器、Python环境或评论数据。

模型由验证Macro-F1选优，原测试指标随包保留用于追溯，上传不是新模型实验。实际新机模型预测和教师验收仍需按README另行执行。

代码与最新获取说明：[项目README](https://github.com/liuguanghui520/NLP/blob/codex/integration/README.md)。来源与许可证据保留在模型包和仓库；Qwen适配器包附基座现存LICENSE。
