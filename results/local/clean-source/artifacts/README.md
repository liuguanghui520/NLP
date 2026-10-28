# 模型、数据与大产物获取

基座来源沿用成员原锁：Qwen/Qwen3-1.7B（ModelScope），hfl/chinese-macbert-base（Hugging Face）。本机Qwen文件SHA256与成员锁一致，原LoRA下载revision未知的历史事实保留；不能把同内容哈希核验改写成已知道原始下载revision。MacBERT本轮固定revision为a986e004d2a7f2a1c2f5a3edef4e20604a974ed1。

原始数据来源AiNiklaus/ChnSentiCorp，raw SHA256及复建规则见data/manifests/manifest.json与既有数据说明。评论再分发许可尚未充分核验，仓库不包含正文。公开JSONL序列化可能与成员副本不同，统一ID及内容通过派生输入哈希核对。

训练产物目前仅在本机核验，不声称已存在可公开下载外链：

|资源|本地路径键|身份清单|
|---|---|---|
|BERT三个len512最佳模型|bert_results|reports/audits/baselines.json|
|TF-IDF三个char_1_3 Pipeline|baseline_results|reports/audits/baselines.json|
|Qwen基座|qwen_model|lora-lock.json和qwen-prompt-model-lock.json|
|普通SFT LoRA三个T01适配器|lora_adapters|lora-lock.json|
|冻结数据|data|data/manifests及原数据复建说明|

paths.example只定义键；实际地址放被忽略的configs/paths.local.json。完整权重和日志不进入主仓库或50MB课程包。上传外链后需补访问权限、大小及SHA256实测记录，未完成此前新机真实推理验收保持未执行。
