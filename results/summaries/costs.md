# 同机推理成本

均值为各训练种子的单模型均值再平均；零/少样本各一次。实际CPU/GPU路线分别标注。

|方法|运行数|设备|端到端均值ms|P95均值ms|allocated峰值最大MiB|
|---|---:|---|---:|---:|---:|
|qwen_zero_shot|1|NVIDIA GeForce RTX 4070 Laptop GPU|97.706|251.918|3365.1|
|qwen_few_shot|1|NVIDIA GeForce RTX 4070 Laptop GPU|79.414|100.536|3409.1|
|qwen_lora|3|NVIDIA GeForce RTX 4070 Laptop GPU|59.056|70.277|3372.6|
|bert|3|NVIDIA GeForce RTX 4070 Laptop GPU|6.133|10.668|416.9|
|tfidf|3|cpu|0.555|0.894|不适用|

峰值范围为模型加载后测试/预热/延迟全过程；RSS为期末采样，不是采样峰值。
训练分项与累计成本见training-costs.json；不同时间边界不叠加为虚构总时长。
