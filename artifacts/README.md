# 模型与复现资源

当前下载状态见[资源清单](reproduction-downloads.json)，一键离线核查见[根README](../README.md)及[完整说明](../docs/reproduction/README.md)。

源码、已有数据、预测和日志已按本人授权随仓库公开。默认离线复现无需模型权重和个人资源路径。数据来源许可的历史未核验项仍保留在`data/manifests/manifest.json`，不因公开而自动改写。

真实模型预测还需以下产物，当前全部下载方式尚未验收：

| 资源 | 数量 | 身份依据 |
|---|---:|---|
| Qwen3-1.7B基座与分词器 | 1套 | `lora-lock.json`、`qwen-prompt-model-lock.json` |
| MacBERT len512最佳模型 | 3个种子 | 基线运行索引的checkpoint_id及模型SHA256 |
| TF-IDF char_1_3 Pipeline | 3个种子 | 基线运行索引的checkpoint_id及Pipeline SHA256 |
| 普通LoRA T01适配器 | 3个种子 | `lora-lock.json`适配器权重与配置SHA256 |

9个训练产物的URL、压缩包SHA256未发布核验时保持null。代码ZIP不能替代这些已训练产物；将来通过GitHub Release分发时补实际链接、大小、解压结构和核查记录。

Qwen基座来源沿用ModelScope `Qwen/Qwen3-1.7B`，以实际本地文件SHA256锁定。MacBERT基座固定revision `a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`；已经微调的模型不能通过重新下载基座替代。原下载信息不完整的历史事实保留，不补造revision。

实际模型位置由成员在`configs/paths.local.json`填写自己的地址，该文件留在个人机器；字段模板为`configs/paths.example.json`。本轮未加载模型或训练，真实模型新机复现和教师验收仍待实际执行。

旧说明原字节保存在`docs/history/artifacts-README-before-reproduction-20261029.md`。
