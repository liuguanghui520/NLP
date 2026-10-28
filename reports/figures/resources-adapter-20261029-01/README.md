# 资源图：补充LoRA新增适配器

A为CUDA allocated峰值；B为完整部署大小；C单独展示T01新增权重。

适配器权重12.87464MB；含配置的完整增量12.875648MB；完整LoRA部署4092.326388MB。

MB按1000000字节，MiB按1048576字节。环境说明见docs/environment/实验运行环境与延迟口径.md。原正式批次及原图保持不变，后续报告使用本补充图。

复跑：python scripts/resource_figure.py --output reports/figures/<new-resource-figure-id>

本轮AI整理已有测量与绘图，未加载模型、重新测量或修改正式数据。
