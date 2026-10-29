# 中文评论情感分析：正式复现限制说明版

日期：2026-10-29。此版替代原课程提交ZIP；旧版Release保留用于追溯。

依据组员回传日志修复：补齐原机158项依赖、固定LLaMA-Factory commit，增加CUDA/BF16与导入预检，修复Windows路径与WSL文件复制，加入下载重试、本地缓存、损坏文件隔离和完整哈希校验，区分环境失败与完成后指标差异，输出逐种子及逐样本对照。

复现机驱动537.70只支持CUDA12.2，不能运行锁定的PyTorch2.14.1+cu130。须先安装与显卡匹配的CUDA13兼容驱动并重启，按包内README建立独立Python3.11环境；不自动换模型、精度或依赖版本。新机已有预测重算已通过、指标差0；真实模型在该机上完成0/11次，须修复环境后重跑。原机修复验证与组员新机验收分别留证。

下载`20245726_liuguanghui_chinese_sentiment.zip`，按课程要求改名为`20245726_刘广徽_中文评论情感分析.zip`。镜像与main下中文文件字节一致，用SHA256SUMS验证。包内仅报告PDF、300字摘要、README、全部工程源码及三份实名贡献声明；数据、权重和录屏均只提供链接。

快速环境检查（不加载模型）：

```text
python src/run.py --preflight --paths src/project/configs/reproduce-models.local.json
```

环境和外部路径准备好后，一条命令重跑11个真实测试运行：

```text
python src/run.py --models --paths src/project/configs/reproduce-models.local.json
```

可加`--input-cache /已有成功工作目录`复用47份冻结输入；输出始终使用新目录。默认`python src/run.py`仅重算已有预测。

9份[原训练产物](https://github.com/liuguanghui520/NLP/releases/tag/trained-artifacts-20261029-01)、基座、数据划分、提示、种子、模型协议与参考指标不变。损坏下载须重取并通过原始SHA256，不修改参考值或锁文件来凑结果。

[完整修复说明](https://github.com/liuguanghui520/NLP/blob/main/docs/reproduction/复现失败定位与修复-20261029.md)。[演示视频](https://pan.baidu.com/s/12nyNQJliIKabozmBmfuR-g?pwd=9vxp)，提取码9vxp。

验证结果：完整仓库66项检查、随包49项检查及Windows解压核查通过；原机11次真实预测各777条，逐种子及主表指标差0、逐样本输出差0。原机验证不替代组员更新驱动后的新机验收。

本次仅明确报告与README的硬件、驱动、软件、下载及验证范围，并加强逐样本一致性门槛。正式模型与指标不改；未通过的CPU／CUDA12兼容方案未进入本包。完整真实模型复现只推荐上述已验证环境。版本号与种子相同不能替代11次×777条的严格实际核查。

报告第7.4节与README明确解释：Qwen贪心解码中，计算栈浮点运算与舍入差异可能改变高分token并影响后续生成；固定种子不能消除该差异。正式验收仍要求主指标和逐样本输出一致，不以此解释放宽标准。
