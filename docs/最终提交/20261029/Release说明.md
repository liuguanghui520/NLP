# 中文评论情感分析课程提交版

提交日期：2026年10月29日。代码主分支：main。

本Release提供唯一课程提交ZIP：`20245726_刘广徽_中文评论情感分析.zip`。报告摘要与独立项目摘要统一为同一份300字文本；三份贡献声明补充具体步骤、实验核查、协作边界、工时与姓名文字署名。

包内根目录保持任务书建议结构：`报告.pdf`、`项目摘要.md`、`README.md`、`src/`及三份`贡献声明_学号_姓名.md`。PDF正文12页、附录8页，7张图片已内嵌。工时分别40h、20h、15h，综合贡献40%、30%、30%。

解压后运行：

```text
python src/run.py
```

默认从固定提交下载47份实验输入，校验SHA256并重算五方案主表，不加载模型。完整模型环境和外部资源配置见包内README，可使用其`--models`入口重跑11份测试。原始训练／测试结果不更改，非法生成保持严格计错。

数据集、模型权重、录屏不进入提交ZIP，只提供链接：

- [9份已训练产物](https://github.com/liuguanghui520/NLP/releases/tag/trained-artifacts-20261029-01)：原文件及校验和保持不变。
- [数据来源](https://modelscope.cn/datasets/AiNiklaus/ChnSentiCorp)；固定划分、预测与日志通过仓库链接下载。
- [演示视频](https://pan.baidu.com/s/12nyNQJliIKabozmBmfuR-g?pwd=9vxp)，提取码`9vxp`，111.317秒。

附带`SHA256SUMS`用于核对ZIP身份；课程收集表只上传ZIP并填写视频链接。当前课程包地址：[course-submission-20261029-01](https://github.com/liuguanghui520/NLP/releases/tag/course-submission-20261029-01)。
