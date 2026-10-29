# ChnSentiCorp_htl_all 说明
0. **下载地址：** [Github](https://github.com/SophonPlus/ChineseNlpCorpus/raw/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv)
1. **数据概览：** 7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论
2. **推荐实验：** 情感/观点/评论 倾向性分析
2. **数据来源：**[携程网](http://www.ctrip.com/)
3. **原数据集：** ChnSentiCorp_htl，由 [谭松波](http://people.ucas.ac.cn/~0012244) 老师整理的一份数据集
4. **加工处理：**
    1. 将原来 1 万个离散的文件整合到 1 个文件中
    2. 将负向评论的 label 从 -1 改成 0
    3. 去重