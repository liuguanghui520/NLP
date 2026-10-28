| 对比 | ΔAccuracy | McNemar b/c | McNemar p | bootstrap Δacc 95% CI | 结论 |
|---|---|---|---|---|---|
| MacBERT ep5（5epoch） vs MacBERT lr_1e5 | -0.13 pp | 13/14 | 1.0000 | [-1.54, +1.16] pp | 不显著 |
| MacBERT len128 vs MacBERT lr_1e5 | +0.39 pp | 21/18 | 0.7493 | [-1.16, +1.93] pp | 不显著 |
| MacBERT len512 vs MacBERT lr_1e5 | +0.51 pp | 18/14 | 0.5966 | [-0.90, +1.93] pp | 不显著 |
| MacBERT lr_5e5 vs MacBERT lr_1e5 | -0.51 pp | 15/19 | 0.6076 | [-2.06, +1.03] pp | 不显著 |
| MacBERT main（lr2e-5, len256） vs MacBERT lr_1e5 | -0.39 pp | 10/13 | 0.6776 | [-1.67, +0.77] pp | 不显著 |
| MacBERT partial（冻下6层，lr=3e-05） vs MacBERT lr_1e5 | -0.39 pp | 11/14 | 0.6900 | [-1.67, +0.90] pp | 不显著 |
| MacBERT weighted（类别加权） vs MacBERT lr_1e5 | +0.13 pp | 11/10 | 1.0000 | [-1.03, +1.29] pp | 不显著 |
| TF-IDF char_1_3 vs MacBERT ep5（5epoch） | -3.73 pp | 23/52 | 0.0012 | [-5.92, -1.54] pp | **显著** |
| TF-IDF char_1_3 vs MacBERT len128 | -4.25 pp | 27/60 | 0.0006 | [-6.56, -1.93] pp | **显著** |
| TF-IDF char_1_3 vs MacBERT len512 | -4.38 pp | 18/52 | 0.0001 | [-6.43, -2.32] pp | **显著** |
| TF-IDF char_1_3 vs MacBERT lr_1e5 | -3.86 pp | 21/51 | 0.0006 | [-6.05, -1.80] pp | **显著** |
| TF-IDF char_1_3 vs MacBERT lr_5e5 | -3.35 pp | 21/47 | 0.0024 | [-5.41, -1.29] pp | **显著** |
| TF-IDF char_1_3 vs MacBERT main（lr2e-5, len256） | -3.47 pp | 22/49 | 0.0020 | [-5.53, -1.29] pp | **显著** |
| TF-IDF char_1_3 vs MacBERT partial（冻下6层，lr=3e-05） | -3.47 pp | 22/49 | 0.0020 | [-5.66, -1.41] pp | **显著** |
| TF-IDF char_1_3 vs MacBERT weighted（类别加权） | -3.99 pp | 23/54 | 0.0006 | [-6.31, -1.80] pp | **显著** |
| MacBERT 长度分桶组 batch vs MacBERT 随机组 batch | +1.16 pp | 14/5 | 0.0636 | [+0.13, +2.32] pp | 不显著 |
| MacBERT n=100 vs MacBERT lr_1e5 | -12.23 pp | 24/119 | 0.0000 | [-15.06, -9.40] pp | **显著** |
| MacBERT n=1000 vs MacBERT lr_1e5 | +0.00 pp | 21/21 | 0.8774 | [-1.67, +1.67] pp | 不显著 |
| MacBERT n=2000 vs MacBERT lr_1e5 | +0.39 pp | 18/15 | 0.7283 | [-1.03, +1.80] pp | 不显著 |
| MacBERT n=3000 vs MacBERT lr_1e5 | -0.39 pp | 13/16 | 0.7111 | [-1.80, +1.03] pp | 不显著 |
| MacBERT n=500 vs MacBERT lr_1e5 | -0.26 pp | 20/22 | 0.8774 | [-1.93, +1.29] pp | 不显著 |
