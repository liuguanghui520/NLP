# -*- coding: utf-8 -*-
"""
选出「代表方案」——四条路线里每条路线都要挑一个代表去跟别的路线比。

规则（全项目统一）
------------------
**代表组 = 该路线所有变体里，验证集准确率最高的那个。**

为什么必须是验证集而不是测试集
------------------------------
用测试集挑代表，等于用最终考场成绩来选谁上场，测出来的高分会被这个选择本身抬高，
后面跟别的路线比的时候就不是公平比较了。验证集是"考场之前"的练习题，
拿它选代表、拿测试集做最终比较，是标准做法。

具体到 MacBERT 全参臂
---------------------
8 组超参配置各跑 3~5 个 epoch，每个 epoch 都在验证集（776 条，与测试集 777 条完全独立）
上评过一次。取每组**最后一个 epoch** 的 val_acc 做比较：

    lr_1e5   0.9162   ← 代表组
    ep5      0.9137
    main     0.9124
    len512   0.9124
    partial  0.9111
    len128   0.9085
    lr_5e5   0.9059
    weighted 0.9059

注意 len512 的**测试集**最高（0.9279），但它的验证集只有 0.9124 ——
按测试集选它就等于拿考试成绩挑代表。这条规则同样适用于：
- TF-IDF 臂（4 组 n-gram，按验证集选 char_1_3）
- LoRA 臂（4 个秩，按验证集选）
- 提示臂（3 个变体，按验证集选）

用法
----
    from representative import pick_representative
    cfg, row, val_acc = pick_representative(bert_dir)
"""
import os
import sys

import pandas as pd


def _last_val_acc(val_log_csv):
    """每组配置的最后一个 epoch 的验证集准确率 -> {config: val_acc}"""
    if not os.path.exists(val_log_csv):
        return {}
    try:
        vl = pd.read_csv(val_log_csv)
    except Exception:
        return {}
    if 'config' not in vl.columns or 'val_acc' not in vl.columns:
        return {}
    vl = vl.copy()
    vl['config'] = vl['config'].astype(str)
    vl['epoch'] = pd.to_numeric(vl['epoch'], errors='coerce')
    vl = vl.dropna(subset=['epoch', 'val_acc'])
    last = vl.sort_values('epoch').groupby('config').tail(1)
    return {str(r['config']): float(r['val_acc']) for _, r in last.iterrows()}


# 公开别名（给 fill_report.py 用）
last_val_acc = _last_val_acc


def pick_representative(bert_dir):
    """挑代表组。

    返回 (config, ablation_row_dict, val_acc)；
    拿不到验证集日志时退回到 ablation_results.csv 的第一行（并让调用方知道）。
    """
    ab_csv = os.path.join(bert_dir, 'ablation_results.csv')
    if not os.path.exists(ab_csv):
        return None, None, None
    df = pd.read_csv(ab_csv)
    if df.empty:
        return None, None, None

    val = _last_val_acc(os.path.join(bert_dir, 'val_log.csv'))
    if not val:
        # 兜底：没有验证集日志就按训练顺序的第一组（与旧行为一致）
        row = df.iloc[0].to_dict()
        return str(row['config']), row, None

    cand = df[df['config'].astype(str).isin(val.keys())]
    if cand.empty:
        row = df.iloc[0].to_dict()
        return str(row['config']), row, None

    scored = [(str(r['config']), float(val[str(r['config'])]), r.to_dict())
              for _, r in cand.iterrows()]
    scored.sort(key=lambda x: (-x[1], x[0]))       # val 高者优先；同分按名字定序，保证可复现
    cfg, val_acc, row = scored[0]
    return cfg, row, val_acc


def representative_table(bert_dir):
    """返回 [(config, val_acc, test_acc)]，按 val_acc 降序 —— 报告里那张"选代表"的表。"""
    val = _last_val_acc(os.path.join(bert_dir, 'val_log.csv'))
    df = pd.read_csv(os.path.join(bert_dir, 'ablation_results.csv'))
    rows = []
    for _, r in df.iterrows():
        cfg = str(r['config'])
        rows.append((cfg, val.get(cfg), float(r['test_acc'])))
    rows.sort(key=lambda x: (-(x[1] if x[1] is not None else -1), x[0]))
    return rows


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(here, '..', 'results', 'results_BERT_FULL')
    d = sys.argv[1] if len(sys.argv) > 1 else default_dir
    print(f'结果目录：{os.path.abspath(d)}\n')
    print('| 配置 | 验证集 acc | 测试集 acc | 是否代表组 |')
    print('|---|---|---|---|')
    cfg, row, v = pick_representative(d)
    for c, va, ta in representative_table(d):
        mark = ' **← 代表组**' if c == cfg else ''
        print(f'| {c} | {"—" if va is None else f"{va:.4f}"} | {ta:.4f} |{mark} |')
    print(f'\n代表组 = {cfg}（验证集 acc {v:.4f}）' if v else f'\n代表组 = {cfg}')
