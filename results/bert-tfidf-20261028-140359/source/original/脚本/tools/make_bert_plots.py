# -*- coding: utf-8 -*-
"""
用 train_bert.md 里【原样抽出来的绘图代码】生成 results_BERT_FULL 的 4 张图。

为什么单独做这个
----------------
train_bert.md 第 7 步「画图」是 2026-10-27 04:31 那次崩溃后**从未被执行过**的代码
（崩溃点在它上一行的 batching_control.csv 落盘）。所以这段代码从来没有被验证过，
而报告 §5.5 直接引用它的 4 张产物：
    loss_curve.png / confusion_matrix.png / ablation_compare.png / data_curve.png

如果让用户花 48 分钟把对照实验跑完，最后才发现绘图代码报错，图还是出不来。

做法
----
不复制粘贴一份绘图代码（那样两份迟早会分叉），而是：
  1. 从 train_bert.md 里按标记行切出「[7/8] 画图」这一段；
  2. 准备它需要的全部输入（都来自已落盘的 CSV）；
  3. exec 这段代码。
所以这里的产物与正式跑第 7 步的产物是**同一份代码跑出来的**，不会不一致。

用法
----
python tools/make_bert_plots.py
"""
import os
import sys
import textwrap

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, 'train_bert.md')
OUT = os.path.join(ROOT, 'src', 'results', 'results_BERT_FULL')

K = 2
CLS = ['negative', 'positive']


def load_env():
    ab = pd.read_csv(os.path.join(OUT, 'ablation_results.csv'))
    RESULTS = ab.to_dict('records')

    curve_csv = os.path.join(OUT, 'data_curve.csv')
    CURVE = pd.read_csv(curve_csv).to_dict('records') if os.path.exists(curve_csv) else []

    # ★ 主配置 = **代表组**（按验证集选的那一组），不是 main。
    #   报告里的 loss 曲线、混淆矩阵都必须对应报告的主结果。
    sys.path.insert(0, os.path.join(ROOT, 'src', 'eval'))
    from representative import pick_representative
    rep_cfg, _row, _val = pick_representative(OUT)
    if rep_cfg is None:
        rep_cfg = 'main'
    print(f'  主配置（代表组）= {rep_cfg}')

    tl = pd.read_csv(os.path.join(OUT, 'train_log.csv'))
    vl = pd.read_csv(os.path.join(OUT, 'val_log.csv'))
    tl = tl[tl['config'].astype(str) == rep_cfg]
    vl = vl[vl['config'].astype(str) == rep_cfg]

    main_res = next(r for r in RESULTS if str(r['config']) == rep_cfg)
    MAIN = (main_res, tl, vl)
    return dict(RESULTS=RESULTS, CURVE=CURVE, MAIN=MAIN,
                pd=pd, np=np, OUT=OUT, K=K, CLS=CLS)


def extract_plot_block():
    with open(MD, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    start = end = None
    for i, l in enumerate(lines):
        if start is None and '[7/8] 画图' in l and l.strip().startswith('print'):
            start = i
        elif start is not None and '[8/8] 存档' in l and l.strip().startswith('print'):
            end = i
            break
    if start is None or end is None:
        raise SystemExit('✗ 没能在 train_bert.md 里定位到「[7/8] 画图」代码段')
    block = textwrap.dedent('\n'.join(lines[start:end]))
    print(f'  已从 train_bert.md 第 {start + 1}~{end} 行切出绘图代码段')
    return block


def main():
    print('=== 生成 BERT 结果图 ===')
    env = load_env()
    block = extract_plot_block()

    before = set(f for f in os.listdir(OUT) if f.endswith('.png'))
    exec(compile(block, 'train_bert.md:[7/8] 画图', 'exec'), env)

    after = set(f for f in os.listdir(OUT) if f.endswith('.png'))
    new = sorted(after - before)
    print('\n  本次新生成:')
    for f in new:
        p = os.path.join(OUT, f)
        print(f'    {f}  ({os.path.getsize(p) / 1024:.0f} KB)')
    if not new:
        print('    (无新文件)')
    missing = [f for f in ('loss_curve.png', 'confusion_matrix.png',
                           'ablation_compare.png', 'data_curve.png')
               if f not in after]
    if missing:
        print(f'  ✗ 仍缺失: {missing}')
        sys.exit(1)
    print('  ✓ 4 张图齐全')


if __name__ == '__main__':
    main()
