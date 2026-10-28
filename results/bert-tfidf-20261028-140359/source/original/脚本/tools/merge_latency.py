# -*- coding: utf-8 -*-
"""
把两臂分别测出的延迟结果合并成报告要读的 latency_baseline.csv。

为什么得分两次测
----------------
本机 Windows 被智能应用控制策略拦了 sklearn 的 C 扩展（WinError 4551），
所以 TF-IDF 臂只能在 WSL 跑；而 WSL 里没有 torch，MacBERT 臂只能在 Windows 跑。
两臂的**协议完全一致**（同一批 200 条冻结 ID、batch=1、预热 20 次），
所以数字可以并排比较。

    Windows:  python src/eval/latency_baseline.py --skip-tfidf \\
                  --out src/results/results_BERT_FULL/latency_bert.csv
    WSL:      python src/eval/latency_baseline.py --skip-bert \\
                  --out src/results/results_BERT_FULL/latency_tfidf.csv
    合并:     python tools/merge_latency.py

用法
----
python tools/merge_latency.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'src', 'results', 'results_BERT_FULL')
SOURCES = ['latency_bert.csv', 'latency_tfidf.csv']
OUT = os.path.join(RES, 'latency_baseline.csv')


def main():
    frames = []
    for name in SOURCES:
        p = os.path.join(RES, name)
        if not os.path.exists(p):
            print(f'  ✗ 缺 {name}，跳过')
            continue
        df = pd.read_csv(p)
        df['measured_on'] = 'Windows (torch)' if 'bert' in name else 'WSL (sklearn)'
        frames.append(df)
        print(f'  ✓ {name}  ({len(df)} 行)')
    if not frames:
        print('!! 没有可合并的延迟文件')
        return 1

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'\n✓ 已合并 -> {OUT}')
    for _, r in out.iterrows():
        print(f'  {r["variant"]:<20} n={int(r["n"]):<4} '
              f'mean={r["mean_ms"]:.3f}ms  p50={r["p50_ms"]:.3f}  '
              f'p95={r["p95_ms"]:.3f}  max={r["max_ms"]:.3f}   [{r["measured_on"]}]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
