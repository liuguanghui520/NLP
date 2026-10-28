# -*- coding: utf-8 -*-
"""
复现预检：花 2 分钟确认「这台机器 + 这套代码 + 这份数据」能不能复现出一样的结果。

为什么需要它
------------
完整复现要跑 26 小时（CPU）或 45 分钟（T4）的 MacBERT 训练。万一是环境、数据、
路径三件事里的一件不对，跑完 26 小时才发现，就太亏了。

这个脚本先把**不耗 CPU** 的那部分做完并判定：
    1. 重建冻结数据集，逐文件 SHA256 与「冻结资产」比对
    2. 跑 verify_repro.py，把你的结果与「期望结果」逐位比对
    3. 打印结论：能不能进入正式训练

用法
----
    # 在项目根目录（work/）下
    python tools/preflight_check.py

    # 参数可改
    python tools/preflight_check.py \
        --source data/ChnSentiCorp/ChnSentiCorp_htl_all.csv \
        --results src/results \
        --expect 期望结果 \
        --frozen 冻结资产/data/processed/v1-20261023

退出码 0 = 预检通过，可以进入正式训练；1 = 有问题。
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

DATA_FILES = ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'excluded.jsonl',
              'invalid_records.jsonl', 'conflicts.jsonl', 'statistics.json',
              'fewshot_ids.json', 'smoke_ids.json', 'latency_ids.json']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='data/ChnSentiCorp/ChnSentiCorp_htl_all.csv')
    ap.add_argument('--results', default='src/results')
    ap.add_argument('--expect', default='期望结果')
    ap.add_argument('--frozen', default='冻结资产/data/processed/v1-20261023')
    ap.add_argument('--skip-verify', action='store_true',
                    help='只重建数据，不跑结果比对（还没训练时用）')
    args = ap.parse_args()

    root = Path(os.getcwd())
    print('=' * 72)
    print('  复现预检')
    print('=' * 72)
    print(f'  工作目录：{root}')

    # ---------- 1. 重建冻结数据 ----------
    print('\n  [1/2] 重建冻结数据集 ...')
    r = subprocess.run([sys.executable, 'src/data/rebuild_data.py',
                        '--source', args.source,
                        '--frozen-dir', args.frozen,
                        '--out', 'data/processed/v1-20261023'],
                       cwd=root, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print('  ❌ 数据重建失败：')
        print('\n'.join((r.stdout or '').splitlines()[-12:]))
        print((r.stderr or '')[-800:])
        return 1
    print('  ✅ 重建完成，与冻结版逐位一致')

    # ---------- 2. 比对结果 ----------
    # 「训练还没跑」和「跑出来但不一致」是两回事，不能混为一谈：
    # 前者是正常状态（预检的阶段 1），后者才是问题（阶段 2）。
    bert_dir = Path(args.results) / 'results_BERT_FULL'
    bert_done = (bert_dir / 'ablation_results.csv').exists()
    paired_done = (Path(args.results) / 'paired_tests.csv').exists()

    if args.skip_verify:
        print('\n  [2/2] 跳过结果比对（--skip-verify）')
        print('\n  结论：✅ 预检通过，可以进入正式训练。')
        return 0

    if not bert_done and not paired_done:
        print('\n  [2/2] 结果比对：跳过')
        print(f'   ℹ 还没找到 {bert_dir}/ablation_results.csv —— 正式训练尚未完成。')
        print('     预检阶段 1（数据 + 环境）通过：可以进入正式训练。')
        print('\n  结论：✅ 预检通过（数据与环境就绪），可以进入正式训练。')
        print('        训练 + 出结果之后，再跑一次本脚本，那时会做逐位比对。')
        return 0

    print('\n  [2/2] 比对结果与期望结果 ...')
    r2 = subprocess.run([sys.executable, 'tools/verify_repro.py',
                         '--results', args.results,
                         '--expect', args.expect,
                         '--frozen', args.frozen,
                         '--data', 'data/processed/v1-20261023'],
                        cwd=root, capture_output=True, text=True,
                        encoding='utf-8', errors='replace')
    out = (r2.stdout or '')
    # 只打结论段，省得刷屏
    for line in out.splitlines():
        if any(k in line for k in ('❌', '⚠', '结论', '➖', '✅ 全部')):
            print('   ' + line.strip())
    if r2.returncode != 0:
        print('\n  结论：❌ 预检未通过 —— 先按 README「复现失败怎么查」排查，再跑正式训练。')
        return 1
    print('\n  结论：✅ 预检通过 —— 结果与期望结果逐位一致。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
