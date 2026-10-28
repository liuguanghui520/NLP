# -*- coding: utf-8 -*-
"""
修复 train_log.csv / val_log.csv 里 config=='main' 的污染行。

背景
----
曲线实验和分组对照实验曾经沿用 CONFIGS[0] 的 name='main'，
于是 run_config 把它们的训练日志也写成 config='main'，
把真 main 的日志行冲淡了（train_log.csv 里 main 有 5184 行，真值应为 2331 行）。

关键性质（已核对）
------------------
真 main 是 3 epoch × 777 step（step 0..776）的完整序列；
所有被污染进来的行（曲线 5 个点 + 对照 1 个点）其 (epoch, step) 组合
**全部是 main 的真子集**（子集 step 上限分别是 12/62/124/249/374，均 < 777）。

因此还原规则唯一且无歧义：
    对 config=='main' 的行，按文件原顺序，按 (epoch, step) 只保留首次出现者。
剩下的必然正好是真 main 的 2331 行 / val_log 的 3 行。

用法
----
python tools/fix_log_pollution.py --dry-run     # 只分析不写
python tools/fix_log_pollution.py --apply        # 真写（写前自动备份）
"""
import argparse
import os
import shutil
import sys

# Windows 控制台默认 GBK，打印 ✓ 会抛 UnicodeEncodeError —— 强制 UTF-8 且不中断
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'src', 'results', 'results_BERT_FULL')
TRAIN_LOG = os.path.join(RES, 'train_log.csv')
VAL_LOG = os.path.join(RES, 'val_log.csv')

# 真 main 的逐条预测副本（detail=True 时写的 predictions.csv / acc_by_length.csv）
MAIN_PRED_SRC = os.path.join(RES, 'predictions.csv')
MAIN_PRED_DST = os.path.join(RES, 'predictions_main.csv')
MAIN_LEN_SRC = os.path.join(RES, 'acc_by_length.csv')
MAIN_LEN_DST = os.path.join(RES, 'acc_by_length_main.csv')

EXPECT_TRAIN_MAIN = 2331   # 3 epoch × 777 step
EXPECT_VAL_MAIN = 3        # 3 epoch


def dedup_main(df, key_cols, expect_rows, label):
    """按 (epoch, step) 保留首次出现的 main 行，并做一致性校验。"""
    mask = df['config'].astype(str) == 'main'
    sub = df[mask]
    before = len(sub)
    kept = sub.drop_duplicates(subset=key_cols, keep='first')
    after = len(kept)

    ok = (after == expect_rows)
    detail = ''
    if 'step' in kept.columns:
        mx = int(pd.to_numeric(kept['step']).max())
        per_ep = kept.groupby('epoch')['step'].nunique().to_dict()
        detail = f'max_step={mx}  每epoch不同step数={per_ep}'
        ok = ok and mx == 776 and all(v == 777 for v in per_ep.values())
    print(f'  [{label}] main 行数 {before} -> {after} (期望 {expect_rows})  {detail}')
    if not ok:
        print(f'  ✗ [{label}] 校验不通过，不写文件')
        return None
    new_df = pd.concat([df[~mask], kept], ignore_index=True)
    return new_df


def restore_main_predictions():
    """从 predictions.csv / acc_by_length.csv 还原被覆盖的 *_main 版本。

    复制前先交叉校验：predictions.csv 的 accuracy 必须等于
    ablation_results.csv 里 main 行的 test_acc，否则说明源文件本身不对。
    """
    ab = pd.read_csv(os.path.join(RES, 'ablation_results.csv'))
    mrow = ab[ab['config'].astype(str) == 'main']
    if mrow.empty:
        print('  ✗ ablation_results.csv 里找不到 main 行，跳过预测文件还原')
        return False
    expect = float(mrow.iloc[0]['test_acc'])

    pred = pd.read_csv(MAIN_PRED_SRC)
    acc = float((pred['true_label'].astype(str) == pred['pred_label'].astype(str)).mean())
    print(f'  predictions.csv 实测 acc={acc:.4f}  ablation_results 记录 main test_acc={expect:.4f}')
    # ablation_results.csv 里存的是 round(x,4)，与实测值最多差 5e-5；
    # 一个样本的权重是 1/777 ≈ 0.0013，所以容差取 1e-3 足够严格。
    if abs(acc - expect) > 1e-3:
        print('  ✗ 两者不一致，源文件可能已被污染，跳过还原')
        return False

    shutil.copy2(MAIN_PRED_SRC, MAIN_PRED_DST)
    shutil.copy2(MAIN_LEN_SRC, MAIN_LEN_DST)
    print(f'  ✓ 已还原 predictions_main.csv / acc_by_length_main.csv')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        args.dry_run = True

    print('=== 修复前 ===')
    for p in (TRAIN_LOG, VAL_LOG):
        df = pd.read_csv(p)
        c = df['config'].astype(str).value_counts().to_dict()
        print(f'  {os.path.basename(p)}: {len(df)} 行  各 config 行数={c}')

    tr = pd.read_csv(TRAIN_LOG)
    vl = pd.read_csv(VAL_LOG)

    new_tr = dedup_main(tr, ['epoch', 'step'], EXPECT_TRAIN_MAIN, 'train_log')
    new_vl = dedup_main(vl, ['epoch'], EXPECT_VAL_MAIN, 'val_log')
    if new_tr is None or new_vl is None:
        print('\n校验失败，已中止，未改动任何文件。')
        sys.exit(1)

    # 交叉校验：真 main 的 val_acc 应与 predictions.csv 一致的那个配置对得上
    m_vl = new_vl[new_vl['config'].astype(str) == 'main']
    print('\n  还原后的 main val_acc:', [round(float(x), 4) for x in m_vl['val_acc']])

    if args.apply:
        for p, new_df in ((TRAIN_LOG, new_tr), (VAL_LOG, new_vl)):
            bak = p + '.polluted.bak'
            if not os.path.exists(bak):
                shutil.copy2(p, bak)
                print(f'  已备份 -> {os.path.basename(bak)}')
            tmp = p + '.tmp'
            new_df.to_csv(tmp, index=False, encoding='utf-8')
            os.replace(tmp, p)
            print(f'  已写入 {os.path.basename(p)}  ({len(new_df)} 行)')
        restore_main_predictions()
    else:
        print('\n[dry-run] 未写任何文件。加 --apply 执行修复。')


if __name__ == '__main__':
    main()
