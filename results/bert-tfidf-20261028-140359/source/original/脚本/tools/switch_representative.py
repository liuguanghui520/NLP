# -*- coding: utf-8 -*-
"""
把「主结果产物」切到代表组（按验证集选出来的那一组）。

为什么需要这个
--------------
代表组从 `main` 换成 `lr_1e5` 之后，所有"主结果"相关的产物都要跟着换：
    predictions.csv        ← 混淆矩阵图读它
    acc_by_length.csv     ← 长度分桶的准确率
    errors_by_class.csv    ← 错误清单
    error_samples.csv      ← 报告 §6.1 要引用的 3 条失败样例（**带正文**）
    bert_report.txt        ← 文字版小结

**不需要重新训练**：每组配置的逐条预测早就存下来了
（`predictions_<config>.csv`，777 行，带两类完整概率），
失败样例的正文按 `sample_id` 回查 `test.jsonl` 就能拿到。
所以这个切换是**纯重算，秒级完成**。

用法
----
python tools/switch_representative.py            # 先看会切到哪一组
python tools/switch_representative.py --apply    # 真切
"""
import argparse
import json
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src', 'eval'))
from representative import pick_representative, representative_table  # noqa: E402

RES = os.path.join(ROOT, 'src', 'results', 'results_BERT_FULL')
DATA = os.path.join(ROOT, 'data', 'processed', 'v1-20261023')
TOKENIZER_DIRS = [os.path.join(ROOT, 'macbert_base'),
                  os.path.join(ROOT, 'src', 'models', 'macbert_base')]
CLS = ['negative', 'positive']
BK = [0, 64, 128, 256, 384, 512, 9999]


def load_jsonl(path):
    out = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_artifacts(pred, cfg, tokenizer_dir=None):
    """从逐条预测重算主结果产物。"""
    tlen = [int(x) for x in pred['token_len']]
    ys = [CLS.index(str(x)) for x in pred['true_label']]
    ps = [CLS.index(str(x)) for x in pred['pred_label']]

    # ---- 错误率 vs 长度分桶（与 train_bert.md 里的逻辑逐位一致）----
    rows_b = []
    for lo, hi in zip(BK[:-1], BK[1:]):
        idx = [i for i, l in enumerate(tlen) if lo < l <= hi]
        if not idx:
            continue
        rows_b.append({'len_range': f'({lo},{hi}]', 'n': len(idx),
                       'acc': round(sum(1 for i in idx if ys[i] == ps[i]) / len(idx), 4),
                       'neg_ratio': round(sum(1 for i in idx if ys[i] == 0) / len(idx), 4)})

    # ---- 错误清单（不含正文）----
    errs = [{'sample_id': pred.iloc[i]['sample_id'],
             'true': CLS[ys[i]], 'pred': CLS[ps[i]],
             'confidence': float(pred.iloc[i]['confidence']),
             'token_len': tlen[i]}
            for i in range(len(pred)) if ys[i] != ps[i]]
    err_df = pd.DataFrame(errs, columns=['sample_id', 'true', 'pred',
                                         'confidence', 'token_len'])

    # ---- 失败样例（带正文，按 sample_id 回查 test.jsonl）----
    te = load_jsonl(os.path.join(DATA, 'test.jsonl'))
    by_id = {x['sample_id']: x for x in te}

    # text_seen = 模型实际看到的 tokenized 文本（decode 掉特殊符号）
    tok = None
    for d in TOKENIZER_DIRS:
        if os.path.isdir(d):
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(d)
                print(f'  ✓ tokenizer: {d}')
                break
            except Exception as e:
                print(f'  ⚠ tokenizer 加载失败（{d}）：{type(e).__name__}')

    samp = []
    for i in range(len(pred)):
        if ys[i] != ps[i]:
            sid = pred.iloc[i]['sample_id']
            text = by_id.get(sid, {}).get('text', '')
            seen = text
            if tok is not None:
                enc = tok([text], truncation=True, max_length=256)
                seen = tok.decode(enc['input_ids'][0], skip_special_tokens=True)
            samp.append({'sample_id': sid, 'true': CLS[ys[i]], 'pred': CLS[ps[i]],
                         'confidence': float(pred.iloc[i]['confidence']),
                         'token_len': tlen[i], 'text': text, 'text_seen': seen})
    samp_df = pd.DataFrame(samp, columns=['sample_id', 'true', 'pred', 'confidence',
                                           'token_len', 'text', 'text_seen'])
    samp_df = samp_df.sort_values('confidence', ascending=False).head(3)

    return rows_b, err_df, samp_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='真写文件（默认只预览）')
    args = ap.parse_args()

    cfg, row, val_acc = pick_representative(RES)
    if cfg is None:
        sys.exit('✗ 读不到 ablation_results.csv')

    print('=== 代表组候选（按验证集 acc 降序）===')
    print('| 配置 | 验证集 acc | 测试集 acc |')
    print('|---|---|---|')
    for c, va, ta in representative_table(RES):
        mark = ' **← 代表组**' if c == cfg else ''
        print(f'| {c}{mark} | {"—" if va is None else f"{va:.4f}"} | {ta:.4f} |')
    print(f'\n★ 代表组 = {cfg}（验证集 acc {val_acc:.4f}）')
    print(f'  测试集 acc={row["test_acc"]}  macroF1={row["macro_f1"]}  '
          f'weightedF1={row["weighted_f1"]}  用时={row["seconds"]}s')

    src_pred = os.path.join(RES, f'predictions_{cfg}.csv')
    if not os.path.exists(src_pred):
        sys.exit(f'✗ 找不到 {src_pred}')
    pred = pd.read_csv(src_pred)
    rows_b, err_df, samp_df = build_artifacts(pred, cfg)

    print(f'\n=== 将要切换的产物 ===')
    print(f'  predictions.csv      ← predictions_{cfg}.csv（{len(pred)} 行）')
    print(f'  acc_by_length.csv    ← {len(rows_b)} 个长度桶')
    print(f'  errors_by_class.csv  ← {len(err_df)} 条错误')
    print(f'  error_samples.csv    ← {len(samp_df)} 条（带正文）')
    for _, r in samp_df.iterrows():
        print(f'      true={r["true"]:<8} pred={r["pred"]:<8} conf={r["confidence"]}  '
              f'{r["text"][:38]}...')

    if not args.apply:
        print('\n[dry-run] 未写任何文件。加 --apply 执行切换。')
        return 0

    # 切换
    shutil.copy2(src_pred, os.path.join(RES, 'predictions.csv'))
    pd.DataFrame(rows_b).to_csv(os.path.join(RES, 'acc_by_length.csv'),
                                index=False, encoding='utf-8')
    err_df.sort_values('confidence', ascending=False).to_csv(
        os.path.join(RES, 'errors_by_class.csv'), index=False, encoding='utf-8')
    samp_df.to_csv(os.path.join(RES, 'error_samples.csv'), index=False, encoding='utf-8')

    # 文字小结也跟着换
    rep_src = os.path.join(RES, f'report_{cfg}.txt')
    if os.path.exists(rep_src):
        shutil.copy2(rep_src, os.path.join(RES, 'bert_report.txt'))
    with open(os.path.join(RES, 'representative.json'), 'w', encoding='utf-8') as fh:
        json.dump({'config': cfg, 'val_acc': val_acc,
                   'rule': '验证集准确率最高（最后一个 epoch），与 TF-IDF/LoRA/提示臂同一规则',
                   'test_acc': row['test_acc'], 'macro_f1': row['macro_f1'],
                   'weighted_f1': row['weighted_f1'], 'seconds': row['seconds']},
                  fh, ensure_ascii=False, indent=2)

    print('\n✓ 已切换到代表组 ' + cfg)
    print('  接下来跑：python tools/make_bert_plots.py   （用新的 predictions.csv 重画混淆矩阵）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
