"""
给预测文件补上 sample_id 主键（并可选去掉评论正文）。

为什么需要这个
--------------
四条路线的 predictions.csv 必须能按**同一个主键**逐条对齐，
才能做配对显著性检验（McNemar / paired bootstrap）。
主键统一用 test.jsonl 里的 `sample_id`（形如 `<文件sha256>:<原始行号>`），
和 fewshot_ids.json / latency_ids.json 用的是同一个字段。

有些早期产物只有 `text` 列（TF-IDF 那份就是），靠正文回查 sample_id 也能对齐——
本脚本就干这件事：读 test.jsonl 建 text → sample_id 索引，给预测文件补上
sample_id / idx，并可选择把正文列丢掉（提交包里不带评论正文）。

用法
----
    # 1) 先看一眼能不能对上（不改文件）
    python src/eval/relink_predictions.py --predictions src/results/results_TFIDF/predictions.csv --dry-run

    # 2) 确认无误后写回（自动备份原文件为 *.bak）
    python src/eval/relink_predictions.py --predictions src/results/results_TFIDF/predictions.csv --drop-text

    # 3) 直接输出到新文件，不动原文件
    python src/eval/relink_predictions.py --predictions a.csv --out b.csv --drop-text
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd

DATA_CANDIDATES = [
    'data/processed/v1-20261023',
    'src/data/processed/v1-20261023',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'v1-20261023'),
]


def find_test_jsonl():
    for c in DATA_CANDIDATES:
        p = os.path.join(c, 'test.jsonl')
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--drop-text', action='store_true',
                    help='丢掉 text 列（提交包不带评论正文）')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--keep-text-out', default='text',
                    help='保留正文的列名（默认 text）')
    args = ap.parse_args()

    tj = find_test_jsonl()
    if tj is None:
        sys.exit('找不到 test.jsonl（试过 ' + ', '.join(DATA_CANDIDATES) + '）')

    rows = []
    with open(tj, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    text2id = {}
    for i, r in enumerate(rows):
        text2id.setdefault(r.get('text', ''), (r.get('sample_id', str(i)), i))
    print(f'✓ test.jsonl: {len(rows)} 行，唯一正文 {len(text2id)} 条')

    df = pd.read_csv(args.predictions)
    print(f'✓ 预测文件: {args.predictions}  {len(df)} 行  列={list(df.columns)}')

    if 'sample_id' in df.columns:
        print('  (已经有 sample_id 列，跳过回填)')
    else:
        if args.keep_text_out not in df.columns:
            sys.exit(f'预测文件里没有 {args.keep_text_out} 列，也没 sample_id，无法回填')
        ids, idxs, miss = [], [], 0
        for t in df[args.keep_text_out]:
            hit = text2id.get(t)
            if hit is None:
                miss += 1
                ids.append(''); idxs.append(-1)
            else:
                ids.append(hit[0]); idxs.append(hit[1])
        if miss:
            print(f'  ✗ 有 {miss} 条正文在 test.jsonl 里找不到，无法回填 sample_id')
            sys.exit(1)
        df.insert(0, 'sample_id', ids)
        df.insert(1, 'idx', idxs)
        print(f'  ✓ 已回填 sample_id / idx（{len(df)} 条全部命中）')

    # 丢正文放在这一步之后，保证「回填」和「去正文」两件事互不影响地都能生效
    if args.drop_text and args.keep_text_out in df.columns:
        df = df.drop(columns=[args.keep_text_out])
        print(f'  ✓ 已丢掉正文列 {args.keep_text_out}')

    # 校验：sample_id 不能重复，且条数要对得上
    if df['sample_id'].duplicated().any():
        sys.exit('✗ 回填后 sample_id 有重复，不能作为主键')
    if len(df) != len(rows):
        print(f'  ⚠ 行数 {len(df)} != test.jsonl 的 {len(rows)}，'
              f'（正常情况应相等；冒烟子集除外）')

    if args.dry_run:
        print('\n--dry-run：没有写任何文件。去掉 --dry-run 就会执行。')
        print(df.head(3).to_string(index=False))
        return 0

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_csv(args.out, index=False, encoding='utf-8-sig')
        print(f'\n✓ 已生成 {args.out}')
    else:
        bak = args.predictions + '.bak'
        shutil.copyfile(args.predictions, bak)
        df.to_csv(args.predictions, index=False, encoding='utf-8-sig')
        print(f'\n✓ 已写回 {args.predictions}（原文件备份为 {bak}）')
    print(df.head(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
