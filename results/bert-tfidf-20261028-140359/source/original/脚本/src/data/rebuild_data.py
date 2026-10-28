# -*- coding: utf-8 -*-
"""本地重建冻结数据集：纯 Python 解析 CSV + 直接复用冻结的 split_ids.json 划分。

为什么不重新抽样
----------------
划分是用 sklearn 的 `train_test_split(random_state=67)` 抽出来的，
但 sklearn 不同版本对 `StratifiedShuffleSplit` 的实现细节有过调整，
换个版本重抽，sample_id 的归属就可能变 —— 一变，后面所有模型的数字全废。

所以这里**不再抽样**：直接读冻结版 `split_ids.json` 里那 7763 条 ID 的归属，
只重建正文（清洗规则是确定性的，源文件 SHA256 锁死）。
这样无论复现者装的是哪个版本的 sklearn/python，划分都一模一样。

不依赖 sklearn（它的 C 扩展在部分 Windows 机器上会被应用控制策略拦截）。

用法
----
    # 默认路径（项目根 = 本文件的上两级）
    python src/data/rebuild_data.py

    # 显式指定
    python src/data/rebuild_data.py \
        --source data/ChnSentiCorp/ChnSentiCorp_htl_all.csv \
        --frozen-dir 冻结资产/data/processed/v1-20261023 \
        --out data/processed/v1-20261023
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

# Windows 控制台默认代码页是 GBK，打印 '✓' / '⚠' 会抛 UnicodeEncodeError。
# 统一把 stdout/stderr 切成 UTF-8 + errors='replace'，任何代码页下都能跑完。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

EXPECT_SHA = '1d414b9c65caf3a836abdb4efded621968502ad3338e1a9d885ae131c86083f0'
MAXCHARS = 512
INSTR = '判断评论整体情感，仅输出0表示负面、1表示正面。'

HERE = Path(__file__).resolve().parent
# 脚本在 <root>/src/data/ 下 → root = 上两级；放别处就退回脚本所在目录
ROOT = HERE.parents[1] if HERE.parent.name == 'src' else HERE


def parse_args():
    ap = argparse.ArgumentParser(description='重建冻结数据集 v1-20261023')
    ap.add_argument('--source', default=str(ROOT / 'data' / 'ChnSentiCorp' / 'ChnSentiCorp_htl_all.csv'),
                    help='原始 ChnSentiCorp CSV（SHA256 会被硬校验）')
    ap.add_argument('--frozen-dir', default=str(ROOT / '冻结资产' / 'data' / 'processed' / 'v1-20261023'),
                    help='冻结资产目录：放 split_ids.json / fewshot_ids.json / smoke_ids.json / latency_ids.json')
    ap.add_argument('--out', default=str(ROOT / 'data' / 'processed' / 'v1-20261023'),
                    help='输出目录')
    return ap.parse_args()


def write_jsonl(p, rows):
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def write_json(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    src = Path(args.source)
    frozen = Path(args.frozen_dir)
    out = Path(args.out)
    print(f'源文件   : {src}')
    print(f'冻结资产 : {frozen}')
    print(f'输出目录 : {out}\n')

    # ---------- 1. 校验源文件 ----------
    if not src.exists():
        raise SystemExit(f'!! 找不到源文件 {src}\n'
                         f'   请先按 README「步骤 1」下载 ChnSentiCorp_htl_all.csv')
    raw_bytes = src.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    print(f'源文件 SHA256: {sha}')
    if sha != EXPECT_SHA:
        raise SystemExit(f'!! SHA256 不匹配！期望 {EXPECT_SHA}\n'
                         f'   下载到的不是同一份数据，后面的数字会对不上。')
    print(f'  ✓ 与冻结版一致 ({len(raw_bytes):,} bytes)\n')

    # ---------- 2. 解析 + 清洗（复刻 process_data.py 的 prepare()） ----------
    text = raw_bytes.decode('utf-8-sig', errors='strict')
    reader = csv.DictReader(io.StringIO(text, newline=''), strict=True)
    assert reader.fieldnames == ['label', 'review'], reader.fieldnames

    records, issues = [], []
    for index, row in enumerate(reader, 1):
        sid = f'{sha}:{index}'
        if None in row or row['review'] is None or row['label'] not in ('0', '1'):
            issues.append(dict(sample_id=sid, source_row=index, reason='invalid_record'))
            continue
        raw = row['review']
        normalized = raw.replace('\r\n', '\n').replace('\r', '\n').strip()
        records.append(dict(sample_id=sid, source_sha256=sha, source_row=index,
                            raw_text=raw, normalized_text=normalized, raw_label=row['label'],
                            label=int(row['label']), raw_char_count=len(raw),
                            normalized_char_count=len(normalized),
                            processing_flags=['minimal_cleaning'] if normalized != raw else []))

    print(f'原始记录 {len(records)+len(issues)}  非法 {len(issues)}')

    # 空文本排除 + 全文精确重复（同标签留最早）
    excluded = []
    groups = {}
    for r in sorted(records, key=lambda x: x['source_row']):
        if not r['normalized_text']:
            excluded.append(dict(r, reason='empty_text'))
            continue
        groups.setdefault(r['normalized_text'], []).append(r)
    valid = []
    for g in groups.values():
        valid.append(g[0])
        for r in g[1:]:
            excluded.append(dict(r, reason='duplicate_same_label', retained_id=g[0]['sample_id']))
    valid.sort(key=lambda r: r['source_row'])

    # 512 截断 + 截断前缀碰撞（保留源序号最小者）
    for r in valid:
        r.update(text=r['normalized_text'][:MAXCHARS],
                 text_char_count=min(MAXCHARS, len(r['normalized_text'])),
                 char_truncated=len(r['normalized_text']) > MAXCHARS)
    seen, retained = set(), []
    for r in valid:
        if r['text'] in seen:
            excluded.append(dict(r, reason='duplicate_truncated_prefix'))
        else:
            seen.add(r['text'])
            retained.append(r)
    valid = retained
    print(f'  排除明细: {dict(Counter(e["reason"] for e in excluded))}')
    print(f'最终有效 {len(valid)}  (期望 7763)\n')
    if len(valid) != 7763:
        raise SystemExit('!! 有效记录数不是 7763，源文件或清洗规则对不上')

    # ---------- 3. 用冻结的 split_ids.json 划分 ----------
    split_path = frozen / 'split_ids.json'
    if not split_path.exists():
        raise SystemExit(f'!! 找不到 {split_path}\n'
                         f'   复现包里的「冻结资产/data/processed/v1-20261023/split_ids.json」必须存在。')
    split_ids = json.loads(split_path.read_text(encoding='utf-8'))
    by_id = {r['sample_id']: r for r in valid}
    for name in ('train', 'validation', 'test'):
        miss = [i for i in split_ids[name] if i not in by_id]
        if miss:
            raise SystemExit(f'!! {name} 有 {len(miss)} 个 id 不在有效集里')
    print('✓ 所有 split_ids 都能在有效集中找到')

    splits = {}
    for name in ('train', 'validation', 'test'):
        rows = [by_id[i] for i in split_ids[name]]
        rows.sort(key=lambda r: r['source_row'])
        for r in rows:
            r['split'] = name
        splits[name] = rows
        print(f'  {name:>10}: {len(rows):>5}  '
              f'负面0={sum(1 for r in rows if r["label"]==0):>5}  '
              f'正面1={sum(1 for r in rows if r["label"]==1):>5}')

    # ---------- 4. 落盘 ----------
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(out / f'{name}.jsonl', rows)
    write_jsonl(out / 'excluded.jsonl', excluded)
    write_jsonl(out / 'invalid_records.jsonl', issues)
    write_jsonl(out / 'conflicts.jsonl', [])

    alpaca = out / 'alpaca'
    alpaca.mkdir(exist_ok=True)
    for name, rows in splits.items():
        write_json(alpaca / f'{name}.json',
                   [dict(instruction=INSTR, input=r['text'], output=str(r['label'])) for r in rows])
        write_jsonl(alpaca / f'{name}_index.jsonl',
                    [dict(index=i, sample_id=r['sample_id']) for i, r in enumerate(rows)])
    write_json(alpaca / 'dataset_info.json', {
        f'sentiment_{s}': dict(file_name=f'{s}.json',
                               columns=dict(prompt='instruction', query='input', response='response'))
        for s in ['train', 'validation']})

    # 固定子集（冒烟 / 少样本 / 延迟）从冻结资产拷过来，不要重新抽
    for name in ('fewshot_ids.json', 'smoke_ids.json', 'latency_ids.json'):
        src_f = frozen / name
        if src_f.exists():
            shutil.copy2(src_f, out / name)
            print(f'  ✓ 拷入固定子集 {name}')
        else:
            print(f'  ⚠ 冻结资产里没有 {name}，跳过')

    # 交叉验证：正文无重复、跨划分无重复
    all_ids, all_texts = set(), set()
    ok = True
    for name, rows in splits.items():
        for r in rows:
            if r['sample_id'] in all_ids or r['normalized_text'] in all_texts:
                ok = False
            all_ids.add(r['sample_id'])
            all_texts.add(r['normalized_text'])
    print(f'✓ 跨划分 id/正文唯一性: {"通过" if ok else "失败"}')
    if not ok:
        raise SystemExit('!! 跨划分出现重复')

    # ---------- 5. 统计 ----------
    lengths = sorted(r['normalized_char_count'] for r in valid)
    stats = dict(
        source_sha256=sha,
        total_records=len(records) + len(issues),
        valid_records=len(valid),
        raw_labels=dict(Counter(r['label'] for r in records)),
        invalid_records=len(issues),
        exclusions=dict(Counter(e['reason'] for e in excluded)),
        conflict_records=0,
        truncated_count=sum(r['char_truncated'] for r in valid),
        truncated_ratio=sum(r['char_truncated'] for r in valid) / len(valid),
        length_distribution={str(q): lengths[round((len(lengths)-1)*q)] for q in [0, .25, .5, .75, .95, 1]},
        splits={s: dict(count=len(rows), labels=dict(Counter(r['label'] for r in rows)))
                for s, rows in splits.items()},
    )
    write_json(out / 'statistics.json', stats)

    # 与冻结版 statistics.json 比对
    ref_path = frozen / 'statistics.json'
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding='utf-8'))
        print('\n===== 与冻结版 statistics.json 比对 =====')
        all_ok = True
        for k in ['source_sha256', 'total_records', 'valid_records', 'raw_labels', 'exclusions',
                  'conflict_records', 'truncated_count', 'truncated_ratio', 'length_distribution', 'splits']:
            # 比的是「落到磁盘上的那份 JSON」：json.dumps 会把 int 键写成字符串，
            # 直接比内存里的 dict 会因为 {1: x} vs {'1': x} 误判为不一致。
            same = (json.dumps(stats.get(k), sort_keys=True, ensure_ascii=False)
                    == json.dumps(ref.get(k), sort_keys=True, ensure_ascii=False))
            all_ok &= same
            print(f'  {"✅" if same else "❌"} {k}: 本地={stats.get(k)}  冻结={ref.get(k)}')
        print(f'\n  {"✅ 全部一致 —— 划分与冻结版逐位相同" if all_ok else "❌ 存在不一致"}')
        if not all_ok:
            raise SystemExit('!! 重建结果与冻结版不一致')

    print('\n===== 长度分布（字符）=====')
    for q, v in stats['length_distribution'].items():
        print(f'  P{float(q)*100:>5.0f}: {v}')
    print(f'  >512字(截断): {stats["truncated_count"]} ({stats["truncated_ratio"]*100:.2f}%)')
    print('\n✓ 数据已就绪 ->', out)
    print('DONE')


if __name__ == '__main__':
    main()
