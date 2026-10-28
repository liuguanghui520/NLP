# -*- coding: utf-8 -*-
"""
复现校验器：把复现者跑出来的结果和「期望结果」逐位对比，输出 PASS/FAIL。

为什么需要这个
--------------
复现的难点不是"跑通了"，而是"跑出来的数字和原来一样"。
本项目里最容易悄悄跑歪的地方有四个：
    1. 数据划分变了（种子/清洗策略/截断长度不同）→ sample_id 对不上；
    2. 指标口径变了（weighted F1 的加权方式不同）→ 数字看着像但不可比；
    3. 选代表组的规则变了（拿测试集选 vs 拿验证集选）→ 主结果对不上；
    4. 随机性没控住 → 同样的配置跑出不一样的数。

这个脚本把上面四件事变成**一次自动判定**：任何一格对不上就报 FAIL。

判定分三档（重要）
------------------
    【严格】指标、逐条预测标签、样本量、检验的 b/c/p
           —— 必须逐位一致，任何一处不符 = 复现失败（退出码 1）。
    【提示】计时类字段（seconds / mean_ms / bootstrap CI）
           —— 跟 CPU 型号、负载、numpy 小版本有关，本来就不该期待一致。
           超阈值只提示（⚠），**不判失败**，但会把「期望 vs 实得」摆出来。
    【结构】行数、列名、划分条数、数据 SHA256
           —— 结构对不上说明流程走错了，属于硬错误。

用法
----
    # 在复现包解出来的项目根目录下
    python tools/verify_repro.py --results src/results --expect 期望结果

    # 顺带核对数据划分是否与冻结版一致
    python tools/verify_repro.py --data data/processed/v1-20261023

退出码 0 = 严格项全部一致；1 = 有不一致。
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd

# 浮点容差：所有写进 CSV 的指标都经过 round(x, 6) 或 round(x, 4)，
# 两边各自取整后应当逐位相同；1e-6 足以判等，同时留一丝余量。
FLOAT_TOL = 1e-6

# (相对路径, 对齐主键列, 严格比对列, 仅提示列, 说明)
CHECKS = [
    ('results_TFIDF/tfidf_ablation.csv',
     ['config'],
     ['val_acc', 'test_acc', 'macro_f1', 'weighted_f1', 'n_features', 'C'],
     ['seconds'],
     'TF-IDF 4 组 n-gram'),
    ('results_BERT_FULL/ablation_results.csv',
     ['config'],
     ['test_acc', 'macro_f1', 'weighted_f1', 'trainable_params', 'trainable_pct'],
     ['seconds'],
     'MacBERT 8 组超参'),
    ('results_BERT_FULL/data_curve.csv',
     ['n_train'],
     ['test_acc', 'macro_f1', 'weighted_f1'],
     [],
     '数据量曲线 6 点'),
    ('results_BERT_FULL/batching_control.csv',
     ['n_train', 'batching'],
     ['test_acc', 'macro_f1', 'weighted_f1'],
     ['seconds'],
     '分组方式对照 2 组'),
    ('results_BERT_FULL/latency_baseline.csv',
     ['variant'],
     ['n'],
     ['mean_ms', 'p50_ms', 'p95_ms', 'max_ms'],
     '基线两臂延迟（硬件相关，只核对样本量）'),
    ('paired_tests.csv',
     ['kind', 'key_a', 'key_b'],
     ['route_a', 'route_b', 'n', 'acc_a', 'acc_b', 'delta_acc_pp',
      'mcnemar_b', 'mcnemar_c', 'mcnemar_p', 'significant_95'],
     ['agree_rate', 'ci95_acc_lo_pp', 'ci95_acc_hi_pp'],
     '配对显著性检验 21 组'),
]

# 逐条预测：每一条都要对得上，这是最硬的证据
# 注意 predictions.csv（代表组）不能被 predictions_*.csv 这个通配符盖住，要单列。
PRED_GLOBS = [
    'results_BERT_FULL/predictions.csv',
    'results_BERT_FULL/predictions_*.csv',
    'results_TFIDF/predictions.csv',
]

# 提示档的阈值：(相对容差, 绝对容差)
LOOSE_RULES = {
    'seconds':           (0.40, 30.0),    # 机器快慢差 40% 以内算正常
    'mean_ms':           (0.60, 5.0),     # 硬件差异
    'p50_ms':            (0.60, 5.0),
    'p95_ms':            (0.60, 5.0),
    'max_ms':            (0.60, 5.0),
    'agree_rate':        (0.0, 1e-6),
    'ci95_acc_lo_pp':    (0.0, 0.5),      # bootstrap 区间，0.5pp 内算一致
    'ci95_acc_hi_pp':    (0.0, 0.5),
}


def _cmp_value(a, b, col):
    """严格比较两个单元格，返回 (是否相同, 期望值, 实得值)。"""
    if isinstance(a, (int, float)) and not isinstance(a, bool) and \
       isinstance(b, (int, float)) and not isinstance(b, bool):
        try:
            return abs(float(a) - float(b)) <= FLOAT_TOL, (a, b)
        except Exception:
            return str(a) == str(b), (a, b)
    return str(a).strip() == str(b).strip(), (a, b)


def _cmp_loose(a, b, col):
    """提示档比较：超过阈值才算"有明显差异"。返回 (是否在阈值内, 期望值, 实得值)。"""
    rel, abs_ = LOOSE_RULES.get(col, (0.0, FLOAT_TOL))
    try:
        fa, fb = float(a), float(b)
    except Exception:
        return str(a).strip() == str(b).strip(), (a, b)
    diff = abs(fa - fb)
    ok = diff <= max(abs_, rel * max(abs(fa), abs(fb)))
    return ok, (a, b)


def _row_key(row, keys):
    return '\u0001'.join(str(row[k]) for k in keys)


def compare_csv(exp_path, got_path, keys, strict_cols, loose_cols):
    """返回 (严格是否全过, 严格说明, 提示项列表)。"""
    if not os.path.exists(got_path):
        return False, f'缺少文件 {os.path.relpath(got_path)}', []
    try:
        e = pd.read_csv(exp_path)
        g = pd.read_csv(got_path)
    except Exception as ex:
        return False, f'读取失败：{type(ex).__name__}: {ex}', []

    if len(e) != len(g):
        return False, f'行数不同：期望 {len(e)} 行，实得 {len(g)} 行', []

    keys = [k for k in keys if k in e.columns and k in g.columns]
    if not keys:
        return False, f'没有可用的对齐主键（期望表列：{list(e.columns)[:6]}）', []

    strict_cols = [c for c in strict_cols if c in e.columns and c in g.columns]
    loose_cols = [c for c in loose_cols if c in e.columns and c in g.columns]
    if not strict_cols:
        return False, f'没有可比较的共同列（期望 {list(e.columns)[:6]}）', []

    emap = {_row_key(r, keys): r for _, r in e.iterrows()}
    gmap = {_row_key(r, keys): r for _, r in g.iterrows()}

    bad, warns = [], []
    for k, er in emap.items():
        if k not in gmap:
            bad.append(f'缺行 {k}')
            continue
        gr = gmap[k]
        for c in strict_cols:
            ok, (ev, gv) = _cmp_value(er[c], gr.get(c), c)
            if not ok:
                bad.append(f'{k}.{c}: 期望 {ev} ≠ 实得 {gv}')
        for c in loose_cols:
            ok, (ev, gv) = _cmp_loose(er[c], gr.get(c), c)
            if not ok:
                warns.append(f'{k}.{c}: 期望 {ev}，实得 {gv}（硬件/计时差异，不判失败）')

    if bad:
        return False, f'{len(bad)} 处不一致，例如：' + '；'.join(bad[:3]), warns
    return True, f'{len(e)} 行 × {len(strict_cols)} 列全部一致', warns


def compare_predictions(exp_path, got_path):
    """逐条预测比对，返回 (是否全过, 说明)。"""
    try:
        e = pd.read_csv(exp_path)
        if not os.path.exists(got_path):
            return False, '缺少文件'
        g = pd.read_csv(got_path)
        if len(e) != len(g):
            return False, f'行数不同 {len(e)} vs {len(g)}'
        cols = [c for c in ('true_label', 'pred_label')
                if c in e.columns and c in g.columns]
        if 'sample_id' not in e.columns or 'sample_id' not in g.columns:
            return False, '缺少 sample_id 列'
        em = e.set_index('sample_id')
        gm = g.set_index('sample_id')
        diff = 0
        first = ''
        for sid, er in em.iterrows():
            if sid not in gm.index:
                diff += 1
                first = first or f'缺 sample_id {sid}'
                continue
            gr = gm.loc[sid]
            if isinstance(gr, pd.DataFrame):
                gr = gr.iloc[0]
            for c in cols:
                ok, (ev, gv) = _cmp_value(er[c], gr.get(c), c)
                if not ok:
                    diff += 1
                    first = first or f'{sid}.{c}: {ev}≠{gv}'
                    break
        if diff == 0:
            return True, f'{len(e)} 条逐条一致'
        return False, f'{diff} 条不一致，例如 {first}'
    except Exception as ex:
        return False, f'{type(ex).__name__}: {ex}'


def check_frozen_data(expect_frozen, data_dir):
    """核对复现者自己生成的数据划分是否与冻结版一致。"""
    if not data_dir or not os.path.isdir(data_dir):
        return None, '跳过（未提供 --data 或目录不存在）'
    exp_stat = os.path.join(expect_frozen, 'statistics.json')
    got_stat = os.path.join(data_dir, 'statistics.json')
    if not os.path.exists(exp_stat):
        return None, '跳过（包里没有冻结版 statistics.json）'
    if not os.path.exists(got_stat):
        return False, f'缺少 {os.path.relpath(got_stat)}（是不是没跑 process_data.py？）'
    try:
        e = json.loads(open(exp_stat, encoding='utf-8').read())
        g = json.loads(open(got_stat, encoding='utf-8').read())
    except Exception as ex:
        return False, f'读取失败：{type(ex).__name__}: {ex}'

    bad = []
    for field in ('source_sha256', 'total_records', 'valid_records',
                  'raw_labels', 'exclusions', 'conflict_records',
                  'truncated_count', 'length_distribution', 'splits'):
        if field not in g:
            bad.append(f'缺字段 {field}')
        elif e.get(field) != g.get(field):
            bad.append(f'{field}: 期望 {e.get(field)} ≠ 实得 {g.get(field)}')
    if bad:
        return False, f'{len(bad)} 处不一致：' + '；'.join(bad[:3])
    return True, ('SHA256 / 记录数 / 排除规则 / 长度分布 / 三划分条数 '
                  '（6210 / 776 / 777）全部一致')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='src/results',
                    help='复现者跑出来的结果目录')
    ap.add_argument('--expect', default='期望结果',
                    help='期望结果目录（包内自带）')
    ap.add_argument('--frozen', default='冻结资产/data/processed/v1-20261023',
                    help='包内的冻结资产目录（用于核对数据划分）')
    ap.add_argument('--data', default=None,
                    help='复现者自己的数据目录，用于核对划分是否一致')
    args = ap.parse_args()

    print('=' * 74)
    print('  复现校验：把跑出来的结果和期望结果逐位对比')
    print('=' * 74)
    print(f'  复现结果：{os.path.abspath(args.results)}')
    print(f'  期望结果：{os.path.abspath(args.expect)}\n')

    passed, failed = 0, []
    warn_count = 0

    print('  【严格】结果文件逐位比对')
    for rel, keys, strict_cols, loose_cols, desc in CHECKS:
        exp = os.path.join(args.expect, rel)
        got = os.path.join(args.results, rel)
        if not os.path.exists(exp):
            print(f'  ➖ {desc:24s} 期望文件缺失，跳过')
            continue
        ok, msg, warns = compare_csv(exp, got, keys, strict_cols, loose_cols)
        warn_count += len(warns)
        print(f'  {"✅" if ok else "❌"} {desc:24s} {msg}')
        for w in warns:
            print(f'       ⚠ {w}')
        if ok:
            passed += 1
        else:
            failed.append(desc)

    # ---- 逐条预测 ----
    print('\n  【严格】逐条预测（777 条 × 每组配置，最硬的证据）')
    import glob
    for pat in PRED_GLOBS:
        for exp in sorted(glob.glob(os.path.join(args.expect, pat))):
            rel = os.path.relpath(exp, args.expect)
            got = os.path.join(args.results, rel)
            name = os.path.basename(rel)
            ok, msg = compare_predictions(exp, got)
            print(f'  {"✅" if ok else "❌"} {name:32s} {msg}')
            if ok:
                passed += 1
            else:
                failed.append(name)

    # ---- 数据划分 ----
    print('\n  【结构】数据划分与冻结版是否一致')
    ok, msg = check_frozen_data(args.frozen, args.data)
    if ok is None:
        print(f'  ➖ {msg}')
    elif ok:
        print(f'  ✅ {msg}')
        passed += 1
    else:
        print(f'  ❌ {msg}')
        failed.append('数据划分')

    print('\n' + '=' * 74)
    if warn_count:
        print(f'  ⚠ 另有 {warn_count} 处计时/硬件类差异，已列出但不影响复现判定。')
    if failed:
        print(f'  结论：❌ {len(failed)} 项不一致 —— ' + '；'.join(failed[:6]))
        print('  排查顺序见 README 的「复现失败怎么查」一节。')
        print('=' * 74)
        return 1
    print(f'  结论：✅ 全部 {passed} 项一致 —— 复现成功')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    sys.exit(main())
