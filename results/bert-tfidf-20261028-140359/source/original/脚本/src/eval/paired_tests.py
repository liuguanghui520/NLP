"""批量配对显著性检验 —— 把所有可用路线的逐条预测两两配对做统计检验。

为什么必须有这一步
------------------
777 条测试集上，单个准确率的标准误约 1.1 个百分点。
也就是说，**表面上一个百分点的差距很可能只是噪声**。
只比汇总数字，很容易把随机波动写成"某超参更好"的结论。
配对检验（McNemar + paired bootstrap）利用的是**同一条样本上的差异**，
才能回答"这个差距是不是真的"。

它做什么
--------
1. 自动发现所有已有的预测文件（TF-IDF / MacBERT 各配置 / LoRA / 提示）；
2. 按规则配对：每个 MacBERT 配置 vs main、TF-IDF vs 每个 MacBERT 配置、
   LoRA/提示 vs main（存在才做）；
3. 逐对做 McNemar 精确检验 + 10000 次配对 bootstrap，出 Δ 与 95% 置信区间；
4. 输出两个文件：
     src/results/paired_tests.csv   机器可读，一行一对
     src/results/paired_tests.md    直接可贴进报告的 Markdown 表格

用法
----
    python src/eval/paired_tests.py
    python src/eval/paired_tests.py --root . --n-boot 10000 --seed 42

注意：某条路线的预测文件不存在时**跳过**，并在输出里写明跳过了什么，
不会用别的数字顶替。
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from evaluate import compare_two, load_predictions  # noqa: E402

ALPHA = 0.05

# MacBERT 各配置的可读名字（拿不到超参表时按这张表退）
BERT_LABEL_FALLBACK = {
    'main': 'MacBERT 全参（基线配置）',
    'partial': 'MacBERT 冻结下 6 层',
    'weighted': 'MacBERT 类别加权',
    'lr_5e5': 'MacBERT lr=5e-5',
    'len128': 'MacBERT len=128',
    'len512': 'MacBERT len=512',
    'lr_1e5': 'MacBERT lr=1e-5',
    'ep5': 'MacBERT 5 epoch',
    # 分组方式对照（不是超参配置，见 _is_ctrl）
    'ctrl_len_group': 'MacBERT 长度分桶组 batch',
    'ctrl_random': 'MacBERT 随机组 batch',
}


def _bert_labels(bert_dir):
    """用 ablation_results.csv 里的真实超参生成名字（要短，表格才读得下去）。"""
    labels = dict(BERT_LABEL_FALLBACK)
    ab = bert_dir / 'ablation_results.csv'
    if not ab.exists():
        return labels
    try:
        df = pd.read_csv(ab)
    except Exception:
        return labels
    for _, r in df.iterrows():
        cfg = str(r['config'])
        if cfg == 'main':
            labels[cfg] = 'MacBERT main（lr2e-5, len256）'
            continue
        desc = []
        if int(r.get('freeze_lower', 0) or 0) > 0:
            desc.append(f"冻下{int(r['freeze_lower'])}层")
        if str(r.get('class_weight', 'False')).lower() in ('true', '1'):
            desc.append('类别加权')
        if int(r.get('maxlen', 256)) != 256 and 'len' not in cfg:
            desc.append(f"len{int(r['maxlen'])}")
        elif float(r.get('lr', 2e-5)) != 2e-5 and 'lr' not in cfg:
            desc.append(f"lr={r['lr']}")
        if int(r.get('epochs', 3)) != 3:
            desc.append(f"{int(r['epochs'])}epoch")
        labels[cfg] = 'MacBERT ' + cfg + (f"（{'，'.join(desc)}）" if desc else '')
    return labels


def discover(root):
    """找出所有可用的预测文件 -> [(key, 展示名, 路径)]"""
    found = []
    rs = root / 'src' / 'results'

    tf = rs / 'results_TFIDF' / 'predictions.csv'
    if tf.exists():
        found.append(('tfidf', 'TF-IDF char_1_3', tf))

    bert_dir = rs / 'results_BERT_FULL'
    if bert_dir.is_dir():
        labels = _bert_labels(bert_dir)
        for p in sorted(bert_dir.glob('predictions_*.csv')):
            cfg = p.stem.replace('predictions_', '')
            if cfg in ('predictions',):      # 兼容旧命名 predictions.csv
                continue
            # 数据量曲线的规模点：显示成 "MacBERT n=100" 这种可读形式
            if cfg.startswith('curve_n'):
                lbl = f'MacBERT n={cfg[len("curve_n"):]}'
            else:
                lbl = labels.get(cfg, f'MacBERT {cfg}')
            found.append((f'bert:{cfg}', lbl, p))

    # 同学的两条臂：目录名不同，文件名也各写各的，这里把能认出来的都收上
    for sub, prefix in (('results_QWEN_FULL', 'Qwen3+LoRA'),
                        ('results_PROMPT_FULL', 'Qwen3 提示')):
        d = rs / sub
        if not d.is_dir():
            continue
        cands = [p for p in sorted(d.glob('*.csv'))
                 if 'pred' in p.stem.lower() and p.stem.lower() != 'predictions']
        if not cands and (d / 'predictions.csv').exists():
            cands = [d / 'predictions.csv']
        for p in cands:
            found.append((f'{sub}:{p.stem}', f'{prefix} {p.stem}', p))

    return found


def _hyper_names(bert_dir):
    """从 ablation_results.csv 读出**真正的超参配置名**集合。

    为什么不能靠前缀判断：predictions_*.csv 里除了 8 组超参，还有两类东西
        - ctrl_*   ：分组方式对照实验（长度分桶 vs 随机）
        - curve_n* ：数据量曲线的各个规模点
    它们都不是超参。如果按"bert: 开头就算超参"来分组，这些会混进去，
    报告里"7 组超参全部不显著"就会变成 12 组，而且和 8 组超参表对不上。
    """
    p = bert_dir / 'ablation_results.csv'
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return {str(x) for x in df['config']}
    except Exception:
        return None


def _is_ctrl(key):
    """分组方式对照实验的配置（ctrl_len_group / ctrl_random）。"""
    return key.startswith('bert:ctrl_')


def _is_curve(key):
    """数据量曲线的各规模点（curve_n100 / curve_n500 / ...）。"""
    return key.startswith('bert:curve_n')


def build_pairs(found, bert_dir=None, rep_key='bert:main'):
    """按规则生成要检验的配对 [(key_a, key_b, 说明)]

    rep_key 是**代表组**（按验证集选出来的那一组，见 representative.py）。
    全项目统一用代表组当基准，四条路线之间才是"代表 vs 代表"地比。
    """
    by_key = {k: (label, path) for k, label, path in found}
    pairs = []

    hyper = _hyper_names(bert_dir) if bert_dir is not None else None

    def _is_hyper(k, include_rep=False):
        """真正的超参配置（代表组单独看：它既是基准组，也是一条要跟 TF-IDF 比的路线）。"""
        if not k.startswith('bert:'):
            return False
        if k == rep_key:
            return include_rep
        if hyper is not None:
            return k.split(':', 1)[1] in hyper
        # 拿不到超参表时退回"排除已知非超参前缀"
        return not (_is_ctrl(k) or _is_curve(k))

    # 1) 每个 MacBERT 超参配置 vs 代表组
    if rep_key in by_key:
        for k in by_key:
            if _is_hyper(k):
                pairs.append((k, rep_key, '超参对比'))

    # 2) TF-IDF vs 每个 MacBERT 超参配置（**含代表组**：
    #    "TF-IDF vs 微调"是整个项目的核心对照，不能漏）
    if 'tfidf' in by_key:
        for k in by_key:
            if _is_hyper(k, include_rep=True):
                pairs.append(('tfidf', k, '基线 vs 微调'))

    # 3) 同学的两条臂 vs 代表组
    for pref in ('results_QWEN_FULL', 'results_PROMPT_FULL'):
        for k in by_key:
            if k.startswith(pref + ':') and rep_key in by_key:
                pairs.append((k, rep_key, '路线对比'))

    # 4) 分组方式对照之间互比：长度分桶 vs 随机组 batch
    #    这一对直接回答"为了提速用的长度分桶，有没有偷走精度"，
    #    必须自己做检验，不能只报两个汇总数的差。
    ctrls = [k for k in by_key if _is_ctrl(k)]
    for i in range(len(ctrls)):
        for j in range(i + 1, len(ctrls)):
            pairs.append((ctrls[i], ctrls[j], '分组方式对照'))

    # 5) 数据量曲线的每个规模点 vs 全量（代表组）
    #    回答"多给几千条标注数据到底换不换得来精度"——
    #    这是提示/少样本方案最核心的对照，曲线点之间的差异都在 1 个百分点内，
    #    只看汇总数字很容易把噪声写成"数据越多越好"。
    if rep_key in by_key:
        for k in by_key:
            if _is_curve(k):
                pairs.append((k, rep_key, '数据量曲线'))

    # 去重且保持顺序
    seen, out = set(), []
    for a, b, why in pairs:
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b, why))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[2]),
                    help='项目根目录（默认自动推断）')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / 'src' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    found = discover(root)
    if not found:
        print('!! 没找到任何预测文件，先跑完至少一条路线')
        return 1

    print('发现的预测文件：')
    for k, label, p in found:
        print(f'  {k:28s} {label:30s} {p.name}')

    by_key = {k: (label, path) for k, label, path in found}
    bert_dir = root / 'src' / 'results' / 'results_BERT_FULL'

    # ★ 基准组 = 代表组（按验证集选出来的那一组，不是"main"也不是测试集最高的）
    rep_cfg, _rep_row, rep_val = None, None, None
    try:
        from representative import pick_representative
        rep_cfg, _rep_row, rep_val = pick_representative(str(bert_dir))
    except Exception as e:
        print(f'  ⚠ 选代表组失败（退回 main）：{type(e).__name__}: {e}')
    rep_key = f'bert:{rep_cfg}' if rep_cfg else 'bert:main'
    if rep_key not in by_key:
        rep_key = 'bert:main'
    if rep_cfg:
        vs = f'（验证集 acc {rep_val:.4f}）' if rep_val is not None else ''
        print(f'\n★ 代表组 = {rep_cfg} {vs}')
        print('  （按验证集选，与 TF-IDF / LoRA / 提示臂用同一条规则）')

    pairs = build_pairs(found, bert_dir, rep_key=rep_key)
    if not pairs:
        print('\n!! 可用路线不足两组，没有可配对的组合')
        return 1

    rows, md = [], []
    print(f'\n做 {len(pairs)} 组配对检验（McNemar 精确检验 + {args.n_boot} 次配对 bootstrap）\n')

    for ka, kb, why in pairs:
        la, pa = by_key[ka]
        lb, pb = by_key[kb]
        try:
            dfa = load_predictions(pa)
            dfb = load_predictions(pb)
            res = compare_two(dfa, dfb, la, lb, n_boot=args.n_boot, seed=args.seed)
        except Exception as e:
            print(f'  ✗ {la} vs {lb}: {type(e).__name__}: {e}')
            continue

        mcn = res['mcnemar']
        boot = res['bootstrap']
        da = res['delta']['accuracy']
        sig = res['significant_95']
        print(f'  {la} vs {lb}')
        print(f'    Δacc = {da["diff_pp"]:+.2f} pp   McNemar b={mcn["b"]} c={mcn["c"]} '
              f'p={mcn["p_value"]:.4f} {"(显著)" if sig else "(不显著)"}')
        print(f'    bootstrap Δacc 95% CI = '
              f'[{boot["accuracy"]["ci95_lo"]*100:+.2f}, '
              f'{boot["accuracy"]["ci95_hi"]*100:+.2f}] pp')

        rows.append(dict(
            kind=why, key_a=ka, key_b=kb, route_a=la, route_b=lb, n=res['n'],
            acc_a=res['routes'][la]['accuracy'], acc_b=res['routes'][lb]['accuracy'],
            delta_acc_pp=round(da['diff_pp'], 4),
            macro_f1_a=res['routes'][la]['macro_f1'],
            macro_f1_b=res['routes'][lb]['macro_f1'],
            delta_macro_f1_pp=round(res['delta']['macro_f1']['diff_pp'], 4),
            weighted_f1_a=res['routes'][la]['weighted_f1'],
            weighted_f1_b=res['routes'][lb]['weighted_f1'],
            delta_weighted_f1_pp=round(res['delta']['weighted_f1']['diff_pp'], 4),
            agree_rate=res['agreement']['agree_rate'],
            mcnemar_b=mcn['b'], mcnemar_c=mcn['c'],
            mcnemar_p=mcn['p_value'],
            ci95_acc_lo_pp=round(boot['accuracy']['ci95_lo'] * 100, 4),
            ci95_acc_hi_pp=round(boot['accuracy']['ci95_hi'] * 100, 4),
            significant_95=sig,
        ))

    if not rows:
        print('\n!! 所有配对都失败了')
        return 1

    df = pd.DataFrame(rows)
    csv_path = out_dir / 'paired_tests.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ---- Markdown 表格（可直接贴进报告）----
    md.append('| 对比 | ΔAccuracy | McNemar b/c | McNemar p | bootstrap Δacc 95% CI | 结论 |')
    md.append('|---|---|---|---|---|---|')
    for r in rows:
        ci = f'[{r["ci95_acc_lo_pp"]:+.2f}, {r["ci95_acc_hi_pp"]:+.2f}] pp'
        verdict = '**显著**' if r['significant_95'] else '不显著'
        md.append(f'| {r["route_a"]} vs {r["route_b"]} | {r["delta_acc_pp"]:+.2f} pp | '
                  f'{r["mcnemar_b"]}/{r["mcnemar_c"]} | {r["mcnemar_p"]:.4f} | {ci} | {verdict} |')
    md_text = '\n'.join(md)
    md_path = out_dir / 'paired_tests.md'
    md_path.write_text(md_text + '\n', encoding='utf-8')

    # ---- 关键结论数字（给 fill_report.py 用） ----
    key = {}
    key['PT_REP_CFG'] = rep_cfg or 'main'
    if rep_val is not None:
        key['PT_REP_VAL'] = f'{rep_val:.4f}'
    # TF-IDF vs 代表组（基线配置）—— 这是整个项目最核心的一对
    tf_rows = [r for r in rows if r['key_a'] == 'tfidf' and r['key_b'] == rep_key]
    if tf_rows:
        r = tf_rows[0]
        key['PT_TFIDF_VS_BERT_DELTA'] = f'{r["delta_acc_pp"]:+.2f}'
        key['PT_TFIDF_VS_BERT_P'] = f'{r["mcnemar_p"]:.4f}'
        key['PT_TFIDF_VS_BERT_CI'] = f'[{r["ci95_acc_lo_pp"]:+.2f}, {r["ci95_acc_hi_pp"]:+.2f}]'
        key['PT_TFIDF_VS_BERT_ABS'] = f'{abs(r["delta_acc_pp"]):.2f}'
        key['PT_TFIDF_VS_BERT_SIG'] = '显著' if r['significant_95'] else '不显著'
        key['PT_TFIDF_VS_BERT_B'] = str(r['mcnemar_b'])
        key['PT_TFIDF_VS_BERT_C'] = str(r['mcnemar_c'])

    # MacBERT 各配置 vs main：是不是全部不显著
    hp_rows = [r for r in rows if r['kind'] == '超参对比']
    if hp_rows:
        key['PT_HP_COUNT'] = str(len(hp_rows))
        key['PT_HP_ALL_NS'] = '全部不显著' if not any(r['significant_95'] for r in hp_rows) else '存在显著差异'
        worst = max(hp_rows, key=lambda r: abs(r['delta_acc_pp']))
        key['PT_HP_MAXABS_DELTA'] = f'{abs(worst["delta_acc_pp"]):.2f}'
        key['PT_HP_MAXABS_PAIR'] = f'{worst["route_a"]} vs {worst["route_b"]}'
        key['PT_HP_MAXABS_P'] = f'{worst["mcnemar_p"]:.4f}'
        key['PT_HP_MAXABS_CI'] = f'[{worst["ci95_acc_lo_pp"]:+.2f}, {worst["ci95_acc_hi_pp"]:+.2f}]'
        key['PT_HP_MAXABS_SIG'] = '显著' if worst['significant_95'] else '不显著'
        key['PT_HP_MIN_P'] = f'{min(r["mcnemar_p"] for r in hp_rows):.4f}'
        key['PT_HP_MAX_P'] = f'{max(r["mcnemar_p"] for r in hp_rows):.4f}'

    # 数据量曲线各规模点 vs 全量 main
    # 回答"多给几千条标注数据到底换不换得来精度"——
    # 曲线点之间的差异都在 1 个百分点上下，只看汇总数字很容易把噪声写成"数据越多越好"。
    cv_rows = [r for r in rows if r['kind'] == '数据量曲线']
    if cv_rows:
        key['PT_CURVE_COUNT'] = str(len(cv_rows))
        key['PT_CURVE_ALL_NS'] = ('全部不显著'
                                  if not any(r['significant_95'] for r in cv_rows)
                                  else '存在显著差异')
        _cw = max(cv_rows, key=lambda r: abs(r['delta_acc_pp']))
        key['PT_CURVE_MAXABS_DELTA'] = f'{abs(_cw["delta_acc_pp"]):.2f}'
        key['PT_CURVE_MAXABS_PAIR'] = f'{_cw["route_a"]} vs {_cw["route_b"]}'
        key['PT_CURVE_MAXABS_P'] = f'{_cw["mcnemar_p"]:.4f}'
        key['PT_CURVE_MAXABS_CI'] = f'[{_cw["ci95_acc_lo_pp"]:+.2f}, {_cw["ci95_acc_hi_pp"]:+.2f}]'
        key['PT_CURVE_MAXABS_SIG'] = '显著' if _cw['significant_95'] else '不显著'
        key['PT_CURVE_MIN_P'] = f'{min(r["mcnemar_p"] for r in cv_rows):.4f}'
        key['PT_CURVE_MAX_P'] = f'{max(r["mcnemar_p"] for r in cv_rows):.4f}'
        # 显著的点单独列出来（如果有）
        _sig = [f'{r["route_a"]}（Δ{r["delta_acc_pp"]:+.2f}pp, p={r["mcnemar_p"]:.4f}）'
                for r in cv_rows if r['significant_95']]
        key['PT_CURVE_SIG_PTS'] = '；'.join(_sig) if _sig else '无'
        # 逐点结论（数据驱动，不预写）
        if not _sig:
            key['PT_CURVE_VERDICT'] = (
                '**所有规模点与全量的差异都落在评测噪声内** —— '
                '所以曲线上 500 条之后那些零点几个百分点的起伏，'
                '**不能**读成"数据越多越好"，它只是这一份测试集上的随机波动。')
        else:
            key['PT_CURVE_VERDICT'] = (
                '与全量存在显著差异的规模点：' + '；'.join(_sig) + '。'
                '其余规模点与全量的差异落在评测噪声内。')
        # 逐点 Markdown 表（直接贴进报告）
        _md = ['| 规模点 vs 全量（6,210 条） | ΔAccuracy | McNemar p | bootstrap Δacc 95% CI | 结论 |',
               '|---|---|---|---|---|']
        for r in sorted(cv_rows, key=lambda x: x['route_a']):
            _ci = f'[{r["ci95_acc_lo_pp"]:+.2f}, {r["ci95_acc_hi_pp"]:+.2f}] pp'
            _vd = '**显著**' if r['significant_95'] else '不显著'
            _md.append(f'| {r["route_a"]} | {r["delta_acc_pp"]:+.2f} pp | '
                       f'{r["mcnemar_p"]:.4f} | {_ci} | {_vd} |')
        key['PT_CURVE_TABLE_MD'] = '\n'.join(_md)

    # 所有配对里最显著的一组
    best = min(rows, key=lambda r: r['mcnemar_p'])
    key['PT_BEST_PAIR'] = f'{best["route_a"]} vs {best["route_b"]}'
    key['PT_BEST_P'] = f'{best["mcnemar_p"]:.4f}'
    key['PT_BEST_DELTA'] = f'{best["delta_acc_pp"]:+.2f}'

    import json
    (out_dir / 'paired_tests_key.json').write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\n✓ {csv_path.name}  ({len(rows)} 行）')
    print(f'✓ {md_path.name}')
    print(f'✓ paired_tests_key.json  → {json.dumps(key, ensure_ascii=False)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
