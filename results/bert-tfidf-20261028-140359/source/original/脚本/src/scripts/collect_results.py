"""汇总四条路线的结果，生成报告第⑤节直接可贴的表格。

用法：
    python src/scripts/collect_results.py [--results-dir results] [--out docs/结果汇总.md]

规则：只读各路线产出的 CSV/TXT，**不重算指标**，保证报告里的每个数字
      都能在代码或输出文件里找到出处。

本文件同时被 `fill_report.py` 当作库调用（`build_sections`），
所以表格的生成逻辑只有一份，改这里报告和汇总就同时更新。
"""
import argparse, os, sys, json, re
import pandas as pd

# 选代表的规则收敛在 src/eval/representative.py 一处实现，四条路线共用
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'eval'))

CLS = ['negative', 'positive']
ZH = {'negative': '负面', 'positive': '正面'}

# 各路线结果目录的候选位置
DIR_CANDIDATES = [
    'results',
    'src/results',
    '/kaggle/working',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'),
]

ARMS = [
    ('TF-IDF + LR',        'results_TFIDF',   'tfidf_ablation.csv'),
    ('MacBERT 全参',       'results_BERT_FULL', 'ablation_results.csv'),
    ('Qwen3-1.7B + LoRA',  'results_QWEN_FULL', 'lora_rank_sweep.csv'),
    ('Qwen3-1.7B 提示',    'results_PROMPT_FULL', 'prompt_variants.csv'),
]


def find_root(explicit):
    if explicit:
        return os.path.abspath(explicit)
    for c in DIR_CANDIDATES:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath('results')


def load(root, sub, fname):
    p = os.path.join(root, sub, fname)
    if not os.path.exists(p):
        return None, p
    try:
        return pd.read_csv(p), p
    except Exception as e:
        print(f'  ⚠ 读取失败 {p}: {e}')
        return None, p


def build_sections(root):
    """读所有结果文件，产出可直接贴进报告的 markdown 表格。

    返回 (sections, summary_df, sources, missing)
      sections : OrderedDict-like dict，key 是 '表1'..'表5'，value 是 markdown 文本
      summary_df: 四条路线主对比表（DataFrame）
      sources   : [(内容, 文件路径), ...]
      missing   : [还没产出的结果项, ...]
    """
    summary, missing, sources = [], [], []
    ablations = []
    sections = {}

    # ---------- 主对比表 ----------
    for label, sub, fname in ARMS:
        df, path = load(root, sub, fname)
        if df is None:
            missing.append(f'{label} → {path}')
            print(f'  ✗ 缺 {label}: {path}')
            continue
        sources.append((label, path))
        print(f'  ✓ {label}: {path}  ({len(df)} 行)')

        if fname == 'tfidf_ablation.csv':
            # 3 组 n-gram 对比，主结果取验证集最好的那组
            best = df.sort_values('val_acc', ascending=False).iloc[0]
            summary.append(dict(方案=label, 变体=best['config'],
                                accuracy=best['test_acc'], macro_F1=best['macro_f1'],
                                weighted_F1=best['weighted_f1'],
                                训练参数='—', 延迟='—'))
            ablations.append((label, df))
        elif fname == 'ablation_results.csv':
            # ★ 主结果取**代表组**（按验证集选），与 TF-IDF / LoRA / 提示臂同一口径
            try:
                from representative import pick_representative
                _cfg, main_row, _val = pick_representative(
                    os.path.join(root, 'results_BERT_FULL'))
                if main_row is None:
                    main_row = df.iloc[0].to_dict()
                else:
                    main_row = pd.Series(main_row)
            except Exception:
                _r = df[df['config'] == 'main']
                main_row = _r.iloc[0] if len(_r) else df.iloc[0]
            summary.append(dict(方案=label, 变体=main_row['config'],
                                accuracy=main_row['test_acc'], macro_F1=main_row['macro_f1'],
                                weighted_F1=main_row['weighted_f1'],
                                训练参数=f"{int(main_row['trainable_params']):,} "
                                      f"({main_row['trainable_pct']}%)", 延迟='—'))
            ablations.append((label, df))
        elif fname == 'lora_rank_sweep.csv':
            # ★ 代表秩按**验证集**选（与其它三条路线同一口径），没有 val_acc 列时退回 r=8
            if 'val_acc' in df.columns and df['val_acc'].notna().any():
                r8 = df.sort_values('val_acc', ascending=False).iloc[0]
            else:
                _r8 = df[df['r'] == 8]
                r8 = _r8.iloc[0] if len(_r8) else df.iloc[0]
            summary.append(dict(方案=label, 变体=f"LoRA r={int(r8['r'])}",
                                accuracy=r8['test_acc'], macro_F1=r8['macro_f1'],
                                weighted_F1=r8['weighted_f1'],
                                训练参数=f"{int(r8['trainable_params']):,} "
                                      f"({r8['trainable_pct']}%)", 延迟='—'))
            ablations.append((label, df))
        elif fname == 'prompt_variants.csv':
            # ★ 表 1 是「四条路线主对比」，每条路线只放**代表**（按验证集选）。
            #   三个变体的完整对比在报告 §5.3(5) 单独有表，不是藏起来。
            if 'val_acc' in df.columns and df['val_acc'].notna().any():
                _rep = df.sort_values('val_acc', ascending=False).iloc[0]
            else:
                _rep = df.iloc[0]
            summary.append(dict(方案=label, 变体=str(_rep['variant']),
                                accuracy=_rep['test_acc'], macro_F1=_rep['macro_f1'],
                                weighted_F1=_rep['weighted_f1'],
                                训练参数='0（不训练）',
                                延迟=f"{_rep.get('ms_per_sample','—')} ms/条"))
            ablations.append((label, df))

    summary_df = pd.DataFrame(summary)

    # ---------- 表1 ----------
    if not summary_df.empty:
        t = ['| 方案 | 变体 | Accuracy | Macro F1 | Weighted F1 | 训练参数 | 平均延迟 |',
             '|---|---|---|---|---|---|---|']
        for _, r in summary_df.iterrows():
            t.append(f"| {r['方案']} | {r['变体']} | {r['accuracy']} | "
                     f"{r['macro_F1']} | {r['weighted_F1']} | {r['训练参数']} | {r['延迟']} |")
        sections['表1'] = '\n'.join(t)

    # ---------- 表2 延迟 ----------
    lat_df, lat_path = load(root, 'results_PROMPT_FULL', 'latency.csv')
    if lat_df is not None:
        sources.append(('延迟', lat_path))
        t = ['| 提示变体 | 条数 | 平均延迟 | P50 | P95 | 最大 | 平均提示 token |',
             '|---|---|---|---|---|---|---|']
        for _, r in lat_df.iterrows():
            t.append(f"| {r['variant']} | {r['n']} | {r['mean_ms']} ms | "
                     f"{r['p50_ms']} ms | {r['p95_ms']} ms | {r['max_ms']} ms | "
                     f"{r['avg_prompt_tokens']} |")
        sections['表2'] = '\n'.join(t)

    # ---------- 表3 数据量曲线 ----------
    curve_df, curve_path = load(root, 'results_BERT_FULL', 'data_curve.csv')
    if curve_df is not None:
        sources.append(('数据量曲线', curve_path))
        t = ['| 训练样本数 | Accuracy | Macro F1 | Weighted F1 |', '|---|---|---|---|']
        for _, r in curve_df.sort_values('n_train').iterrows():
            t.append(f"| {int(r['n_train'])} | {r['test_acc']} | "
                     f"{r['macro_f1']} | {r['weighted_f1']} |")
        sections['表3'] = '\n'.join(t)

    # ---------- 表4 LoRA 秩 ----------
    rank_df, rank_path = load(root, 'results_QWEN_FULL', 'lora_rank_sweep.csv')
    if rank_df is not None:
        sources.append(('LoRA秩对比', rank_path))
        t = ['| LoRA 秩 r | 训练参数 | 占比 | Accuracy | Macro F1 |', '|---|---|---|---|---|']
        for _, r in rank_df.sort_values('r').iterrows():
            t.append(f"| {int(r['r'])} | {int(r['trainable_params']):,} | "
                     f"{r['trainable_pct']}% | {r['test_acc']} | {r['macro_f1']} |")
        sections['表4'] = '\n'.join(t)

    # ---------- 表5 MacBERT 8 组超参 ----------
    for label, df in ablations:
        if label == 'MacBERT 全参':
            cols = ['config','lr','epochs','freeze_lower','class_weight','maxlen',
                    'trainable_pct','test_acc','macro_f1','weighted_f1','seconds']
            cols = [c for c in cols if c in df.columns]
            t = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
            for _, r in df.iterrows():
                t.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
            sections['表5'] = '\n'.join(t)

    return sections, summary_df, sources, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--out', default='docs/结果汇总.md')
    args = ap.parse_args()

    root = find_root(args.results_dir)
    print(f'结果根目录: {root}\n')
    if not os.path.isdir(root):
        print('!! 找不到结果目录'); sys.exit(1)

    sections, summary_df, sources, missing = build_sections(root)

    # ---------- 输出 ----------
    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    lines = ['# 实验结果汇总（自动生成，勿手改）', '',
             f'> 由 `src/scripts/collect_results.py` 从各路线的输出文件生成。',
             f'> 结果根目录：`{root}`', '']

    titles = {
        '表1': '表1 · 四条路线主对比',
        '表2': '表2 · 延迟与 token 成本',
        '表3': '表3 · 数据量曲线（MacBERT 全参，main 配置）',
        '表4': '表4 · LoRA 秩对比（Qwen3-1.7B）',
        '表5': '表5 · MacBERT 全参微调参数对比（8 组）',
    }
    for key in ['表1', '表2', '表3', '表4', '表5']:
        if key in sections:
            lines += [f'## {titles[key]}', '', sections[key], '']

    if not summary_df.empty:
        summary_df.to_csv(os.path.join(root, '表1_主对比.csv'), index=False,
                          encoding='utf-8-sig')

    if missing:
        lines += ['## 尚未产出（跑完对应脚本后重新生成本文件）', '']
        for m in missing:
            lines.append(f'- {m}')
        lines.append('')

    lines += ['## 数据出处', '', '| 内容 | 文件 |', '|---|---|']
    for label, path in sources:
        lines.append(f'| {label} | `{path}` |')
    lines.append('')

    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    print(f'\n✓ 已生成 {args.out}')
    if not summary_df.empty:
        print('\n' + summary_df.to_string(index=False))
    if missing:
        print(f'\n⚠ 缺 {len(missing)} 项结果（见上面 ✗ 行）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
