"""
统一评测实现 —— 四条路线共用同一个口径。

为什么必须有这个文件
--------------------
1) **口径对齐是硬门槛。** weighted F1 的权重有坑：
   - 按【训练集】类别数加权  → 和 sklearn 的 average='weighted' 不一致，能差好几个点
   - 按【测试集真实标签数】加权 → = sklearn average='weighted'（本项目统一用这个）
   之前 BERT / LoRA / sklearn 三边的 weighted F1 对不上，就是栽在这里。
   四条路线必须用同一个脚本算，否则表 1 里的数字不是同一个东西，结论直接作废。

2) **数字可复算。** 报告里任何一个数，都能用这个脚本从 predictions.csv 重算一遍，
   证明"报告里的数不是手敲的"（任务书硬门槛 2）。

输入格式（每条路线都一样）
--------------------------
CSV 至少要有这几列：

    sample_id, true_label, pred_label

可选列（强烈建议都带，缺了就没法做概率层面的比较）：

    p_negative, p_positive, confidence, token_len, idx

- sample_id : 与 data/processed/<版本>/test.jsonl 里的 sample_id 一致，**主键**，
              两条路线靠它逐条对齐。
- 标签取值 : 统一写 negative / positive（本脚本也接受 0/1，会自动转换）。

用法
----
    # 单条路线评测
    python src/eval/evaluate.py --predictions src/results/results_BERT_FULL/predictions_main.csv \
                                --label-name "MacBERT 全参" --out src/results/results_BERT_FULL/eval_report_main.json

    # 两条路线对比（配对检验）
    python src/eval/evaluate.py --compare A.csv B.csv --label-a "MacBERT 全参" --label-b "Qwen3+LoRA"

    # 自检：拿随机数据验证实现和 sklearn 一致
    python src/eval/evaluate.py --selfcheck
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit('需要 pandas：pip install pandas')

CLS = ['negative', 'positive']
LABEL_ALIASES = {
    'negative': 'negative', 'neg': 'negative', '0': 'negative', 'n': 'negative',
    'false': 'negative', '差评': 'negative', '负面': 'negative',
    'positive': 'positive', 'pos': 'positive', '1': 'positive', 'p': 'positive',
    'true': 'positive', '好评': 'positive', '正面': 'positive',
}


def canon_label(v):
    """把 0/1、negative/positive、中文 统一成 negative / positive。"""
    s = str(v).strip().lower()
    if s in LABEL_ALIASES:
        return LABEL_ALIASES[s]
    raise ValueError(f'无法识别的标签取值: {v!r}（应为 0/1 或 negative/positive）')


# ----------------------------------------------------------------------
# 指标实现（纯手写，不依赖 sklearn，保证任何解释器都能跑）
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """y_true / y_pred 都是 0/1 整数列表（0=negative, 1=positive）。

    返回 acc, macro_f1, weighted_f1, per_class, confusion。
    weighted F1 按【测试集真实标签数】加权 —— 与 sklearn average='weighted' 定义一致。
    """
    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]
    n = len(y_true)
    if n == 0:
        raise ValueError('空预测')
    if n != len(y_pred):
        raise ValueError(f'长度不一致: true {n} vs pred {len(y_pred)}')

    tp = [0, 0]; fp = [0, 0]; fn = [0, 0]; support = [0, 0]
    for a, b in zip(y_true, y_pred):
        support[a] += 1
        if a == b:
            tp[a] += 1
        else:
            fp[b] += 1
            fn[a] += 1

    per_class = []
    f1s_raw = []
    for c in (0, 1):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s_raw.append(f)
        per_class.append(dict(label=CLS[c], precision=round(p, 6), recall=round(r, 6),
                               f1=round(f, 6), support=support[c]))

    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    # ★ 聚合用【未四舍五入】的 per-class F1，和 sklearn 的定义逐位一致；
    #   展示用的 round 只作用在 per_class 字典里。
    macro_f1 = sum(f1s_raw) / len(f1s_raw)
    weighted_f1 = sum(f1s_raw[c] * support[c] for c in (0, 1)) / n
    cm = [[0, 0], [0, 0]]
    for a, b in zip(y_true, y_pred):
        cm[a][b] += 1
    return dict(n=n, accuracy=round(acc, 6), macro_f1=round(macro_f1, 6),
                weighted_f1=round(weighted_f1, 6),
                per_class=per_class, confusion_matrix=cm)


# ----------------------------------------------------------------------
# 读预测文件
# ----------------------------------------------------------------------
REQUIRED = ['true_label', 'pred_label']
PREFERRED = ['sample_id', 'p_negative', 'p_positive', 'confidence', 'token_len', 'idx']


def load_predictions(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: 缺少必需列 {missing}')
    df = df.copy()
    df['_true'] = [canon_label(v) for v in df['true_label']]
    df['_pred'] = [canon_label(v) for v in df['pred_label']]
    df['_y_true'] = [0 if v == 'negative' else 1 for v in df['_true']]
    df['_y_pred'] = [0 if v == 'negative' else 1 for v in df['_pred']]
    if 'sample_id' not in df.columns:
        raise ValueError(f'{path}: 缺少 sample_id 列，无法与其它路线逐条对齐')
    if df['sample_id'].duplicated().any():
        raise ValueError(f'{path}: sample_id 有重复，无法作为主键')
    for c in PREFERRED:
        if c not in df.columns:
            df[c] = None
    return df


# ----------------------------------------------------------------------
# 配对比较：McNemar + paired bootstrap
# ----------------------------------------------------------------------
def mcnemar_exact_p(b, c):
    """双侧精确 McNemar（二项检验）。b/c 是不一致对的两类计数。

    零假设：b 和 c 来自同一个 p=0.5 的二项分布。
    这里用正态近似 + 小样本精确二项和兜底。

    b, c 较大时用正态近似（chi2, 1 df）：p = erfc(sqrt(x2/2))
    b+c 较小时用精确二项检验（穷举，n<=40 时精确）
    """
    n = b + c
    if n == 0:
        return 1.0, 0.0
    if n <= 40:
        # 精确二项检验：sum over k of C(n,k)*0.5^n, 取双侧
        from math import comb
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        return min(1.0, 2 * tail), 0.0
    x2 = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(x2 / 2.0)), x2


def paired_bootstrap(y_true, pred_a, pred_b, n_boot=10000, seed=42):
    """对 Δ = 指标(A) - 指标(B) 做配对 bootstrap，返回 (mean, lo95, hi95)。"""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true, dtype=int)
    pa = np.asarray(pred_a, dtype=int)
    pb = np.asarray(pred_b, dtype=int)
    n = len(yt)
    idx = rng.integers(0, n, size=(n_boot, n))

    def acc(p):
        return (p[idx] == yt[idx]).mean(axis=1)

    def macro_f1(p):
        out = np.zeros(n_boot)
        for c in (0, 1):
            tp = ((p[idx] == c) & (yt[idx] == c)).sum(axis=1)
            fp = ((p[idx] == c) & (yt[idx] != c)).sum(axis=1)
            fn = ((p[idx] != c) & (yt[idx] == c)).sum(axis=1)
            prec = np.divide(tp, tp + fp, out=np.zeros(n_boot), where=(tp + fp) > 0)
            rec = np.divide(tp, tp + fn, out=np.zeros(n_boot), where=(tp + fn) > 0)
            f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n_boot), where=(prec + rec) > 0)
            out += f1
        return out / 2.0

    res = {}
    for name, fn_ in (('accuracy', acc), ('macro_f1', macro_f1)):
        d = fn_(pa) - fn_(pb)
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[name] = dict(mean=float(d.mean()), ci95_lo=float(lo), ci95_hi=float(hi))
    return res


def compare_two(df_a, df_b, name_a, name_b, n_boot=10000, seed=42):
    """按 sample_id 对齐后做配对比较。"""
    m = df_a.merge(df_b, on='sample_id', suffixes=('_a', '_b'))
    if len(m) != len(df_a) or len(m) != len(df_b):
        raise ValueError(
            f'sample_id 对不上：{name_a} {len(df_a)} 条 / {name_b} {len(df_b)} 条，'
            f'交集 {len(m)} 条。两条路线必须用同一份 test.jsonl。')
    yt = m['_y_true_a'].tolist()
    pa = m['_y_pred_a'].tolist()
    pb = m['_y_pred_b'].tolist()

    ma = compute_metrics(yt, pa)
    mb = compute_metrics(yt, pb)

    agree = sum(1 for x, y in zip(pa, pb) if x == y)
    both_right = sum(1 for t, x, y in zip(yt, pa, pb) if x == y == t)
    both_wrong = sum(1 for t, x, y in zip(yt, pa, pb) if x != t and y != t)
    a_only = sum(1 for t, x, y in zip(yt, pa, pb) if x == t and y != t)  # b
    b_only = sum(1 for t, x, y in zip(yt, pa, pb) if x != t and y == t)  # c
    p_val, chi2 = mcnemar_exact_p(a_only, b_only)
    boot = paired_bootstrap(yt, pa, pb, n_boot=n_boot, seed=seed)

    return dict(
        n=len(m),
        routes={name_a: ma, name_b: mb},
        agreement=dict(agree=agree, agree_rate=round(agree / len(m), 6),
                        both_right=both_right, both_wrong=both_wrong,
                        a_only_correct=a_only, b_only_correct=b_only),
        mcnemar=dict(b=a_only, c=b_only, chi2=round(chi2, 4), p_value=round(p_val, 6)),
        delta={k: dict(a=ma[k], b=mb[k], diff=round(ma[k] - mb[k], 6),
                       diff_pp=round((ma[k] - mb[k]) * 100, 4))
               for k in ('accuracy', 'macro_f1', 'weighted_f1')},
        bootstrap=boot,
        significant_95=bool(p_val < 0.05),
    )


# ----------------------------------------------------------------------
# 自检：和 sklearn 对一遍
# ----------------------------------------------------------------------
def selfcheck():
    """自检：随机数据上把手写实现和 sklearn 对一遍。

    sklearn 不可用时（部分 Windows 环境会被应用控制策略拦掉 sklearn.metrics 的 C 扩展）
    明确报告"跳过"，不谎称通过。
    """
    try:
        from sklearn.metrics import f1_score, accuracy_score
    except Exception as e:
        print(f'  ⚠ sklearn 不可用（{type(e).__name__}: {str(e)[:120]}）')
        print('    → 本脚本本身不依赖 sklearn，手写实现仍可直接使用；')
        print('    → 但"与 sklearn 交叉校验"这一步需要在一个能导入 sklearn 的环境里跑')
        print('    → 正常机器上：pip install scikit-learn 后重跑本自检')
        return False

    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(20, 400))
        # 故意造不平衡（贴近本数据集 31.5% / 68.5%）
        yt = rng.binomial(1, 0.68, size=n)
        pb = np.where(rng.random(n) < 0.75, yt, 1 - yt)
        yt, pb = yt.tolist(), pb.tolist()
        mine = compute_metrics(yt, pb)
        sk_acc = accuracy_score(yt, pb)
        sk_mf1 = f1_score(yt, pb, average='macro', zero_division=0)
        sk_wf1 = f1_score(yt, pb, average='weighted', zero_division=0)
        # 本实现把结果 round 到 6 位小数，所以拿 sklearn 的值也 round 到 6 位再比
        # （否则 0.781341 vs 0.7813411078717201 会被误判成不一致）
        if not (round(sk_acc, 6) == mine['accuracy']
                and round(sk_mf1, 6) == mine['macro_f1']
                and round(sk_wf1, 6) == mine['weighted_f1']):
            print(f'  ✗ 第 {trial} 次不一致: {mine} vs sklearn '
                  f'{sk_acc}/{sk_mf1}/{sk_wf1}')
            return False
    print('  ✓ 自检通过：200 组随机数据上，手写实现与 sklearn '
          'accuracy / f1_score(macro) / f1_score(weighted) 完全一致')
    return True


# ----------------------------------------------------------------------
def md_table(res):
    a, b = list(res['routes'].keys())
    ma, mb = res['routes'][a], res['routes'][b]
    L = []
    L.append(f'| 指标 | {a} | {b} | 差值 |')
    L.append('|---|---|---|---|')
    for k, name in (('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1'),
                    ('weighted_f1', 'Weighted F1')):
        L.append(f'| {name} | {ma[k]:.4f} | {mb[k]:.4f} | '
                 f'{res["delta"][k]["diff_pp"]:+.2f} pp |')
    bs = res['bootstrap']
    L.append('')
    L.append(f'- 一致率：{res["agreement"]["agree_rate"]*100:.2f}%'
             f'（{res["agreement"]["agree"]}/{res["n"]}）')
    L.append(f'- 两者都对 {res["agreement"]["both_right"]} 条，'
             f'两者都错 {res["agreement"]["both_wrong"]} 条，'
             f'仅 {a} 对 {res["agreement"]["a_only_correct"]} 条，'
             f'仅 {b} 对 {res["agreement"]["b_only_correct"]} 条')
    L.append(f'- McNemar：b={res["mcnemar"]["b"]}, c={res["mcnemar"]["c"]}, '
             f'p={res["mcnemar"]["p_value"]:.4f}'
             f'（{"显著" if res["significant_95"] else "不显著"}，α=0.05）')
    L.append(f'- paired bootstrap（10000 次重采样）ΔAccuracy：'
             f'{bs["accuracy"]["mean"]*100:+.2f} pp，'
             f'95% CI [{bs["accuracy"]["ci95_lo"]*100:+.2f}, '
             f'{bs["accuracy"]["ci95_hi"]*100:+.2f}] pp')
    L.append(f'- paired bootstrap（10000 次重采样）ΔMacro F1：'
             f'{bs["macro_f1"]["mean"]*100:+.2f} pp，'
             f'95% CI [{bs["macro_f1"]["ci95_lo"]*100:+.2f}, '
             f'{bs["macro_f1"]["ci95_hi"]*100:+.2f}] pp')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', action='append', default=[],
                    help='预测 CSV，可给多个')
    ap.add_argument('--label-name', default=None)
    ap.add_argument('--compare', action='store_true',
                    help='给两条 --predictions 做配对比较')
    ap.add_argument('--label-a', default='路线 A')
    ap.add_argument('--label-b', default='路线 B')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    ap.add_argument('--selfcheck', action='store_true')
    args = ap.parse_args()

    if args.selfcheck:
        print('=== 自检：手写实现 vs sklearn ===')
        selfcheck()
        return 0

    if not args.predictions:
        ap.error('至少要给一个 --predictions')

    if args.compare:
        if len(args.predictions) != 2:
            ap.error('--compare 需要正好两个 --predictions')
        df_a = load_predictions(args.predictions[0])
        df_b = load_predictions(args.predictions[1])
        res = compare_two(df_a, df_b, args.label_a, args.label_b,
                          n_boot=args.n_boot, seed=args.seed)
        print(f'\n=== {args.label_a}  vs  {args.label_b} ===')
        print(md_table(res))
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, 'w', encoding='utf-8') as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)
            print(f'\n✓ 已生成 {args.out}')
        return 0

    # 单条路线评测
    out_all = {}
    for i, path in enumerate(args.predictions):
        name = args.label_name if (args.label_name and len(args.predictions) == 1) \
            else (args.label_name or f'路线{i+1}')
        df = load_predictions(path)
        m = compute_metrics(df['_y_true'].tolist(), df['_y_pred'].tolist())
        out_all[name] = m
        print(f'\n=== {name} ===')
        print(f'  文件: {path}')
        print(f'  n={m["n"]}  acc={m["accuracy"]:.4f}  '
              f'macroF1={m["macro_f1"]:.4f}  weightedF1={m["weighted_f1"]:.4f}')
        for d in m['per_class']:
            print(f'    {d["label"]:>9}: P={d["precision"]:.4f} R={d["recall"]:.4f} '
                  f'F1={d["f1"]:.4f}  support={d["support"]}')
        print('  混淆矩阵 (行=真实, 列=预测):')
        print('            negative  positive')
        for i_, row in enumerate(m['confusion_matrix']):
            print(f'    {CLS[i_]:>9}  {row[0]:>8}  {row[1]:>8}')
        if 'p_negative' in df.columns and df['p_negative'].notna().all():
            y = np.asarray(df['_y_true'], dtype=int)
            p1 = np.asarray(df['p_positive'], dtype=float)
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y, p1)
                print(f'  AUC（用 p_positive）= {auc:.4f}')
                m['auc'] = round(float(auc), 6)
            except Exception:
                pass

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(out_all, fh, ensure_ascii=False, indent=2)
        print(f'\n✓ 已生成 {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
