"""把结果文件里的数字填进报告模板，生成终版报告。

用法：
    python src/scripts/fill_report.py [--results-dir src/results] \
        [--template 项目报告.md] [--out 项目报告_终版.md]

设计原则：
  1. **不重算指标**。所有数字都直接读自各路线产出的 CSV/TXT，
     所以报告里的每个数字都能在输出文件里找到出处（任务书硬要求）。
  2. 缺任何一路结果时，对应占位符保留成【待补：xxx】并在最后列清单，
     不会静默填 0 或编数字。
  3. 表格段落用 <!--AUTO:表N--> 标记，整段由 collect_results.py 生成，
     保证「结果汇总.md」和「报告正文」里同一张表永远一致。

占位符清单见文件末尾的 PLACEHOLDERS 说明。
"""
import argparse, os, sys, re, io, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import build_sections, find_root, load  # noqa: E402

import pandas as pd  # noqa: E402

MISS = '【待补：{}】'


def _num(v, fmt='{:.4f}'):
    try:
        return fmt.format(float(v))
    except Exception:
        return str(v)


def _f(v, fmt='{:.4f}'):
    """把 CSV 里读出来的可能是 str 的数字格式化。"""
    try:
        return fmt.format(float(v))
    except Exception:
        return str(v)


def sensitivity_conclusion(diff_pp, subject, unit='个百分点'):
    """按实测幅度生成"这个超参敏不敏感"的结论句。

    ★ 为什么要有这个函数：
      报告里原来预写死了"该任务对学习率不敏感"、"LoRA 秩不是敏感超参"这类结论。
      那是**实验前写死的假设**——如果数据不支持，报告就变成了编结论
      （任务书 §九 明令禁止"用 AI 代写实验设计与结果分析"，硬门槛 2 也要求
      报告里的判断在数据里能找到出处）。

      所以这里改成：先算出实际波动幅度，再按幅度选措辞。
      同一个句式，数据变了结论就跟着变——这样报告的结论永远和数据一致。
    """
    try:
        d = abs(float(diff_pp))
    except Exception:
        return MISS.format(f'{subject}的敏感性结论')
    if d < 1.0:
        return f'说明{subject}**不敏感**，不需要精细搜索'
    if d < 2.0:
        return f'说明{subject}**有影响但不需要精细搜索**（波动不到 2 个{unit}）'
    return f'说明{subject}是**敏感超参**，必须精细搜索（波动 {d:.2f} 个{unit}）'


def add_cmp(ph, name, a, b):
    """给两个数生成"高/低 + 绝对差"的占位符，避免报告里手写结论。

    a/b 可能是字符串（还没填上）或数字。填不上时留空。
    """
    try:
        fa, fb = float(a), float(b)
    except Exception:
        return
    diff = fa - fb
    ph[f'CMP_{name}'] = '高' if diff > 0 else ('低' if diff < 0 else '持平')
    ph[f'ABS_{name}'] = f'{abs(diff) * 100:.2f}'
    ph[f'SGN_{name}'] = '+' if diff > 0 else ('-' if diff < 0 else '±')


def collect(root):
    """读所有结果，产出 (ph, sections, missing)。ph 是占位符 -> 文本 的字典。"""
    ph, missing = {}, []
    sections, summary_df, sources, _ = build_sections(root)

    # ---------------- 表1 主对比（用于"最优路线"等叙述） ----------------
    best_row = None
    if not summary_df.empty:
        try:
            best_row = summary_df.sort_values('weighted_F1', ascending=False).iloc[0]
            ph['BEST_ROUTE'] = f"{best_row['方案']}（{best_row['变体']}）"
            ph['BEST_WF1'] = str(best_row['weighted_F1'])
            ph['BEST_ACC'] = str(best_row['accuracy'])
        except Exception:
            pass
    ph['ROUTE_COUNT'] = str(len(summary_df)) if not summary_df.empty else MISS.format('路线结果')

    # ---------------- 总体排序（摘要里那句结论，必须数据驱动） ----------------
    # 报告模板原来把 "全参微调 > LoRA > 零样本提示 > 传统基线" 写死在摘要里。
    # 那是实验前的假设：如果数据不是这个顺序，那句话就是编的。
    # 这里改成按各路线实测 weighted F1 现算，顺序随数据变。
    _order = []
    for _k, _label in (('BERT_WF1', 'MacBERT 全参微调'),
                        ('LORA_WF1', 'Qwen3-1.7B + LoRA'),
                        ('P_BEST_WF1', 'Qwen3-1.7B 提示'),
                        ('TFIDF_WF1', 'TF-IDF + LR')):
        if _k in ph:
            try:
                _order.append((float(ph[_k]), _label))
            except Exception:
                pass
    if len(_order) >= 2:
        _order.sort(key=lambda x: -x[0])
        ph['OVERALL_RANK'] = ' > '.join(l for _, l in _order)
        ph['OVERALL_RANK_WF1'] = '；'.join(f'{l} {v:.4f}' for v, l in _order)
        _spread = (_order[0][0] - _order[-1][0]) * 100
        ph['OVERALL_SPREAD'] = f'{_spread:.2f}'
        ph['OVERALL_SPREAD_CMP'] = ('差距明显' if _spread >= 5 else
                                    ('差距有限' if _spread >= 2 else '差距很小'))
    else:
        for _k, _lbl in (('OVERALL_RANK', '总体排序'),
                         ('OVERALL_RANK_WF1', '各路线 weighted F1'),
                         ('OVERALL_SPREAD', '最高与最低差距'),
                         ('OVERALL_SPREAD_CMP', '差距评价')):
            ph[_k] = MISS.format(_lbl)

    # ---------------- ① TF-IDF ----------------
    df, p = load(root, 'results_TFIDF', 'tfidf_ablation.csv')
    if df is not None:
        b = df.sort_values('val_acc', ascending=False).iloc[0]
        ph['TFIDF_CFG'] = str(b['config'])
        ph['TFIDF_ACC'] = _f(b['test_acc'])
        ph['TFIDF_MF1'] = _f(b['macro_f1'])
        ph['TFIDF_WF1'] = _f(b['weighted_f1'])
        ph['TFIDF_VACC'] = _f(b['val_acc'])
        ph['TFIDF_FEAT'] = f"{int(b['n_features']):,}"
        ph['TFIDF_SEC'] = _f(b['seconds'], '{:.0f}')
        ph['TFIDF_NGRAMS'] = ' / '.join(
            f"{r['config']}({_f(r['test_acc'])})" for _, r in df.iterrows())
        if 'f1_negative' in df.columns:
            ph['TFIDF_F1NEG'] = _f(b['f1_negative'])
            ph['TFIDF_F1POS'] = _f(b['f1_positive'])

        # ★ 词级分词对照组（jieba）：直接回应任务书原理点 1
        #   "分词 / 子词与词表选择对效果的影响"
        #   注意这里有个值得写进报告的现象：词级在【测试集】上更好、在【验证集】上更差，
        #   而主结果按验证集选（不拿测试集选模型），这两条要一起说清楚。
        w = df[df['analyzer'] == 'word'] if 'analyzer' in df.columns else df.iloc[0:0]
        if len(w):
            w = w.iloc[0]
            ph['TFIDF_WORD_CFG'] = str(w['config'])
            ph['TFIDF_WORD_ACC'] = _f(w['test_acc'])
            ph['TFIDF_WORD_MF1'] = _f(w['macro_f1'])
            ph['TFIDF_WORD_WF1'] = _f(w['weighted_f1'])
            ph['TFIDF_WORD_VACC'] = _f(w['val_acc'])
            ph['TFIDF_WORD_FEAT'] = f"{int(w['n_features']):,}"
            if 'f1_negative' in w.index:
                ph['TFIDF_WORD_F1NEG'] = _f(w['f1_negative'])
            # 差值方向：词级 - 字符级
            d_acc = float(w['test_acc']) - float(b['test_acc'])
            d_vacc = float(w['val_acc']) - float(b['val_acc'])
            ph['TFIDF_WORD_DACC'] = f'{d_acc * 100:+.2f}'
            ph['TFIDF_WORD_DVACC'] = f'{d_vacc * 100:+.2f}'
        else:
            # 环境里没装 jieba → 词级那一组没跑。报告里如实标注，不留 {{占位符}}，
            # 也不让整个报告生成挂掉。
            for _k in ('TFIDF_WORD_CFG', 'TFIDF_WORD_ACC', 'TFIDF_WORD_MF1',
                       'TFIDF_WORD_WF1', 'TFIDF_WORD_VACC', 'TFIDF_WORD_FEAT',
                       'TFIDF_WORD_F1NEG'):
                ph[_k] = '（本次运行未含词级对照：环境缺 jieba）'
            ph['TFIDF_WORD_DACC'] = 'n/a'
            ph['TFIDF_WORD_DVACC'] = 'n/a'
    else:
        missing.append(f'TF-IDF 结果 → {p}')

    # ---------------- 长度分桶 vs 随机组 batch 的对照 ----------------
    # 这组对照回答："为了在本机 CPU 上把 26 小时压下来用的长度分桶提速，
    # 有没有偷偷牺牲精度？" —— 是 §4 工程优化不改变结论的证据。
    cdf2, p2 = load(root, 'results_BERT_FULL', 'batching_control.csv')
    if cdf2 is not None and len(cdf2) >= 2:
        g = cdf2[cdf2['batching'] == 'len_group']
        r = cdf2[cdf2['batching'] == 'random']
        g = g.iloc[0] if len(g) else cdf2.iloc[0]
        r = r.iloc[0] if len(r) else cdf2.iloc[1]
        ph['CTRL_N'] = str(int(g['n_train']))
        ph['CTRL_GROUP_ACC'] = _f(g['test_acc'])
        ph['CTRL_RANDOM_ACC'] = _f(r['test_acc'])
        ph['CTRL_GROUP_MF1'] = _f(g['macro_f1'])
        ph['CTRL_RANDOM_MF1'] = _f(r['macro_f1'])
        ph['CTRL_GROUP_SEC'] = _f(g['seconds'], '{:.0f}')
        ph['CTRL_RANDOM_SEC'] = _f(r['seconds'], '{:.0f}')
        ph['CTRL_DACC'] = f'{(float(g["test_acc"]) - float(r["test_acc"])) * 100:+.2f}'
        ph['CTRL_DMF1'] = f'{(float(g["macro_f1"]) - float(r["macro_f1"])) * 100:+.2f}'
        ph['CTRL_DTIME'] = f'{(float(g["seconds"]) - float(r["seconds"])) / 60:+.1f}'
        ph['CTRL_SPEEDUP'] = f'{float(r["seconds"]) / max(float(g["seconds"]), 1e-9):.2f}'
    else:
        missing.append(f'分桶对照 → {p2}')

    # 分桶对照的配对显著性检验：
    # "快 1.6 倍但精度高 1.16 个百分点" 这句话如果说成结论就危险了 ——
    # 1.16 个百分点在 777 条测试集上完全可能是噪声。必须先检验再下结论。
    ptdf, p3 = load(root, '.', 'paired_tests.csv')
    if ptdf is not None and 'kind' in ptdf.columns:
        _c = ptdf[ptdf['kind'] == '分组方式对照']
        if len(_c):
            _row = _c.iloc[0]
            ph['CTRL_MCN_B'] = str(int(_row['mcnemar_b']))
            ph['CTRL_MCN_C'] = str(int(_row['mcnemar_c']))
            ph['CTRL_MCN_P'] = f"{float(_row['mcnemar_p']):.4f}"
            ph['CTRL_MCN_SIG'] = '显著' if bool(_row['significant_95']) else '不显著'
            ph['CTRL_CI'] = (f'[{float(_row["ci95_acc_lo_pp"]):+.2f}, '
                             f'{float(_row["ci95_acc_hi_pp"]):+.2f}]')
            _lo = float(_row['ci95_acc_lo_pp'])
            _hi = float(_row['ci95_acc_hi_pp'])
            ph['CTRL_CI_VERDICT'] = '不跨 0' if (_lo > 0 or _hi < 0) else '跨 0'
            # 离 0 最近的那一端有多远 —— 用来如实说明"处在统计边缘"
            ph['CTRL_CI_MARGIN'] = f'{min(abs(_lo), abs(_hi)):.2f}'
            ph['CTRL_AGREE'] = f'{float(_row["agree_rate"]) * 100:.1f}%'
            ph['CTRL_DISAGREE_N'] = str(int(round((1 - float(_row['agree_rate'])) * int(_row['n']))))
    else:
        missing.append(f'分桶对照的配对检验 → {p3}')

    # ---------------- ② MacBERT 全参 ----------------
    df, p = load(root, 'results_BERT_FULL', 'ablation_results.csv')
    if df is not None:
        # ★ 主结果取**代表组**：8 组超参里验证集准确率最高的那个。
        #   选代表用验证集、评最终用测试集，四条路线（TF-IDF / MacBERT / LoRA / 提示）
        #   全部用同一条规则，这样才是"代表 vs 代表"地比。
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'eval'))
        try:
            from representative import pick_representative
            _rep_cfg, _rep_row, _rep_val = pick_representative(
                os.path.join(root, 'results_BERT_FULL'))
        except Exception as _e:
            print(f'  ⚠ 选代表组失败（退回 main）：{_e}')
            _rep_cfg, _rep_row, _rep_val = 'main', None, None
        m = _rep_row if _rep_row is not None else df.iloc[0].to_dict()
        ph['BERT_CFG'] = str(m['config'])
        ph['BERT_VAL'] = f'{_rep_val:.4f}' if _rep_val is not None else MISS.format('代表组验证集 acc')
        ph['BERT_ACC'] = _f(m['test_acc'])
        ph['BERT_MF1'] = _f(m['macro_f1'])
        ph['BERT_WF1'] = _f(m['weighted_f1'])
        ph['BERT_PARAMS'] = f"{int(m['trainable_params']):,}"
        ph['BERT_PCT'] = f"{m['trainable_pct']}"
        ph['BERT_SEC'] = _f(m['seconds'], '{:.0f}')
        ph['BERT_LR'] = str(m['lr'])
        ph['BERT_EP'] = str(m['epochs'])
        ph['BERT_LEN'] = str(m['maxlen'])
        if 'f1_negative' in df.columns:
            ph['BERT_F1NEG'] = _f(m['f1_negative'])
            ph['BERT_F1POS'] = _f(m['f1_positive'])
        # 8 组里按 weighted_f1 最好的那组（**只是报告出来，不是代表组**）
        # 代表组是按验证集选的，见上面 pick_representative。
        bb = df.sort_values('weighted_f1', ascending=False).iloc[0]
        ph['BERT_BESTCFG'] = str(bb['config'])
        ph['BERT_BESTWF1'] = _f(bb['weighted_f1'])
        ph['BERT_BESTACC'] = _f(bb['test_acc'])
        ph['BERT_BEST_SEC'] = _f(bb['seconds'], '{:.0f}')
        ph['BERT_REP_SEC'] = _f(m['seconds'], '{:.0f}')
        # 最高组的验证集准确率（用它说明"为什么测试集最高的不是代表组"）
        try:
            from representative import last_val_acc
            _bv = last_val_acc(os.path.join(root, 'results_BERT_FULL', 'val_log.csv'))
            ph['BERT_BEST_VAL'] = (f'{_bv[str(bb["config"])]:.4f}'
                                   if str(bb['config']) in _bv else '—')
        except Exception:
            ph['BERT_BEST_VAL'] = '—'
        try:
            ph['BERT_SEC_RATIO'] = f"{float(bb['seconds']) / float(m['seconds']):.2f}"
            ph['BERT_BEST_DELTA_VS_REP'] = (
                f"{abs(float(bb['weighted_f1']) - float(m['weighted_f1'])) * 100:.2f}")
        except Exception:
            ph['BERT_SEC_RATIO'] = '—'
            ph['BERT_BEST_DELTA_VS_REP'] = '—'
        ph['BERT_ABL_ROWS'] = ' / '.join(
            f"{r['config']}:{_f(r['weighted_f1'])}" for _, r in df.iterrows())

        # ---- 派生的对比量（供 §5.3 结论句用） ----
        def row(name):
            s = df[df['config'] == name]
            return s.iloc[0] if len(s) else None

        main_r = row('main')
        # 学习率三档的波动幅度
        lrs = [r for r in (row('main'), row('lr_5e5'), row('lr_1e5')) if r is not None]
        if len(lrs) >= 2:
            vals = [float(r['weighted_f1']) for r in lrs]
            _d = (max(vals) - min(vals)) * 100
            ph['ABS_BERT_LR'] = f'{_d:.2f}'
            ph['BERT_LR_CONC'] = sensitivity_conclusion(_d, '学习率')
        # partial（冻结底部 6 层）
        pr = row('partial')
        if pr is not None and main_r is not None:
            ph['BERT_PARTIAL_PCT'] = f"{pr['trainable_pct']}"
            add_cmp(ph, 'PARTIAL', pr['weighted_f1'], main_r['weighted_f1'])
        # len512 相对 len256
        l5 = row('len512')
        if l5 is not None and main_r is not None:
            d = float(l5['weighted_f1']) - float(main_r['weighted_f1'])
            ph['ABS_LEN512'] = f'{abs(d) * 100:.2f}'
            ph['SGN_LEN512'] = '+' if d > 0 else ('-' if d < 0 else '±')
            ph['BERT_LEN512_CONC'] = sensitivity_conclusion(abs(d) * 100, '序列长度')
    else:
        missing.append(f'MacBERT 结果 → {p}')

    # MacBERT 错误样例（报告第⑥节）
    # 优先读 error_samples.csv（只含报告要引用的 3 条，带正文）；
    # 没有就退回 errors_by_class.csv（旧格式，正文在 text 列里）。
    edf, p = load(root, 'results_BERT_FULL', 'error_samples.csv')
    if edf is None or not len(edf):
        edf, p = load(root, 'results_BERT_FULL', 'errors_by_class.csv')
    if edf is not None and len(edf):
        # 按置信度从高到低取：模型最确信却仍然错的样例最值得分析
        if 'confidence' in edf.columns:
            edf = edf.sort_values('confidence', ascending=False)
        for i in range(3):
            if i >= len(edf):
                break
            r = edf.iloc[i]
            k = i + 1
            ph[f'ERR{k}_TEXT'] = str(r['text']).replace('\n', ' ').replace('|', '｜')
            txt = str(r['text']).replace('\n', ' ').replace('|', '｜')
            ph[f'ERR{k}_SHORT'] = (txt[:60] + '…') if len(txt) > 60 else txt
            ph[f'ERR{k}_TRUE'] = ZH_MAP.get(str(r['true']), str(r['true']))
            ph[f'ERR{k}_PRED'] = ZH_MAP.get(str(r['pred']), str(r['pred']))
            ph[f'ERR{k}_CONF'] = _f(r['confidence'], '{:.3f}')
            if 'token_len' in edf.columns:
                ph[f'ERR{k}_LEN'] = str(int(r['token_len']))
            else:
                ph[f'ERR{k}_LEN'] = str(len(str(r['text'])))
            ph[f'ERR{k}_DIAG'] = diagnose(r)
    else:
        missing.append(f'MacBERT 错误样例 → {p}')

    # 按长度准确率
    ldf, p = load(root, 'results_BERT_FULL', 'acc_by_length.csv')
    if ldf is not None:
        rows = []
        for _, r in ldf.iterrows():
            rows.append(f"长度 {r.iloc[0]}：n={int(r.iloc[1])}，acc={_f(r.iloc[2])}")
        ph['BERT_LENACC'] = '；'.join(rows)
    else:
        missing.append(f'MacBERT 分长度准确率 → {p}')

    # ---------------- ③ Qwen3 提示 ----------------
    best_prompt_wf1 = None
    pdf, p = load(root, 'results_PROMPT_FULL', 'prompt_variants.csv')
    if pdf is not None:
        for _, r in pdf.iterrows():
            v = str(r['variant'])
            ph[f'P_{v}_ACC'] = _f(r['test_acc'])
            ph[f'P_{v}_MF1'] = _f(r['macro_f1'])
            ph[f'P_{v}_WF1'] = _f(r['weighted_f1'])
            if 'ifr' in pdf.columns:
                ph[f'P_{v}_IFR'] = _f(r['ifr'], '{:.1%}')
            if 'avg_prompt_tokens' in pdf.columns:
                ph[f'P_{v}_TOK'] = _f(r['avg_prompt_tokens'], '{:.0f}')
            ph[f'P_{v}_NAME'] = VARIANT_ZH.get(v, v)
            if 'val_acc' in pdf.columns:
                ph[f'P_{v}_VAL'] = _f(r['val_acc'])
            else:
                ph.setdefault(f'P_{v}_VAL', MISS.format('提示臂验证集 acc'))
            try:
                # ★ 代表变体按**验证集**准确率选（与其它三条路线同一口径）；
                #   报告里引用的仍然是它在**测试集**上的指标。
                if 'val_acc' in pdf.columns:
                    _sel = float(r['val_acc'])
                else:
                    _sel = float(r['weighted_f1'])      # 兜底：没有验证集列时退回
                if best_prompt_wf1 is None or _sel > best_prompt_wf1:
                    best_prompt_wf1 = _sel
                    ph['P_BEST_NAME'] = VARIANT_ZH.get(v, v)
                    ph['P_BEST_ACC'] = _f(r['test_acc'])
                    ph['P_BEST_MF1'] = _f(r['macro_f1'])
                    ph['P_BEST_WF1'] = _f(r['weighted_f1'])
                    ph['P_BEST_VAL'] = (_f(r['val_acc'])
                                        if 'val_acc' in pdf.columns
                                        else MISS.format('提示臂验证集 acc'))
            except Exception:
                pass
        if 'ifr' in pdf.columns:
            zs = pdf[pdf['variant'] == 'zero']
            if len(zs):
                ph['P_ZERO_IFR'] = _f(zs.iloc[0]['ifr'], '{:.1%}')
            # ★ "格式约束修的是听不听话还是准不准"——按实测分开算，不写死结论
            zf = pdf[pdf['variant'] == 'zero_fmt']
            if len(zs) and len(zf):
                d_ifr = (float(zf.iloc[0]['ifr']) - float(zs.iloc[0]['ifr'])) * 100
                d_wf1 = abs(float(zf.iloc[0]['weighted_f1'])
                            - float(zs.iloc[0]['weighted_f1'])) * 100
                ph['P_FMT_DIFR'] = f'{d_ifr:+.1f}'
                ph['P_FMT_DWF1'] = f'{d_wf1:.2f}'
                if d_ifr >= 5 and d_wf1 < 1:
                    ph['P_FMT_CONC'] = ('**格式约束主要修的是「听不听话」，不是「准不准」**'
                                        '——指令遵循率涨了不少，weighted F1 基本没动')
                elif d_ifr >= 5:
                    ph['P_FMT_CONC'] = ('**格式约束同时提升了指令遵循率和准确率**，'
                                        f'不是只修「听不听话」')
                elif d_wf1 >= 1:
                    ph['P_FMT_CONC'] = ('**格式约束主要改善的是「准不准」**，'
                                        f'指令遵循率只变了 {d_ifr:+.1f} 个百分点')
                else:
                    ph['P_FMT_CONC'] = ('**格式约束在两个维度上都没有明显效果**'
                                        f'（指令遵循率 {d_ifr:+.1f} 个百分点，'
                                        f'weighted F1 {d_wf1:.2f} 个百分点）')
    else:
        missing.append(f'Qwen3 提示结果 → {p}')

    # 延迟
    ldf2, p = load(root, 'results_PROMPT_FULL', 'latency.csv')
    if ldf2 is not None:
        r0 = ldf2.iloc[0]
        ph['LAT_N'] = str(int(r0['n']))
        ph['LAT_MEAN'] = _f(r0['mean_ms'], '{:.0f}')
        ph['LAT_P50'] = _f(r0['p50_ms'], '{:.0f}')
        ph['LAT_P95'] = _f(r0['p95_ms'], '{:.0f}')
        ph['LAT_MAX'] = _f(r0['max_ms'], '{:.0f}')
        if 'avg_prompt_tokens' in ldf2.columns:
            ph['LAT_TOK'] = _f(r0['avg_prompt_tokens'], '{:.0f}')
        # 整个测试集（777 条）纯推理耗时（分钟）
        try:
            ph['LAT_TOTAL_MIN'] = f"{float(r0['mean_ms']) * 777 / 1000 / 60:.1f}"
        except Exception:
            pass
    else:
        missing.append(f'提示臂延迟 → {p}')

    # LoRA 臂的延迟（分类模型，用 src/eval/latency_model.py 测）
    ldl, p4 = load(root, 'results_QWEN_FULL', 'latency.csv')
    if ldl is not None and len(ldl):
        _l0 = ldl.iloc[0]
        ph['LAT_LORA_MEAN'] = _f(_l0['mean_ms'], '{:.1f}')
        ph['LAT_LORA_P50'] = _f(_l0['p50_ms'], '{:.1f}')
        ph['LAT_LORA_P95'] = _f(_l0['p95_ms'], '{:.1f}')
        ph['LAT_LORA_MAX'] = _f(_l0['max_ms'], '{:.1f}')
        try:
            ph['LAT_LORA_OVER_PROMPT'] = (
                f"{float(_l0['mean_ms']) / float(ph['LAT_MEAN']):.2f}")
        except Exception:
            pass
    else:
        missing.append(f'LoRA 臂延迟 → {p4}')
        for _k in ('LAT_LORA_MEAN', 'LAT_LORA_P50', 'LAT_LORA_P95', 'LAT_LORA_MAX'):
            ph.setdefault(_k, MISS.format('LoRA 延迟'))

    # ---------------- 基线两臂的延迟（TF-IDF / MacBERT）----------------
    # 与提示臂用同一套协议（latency_ids.json 冻结的 200 条、batch=1、预热 20 次），
    # 数字同样只搬运不重算。MacBERT 那一臂必须在训练进程退出、机器空闲时测，
    # 否则测出来的延迟是被训练抢走的 CPU 抬高的虚值。
    ldb, pb = load(root, 'results_BERT_FULL', 'latency_baseline.csv')
    if ldb is not None:
        for _, _row in ldb.iterrows():
            _v = str(_row.get('variant', ''))
            _tag = ('LAT_BERT' if ('BERT' in _v or 'MacBERT' in _v)
                    else 'LAT_TFIDF' if 'TF-IDF' in _v else None)
            if _tag is None:
                continue
            ph[f'{_tag}_MEAN'] = _f(_row['mean_ms'], '{:.1f}')
            ph[f'{_tag}_P50'] = _f(_row['p50_ms'], '{:.1f}')
            ph[f'{_tag}_P95'] = _f(_row['p95_ms'], '{:.1f}')
            ph[f'{_tag}_MAX'] = _f(_row['max_ms'], '{:.1f}')
            ph[f'{_tag}_N'] = str(int(_row['n']))
        # 同为本机 CPU 测的两臂，可以直接比倍数
        try:
            _b = float(ph['LAT_BERT_MEAN']); _t = float(ph['LAT_TFIDF_MEAN'])
            ph['LAT_BERT_OVER_TFIDF'] = f'{_b / max(_t, 1e-9):.0f}'
        except Exception:
            pass
    else:
        missing.append(f'基线两臂延迟 → {pb}')
        for _k in ('LAT_BERT_MEAN', 'LAT_BERT_P50', 'LAT_BERT_P95', 'LAT_BERT_MAX',
                   'LAT_TFIDF_MEAN', 'LAT_TFIDF_P50', 'LAT_TFIDF_P95', 'LAT_TFIDF_MAX'):
            ph.setdefault(_k, MISS.format('基线延迟'))

    # ---------------- ④ Qwen3 + LoRA ----------------
    ldf3, p = load(root, 'results_QWEN_FULL', 'lora_rank_sweep.csv')
    if ldf3 is not None:
        # ★ 代表秩按**验证集**准确率选（与其它三条路线同一口径）；
        #   没有 val_acc 列时退回 r=8（组里预先约定的默认档）。
        if 'val_acc' in ldf3.columns and ldf3['val_acc'].notna().any():
            r8 = ldf3.sort_values('val_acc', ascending=False).iloc[0]
        else:
            _r8 = ldf3[ldf3['r'] == 8]
            r8 = _r8.iloc[0] if len(_r8) else ldf3.iloc[0]
        ph['LORA_R'] = str(int(r8['r']))
        ph['LORA_ACC'] = _f(r8['test_acc'])
        ph['LORA_MF1'] = _f(r8['macro_f1'])
        ph['LORA_WF1'] = _f(r8['weighted_f1'])
        ph['LORA_PARAMS'] = f"{int(r8['trainable_params']):,}"
        ph['LORA_PCT'] = f"{r8['trainable_pct']}"
        ph['LORA_SEC'] = _f(r8['seconds'], '{:.0f}') if 'seconds' in ldf3.columns else MISS.format('LoRA 耗时')
        ph['LORA_RANKS'] = ' / '.join(
            f"r={int(r['r'])}:{_f(r['test_acc'])}" for _, r in ldf3.sort_values('r').iterrows())
        # 验证集准确率（选代表的依据），单独给一行，方便读者核对
        if 'val_acc' in ldf3.columns:
            ph['LORA_RANK_VALS'] = ' / '.join(
                f"r={int(r['r'])}:{_f(r['val_acc'])}"
                for _, r in ldf3.sort_values('r').iterrows()
                if pd.notna(r['val_acc']))
        else:
            ph['LORA_RANK_VALS'] = MISS.format('LoRA 各档验证集 acc')
        try:
            vals = [float(r['weighted_f1']) for _, r in ldf3.iterrows()]
            _d = (max(vals) - min(vals)) * 100
            ph['ABS_LORA_RANK'] = f'{_d:.2f}'
            ph['LORA_RANK_CONC'] = sensitivity_conclusion(_d, 'LoRA 秩')
        except Exception:
            pass
    else:
        missing.append(f'Qwen3 LoRA 结果 → {p}')

    # 数据量曲线
    cdf, p = load(root, 'results_BERT_FULL', 'data_curve.csv')
    if cdf is not None:
        cdf = cdf.sort_values('n_train')
        ph['CURVE_N'] = ' / '.join(str(int(x)) for x in cdf['n_train'])
        ph['CURVE_ACC'] = ' / '.join(_f(x) for x in cdf['test_acc'])
        r0, rN = cdf.iloc[0], cdf.iloc[-1]
        ph['CURVE_LO_N'] = str(int(r0['n_train']))
        ph['CURVE_LO_ACC'] = _f(r0['test_acc'])
        ph['CURVE_HI_N'] = str(int(rN['n_train']))
        ph['CURVE_HI_ACC'] = _f(rN['test_acc'])
    else:
        missing.append(f'数据量曲线 → {p}')

    # ---------------- 路线间对比（避免在报告里手写结论） ----------------
    if 'BERT_WF1' in ph and 'TFIDF_WF1' in ph:
        add_cmp(ph, 'BERT_TFIDF', ph['BERT_WF1'], ph['TFIDF_WF1'])
    if 'LORA_WF1' in ph and 'TFIDF_WF1' in ph:
        add_cmp(ph, 'LORA_TFIDF', ph['LORA_WF1'], ph['TFIDF_WF1'])
    if 'LORA_WF1' in ph and 'P_BEST_WF1' in ph:
        add_cmp(ph, 'LORA_PROMPT', ph['LORA_WF1'], ph['P_BEST_WF1'])
    if 'BERT_WF1' in ph and 'P_BEST_WF1' in ph:
        add_cmp(ph, 'BERT_PROMPT', ph['BERT_WF1'], ph['P_BEST_WF1'])
    if 'BERT_WF1' in ph and 'LORA_WF1' in ph:
        add_cmp(ph, 'BERT_LORA', ph['BERT_WF1'], ph['LORA_WF1'])
    if 'P_fewshot4_WF1' in ph and 'P_zero_WF1' in ph:
        add_cmp(ph, 'FEWSHOT_ZERO', ph['P_fewshot4_WF1'], ph['P_zero_WF1'])
    if 'P_zero_fmt_WF1' in ph and 'P_zero_WF1' in ph:
        add_cmp(ph, 'FMT_ZERO', ph['P_zero_fmt_WF1'], ph['P_zero_WF1'])
    if 'P_BEST_WF1' in ph and 'TFIDF_WF1' in ph:
        add_cmp(ph, 'PROMPT_TFIDF', ph['P_BEST_WF1'], ph['TFIDF_WF1'])

    # ---------------- 配对显著性检验（§5.3(7)） ----------------
    # 表和结论数字都由 src/eval/paired_tests.py 生成，
    # 报告里只做搬运，不在这里重算 p 值，避免两处实现不一致。
    _pt_md = os.path.join(root, 'paired_tests.md')
    _pt_json = os.path.join(root, 'paired_tests_key.json')
    if os.path.exists(_pt_md):
        with open(_pt_md, encoding='utf-8') as fh:
            ph['PAIRED_TEST_TABLE'] = fh.read().strip()
    else:
        ph['PAIRED_TEST_TABLE'] = ('（还没跑 `python src/eval/paired_tests.py`，'
                                   '本节的检验表待补）')
        missing.append('配对显著性检验表 → paired_tests.md')
    if os.path.exists(_pt_json):
        try:
            with open(_pt_json, encoding='utf-8') as fh:
                _pt = json.load(fh)
            ph.update({k: str(v) for k, v in _pt.items()})
        except Exception as e:
            print(f'  ⚠ 读取 paired_tests_key.json 失败: {e}')

    return ph, sections, missing


ZH_MAP = {'negative': '负面', 'positive': '正面'}
VARIANT_ZH = {
    'zero': '零样本（zero-shot）',
    'zero_fmt': '零样本 + 严格格式约束',
    'fewshot4': '4-shot 少样本',
}


def diagnose(r):
    """根据可测量的特征给错误样例生成一段客观诊断（不臆测语义）。"""
    conf = float(r['confidence'])
    parts = []
    if conf >= 0.9:
        parts.append(f'高置信度误判（conf={conf:.3f}），排除「模型自己不确定」这一解释')
    elif conf >= 0.6:
        parts.append(f'中等置信度误判（conf={conf:.3f}），模型处于边界状态')
    else:
        parts.append(f'低置信度误判（conf={conf:.3f}），模型实际上已经倾向正确类')
    t = str(r['true']); p_ = str(r['pred'])
    if t == 'negative' and p_ == 'positive':
        parts.append('真实为负面却判为正面 → 属于「漏判负面」方向，'
                     '会直接压低少数类的 F1，对 Macro F1 的伤害大于对 Accuracy 的伤害')
    elif t == 'positive' and p_ == 'negative':
        parts.append('真实为正面却判为负面 → 属于「误报负面」方向')
    n = len(str(r['text']))
    if n <= 30:
        parts.append(f'文本很短（{n} 字），上下文线索少')
    elif n >= 200:
        parts.append(f'文本很长（{n} 字），存在截断风险')
    else:
        parts.append(f'文本长度中等（{n} 字）')
    return '；'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--template', default='项目报告.md')
    ap.add_argument('--out', default='项目报告_终版.md')
    ap.add_argument('--summary-template', default='项目摘要_模板.md')
    ap.add_argument('--summary-out', default='项目摘要.md')
    args = ap.parse_args()

    root = find_root(args.results_dir)
    if not os.path.isdir(root):
        print(f'!! 找不到结果目录: {root}'); return 1
    print(f'结果根目录: {root}')

    tpl_path = args.template
    if not os.path.exists(tpl_path):
        print(f'!! 找不到报告模板: {tpl_path}'); return 1
    with open(tpl_path, encoding='utf-8') as fh:
        tpl = fh.read()

    ph, sections, missing = collect(root)

    # 表格：<!--AUTO:表N--> → 整段表格
    def sub_table(m):
        key = m.group(1).strip()
        if key in sections:
            return '\n\n' + sections[key] + '\n\n'
        return f'\n\n【待补：{key}（对应结果文件还没产出）】\n\n'

    out = re.sub(r'<!--AUTO:(表\d+)-->', sub_table, tpl)

    # 占位符：{{KEY}} → 文本
    used, unresolved = set(), []
    def sub_ph(m):
        k = m.group(1).strip()
        used.add(k)
        if k in ph:
            return ph[k]
        unresolved.append(k)
        return MISS.format(k)
    out = re.sub(r'\{\{([^}]+)\}\}', sub_ph, out)

    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write(out)

    # 提醒还没人工填的地方
    todo = [m.group(0) for m in re.finditer(r'<!--\s*人工[^>]*-->', out)]
    if todo:
        print('\n⚠ 报告里还有待人工填写的位置（HTML 注释）：')
        for t in sorted(set(todo)):
            print(f'    {t}')

    print(f'\n✓ 已生成 {args.out}')

    # ---- 顺带把项目摘要也填一遍（交付要求文件名就是「项目摘要.md」） ----
    if os.path.exists(args.summary_template):
        with open(args.summary_template, encoding='utf-8') as fh:
            stpl = fh.read()
        s_out = re.sub(r'<!--AUTO:(表\d+)-->', sub_table, stpl)
        s_unres = []

        def sub_ph_s(m):
            k = m.group(1).strip()
            return ph[k] if k in ph else (s_unres.append(k), MISS.format(k))[1]
        s_out = re.sub(r'\{\{([^}]+)\}\}', sub_ph_s, s_out)
        if s_unres:
            # 还没填完 → 顶部加醒目标记，防止误当终版交上去
            s_out = ('> ⚠ **本文件尚未完成**：以下 `【待补：…】` 是还没跑出结果的数字占位。\n'
                     '> 跑完 Kaggle 三条路线后重新执行 `fill_report.py` 即可自动补全。\n\n'
                     + s_out)
        with open(args.summary_out, 'w', encoding='utf-8') as fh:
            fh.write(s_out)
        body = re.sub(r'[#*`\->\s]', '', s_out)
        n = len([c for c in body if '一' <= c <= '鿿'])
        if s_unres:
            print(f'⚠ 已生成 {args.summary_out}，但还差 {len(set(s_unres))} 个数字：'
                  f'{sorted(set(s_unres))}（顶部已加未完成标记）')
        else:
            print(f'✓ 已生成 {args.summary_out}（中文正文约 {n} 字，限 300）')
    uniq_unresolved = set(unresolved)
    print(f'  填充占位符 {len(used) - len(uniq_unresolved)}/{len(used)} 个')
    if unresolved:
        print('\n⚠ 还没填上的占位符（说明对应结果还没跑出来）：')
        for u in sorted(set(unresolved)):
            print(f'    {{{{{u}}}}}')
    if missing:
        print('\n⚠ 缺的结果文件：')
        for m in missing:
            print(f'    {m}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
