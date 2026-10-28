
import os, json, math, time, random
import numpy as np
import pandas as pd

# ==================== 0. 找数据 ====================
print('=== [0/4] 找数据集 ===', flush=True)
def find_data_dir():
    here = os.getcwd()
    for c in [os.path.join(here,'data','processed','v1-20261023'),
              '/kaggle/input/chunksenti-v1-20261023',
              '/kaggle/input/chunksenti_v1_20261023']:
        if os.path.exists(os.path.join(c,'train.jsonl')): return c
    for root in ['/kaggle/input', here]:
        if not os.path.isdir(root): continue
        for dp, dn, fn in os.walk(root):
            if 'train.jsonl' in fn and 'validation.jsonl' in fn and 'test.jsonl' in fn:
                return dp
    import zipfile
    for root in ['/kaggle/input', here]:
        if not os.path.isdir(root): continue
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if f.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(os.path.join(dp,f)) as z:
                            z.extractall('/kaggle/working/_data')
                        for dp2, dn2, fn2 in os.walk('/kaggle/working/_data'):
                            if 'train.jsonl' in fn2: return dp2
                    except Exception: pass
    return None

DATA = find_data_dir()
if DATA is None:
    raise SystemExit('!!! 找不到数据集')
print('  ✓', DATA, flush=True)
# 数据不是从「当前目录 ./data/processed/v1-20261023」找到的 —— 多半是不在项目根目录跑。
# 这种情况下结果会被写到别处，或者拿错一份数据，必须喊出来。
_CANON = os.path.join(os.getcwd(), 'data', 'processed', 'v1-20261023')
if os.path.abspath(DATA) != os.path.abspath(_CANON):
    print(f'  ⚠ 数据不在 {_CANON} 下，而是在 {DATA}', flush=True)
    print('    → 请先 cd 到项目根目录再跑本脚本，否则结果会写到错误位置。', flush=True)

def load_jsonl(p):
    out = []
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

tr = load_jsonl(os.path.join(DATA,'train.jsonl'))
va = load_jsonl(os.path.join(DATA,'validation.jsonl'))
te = load_jsonl(os.path.join(DATA,'test.jsonl'))
Xtr = [r['text'] for r in tr]; ytr = [int(r['label']) for r in tr]
Xva = [r['text'] for r in va]; yva = [int(r['label']) for r in va]
Xte = [r['text'] for r in te]; yte = [int(r['label']) for r in te]
# ★ 测试集主键：和 MacBERT / LoRA 两条臂用同一个 sample_id，
#   这样四条路线的逐条预测能按主键对齐，做配对显著性检验。
te_ids = [r.get('sample_id', str(i)) for i, r in enumerate(te)]
print(f'  train {len(Xtr)} val {len(Xva)} test {len(Xte)}', flush=True)

CLS = ['negative','positive']
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
# ★ 结果目录：统一落到 <项目根>/src/results/results_TFIDF，和 MacBERT 臂放一起
#   （之前是靠"在哪个目录下跑就写到哪"来定的，换个目录跑就散掉了；
#    现在无论从项目根还是 src/results 跑，结果都回到同一个地方）
_out_root = os.environ.get('TFIDF_OUT_ROOT')
if _out_root:
    OUT = os.path.join(_out_root, 'results_TFIDF')
else:
    _r = os.getcwd()
    if os.path.isdir(os.path.join(_r, 'src', 'results')):
        _r = os.path.join(_r, 'src', 'results')
    elif os.path.isdir(os.path.join(_r, 'src')):
        # 项目根下有 src/ 但还没建 results/ —— 建在 src/results/，
        # 这样校验器（--results src/results）和汇总脚本能找到同一份产物。
        _r = os.path.join(_r, 'src', 'results')
    OUT = os.path.join(_r, 'results_TFIDF')
os.makedirs(OUT, exist_ok=True)
print('  ✓ 结果目录:', OUT, flush=True)

# ==================== 1. TF-IDF（字符 n-gram，中文友好） ====================
print('\n=== [1/4] TF-IDF + LR ===', flush=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 中文没有空格分词，用 analyzer='char' 的字符 n-gram（2~4 gram）
# 再对比一组 word 级（jieba 分词）做参数对比
#
# ★ word 级这一组是为了回应任务书原理点 1「分词 / 子词与词表选择对效果的影响」：
#   同一份数据、同一个分类器，只把"怎么切词"从字符换成词，看差别有多大。
#   没有 jieba 就跳过这一组（其余三组照跑，不影响已有结论）。
_WORD_TOKENIZER = None
try:
    import jieba
    jieba.setLogLevel(60)                 # 关掉 jieba 的初始化日志
    _WORD_TOKENIZER = lambda s: jieba.lcut(s)
except Exception as e:
    print(f'  ⚠ 没装 jieba，跳过 word 级对照（pip install jieba）：{e}', flush=True)

CONFIGS = [
    dict(name='char_2_4',   ngram=(2,4), analyzer='char'),
    dict(name='char_1_3',   ngram=(1,3), analyzer='char'),
    dict(name='char_2_5',   ngram=(2,5), analyzer='char'),
]
if _WORD_TOKENIZER is not None:
    CONFIGS.append(dict(name='word_1_2', ngram=(1,2), analyzer='word'))

RESULTS = []
FITTED = {}          # name -> (vec, clf, ps, pv)
for cfg in CONFIGS:
    t0 = time.time()
    _vkw = dict(analyzer=cfg['analyzer'], ngram_range=cfg['ngram'],
                min_df=2, sublinear_tf=True, max_features=200000)
    if cfg['analyzer'] == 'word':
        _vkw['tokenizer'] = _WORD_TOKENIZER
        _vkw['token_pattern'] = None      # 用了自定义 tokenizer 就不能再用 token_pattern
    vec = TfidfVectorizer(**_vkw)
    Xt = vec.fit_transform(Xtr)
    Xv = vec.transform(Xva); Xs = vec.transform(Xte)
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight=None)
    clf.fit(Xt, ytr)
    pv = clf.predict(Xv); ps = clf.predict(Xs)
    acc  = float((ps == np.array(yte)).mean())
    mf1  = float(f1_score(yte, ps, average='macro'))
    wf1  = float(f1_score(yte, ps, average='weighted'))
    vacc = float((pv == np.array(yva)).mean())
    res = dict(config=cfg['name'], analyzer=cfg['analyzer'], ngram=str(cfg['ngram']),
               tokenizer=('jieba' if cfg['analyzer'] == 'word' else 'none'),
               n_features=int(Xt.shape[1]), C=4.0,
               val_acc=round(vacc,4), test_acc=round(acc,4),
               macro_f1=round(mf1,4), weighted_f1=round(wf1,4),
               seconds=round(time.time()-t0,1))
    for i,c in enumerate(CLS):
        res[f'f1_{c}'] = round(float(f1_score(yte, ps, labels=[i], average='macro')),4)
    RESULTS.append(res)
    FITTED[cfg['name']] = (vec, clf, ps, pv)
    print(f'  [{cfg["name"]}] 特征数={Xt.shape[1]:,}  val_acc={vacc:.4f}  '
          f'test_acc={acc:.4f}  macroF1={mf1:.4f}  weightedF1={wf1:.4f}', flush=True)

ab = pd.DataFrame(RESULTS)
ab.to_csv(f'{OUT}/tfidf_ablation.csv', index=False, encoding='utf-8-sig')
print('\n' + ab.to_string(index=False), flush=True)

# ★ 主结果 = 验证集最好的那一组（明细/报告必须和主表来自同一个模型）
best_name = ab.sort_values('val_acc', ascending=False).iloc[0]['config']
cfg = next(c for c in CONFIGS if c['name'] == best_name)
vec, clf, ps, pv = FITTED[best_name]
print(f'\n  ★ 主结果取验证集最优: {best_name}', flush=True)

ab = pd.DataFrame(RESULTS)
ab.to_csv(f'{OUT}/tfidf_ablation.csv', index=False, encoding='utf-8-sig')
print('\n' + ab.to_string(index=False), flush=True)

# ==================== 2. 明细 ====================
rep = classification_report(yte, ps, target_names=CLS, digits=4)
cm  = confusion_matrix(yte, ps)
print('\n' + rep)

proba = clf.predict_proba(vec.transform(Xte))
conf  = proba.max(1)
# ★ 逐条预测：带 sample_id 主键 + 两个类的完整概率；
#   **不带评论正文**（提交包不含评论正文，正文只在报告引用的样例里保留）。
pd.DataFrame({'sample_id':te_ids,
              'idx':list(range(len(yte))),
              'true_label':[CLS[i] for i in yte],
              'pred_label':[CLS[i] for i in ps],
              'p_negative':[round(float(x),6) for x in proba[:,0]],
              'p_positive':[round(float(x),6) for x in proba[:,1]],
              'confidence':[round(float(x),4) for x in conf],
              'correct':[a==b for a,b in zip(yte,ps)]}).to_csv(
    f'{OUT}/predictions.csv', index=False, encoding='utf-8-sig')
# 错误清单：只留主键和置信度，不留正文
errs=[{'sample_id':te_ids[i],'true':CLS[a],'pred':CLS[b],'confidence':round(float(c),4)}
      for i,(a,b,c) in enumerate(zip(yte,ps,conf)) if a!=b]
pd.DataFrame(errs, columns=['sample_id','true','pred','confidence']).sort_values(
    'confidence', ascending=False).to_csv(
    f'{OUT}/errors_by_class.csv', index=False, encoding='utf-8-sig')

with open(f'{OUT}/tfidf_report.txt','w',encoding='utf-8') as fh:
    fh.write(f'TF-IDF + LogisticRegression 基线\n')
    fh.write(f'配置: analyzer={cfg["analyzer"]} ngram={cfg["ngram"]} C=4.0 min_df=2 sublinear_tf\n')
    fh.write(f'特征数: {int(vec.transform(Xtr).shape[1]):,}\n')
    _b = ab[ab['config'] == best_name].iloc[0]
    fh.write(f'测试集 acc={_b["test_acc"]} macroF1={_b["macro_f1"]} '
             f'weightedF1={_b["weighted_f1"]}\n\n')
    fh.write(rep)
    fh.write('\n混淆矩阵 (行=真实, 列=预测)\n')
    fh.write('        '+' '.join(f'{c:>9}' for c in CLS)+'\n')
    for i,c in enumerate(CLS):
        fh.write(f'{c:>8} '+' '.join(f'{v:>9}' for v in cm[i])+'\n')

# ==================== 3. 画图 ====================
print('\n=== [3/4] 画图 ===', flush=True)
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2, figsize=(11,4.2))
    x = np.arange(len(ab))
    ax[0].bar(x, ab['test_acc'], color='steelblue', alpha=.85)
    ax[0].set_xticks(x); ax[0].set_xticklabels(ab['config'])
    ax[0].set_ylabel('Test Accuracy'); ax[0].set_ylim(0,1)
    ax[0].set_title('TF-IDF n-gram vs Accuracy'); ax[0].grid(alpha=.3, axis='y')
    for i,v in enumerate(ab['test_acc']): ax[0].text(i,v+.02,f'{v:.3f}',ha='center',fontsize=9)
    ax[1].bar(x, ab['macro_f1'], color='crimson', alpha=.85)
    ax[1].set_xticks(x); ax[1].set_xticklabels(ab['config'])
    ax[1].set_ylabel('Macro F1'); ax[1].set_ylim(0,1)
    ax[1].set_title('TF-IDF n-gram vs Macro F1'); ax[1].grid(alpha=.3, axis='y')
    for i,v in enumerate(ab['macro_f1']): ax[1].text(i,v+.02,f'{v:.3f}',ha='center',fontsize=9)
    plt.tight_layout(); plt.savefig(f'{OUT}/tfidf_compare.png', dpi=150); plt.close()
    print('  ✓ tfidf_compare.png', flush=True)
except Exception as e:
    print(f'  画图跳过: {e}', flush=True)

# ==================== 4. 存档 ====================
print('\n=== [4/4] 存档 ===', flush=True)
import zipfile, shutil
# zip 放在结果目录的同级，方便一起拷走
_zip_root = os.path.dirname(OUT)
zp = os.path.join(_zip_root, 'results_TFIDF.zip')
try:
    subprocess.run(['zip','-r','-q',zp,os.path.basename(OUT)], cwd=_zip_root)
except Exception:
    shutil.make_archive(zp[:-4],'zip',_zip_root, os.path.basename(OUT))
print('\n' + '='*60)
m = ab[ab['config'] == best_name].iloc[0]
print(f"★ TF-IDF 基线[{best_name}]:  acc={m['test_acc']}  macroF1={m['macro_f1']}  weightedF1={m['weighted_f1']}")
print(f"★ 结果包 → {zp}  ({os.path.getsize(zp)/1e6:.1f} MB)")
print('='*60)
print('DONE', flush=True)
