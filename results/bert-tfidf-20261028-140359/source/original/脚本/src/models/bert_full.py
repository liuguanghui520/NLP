
import os, sys, gc, json, math, time, random, zipfile, shutil, subprocess
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ★★★ 开关 ★★★
SMOKE       = True      # True=冒烟(3分钟) / False=正式(45分钟)
RUN_ABLATION = True     # 超参对比实验
RUN_CURVE    = True     # 数据量曲线
MODEL_ID    = 'hfl/chinese-macbert-base'
SEED        = 42
NUM_LABELS  = 2
CLS         = ['negative', 'positive']
BASE_MAXLEN = 256
BATCH_GPU   = 16
BATCH_CPU   = 8
EVAL_BATCH  = 32
CURVE_SIZES = [100, 500, 1000, 2000, 3000]   # 加上全量 = 6 个点

# ==================== 0. 环境 ====================
print('=== [0/8] 环境 ===', flush=True)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

USE_CUDA = torch.cuda.is_available()
DEV = 'cuda' if USE_CUDA else 'cpu'
if USE_CUDA:
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}  '
          f'{(torch.cuda.get_device_properties(0).total_memory/1e9):.1f}GB', flush=True)
else:
    torch.set_num_threads(os.cpu_count())
    print(f'  ⚠ 无 GPU → CPU 模式（{os.cpu_count()} 线程）。'
          f'全量跑完约需 40 小时，建议挂机或改用 Kaggle。', flush=True)
print(f'  torch {torch.__version__}  seed={SEED}', flush=True)

WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
TAG  = 'SMOKE' if SMOKE else 'FULL'
OUT  = os.path.join(WORK, f'results_BERT_{TAG}')
os.makedirs(OUT, exist_ok=True)

# ==================== 1. 找数据 ====================
print('\n=== [1/8] 找数据集 ===', flush=True)
def find_data_dir():
    here = os.getcwd()
    fixed = [
        os.path.join(here, 'data', 'processed', 'v1-20261023'),
        os.path.join(here, 'data', 'processed', 'v1-20261023'),
        '/kaggle/input/chunksenti-v1-20261023',
        '/kaggle/input/chunksenti_v1_20261023',
    ]
    for c in fixed:
        if os.path.exists(os.path.join(c, 'train.jsonl')):
            return c
    for root in ['/kaggle/input', here]:
        if not os.path.isdir(root): continue
        for dp, dn, fn in os.walk(root):
            if 'train.jsonl' in fn and 'validation.jsonl' in fn and 'test.jsonl' in fn:
                return dp
    # 还没解压就自己解
    for root in ['/kaggle/input', here]:
        if not os.path.isdir(root): continue
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if f.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(os.path.join(dp, f)) as z:
                            z.extractall(os.path.join(WORK, '_data'))
                        for dp2, dn2, fn2 in os.walk(os.path.join(WORK, '_data')):
                            if 'train.jsonl' in fn2: return dp2
                    except Exception as e:
                        print(f'  zip 解压失败 {f}: {e}', flush=True)
    return None

DATA = find_data_dir()
if DATA is None:
    raise SystemExit('!!! 找不到数据集。Kaggle 请 Add Input → Upload 上传 '
                     'chunksenti_v1-20261023.zip；本地请确认 data/processed/v1-20261023 存在')
print('  ✓ 数据目录:', DATA, flush=True)

def load_jsonl(path):
    out = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

tr_rows = load_jsonl(os.path.join(DATA, 'train.jsonl'))
va_rows = load_jsonl(os.path.join(DATA, 'validation.jsonl'))
te_rows = load_jsonl(os.path.join(DATA, 'test.jsonl'))
print(f'  train {len(tr_rows)}  val {len(va_rows)}  test {len(te_rows)}', flush=True)

tr_texts = [r['text'] for r in tr_rows]; tr_labels = [int(r['label']) for r in tr_rows]
va_texts = [r['text'] for r in va_rows]; va_labels = [int(r['label']) for r in va_rows]
te_texts = [r['text'] for r in te_rows]; te_labels = [int(r['label']) for r in te_rows]

# ==================== 2. 模型 ====================
print('\n=== [2/8] 准备模型 ===', flush=True)
BASE_DIR = os.path.join(WORK, 'macbert_base')

def resolve_model():
    # A) 直连 HF（本地走 hf-mirror，Kaggle 走 huggingface.co）
    for attempt in range(2):
        try:
            tok = AutoTokenizer.from_pretrained(MODEL_ID)
            m = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID, num_labels=NUM_LABELS)
            m.save_pretrained(BASE_DIR); tok.save_pretrained(BASE_DIR)
            print('  ✓ 直连下载成功', flush=True)
            return tok, BASE_DIR
        except Exception as e:
            print(f'  直连失败(第{attempt+1}次): {type(e).__name__}: {str(e)[:140]}', flush=True)
            os.environ['HF_ENDPOINT'] = 'https://huggingface.co'   # 换源再试
    # B) curl 落到本地目录（Kaggle 老办法）
    d = os.path.join(WORK, 'macbert_local'); os.makedirs(d, exist_ok=True)
    need = ['config.json', 'vocab.txt', 'tokenizer.json', 'tokenizer_config.json',
            'pytorch_model.bin', 'special_tokens_map.json']
    bases = ['https://hf-mirror.com/hfl/chinese-macbert-base/resolve/main',
             'https://huggingface.co/hfl/chinese-macbert-base/resolve/main']
    for f in need:
        out = os.path.join(d, f)
        if os.path.exists(out) and os.path.getsize(out) > 1000: continue
        for b in bases:
            subprocess.run(['curl', '-L', '-s', '--max-time', '900', '-o', out, f'{b}/{f}'])
            if os.path.exists(out) and os.path.getsize(out) > 1000:
                print(f'  ✓ {f} ({os.path.getsize(out)/1e6:.1f}MB)', flush=True); break
        else:
            if f in ('config.json', 'vocab.txt'): raise SystemExit(f'!!! {f} 下载失败')
    tok = AutoTokenizer.from_pretrained(d)
    m = AutoModelForSequenceClassification.from_pretrained(d, num_labels=NUM_LABELS)
    m.save_pretrained(BASE_DIR); tok.save_pretrained(BASE_DIR)
    return tok, BASE_DIR

tok, model_src = resolve_model()
print(f'  ✓ 模型就绪 -> {model_src}', flush=True)

# ==================== 3. 类别权重 ====================
cnt = {}
for y in tr_labels: cnt[y] = cnt.get(y, 0) + 1
N = len(tr_labels); K = NUM_LABELS
CLASS_W = torch.tensor([N / (K * cnt.get(i, 1)) for i in range(K)], dtype=torch.float)
print('  类别权重:', [round(x, 4) for x in CLASS_W.tolist()], flush=True)

# ==================== 4. 数据集 / 批处理 ====================
class DS(Dataset):
    def __init__(s, texts, labels):
        s.t = texts; s.y = labels
    def __len__(s): return len(s.t)
    def __getitem__(s, i): return s.t[i], s.y[i]

def make_col(maxlen):
    def col(b):
        ts = [x[0] for x in b]
        ys = torch.tensor([x[1] for x in b], dtype=torch.long)
        e = tok(ts, padding=True, truncation=True, max_length=maxlen, return_tensors='pt')
        return e.input_ids, e.attention_mask, ys
    return col

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if USE_CUDA: torch.cuda.manual_seed_all(s)

# ==================== 5. 训练函数 ====================
print('\n=== [3/8] 定义训练函数 ===', flush=True)

def freeze_lower_layers(model, n_lower):
    if n_lower <= 0: return 0
    frozen = 0
    for name, p in model.named_parameters():
        if 'embeddings' in name:
            p.requires_grad = False; frozen += p.numel()
    enc = None
    for mname, mod in model.named_modules():
        if mname.endswith('encoder') and hasattr(mod, 'layer'):
            enc = mod; break
    if enc is None:
        for mod in model.modules():
            if hasattr(mod, 'layer') and isinstance(mod.layer, nn.ModuleList):
                enc = mod; break
    if enc is not None:
        for i in range(min(n_lower, len(enc.layer))):
            for p in enc.layer[i].parameters():
                p.requires_grad = False; frozen += p.numel()
    return frozen

def run_config(cfg, tr_texts_=None, tr_labels_=None, detail=False, save_ckpt=False):
    name = cfg['name']; t0 = time.time()
    set_seed(SEED)
    ML   = cfg['maxlen']
    TT   = tr_texts_ if tr_texts_ is not None else tr_texts
    LL   = tr_labels_ if tr_labels_ is not None else tr_labels
    BATCH = cfg.get('batch', BATCH_GPU if USE_CUDA else BATCH_CPU)
    # 长序列降 batch，避免 T4 16GB 上 OOM（len512 那组最容易爆）
    if ML >= 512 and USE_CUDA: BATCH = min(BATCH, 8)
    if ML >= 512 and not USE_CUDA: BATCH = min(BATCH, 4)
    col  = make_col(ML)
    tr = DataLoader(DS(TT, LL), batch_size=BATCH, shuffle=True, collate_fn=col)
    va = DataLoader(DS(va_texts, va_labels), batch_size=EVAL_BATCH, collate_fn=col)
    te = DataLoader(DS(te_texts, te_labels), batch_size=EVAL_BATCH, collate_fn=col)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_src, num_labels=NUM_LABELS)
    frozen = freeze_lower_layers(model, cfg['freeze_lower'])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    model.to(DEV)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['lr'], weight_decay=0.01)
    cw = CLASS_W.to(DEV) if cfg['weighted'] else None
    sched = get_linear_schedule_with_warmup(
        opt, int(len(tr) * cfg['epochs'] * 0.1), len(tr) * cfg['epochs'])

    LOSS, VAL = [], []
    for ep in range(cfg['epochs']):
        model.train(); ls = []; opt.zero_grad()
        for i, (ids, msk, y) in enumerate(tr):
            ids, msk, y = ids.to(DEV), msk.to(DEV), y.to(DEV)
            logits = model(input_ids=ids, attention_mask=msk).logits
            loss_t = F.cross_entropy(logits.float(), y, weight=cw)
            lv = loss_t.item()
            if math.isnan(lv) or math.isinf(lv):
                opt.zero_grad(); continue
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            ls.append(lv); LOSS.append({'config': name, 'epoch': ep + 1, 'step': i, 'loss': lv})
            if i % 100 == 0:
                print(f'    [{name}] Ep{ep+1} {i}/{len(tr)} loss={round(lv,4)}', flush=True)
        model.eval(); c = t = 0
        with torch.no_grad():
            for ids, msk, y in va:
                o = model(input_ids=ids.to(DEV), attention_mask=msk.to(DEV))
                c += (o.logits.argmax(-1) == y.to(DEV)).sum().item(); t += y.size(0)
        vacc = c / t
        VAL.append({'config': name, 'epoch': ep + 1,
                    'train_loss': sum(ls) / max(len(ls), 1), 'val_acc': vacc})
        print(f'  >>> [{name}] Epoch {ep+1} loss={round(sum(ls)/max(len(ls),1),4)} '
              f'val_acc={round(vacc,4)}', flush=True)

    # ---- 测试集 ----
    model.eval(); ps = []; ys_ = []; cfp = []; txt = []; tlen = []
    with torch.no_grad():
        for ids, msk, y in te:
            o = model(input_ids=ids.to(DEV), attention_mask=msk.to(DEV))
            pr = torch.softmax(o.logits.float(), -1).cpu()
            ps.extend(pr.argmax(-1).tolist())
            cfp.extend(pr.max(-1).values.tolist())
            ys_.extend(y.tolist())
            for j in range(ids.shape[0]):
                txt.append(tok.decode(ids[j], skip_special_tokens=True))
                tlen.append(int(msk[j].sum().item()))
    acc = sum(1 for a, b in zip(ys_, ps) if a == b) / len(ys_)
    # ★ weighted F1 按「测试集真实标签数」加权 —— 和 sklearn average='weighted' 定义一致，
    #   否则和 LoRA 那一版的数字没法并排比
    tcnt = {}
    for a in ys_: tcnt[a] = tcnt.get(a, 0) + 1
    f1s = []
    for c_ in range(K):
        tp = sum(1 for a, b in zip(ys_, ps) if a == c_ and b == c_)
        fp = sum(1 for a, b in zip(ys_, ps) if a != c_ and b == c_)
        fn = sum(1 for a, b in zip(ys_, ps) if a == c_ and b != c_)
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
    mf1 = sum(f1s) / K
    wf1 = sum(f1s[i] * tcnt.get(i, 0) for i in range(K)) / max(len(ys_), 1)
    dt = time.time() - t0
    print(f'\n  ===== {name} =====')
    for i, cname in enumerate(CLS):
        print(f'    {cname:>9}: F1={f1s[i]:.4f}  测试集支持数={tcnt.get(i,0)}')
    print(f'  acc={acc:.4f}  macroF1={mf1:.4f}  weightedF1={wf1:.4f}  '
          f'训练参数={n_train:,} ({n_train/n_total*100:.2f}%)  用时={dt:.0f}s', flush=True)

    res = dict(config=name, lr=cfg['lr'], epochs=cfg['epochs'],
               freeze_lower=cfg['freeze_lower'], class_weight=cfg['weighted'],
               maxlen=ML, n_train_samples=len(TT),
               trainable_params=n_train, total_params=n_total,
               trainable_pct=round(n_train / n_total * 100, 4),
               test_acc=round(acc, 4), macro_f1=round(mf1, 4),
               weighted_f1=round(wf1, 4), seconds=round(dt))
    for i, cname in enumerate(CLS):
        res[f'f1_{cname}'] = round(f1s[i], 4)

    if detail:
        import pandas as pd
        pd.DataFrame(LOSS).to_csv(f'{OUT}/train_log.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(VAL).to_csv(f'{OUT}/val_log.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame({'text': txt, 'true_label': [CLS[i] for i in ys_],
                      'pred_label': [CLS[i] for i in ps],
                      'confidence': [round(x, 4) for x in cfp],
                      'token_len': tlen,
                      'correct': [a == b for a, b in zip(ys_, ps)]}).to_csv(
            f'{OUT}/predictions.csv', index=False, encoding='utf-8-sig')
        errs = []
        for i, a in enumerate(ys_):
            if ps[i] != a:
                errs.append({'true': CLS[a], 'pred': CLS[ps[i]],
                             'confidence': round(cfp[i], 4), 'token_len': tlen[i],
                             'text': txt[i]})
        err_df = pd.DataFrame(errs, columns=['true', 'pred', 'confidence', 'token_len', 'text'])
        err_df.sort_values('confidence', ascending=False).to_csv(
            f'{OUT}/errors_by_class.csv', index=False, encoding='utf-8-sig')
        # 错误率 vs 长度分桶（负样本更长 → 看是否长度偏置）
        bk = [0, 64, 128, 256, 384, 512, 9999]
        rows_b = []
        for lo, hi in zip(bk[:-1], bk[1:]):
            idx = [i for i, l in enumerate(tlen) if lo < l <= hi]
            if not idx: continue
            rows_b.append({'len_range': f'({lo},{hi}]', 'n': len(idx),
                           'acc': round(sum(1 for i in idx if ys_[i] == ps[i]) / len(idx), 4),
                           'neg_ratio': round(sum(1 for i in idx if ys_[i] == 0) / len(idx), 4)})
        pd.DataFrame(rows_b).to_csv(f'{OUT}/acc_by_length.csv', index=False, encoding='utf-8-sig')
        with open(f'{OUT}/bert_report.txt', 'w', encoding='utf-8') as fh:
            fh.write(f'{MODEL_ID} 全参微调 [{name}]\n')
            fh.write(f'超参: lr={cfg["lr"]} epochs={cfg["epochs"]} batch={BATCH} '
                     f'maxlen={ML} freeze_lower={cfg["freeze_lower"]} '
                     f'class_weight={cfg["weighted"]} seed={SEED}\n')
            fh.write(f'训练参数: {n_train:,} / {n_total:,} ({n_train/n_total*100:.2f}%)\n')
            fh.write(f'测试集 acc={acc:.4f} macroF1={mf1:.4f} weightedF1={wf1:.4f}\n\n')
            for i, cname in enumerate(CLS):
                fh.write(f'{cname}: F1={f1s[i]:.4f}\n')
            fh.write('\n按长度的准确率:\n')
            for r_ in rows_b:
                fh.write(f'  {r_["len_range"]:>12}: n={r_["n"]:<5} acc={r_["acc"]:.4f}\n')
    if save_ckpt:
        torch.save(model.state_dict(), os.path.join(WORK, f'bert_{TAG}.pt'))
        print(f'  ✓ 权重已存 {WORK}/bert_{TAG}.pt', flush=True)

    del model, opt, tr, va, te
    gc.collect()
    if USE_CUDA: torch.cuda.empty_cache()
    return res, LOSS, VAL

# ==================== 6. 超参对比 ====================
CONFIGS = [
    dict(name='main',     lr=2e-5, epochs=3, freeze_lower=0, weighted=False, maxlen=BASE_MAXLEN),
    dict(name='lr_5e5',  lr=5e-5, epochs=3, freeze_lower=0, weighted=False, maxlen=BASE_MAXLEN),
    dict(name='lr_1e5',  lr=1e-5, epochs=3, freeze_lower=0, weighted=False, maxlen=BASE_MAXLEN),
    dict(name='ep5',     lr=2e-5, epochs=5, freeze_lower=0, weighted=False, maxlen=BASE_MAXLEN),
    dict(name='partial', lr=3e-5, epochs=3, freeze_lower=6, weighted=False, maxlen=BASE_MAXLEN),
    dict(name='weighted',lr=2e-5, epochs=3, freeze_lower=0, weighted=True,  maxlen=BASE_MAXLEN),
    dict(name='len512',  lr=2e-5, epochs=3, freeze_lower=0, weighted=False, maxlen=512),
    dict(name='len128',  lr=2e-5, epochs=3, freeze_lower=0, weighted=False, maxlen=128),
]

if SMOKE:
    # ★ 分层取样：直接取前 N 条会全是正面（正面占 68.5%），macro F1 会假性掉到 0.5
    def stratified_first(texts, labels, n_total):
        out_t, out_l = [], []
        for l_ in [0, 1]:
            k_ = int(round(n_total * cnt.get(l_, 1) / N))
            got = [t for t, l in zip(texts, labels) if l == l_][:k_]
            out_t += got; out_l += [l_] * len(got)
        return out_t, out_l
    CONFIGS = [CONFIGS[0]]
    tr_texts, tr_labels = stratified_first(tr_texts, tr_labels, 200)
    va_texts, va_labels = stratified_first(va_texts, va_labels, 100)
    te_texts, te_labels = stratified_first(te_texts, te_labels, 100)
    for c in CONFIGS: c['epochs'] = 1
    CURVE_SIZES = [50, 100]
    print(f'  ★ 冒烟：train {len(tr_texts)} val {len(va_texts)} test {len(te_texts)}（分层）', flush=True)

import pandas as pd
RESULTS = []; MAIN = None
if RUN_ABLATION:
    print('\n=== [4/8] 超参对比实验 ===', flush=True)
    for k, cfg in enumerate(CONFIGS):
        print(f'\n{"="*70}\n  配置 {k+1}/{len(CONFIGS)}: {cfg["name"]}  '
              f'(lr={cfg["lr"]} ep={cfg["epochs"]} freeze={cfg["freeze_lower"]} '
              f'w={cfg["weighted"]} len={cfg["maxlen"]})\n{"="*70}', flush=True)
        r, L, V = run_config(cfg, detail=(cfg['name'] == 'main'),
                             save_ckpt=(cfg['name'] == 'main'))
        RESULTS.append(r)
        if cfg['name'] == 'main': MAIN = (r, L, V)

# ==================== 7. 数据量曲线 ====================
CURVE = []
if RUN_CURVE:
    print('\n=== [5/8] 数据量曲线 ===', flush=True)
    main_cfg = dict(CONFIGS[0])
    main_cfg['epochs'] = 1 if SMOKE else 3
    # 复现 main 的结果作为"全量"点
    full_res = next((r for r in RESULTS if r['config'] == 'main'), None)
    if full_res is not None and not SMOKE:
        CURVE.append(dict(n_train=full_res['n_train_samples'],
                          test_acc=full_res['test_acc'],
                          macro_f1=full_res['macro_f1'],
                          weighted_f1=full_res['weighted_f1']))
        print(f"  ✓ 全量点直接复用 main: acc={full_res['test_acc']}", flush=True)
    # 分层子采样
    for n in CURVE_SIZES:
        if n >= len(tr_texts):
            continue
        rem = list(zip(tr_texts, tr_labels))
        random.Random(SEED).shuffle(rem)
        n0 = int(round(n * cnt.get(0, 1) / N))
        n1 = n - n0
        c0 = [t for t, l in rem if l == 0][:n0]
        c1 = [t for t, l in rem if l == 1][:n1]
        sub = c0 + c1
        sub_lab = [0] * len(c0) + [1] * len(c1)
        print(f'\n  --- 数据量 {len(sub)} 条 ---', flush=True)
        r, _, _ = run_config(main_cfg, tr_texts_=sub, tr_labels_=sub_lab)
        CURVE.append(dict(n_train=len(sub), test_acc=r['test_acc'],
                          macro_f1=r['macro_f1'], weighted_f1=r['weighted_f1']))
        print(f"  ✓ n={len(sub)}  acc={r['test_acc']}  macroF1={r['macro_f1']}", flush=True)

CURVE.sort(key=lambda r: r['n_train'])
if CURVE:
    pd.DataFrame(CURVE).to_csv(f'{OUT}/data_curve.csv', index=False, encoding='utf-8-sig')
    print('\n===== 数据量曲线 =====')
    print(pd.DataFrame(CURVE).to_string(index=False), flush=True)

# ==================== 8. 汇总 + 画图 + 存档 ====================
print('\n=== [6/8] 汇总 ===', flush=True)
if RESULTS:
    ab = pd.DataFrame(RESULTS)
    ab.to_csv(f'{OUT}/ablation_results.csv', index=False, encoding='utf-8-sig')
    print(ab[['config','lr','epochs','freeze_lower','class_weight','maxlen',
              'trainable_params','trainable_pct','test_acc','macro_f1',
              'weighted_f1','seconds']].to_string(index=False), flush=True)

print('\n=== [7/8] 画图 ===', flush=True)
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if RESULTS and MAIN:
        res, LOSS, VAL = MAIN
        df = pd.DataFrame(LOSS); vd = pd.DataFrame(VAL)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(df['step'], df['loss'], alpha=.5, lw=.7, label='batch loss')
        ax[0].plot(df['step'], df['loss'].rolling(100, min_periods=1).mean(),
                   lw=2.2, color='crimson', label='smoothed(100)')
        ax[0].set_xlabel('Step'); ax[0].set_ylabel('Training Loss')
        ax[0].set_title('MacBERT Full Fine-tuning Loss'); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(vd['epoch'], vd['val_acc'], 'o-', lw=2.2, color='crimson')
        ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Validation Accuracy')
        ax[1].set_ylim(0, 1); ax[1].set_title('Validation Accuracy'); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f'{OUT}/loss_curve.png', dpi=150); plt.close()

        # 混淆矩阵
        import numpy as _np
        cm = _np.zeros((K, K), dtype=int)
        pr_df = pd.read_csv(f'{OUT}/predictions.csv')
        for a_, b_ in zip(pr_df['true_label'], pr_df['pred_label']):
            cm[CLS.index(a_), CLS.index(b_)] += 1
        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(['Negative','Positive'], rotation=20, ha='right')
        ax.set_yticklabels(['Negative','Positive'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'MacBERT Confusion Matrix (acc={res["test_acc"]:.4f})')
        for i in range(K):
            for j in range(K):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        fig.colorbar(im); plt.tight_layout()
        plt.savefig(f'{OUT}/confusion_matrix.png', dpi=150); plt.close()

        if len(RESULTS) > 1:
            fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
            x = _np.arange(len(ab)); lbl = ab['config']
            for a_, col_, ttl in [(ax[0], 'test_acc', 'Test Accuracy'),
                                  (ax[1], 'macro_f1', 'Macro F1'),
                                  (ax[2], 'weighted_f1', 'Weighted F1')]:
                a_.bar(x, ab[col_], color='crimson', alpha=.85)
                a_.set_xticks(x); a_.set_xticklabels(lbl, rotation=25, ha='right')
                a_.set_ylabel(ttl); a_.set_ylim(0, 1); a_.set_title(ttl)
                a_.grid(alpha=.3, axis='y')
                for i, v in enumerate(ab[col_]):
                    a_.text(i, v + .02, f'{v:.3f}', ha='center', fontsize=8)
            plt.tight_layout(); plt.savefig(f'{OUT}/ablation_compare.png', dpi=150); plt.close()
    if len(CURVE) > 1:
        cd = pd.DataFrame(CURVE)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ax[0].plot(cd['n_train'], cd['test_acc'], 'o-', lw=2, color='crimson', label='accuracy')
        ax[0].plot(cd['n_train'], cd['macro_f1'], 's--', lw=2, color='steelblue', label='macro F1')
        ax[0].set_xscale('log'); ax[0].set_xlabel('Training Samples'); ax[0].set_ylabel('Score')
        ax[0].set_title('Data Quantity Curve'); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(cd['n_train'], cd['weighted_f1'], '^-', lw=2, color='darkgreen')
        ax[1].set_xscale('log'); ax[1].set_xlabel('Training Samples')
        ax[1].set_ylabel('Weighted F1'); ax[1].set_ylim(0, 1)
        ax[1].set_title('Weighted F1 vs Data Size'); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f'{OUT}/data_curve.png', dpi=150); plt.close()
    print('  ✓ 图已保存', flush=True)
except Exception as e:
    print(f'  画图跳过: {type(e).__name__}: {e}', flush=True)

print('\n=== [8/8] 存档 ===', flush=True)
with open(f'{OUT}/ablation_summary.txt', 'w', encoding='utf-8') as fh:
    fh.write(f'{MODEL_ID} 全参微调 - 参数对比 + 数据量曲线\n')
    fh.write(f'seed={SEED}  train={len(tr_texts)} val={len(va_texts)} test={len(te_texts)}  '
             f'设备={DEV}\n\n')
    if RESULTS:
        fh.write(ab.to_string(index=False)); fh.write('\n\n')
    if CURVE:
        fh.write('数据量曲线:\n'); fh.write(pd.DataFrame(CURVE).to_string(index=False)); fh.write('\n')

zp = os.path.join(WORK, f'results_BERT_{TAG}.zip')
try:
    subprocess.run(['zip', '-r', '-q', zp, os.path.basename(OUT)], cwd=WORK)
except Exception:
    import shutil
    shutil.make_archive(zp[:-4], 'zip', WORK, os.path.basename(OUT))

print('\n' + '=' * 70)
if RESULTS:
    m = next(r for r in RESULTS if r['config'] == 'main')
    best = max(RESULTS, key=lambda r: r['macro_f1'])
    print(f"★ 主实验(main):  acc={m['test_acc']}  macroF1={m['macro_f1']}  weightedF1={m['weighted_f1']}")
    print(f"★ 训练参数: {m['trainable_params']:,} ({m['trainable_pct']}%)  总参数 {m['total_params']:,}")
    print(f"★ 最佳(macroF1): {best['config']}  macroF1={best['macro_f1']}")
print(f"★ 结果包: {zp}  ({os.path.getsize(zp)/1e6:.1f} MB)")
print('=' * 70)
print('✅ 冒烟通过！把 SMOKE 改成 False 再跑正式' if SMOKE else 'DONE', flush=True)
