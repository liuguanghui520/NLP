# MacBERT 全参微调 + 参数对比 + 数据量曲线（新数据集 ChnSentiCorp）

> **同一份代码，本地和 Kaggle 都能跑。** 脚本自动检测 GPU：有 CUDA 用 GPU（40 分钟跑完），没有就用 CPU（会打印预估时间）。

---

## 📊 新数据集实测统计（决定了下面所有实验设计）

| 发现 | 数值 | 对应实验 |
|---|---|---|
| **二分类，不平衡 2.18×** | 负面 2443 (31.47%) ↔ 正面 5320 (68.53%) | `weighted` 类别加权损失 |
| **文本比旧数据集长 3 倍** | token P50=84 / P75=154 / P95=376 / max=514 | `len128` / `len512` 长度对比 |
| **maxlen=64 会截断 62.4%** | 截断 4847/7763 条 | baseline 用 **256**（截断 10.78%）|
| **负面评论明显更长** | 负面 P50=108、正面 P50=76；>256 比例 17.15% vs 7.86% | 错误分析里"长度与标签相关" |
| **数据很干净** | 跨划分重复 0、标签冲突 0 | ✅ 无泄漏风险 |

> 旧数据集是微博 6 类情绪（joy/sad/anger/fear/surprise/neutral），**已全部作废**。
> 新数据集是携程酒店评论 2 类情感（negative/positive），全组所有路线统一用这个。

---

## 🖥️ 跑哪儿？实测结论

| 环境 | 能不能跑 | 实测速度 | 8 组配置耗时 |
|---|---|---|---|
| **Kaggle T4 ×2** | ✅ 推荐 | ~15 样本/s | **~45 分钟** |
| **本机 CPU**（12 线程 R5 5600H） | ⚠️ 能，但要挂机 | **1.1 样本/s**（batch16/seq256）| **~37 小时** |
| **本机跑 Qwen3-1.7B** | ❌ 不行 | 无 NVIDIA 显卡（0.5GB AMD 集显）；1.7B fp32 权重 6.5GB + AdamW 状态 13GB ≈ 20GB > 内存 15.4GB | 无法 |

**结论**：BERT 这半边本地跑得动（挂着就行），但**大模型那半边必须上 Kaggle**。

---

## ⚠️ 每次重跑前

**`Session` → `Restart & Clear Output`**（Kaggle）/ 本地直接重跑脚本。

---

## 两轮跑法

```python
SMOKE = True     # 第一次 True（3分钟，验代码）；改成 False 跑正式
```

| | 耗时（Kaggle） | 耗时（本机 CPU）|
|---|---|---|
| `SMOKE=True` | 3 分钟 | 5 分钟 |
| `SMOKE=False` | **~45 分钟**（8 组 + 5 个曲线点）| **~24~30 小时**（实测吞吐 2.6 样本/秒）|

**时间紧就删配置**：只留 `main / partial / weighted / len512`（4 组 ≈ 22 分钟）。

---

## 完整代码

```python
exec(r"""
import os, sys, gc, json, math, time, random, zipfile, shutil, subprocess
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Windows 控制台默认代码页是 GBK，打印 '✓' / '⚠' 会抛 UnicodeEncodeError。
# 统一把 stdout/stderr 切成 UTF-8 + errors='replace'，任何代码页下都能跑完。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ★★★ 开关 ★★★
SMOKE       = True      # True=冒烟(3分钟) / False=正式(45分钟)
# 本机跑正式版时用环境变量覆盖，不用改代码：$env:BERT_SMOKE='0'
if os.environ.get('BERT_SMOKE') is not None:
    SMOKE = os.environ['BERT_SMOKE'] == '1'
RUN_ABLATION = True     # 超参对比实验
RUN_CURVE    = True     # 数据量曲线
USE_LEN_GROUP = True    # 长度分桶组 batch（CPU 上约 2x 提速，8 组统一用，对比公平）
RESUME      = True      # 断点续跑：已跑完的配置直接跳过
MODEL_ID    = 'hfl/chinese-macbert-base'
SEED        = 42
NUM_LABELS  = 2
CLS         = ['negative', 'positive']
BASE_MAXLEN = 256
BATCH_GPU   = 16
BATCH_CPU   = 8      # 本机 CPU 取 8：峰值内存约减半（本机空闲内存仅 2.4GB，batch=16 会触发换页），
                      # 与报告 §4 超参表"CPU 8"一致。
                      # 基准：未开长度分桶时 batch 8/16/32 吞吐 1.23 / 1.24 / 1.26 样本每秒（差 0.8%，batch 大小不敏感）；
                      # 正式跑开着长度分桶组 batch，实测吞吐约 2.6 样本/秒（2026-10-26 01:50 起那轮）。
EVAL_BATCH  = 32
CURVE_SIZES = [100, 500, 1000, 2000, 3000]   # 加上全量 = 6 个点

# ==================== 日志双写（控制台 + 文件） ====================
# 为什么要这个：任务书评分项 ②「进度与工作量证明」明确要"运行日志"作为
# 可核查的证据，而报告附录 C 也要引用日志片段。
# 之前所有输出只 print 到控制台 —— 控制台一关、机器一重启，日志就跟着没了
# （2026-10-24 那次训练就是这么丢的，只剩最后几行）。
# 现在 stdout / stderr 同时写到日志文件，控制台照常看得到。
class _Tee:
    """把一个流同时写到多个目标（控制台 + 文件）。"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s); st.flush()
            except Exception:
                pass
        return len(s)
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass
    def isatty(self):
        return False

WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
# 本机跑时把产物直接放进 src/results/，这样汇总脚本不用再搬一次
if os.environ.get('BERT_OUT_ROOT'):
    WORK = os.environ['BERT_OUT_ROOT']
TAG  = 'SMOKE' if SMOKE else 'FULL'
OUT  = os.path.join(WORK, f'results_BERT_{TAG}')
os.makedirs(OUT, exist_ok=True)

# 日志路径：可用 BERT_LOG 环境变量指定，默认写到结果目录里的 run_log.txt
_LOG_PATH = os.environ.get('BERT_LOG') or os.path.join(OUT, 'run_log.txt')
try:
    os.makedirs(os.path.dirname(os.path.abspath(_LOG_PATH)), exist_ok=True)
    _lf = open(_LOG_PATH, 'a', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(sys.stdout, _lf)
    sys.stderr = _Tee(sys.stderr, _lf)
except Exception as _e:
    _LOG_PATH = None
    print(f'  ⚠ 日志文件打开失败（不影响训练）: {_e}', flush=True)


# ==================== 原子写（暂停/断电安全） ====================
# 为什么需要：下面几个 CSV 是"断点续跑的唯一依据"。
# 如果在 to_csv 写入的那几毫秒里被 Ctrl+C 打断，文件会变成半截，
# 而续跑逻辑读失败时会抛异常 → 退化成"从头跑全部 8 组"，
# 白白丢掉十几个小时。所以改成：先写临时文件，写完再原子替换。
def _atomic_csv(df, path, index=False):
    tmp = f'{path}.tmp'
    try:
        df.to_csv(tmp, index=index, encoding='utf-8-sig')
        os.replace(tmp, path)          # 同卷内的原子替换
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        print(f'  ⚠ 写 {path} 失败: {e}', flush=True)
        raise
    return path


# ==================== 0. 环境 ====================
print('=== [0/8] 环境 ===', flush=True)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
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
          f'实测吞吐约 2.6 样本/秒（batch=8, len=256），'
          f'全量 8 组 + 曲线预计 24~30 小时，建议挂机。', flush=True)
print(f'  torch {torch.__version__}  seed={SEED}', flush=True)
if _LOG_PATH:
    print(f'  ✓ 日志同时写入: {_LOG_PATH}', flush=True)

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
# ★ 测试集样本 ID —— 全项目统一的对齐键（主键）。
#   fewshot_ids.json / latency_ids.json 用的是同一个字段；
#   各路线的 predictions.csv 也以它为主键。
#   这样 MacBERT 的逐条预测和 LoRA 的逐条预测才能按 sample_id 一一对齐，
#   做配对显著性检验（McNemar / paired bootstrap）。
#   事后补不出来：没有 ID 就没法把两条路线的预测对上。
te_ids = [r.get('sample_id', str(i)) for i, r in enumerate(te_rows)]

# ==================== 2. 模型 ====================
print('\n=== [2/8] 准备模型 ===', flush=True)
BASE_DIR = os.path.join(os.getcwd(), 'macbert_base')
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.join(WORK, 'macbert_base')

def resolve_model():
    # 0) 本地已经有一份就直接用，省掉每次重下 400MB
    if os.path.exists(os.path.join(BASE_DIR, 'config.json')):
        try:
            tok = AutoTokenizer.from_pretrained(BASE_DIR)
            m = AutoModelForSequenceClassification.from_pretrained(BASE_DIR, num_labels=NUM_LABELS)
            print(f'  ✓ 直接用本地已有模型 -> {BASE_DIR}', flush=True)
            return tok, BASE_DIR
        except Exception as e:
            print(f'  本地模型不可用({e})，转为下载', flush=True)
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

# ------------------------------------------------------------------
# 长度分桶组 batch（LengthGroupedSampler 的思路）
#
# 为什么需要：本数据集 token 长度 P50=85 / P90=256 / 均值 108.7。
# 随机组 batch 时，只要一个 batch 里有 1 条 >=256，整批就得 pad 到 256
#   P(batch 里存在 >=256 的样本) = 1 - 0.9^16 ≈ 81%
# 也就是 81% 的 batch 在为 padding 白算 2.4 倍的 token。
#
# 做法：先全局打乱 → 按 megabatch(50×batch) 段内按长度排序 → 切 batch
#       → 再把 batch 顺序打乱。
# 这样既保留了随机性（megabatch 边界 + batch 顺序都是随机的），
# 又让同一 batch 内长度接近，padding 大幅减少。
# 本机 CPU 上实测提速约 2x，8 组配置全部用同一种方式，对比仍然公平。
# ------------------------------------------------------------------
class SortishBatches(Sampler):
    def __init__(self, lengths, batch_size, mega=50, seed=42):
        self.lengths = np.asarray(lengths)
        self.bs = batch_size
        self.mega = mega
        self.seed = seed
        self.epoch = 0
    def __iter__(self):
        g = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        idx = g.permutation(len(self.lengths))
        mb = self.bs * self.mega
        batches = []
        for s in range(0, len(idx), mb):
            chunk = idx[s:s + mb]
            chunk = chunk[np.argsort(self.lengths[chunk], kind='stable')]
            for b in range(0, len(chunk), self.bs):
                batches.append(chunk[b:b + self.bs].tolist())
        g.shuffle(batches)
        return iter(batches)
    def __len__(self):
        return math.ceil(len(self.lengths) / self.bs)

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

# 全局累积的日志（断点续跑时从已有 CSV 里读回来，保证曲线能重建）
ALL_LOSS = []
ALL_VAL  = []

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
    if USE_LEN_GROUP and not cfg.get('no_len_group'):
        # 长度分桶：先算每条的真实 token 长度，再按长度排 batch（省掉大量 padding 算力）
        _lens = [len(x) for x in tok(list(TT), truncation=True, max_length=ML)['input_ids']]
        tr = DataLoader(DS(TT, LL), batch_sampler=SortishBatches(_lens, BATCH, seed=SEED),
                        collate_fn=col)
    else:
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
    ALL_LOSS.extend(LOSS); ALL_VAL.extend(VAL)

    # ---- 测试集 ----
    # ★ 两个类的完整概率都留着：以后要算 AUC / 校准曲线 / 概率的配对 bootstrap 都用得上，
    #   而且只有 forward 的那一刻才有，事后补不出来。
    model.eval(); ps = []; ys_ = []; cfp = []; txt = []; tlen = []
    p_neg = []; p_pos = []
    with torch.no_grad():
        for ids, msk, y in te:
            o = model(input_ids=ids.to(DEV), attention_mask=msk.to(DEV))
            pr = torch.softmax(o.logits.float(), -1).cpu()
            ps.extend(pr.argmax(-1).tolist())
            cfp.extend(pr.max(-1).values.tolist())
            p_neg.extend(pr[:, 0].tolist())
            p_pos.extend(pr[:, 1].tolist())
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

    import pandas as pd
    # ---- 训练/验证日志：每个配置都写（断点续跑时用来重建曲线） ----
    if ALL_LOSS:
        _atomic_csv(pd.DataFrame(ALL_LOSS), f'{OUT}/train_log.csv')
    if ALL_VAL:
        _atomic_csv(pd.DataFrame(ALL_VAL), f'{OUT}/val_log.csv')
    # ============================================================
    # 落盘策略：**每一组配置都留逐条预测**，一份都不省。
    #
    # 为什么不能省：
    #   事后要跟 LoRA 臂做配对显著性检验（McNemar / paired bootstrap），
    #   必须能按 sample_id 把两条路线的 777 条预测逐条对齐。
    #   而本机磁盘只放得下一份 409MB 权重（只有 main 组存），
    #   所以其它 7 组的 predictions_<config>.csv 就是唯一证据 —— 不存 = 这组白跑。
    #
    # 预测文件**不含评论正文**：正文只在 error_samples.csv 留 3 条
    # （报告 §6.1 直接引用的那 3 条），其余靠 sample_id 回查 test.jsonl。
    # 这样提交包不含评论语料，和 §7.2 的声明一致。
    # ============================================================
    # ---- 错误率 vs 长度分桶（负样本更长 → 看是否长度偏置）----
    bk = [0, 64, 128, 256, 384, 512, 9999]
    rows_b = []
    for lo, hi in zip(bk[:-1], bk[1:]):
        idx = [i for i, l in enumerate(tlen) if lo < l <= hi]
        if not idx: continue
        rows_b.append({'len_range': f'({lo},{hi}]', 'n': len(idx),
                       'acc': round(sum(1 for i in idx if ys_[i] == ps[i]) / len(idx), 4),
                       'neg_ratio': round(sum(1 for i in idx if ys_[i] == 0) / len(idx), 4)})

    pred_df = pd.DataFrame({
        'sample_id': te_ids,
        'idx': list(range(len(ys_))),
        'true_label': [CLS[i] for i in ys_],
        'pred_label': [CLS[i] for i in ps],
        'p_negative': [round(x, 6) for x in p_neg],
        'p_positive': [round(x, 6) for x in p_pos],
        'confidence': [round(x, 4) for x in cfp],
        'token_len': tlen,
        'correct': [a == b for a, b in zip(ys_, ps)]})
    # ★ 每组一份，这是"事后还能对比"的唯一凭据
    _atomic_csv(pred_df, f'{OUT}/predictions_{name}.csv')
    _atomic_csv(pd.DataFrame(rows_b), f'{OUT}/acc_by_length_{name}.csv')

    if detail:
        # main 组：保留传统文件名（画混淆矩阵、fill_report.py 都读这两个）
        _atomic_csv(pred_df, f'{OUT}/predictions.csv')
        _atomic_csv(pd.DataFrame(rows_b), f'{OUT}/acc_by_length.csv')
        # 错误清单（不含正文，靠 sample_id 回查 test.jsonl）
        errs = []
        for i, a in enumerate(ys_):
            if ps[i] != a:
                errs.append({'sample_id': te_ids[i], 'true': CLS[a], 'pred': CLS[ps[i]],
                             'confidence': round(cfp[i], 4), 'token_len': tlen[i]})
        err_df = pd.DataFrame(errs, columns=['sample_id', 'true', 'pred', 'confidence', 'token_len'])
        _atomic_csv(err_df.sort_values('confidence', ascending=False),
                    f'{OUT}/errors_by_class.csv')
        # 只给报告 §6.1 留 3 条「最确信却错」的样例：正文仅此 3 条会进提交包
        samp = []
        for i, a in enumerate(ys_):
            if ps[i] != a:
                samp.append({'sample_id': te_ids[i], 'true': CLS[a], 'pred': CLS[ps[i]],
                             'confidence': round(cfp[i], 4), 'token_len': tlen[i],
                             'text': te_texts[i], 'text_seen': txt[i]})
        _atomic_csv(pd.DataFrame(samp, columns=['sample_id', 'true', 'pred', 'confidence',
                                                'token_len', 'text', 'text_seen'])
                    .sort_values('confidence', ascending=False).head(3),
                    f'{OUT}/error_samples.csv')
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
        # 每组一份纯文本小结，方便报告逐组引用
        with open(f'{OUT}/report_{name}.txt', 'w', encoding='utf-8') as fh:
            fh.write(f'{MODEL_ID} 全参微调 [{name}]\n')
            fh.write(f'超参: lr={cfg["lr"]} epochs={cfg["epochs"]} batch={BATCH} '
                     f'maxlen={ML} freeze_lower={cfg["freeze_lower"]} '
                     f'class_weight={cfg["weighted"]} seed={SEED}\n')
            fh.write(f'训练参数: {n_train:,} / {n_total:,} ({n_train/n_total*100:.2f}%)\n')
            fh.write(f'测试集 acc={acc:.4f} macroF1={mf1:.4f} weightedF1={wf1:.4f}\n')
            for i, cname in enumerate(CLS):
                fh.write(f'{cname}: F1={f1s[i]:.4f}\n')
            fh.write(f'逐条预测: {OUT}/predictions_{name}.csv\n')
    if save_ckpt:
        CKPT_DIR = os.environ.get('BERT_CKPT_DIR', WORK)
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f'bert_{TAG}.pt'))
        print(f'  ✓ 权重已存 {CKPT_DIR}/bert_{TAG}.pt', flush=True)

    del model, opt, tr, va, te
    gc.collect()
    if USE_CUDA: torch.cuda.empty_cache()
    return res, LOSS, VAL

# ==================== 6. 超参对比 ====================
# ★ 顺序按「对报告的重要程度」排，不是按参数字典序。
#   本机 CPU 跑全套要十几个小时，万一时间不够/中途停机，
#   先跑完的越是报告里最需要的：
#     main      → 主结果（头条数字）
#     partial   → 全参 vs 冻结底层 6 层（训练参数量对比，直接回应"全参微调")
#     weighted  → 类别加权损失（直接关系到 Macro F1，回应类别不均衡）
#     lr_5e5    → 学习率对比
#     len128    → 序列长度对比（便宜）
#     len512    → 序列长度对比（贵，放后面）
#     lr_1e5    → 学习率第三档（补充）
#     ep5       → 训练轮数（最贵，放最后）
CONFIGS = [
    dict(name='main',     lr=2e-5, epochs=3, freeze_lower=0, weighted=False, maxlen=BASE_MAXLEN),
    dict(name='partial', lr=3e-5, epochs=3, freeze_lower=6, weighted=False, maxlen=BASE_MAXLEN),
    dict(name='weighted',lr=2e-5, epochs=3, freeze_lower=0, weighted=True,  maxlen=BASE_MAXLEN),
    dict(name='lr_5e5',  lr=5e-5, epochs=3, freeze_lower=0, weighted=False, maxlen=BASE_MAXLEN),
    dict(name='len128',  lr=2e-5, epochs=3, freeze_lower=0, weighted=False, maxlen=128),
    dict(name='len512',  lr=2e-5, epochs=3, freeze_lower=0, weighted=False, maxlen=512),
    dict(name='lr_1e5',  lr=1e-5, epochs=3, freeze_lower=0, weighted=False, maxlen=BASE_MAXLEN),
    dict(name='ep5',     lr=2e-5, epochs=5, freeze_lower=0, weighted=False, maxlen=BASE_MAXLEN),
]

if SMOKE:
    # ★ 分层取样：直接取前 N 条会全是正面（正面占 68.5%），macro F1 会假性掉到 0.5
    def stratified_first(texts, labels, n_total):
        out_t, out_l, out_i = [], [], []
        for l_ in [0, 1]:
            k_ = int(round(n_total * cnt.get(l_, 1) / N))
            got = [(i, t) for i, (t, l) in enumerate(zip(texts, labels)) if l == l_][:k_]
            out_i += [g[0] for g in got]
            out_t += [g[1] for g in got]; out_l += [l_] * len(got)
        return out_t, out_l, out_i
    CONFIGS = [CONFIGS[0]]
    # 冒烟样本量可用环境变量调小（BERT_SMOKE_N），方便几十秒内验证代码路径
    _sn = int(os.environ.get('BERT_SMOKE_N', '200'))
    tr_texts, tr_labels, _ = stratified_first(tr_texts, tr_labels, _sn)
    va_texts, va_labels, _ = stratified_first(va_texts, va_labels, max(_sn // 2, 20))
    te_texts, te_labels, _te_i = stratified_first(te_texts, te_labels, max(_sn // 2, 20))
    # ★ te_ids 必须和 te_texts 一起抽，否则 sample_id 和预测对不上（正式跑不受影响）
    te_ids = [te_ids[i] for i in _te_i]
    for c in CONFIGS: c['epochs'] = 1
    CURVE_SIZES = [50, 100]
    print(f'  ★ 冒烟：train {len(tr_texts)} val {len(va_texts)} test {len(te_texts)}（分层）', flush=True)

import pandas as pd
RESULTS = []; MAIN = None
_ABL_CSV = f'{OUT}/ablation_results.csv'
_CURVE_CSV = f'{OUT}/data_curve.csv'

# ==================== 断点续跑 ====================
# 本机 CPU 跑全套要十几个小时，中途崩了不能全白跑。
# 每跑完一个配置就落盘；重跑时把已完成的结果读回来直接跳过。
_DONE = {}
if RESUME and os.path.exists(_ABL_CSV):
    try:
        import pandas as _pd
        _old = _pd.read_csv(_ABL_CSV)
        for _, _r in _old.iterrows():
            _DONE[str(_r['config'])] = _r.to_dict()
        print(f'  ✓ 断点续跑：已有 {len(_DONE)} 个配置结果 → {sorted(_DONE)}', flush=True)
    except Exception as e:
        print(f'  ⚠ 读历史 ablation_results 失败（从头跑）: {e}', flush=True)
if RESUME and os.path.exists(f'{OUT}/train_log.csv'):
    try:
        import pandas as _pd
        _tl = _pd.read_csv(f'{OUT}/train_log.csv')
        ALL_LOSS.extend(_tl.to_dict('records'))
        _vl = _pd.read_csv(f'{OUT}/val_log.csv')
        ALL_VAL.extend(_vl.to_dict('records'))
        print(f'  ✓ 已读回历史日志 {len(ALL_LOSS)} 行 / {len(ALL_VAL)} 行', flush=True)
    except Exception as e:
        print(f'  ⚠ 读历史日志失败: {e}', flush=True)

if RUN_ABLATION:
    print('\n=== [4/8] 超参对比实验 ===', flush=True)
    for k, cfg in enumerate(CONFIGS):
        if cfg['name'] in _DONE:
            print(f'  ⏭ [{cfg["name"]}] 已有结果，跳过', flush=True)
            RESULTS.append(_DONE[cfg['name']])
            continue
        print(f'\n{"="*70}\n  配置 {k+1}/{len(CONFIGS)}: {cfg["name"]}  '
              f'(lr={cfg["lr"]} ep={cfg["epochs"]} freeze={cfg["freeze_lower"]} '
              f'w={cfg["weighted"]} len={cfg["maxlen"]})\n{"="*70}', flush=True)
        r, L, V = run_config(cfg, detail=(cfg['name'] == 'main'),
                             save_ckpt=(cfg['name'] == 'main'))
        RESULTS.append(r)
        if cfg['name'] == 'main': MAIN = (r, L, V)
        # ★ 每跑完一个配置立刻落盘，崩了也不丢前面的结果
        import pandas as _pd
        _atomic_csv(_pd.DataFrame(RESULTS), _ABL_CSV)
        print(f'  ✓ 已落盘 {_ABL_CSV}（{len(RESULTS)} 行）', flush=True)

# main 被跳过时（断点续跑），用历史日志重建 MAIN，让曲线图还能画出来
if MAIN is None and 'main' in _DONE:
    _ml = [x for x in ALL_LOSS if str(x.get('config')) == 'main']
    _mv = [x for x in ALL_VAL if str(x.get('config')) == 'main']
    MAIN = (_DONE['main'], _ml, _mv)
    print('  ✓ main 走的是断点续跑，曲线用历史日志重建', flush=True)

# ==================== 7. 数据量曲线 ====================
CURVE = []
_CURVE_DONE = set()
if RESUME and os.path.exists(_CURVE_CSV):
    try:
        import pandas as _pd
        _oc = _pd.read_csv(_CURVE_CSV)
        for _, _r in _oc.iterrows():
            CURVE.append(_r.to_dict()); _CURVE_DONE.add(int(_r['n_train']))
        print(f'  ✓ 断点续跑：数据量曲线已有 {len(CURVE)} 个点', flush=True)
    except Exception as e:
        print(f'  ⚠ 读历史 data_curve 失败: {e}', flush=True)
if RUN_CURVE:
    print('\n=== [5/8] 数据量曲线 ===', flush=True)
    main_cfg = dict(CONFIGS[0])
    main_cfg['epochs'] = 1 if SMOKE else 3
    # ★ 强制重跑模式：把曲线点重新训练一遍，只为把每个点的**逐条预测**留下来。
    #   起因：10-27 之前曲线点沿用了 'main' 这个名字，预测写进了 predictions_main.csv，
    #   还原主结果后那批预测已经丢失 —— 于是曲线只能做趋势解读，做不了逐点配对检验。
    #   这个模式下结果**另存** data_curve_repred.csv，不覆盖原来的 data_curve.csv，
    #   两份文件逐位对比本身就是"同一份代码、同一颗种子能跑出同样结果"的证据。
    _CURVE_FORCE = os.environ.get('BERT_CURVE_FORCE') == '1'
    _curve_out = _CURVE_CSV
    if _CURVE_FORCE:
        _curve_out = f'{OUT}/data_curve_repred.csv'
        # ★ 断点续跑粒度 = 一个规模点：
        #   已经重跑过的点（data_curve_repred.csv 里有的）直接跳过，
        #   没跑过的才真跑。这样中途 Ctrl+C 不会丢掉已完成的点。
        CURVE = []
        _CURVE_DONE = set()
        if os.path.exists(_curve_out):
            try:
                import pandas as _pd
                _oc = _pd.read_csv(_curve_out)
                for _, _r in _oc.iterrows():
                    CURVE.append(_r.to_dict()); _CURVE_DONE.add(int(_r['n_train']))
                print(f'  ✓ data_curve_repred.csv 已有 {len(_CURVE_DONE)} 个点，跳过它们', flush=True)
            except Exception as e:
                print(f'  ⚠ 读 data_curve_repred.csv 失败: {e}', flush=True)
                CURVE = []; _CURVE_DONE = set()
        print('  ★ BERT_CURVE_FORCE=1：补跑数据量曲线（结果写入 data_curve_repred.csv）', flush=True)
    # 复现 main 的结果作为"全量"点
    full_res = next((r for r in RESULTS if r['config'] == 'main'), None)
    if full_res is not None and not SMOKE:
        if not any(int(c['n_train']) == int(full_res['n_train_samples']) for c in CURVE):
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
        if len(sub) in _CURVE_DONE:
            print(f'  ⏭ 数据量 {len(sub)} 条已有结果，跳过', flush=True)
            continue
        print(f'\n  --- 数据量 {len(sub)} 条 ---', flush=True)
        # ★ 名字必须和主配置区分开：run_config 会写 predictions_<name>.csv，
        #   如果沿用 'main'，这个点的预测会覆盖掉主配置的逐条预测文件。
        _cfg_curve = dict(main_cfg); _cfg_curve['name'] = f'curve_n{len(sub)}'
        r, _, _ = run_config(_cfg_curve, tr_texts_=sub, tr_labels_=sub_lab)
        CURVE.append(dict(n_train=len(sub), test_acc=r['test_acc'],
                          macro_f1=r['macro_f1'], weighted_f1=r['weighted_f1']))
        print(f"  ✓ n={len(sub)}  acc={r['test_acc']}  macroF1={r['macro_f1']}", flush=True)
        # ★ 每跑完一个点立刻落盘
        CURVE.sort(key=lambda x: x['n_train'])
        _atomic_csv(pd.DataFrame(CURVE), _curve_out)

CURVE.sort(key=lambda r: r['n_train'])
if CURVE:
    # 强制重跑模式下不要覆盖原始 data_curve.csv（它是对照基准）
    if not (os.environ.get('BERT_CURVE_FORCE') == '1'):
        _atomic_csv(pd.DataFrame(CURVE), f'{OUT}/data_curve.csv')
    print('\n===== 数据量曲线 =====')
    print(pd.DataFrame(CURVE).to_string(index=False), flush=True)

# ==================== 7b. 分组方式对照实验 ====================
# 长度分桶是个「工程提速」手段，必须验证它没有偷走精度。
# 同样的数据、同样的超参，只改组 batch 的方式，跑两遍比结果。
# 结论写进报告第④节：提速换来的时间，精度代价是多少。
print('\n=== [5b/8] 分组方式对照（长度分桶 vs 随机组 batch）===')
CTRL = []
_ctrl_csv = f'{OUT}/batching_control.csv'
_CTRL_N = int(os.environ.get('BERT_CTRL_N', '1000'))
if RESUME and os.path.exists(_ctrl_csv):
    try:
        import pandas as _pd
        _oc = _pd.read_csv(_ctrl_csv)
        CTRL = _oc.to_dict('records')
        print(f'  ✓ 断点续跑：对照组已有 {len(CTRL)} 行', flush=True)
    except Exception as e:
        print(f'  ⚠ 读历史对照失败: {e}', flush=True)

if not CTRL and not SMOKE and RUN_CURVE:
    main_cfg = dict(CONFIGS[0]); main_cfg['epochs'] = 3
    for _tag, _nogroup in [('len_group', False), ('random', True)]:
        _rem = list(zip(tr_texts, tr_labels))
        random.Random(SEED).shuffle(_rem)
        _n0 = int(round(_CTRL_N * cnt.get(0, 1) / N)); _n1 = _CTRL_N - _n0
        _c0 = [t for t, l in _rem if l == 0][:_n0]
        _c1 = [t for t, l in _rem if l == 1][:_n1]
        _sub = _c0 + _c1
        _sub_lab = [0] * len(_c0) + [1] * len(_c1)
        # ★ 名字必须和主配置区分开：run_config 会无条件写 predictions_<name>.csv，
        #   如果沿用 'main'，这个对照的预测会覆盖掉主配置的逐条预测文件。
        _cfg = dict(main_cfg); _cfg['no_len_group'] = _nogroup
        _cfg['name'] = f'ctrl_{_tag}'
        print(f'\n  --- 对照: n={len(_sub)}  分组方式={_tag} ---', flush=True)
        _r, _, _ = run_config(_cfg, tr_texts_=_sub, tr_labels_=_sub_lab)
        CTRL.append(dict(n_train=len(_sub), batching=_tag,
                         test_acc=_r['test_acc'], macro_f1=_r['macro_f1'],
                         weighted_f1=_r['weighted_f1'], seconds=_r['seconds']))
        import pandas as _pd
        _atomic_csv(pd.DataFrame(CTRL), _ctrl_csv)
        print(f"  ✓ {_tag}: acc={_r['test_acc']}  macroF1={_r['macro_f1']}  "
              f"用时={_r['seconds']}s", flush=True)
    if len(CTRL) == 2:
        _d = (float(CTRL[0]['test_acc']) - float(CTRL[1]['test_acc'])) * 100
        _dt = (float(CTRL[0]['seconds']) - float(CTRL[1]['seconds'])) / 60
        print(f'\n  ★ 长度分桶 vs 随机组 batch: 精度差 {_d:+.2f} 个百分点，'
              f'省时 {_dt:.1f} 分钟', flush=True)

# ==================== 8. 汇总 + 画图 + 存档 ====================
print('\n=== [6/8] 汇总 ===', flush=True)
if RESULTS:
    ab = pd.DataFrame(RESULTS)
    _atomic_csv(ab, f'{OUT}/ablation_results.csv')
    print(ab[['config','lr','epochs','freeze_lower','class_weight','maxlen',
              'trainable_params','trainable_pct','test_acc','macro_f1',
              'weighted_f1','seconds']].to_string(index=False), flush=True)

print('\n=== [7/8] 画图 ===', flush=True)
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # ★ 画图段要自足：ab 在上面的汇总步骤里定义，但那一段包在 if RESULTS 里，
    #   这里重新建一份，避免"画图"依赖"汇总"的执行顺序。
    ab = pd.DataFrame(RESULTS)
    if RESULTS and MAIN:
        res, LOSS, VAL = MAIN
        df = pd.DataFrame(LOSS); vd = pd.DataFrame(VAL)
        # ★ step 是「每个 epoch 内」的步号，三个 epoch 都从 0 开始。
        #   直接拿来当 x 轴会叠成三段锯齿，这里换算成全局步号再看趋势。
        #   注意：偏移量要按 epoch 聚合后再 cumsum，不能对逐行的 span 直接 cumsum
        #   （那样会累加成 2331×777≈1.8e6，x 轴直接变成科学计数法）。
        if len(df):
            _spans = df.groupby('epoch')['step'].max() + 1        # epoch -> 777
            _offset = (_spans.cumsum() - _spans)                  # epoch -> 0/777/1554
            df = df.copy()
            df['gstep'] = df['step'] + df['epoch'].map(_offset)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(df['gstep'], df['loss'], alpha=.5, lw=.7, label='batch loss')
        ax[0].plot(df['gstep'], df['loss'].rolling(100, min_periods=1).mean(),
                   lw=2.2, color='crimson', label='smoothed(100)')
        ax[0].set_xlabel('Global Step'); ax[0].set_ylabel('Training Loss')
        ax[0].set_title('MacBERT Full Fine-tuning Loss'); ax[0].legend(); ax[0].grid(alpha=.3)
        if len(vd):
            ax[1].plot(vd['epoch'], vd['val_acc'], 'o-', lw=2.2, color='crimson')
            # 三个 epoch 的 val_acc 只差零点几个百分点，压在 0~1 全量程里看不出差别，
            # 这里自适应 y 轴并标出数值；真实数值同时写在图上，不靠视觉放大误导。
            _lo = float(vd['val_acc'].min()); _hi = float(vd['val_acc'].max())
            _pad = max((_hi - _lo) * 0.35, 0.004)
            ax[1].set_ylim(max(0, _lo - _pad), min(1.0, _hi + _pad))
            for _e, _v in zip(vd['epoch'], vd['val_acc']):
                ax[1].annotate(f'{float(_v):.4f}', (_e, float(_v)),
                               textcoords='offset points', xytext=(0, 9),
                               ha='center', fontsize=9)
        ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Validation Accuracy')
        ax[1].set_title('Validation Accuracy'); ax[1].grid(alpha=.3)
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
""")
```

---

## 📍 数据放哪

| 环境 | 做法 |
|---|---|
| **Kaggle** | `Add Input` → `Upload` → 选 `chunksenti_v1-20261023.zip`（**session 私有，不要建公开 Dataset**——原仓库 `redistribution_verified: false`，评论正文再分发权没核实）→ 脚本自动在 `/kaggle/input` 下找 |
| **本机** | 已经有了：`资料\data\processed\v1-20261023\`（脚本会自动找）|

---

## 操作

### 第一轮 · 冒烟（3 分钟）
1. `Restart & Clear Output`
2. 复制粘贴运行
3. 看到 `★ 结果包: ...` 就是通了

### 第二轮 · 正式（~45 分钟）
1. `SMOKE = False`
2. `Restart & Clear Output`
3. 运行
4. 跑完 → 右侧边栏文件浏览器下载 `results_BERT_FULL.zip`

---

## 产出对照（报告哪里用）

| 文件 | 报告哪里 |
|---|---|
| `ablation_results.csv` / `ablation_summary.txt` | **⑤评估 - 参数对比表（8 组）** |
| `ablation_compare.png` | ⑤评估 - 三联柱状图 |
| `data_curve.csv` / `data_curve.png` | **⑤评估 - 数据量影响曲线** |
| `predictions_<config>.csv` | ⑥**每一组配置各一份**（8 份）。带 `sample_id` 主键 + `p_negative/p_positive` 完整概率，是跟 LoRA 臂做配对检验（McNemar / paired bootstrap）的唯一凭据，事后补不出来 |
| `predictions.csv` | ⑥main 组的逐条预测（= `predictions_main.csv` 的副本，兼容旧流程）|
| `acc_by_length_<config>.csv` | ⑥长度偏置（负面评论更长 → 短文本更容易错），每组一份 |
| `errors_by_class.csv` | ⑥**≥3 类错误归因**（不含评论正文，靠 `sample_id` 回查 test.jsonl）|
| `error_samples.csv` | ⑥报告 §6.1 直接引用的 3 条「最确信却错」样例（**全项目只有这里 3 条正文会进提交包**）|
| `report_<config>.txt` | 每组配置的纯文本小结（超参 + 成绩 + 分类别 F1）|
| `loss_curve.png` / `confusion_matrix.png` | ⑤图表 |
| `bert_FULL.pt` | 复现用（不进交付 zip）|

---

## ⏱ 时间紧的话，删掉这几组

```python
CONFIGS = [
    dict(name='main',     ...),   # 必留
    dict(name='lr_5e5',  ...),   # 可删
    dict(name='lr_1e5',  ...),   # 可删
    dict(name='ep5',     ...),   # 可删
    dict(name='partial', ...),   # 必留（全参 vs 部分冻结）
    dict(name='weighted',...),   # 必留（类别不平衡）
    dict(name='len512',  ...),   # 必留（长度对比）
    dict(name='len128',  ...),   # 可删
]
```

**最少 4 组 ≈ 22 分钟**，且 4 组都能讲出道理。

---

## 🧠 这 8 组实验分别支撑哪个「课程原理点」

| 配置 | 支撑的原理点 | 报告位置 |
|---|---|---|
| `main` vs `partial` | 预训练目标（MLM）对下游任务的影响；全参 vs 冻结 | 第⑤节 |
| `main` vs `weighted` | 评测指标与协议选择（accuracy 掩盖小类）| 第⑤⑥节 |
| `main` vs `len128`/`len512` | 分词/子词与输入截断对效果的影响 | 第⑤节 |
| `data_curve` | 预训练模型在下游任务上的样本效率 | 第⑤节 |

---

去跑冒烟，3 分钟。跑完把输出贴给我，我给你写报告的第⑤节表格。
