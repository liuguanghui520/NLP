# Qwen3-1.7B 零样本 / 少样本提示（新数据集 ChnSentiCorp · 二分类）

> 方向 C 的另一半：**提示 vs 微调**。和 LoRA 用**同一个底座 Qwen3-1.7B**，只换「怎么问」。
> 用组里冻结的 `fewshot_ids.json`（4 条，负/正/负/正）和 `latency_ids.json`（测试集分层 200 条），全组口径一致。

---

## ⚠️ 每次重跑前

**`Session` → `Restart & Clear Output`**

---

## 两轮跑法

```python
SMOKE = True     # 第一次 True（6分钟，验代码）；改成 False 跑正式
```

| | 耗时 | 说明 |
|---|---|---|
| `SMOKE=True` | 6 分钟 | 40 条测试样本 / 3 种提示 |
| `SMOKE=False` | **~25 分钟** | 全测试集 777 条 × 3 种提示 + 延迟计时 200×3 |

> **本机跑不了**：1.7B 模型在 CPU 上单条要几十秒，777×3 条要一整天。**必须在 Kaggle 跑。**

---

## 三种提示变体（这就是「提示 vs 微调」里的提示侧参数对比）

| 变体 | 怎么问 | 考什么 |
|---|---|---|
| `zero` | 只给任务说明 + 评论 | 零样本基线 |
| `zero_fmt` | 任务说明 + **「只输出一个字符 0 或 1，不要任何解释」** | 指令格式对指令遵循率的影响 |
| `fewshot4` | 4 条固定示例（负/正/负/正）+ 评论 | 少样本能否补上零样本缺的知识 |

---

## 完整代码

```python
exec(r"""
import os, gc, subprocess, json, math, time, re, random
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ★★★ 开关 ★★★
SMOKE = True
MODEL_ID = 'Qwen/Qwen3-1.7B'
SEED = 42
NUM_LABELS = 2
CLS = ['negative', 'positive']
INSTR = '判断评论整体情感，仅输出0表示负面、1表示正面。'
INSTR_STRICT = ('判断评论整体情感，仅输出0表示负面、1表示正面。'
                 '请只输出一个字符（0 或 1），不要输出任何解释、标点或其他文字。')
MAX_NEW = 8
GEN_BATCH = 8

# ==================== 0. 清理 ====================
print('=== [0/7] 清理显存 ===', flush=True)
try:
    import torch
    _keep = {'os','gc','subprocess','json','math','time','re','random','torch','__builtins__','exec'}
    for _k in list(globals().keys()):
        if _k in _keep or _k.startswith('_'): continue
        try:
            if torch.is_tensor(globals()[_k]) or 'Model' in type(globals()[_k]).__name__:
                globals().pop(_k, None)
        except Exception: pass
    gc.collect(); torch.cuda.empty_cache()
    print(f'  显存 {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)
except Exception as e:
    print(f'  {e}', flush=True)

# ==================== 1. 数据 ====================
print('=== [1/7] 找数据集 ===', flush=True)
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
    raise SystemExit('!!! 找不到数据集，请 Add Input → Upload 上传 chunksenti_v1-20261023.zip')
print('  ✓', DATA, flush=True)

def load_jsonl(p):
    out = []
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

tr_rows = load_jsonl(os.path.join(DATA,'train.jsonl'))
te_rows = load_jsonl(os.path.join(DATA,'test.jsonl'))
by_id = {r['sample_id']: r for r in tr_rows}
te_texts = [r['text'] for r in te_rows]
te_labels = [int(r['label']) for r in te_rows]
te_ids    = [r['sample_id'] for r in te_rows]

# ---- 少样本示例：组里冻结的 4 条（负/正/负/正），不要自己重新抽 ----
FS_PATH = os.path.join(DATA, 'fewshot_ids.json')
if os.path.exists(FS_PATH):
    FS_IDS = json.load(open(FS_PATH, encoding='utf-8'))
    print(f'  ✓ fewshot_ids: {len(FS_IDS)} 条', flush=True)
else:
    FS_IDS = ['1d414b9c65caf3a836abdb4efded621968502ad3338e1a9d885ae131c86083f0:5542',
              '1d414b9c65caf3a836abdb4efded621968502ad3338e1a9d885ae131c86083f0:3812',
              '1d414b9c65caf3a836abdb4efded621968502ad3338e1a9d885ae131c86083f0:5693',
              '1d414b9c65caf3a836abdb4efded621968502ad3338e1a9d885ae131c86083f0:4407']
    print('  ⚠ 找不到 fewshot_ids.json，用内置兜底 ID', flush=True)
FS = [by_id[i] for i in FS_IDS if i in by_id]
print('  少样本示例标签顺序:', [int(r['label']) for r in FS], flush=True)

# ---- 延迟测试集：组里冻结的 200 条 ----
LAT_PATH = os.path.join(DATA, 'latency_ids.json')
if os.path.exists(LAT_PATH):
    LAT_IDS = json.load(open(LAT_PATH, encoding='utf-8'))
else:
    LAT_IDS = []
LAT_IDS = [i for i in LAT_IDS if i in set(te_ids)]
print(f'  ✓ latency_ids: {len(LAT_IDS)} 条', flush=True)

if SMOKE:
    idx = list(range(0, len(te_rows), 10))[:40]
    te_texts = [te_texts[i] for i in idx]; te_labels = [te_labels[i] for i in idx]
    te_ids    = [te_ids[i] for i in idx]
    LAT_IDS = LAT_IDS[:20]

# ==================== 2. 模型 ====================
print('=== [2/7] 准备模型 ===', flush=True)
model_dir = '/kaggle/working/qwen3_gen_local'
os.makedirs(model_dir, exist_ok=True)
need = ['config.json','tokenizer.json','tokenizer_config.json',
        'vocab.json','merges.txt','model.safetensors']
bases = [f'https://hf-mirror.com/{MODEL_ID}/resolve/main',
         f'https://huggingface.co/{MODEL_ID}/resolve/main',
         f'https://www.modelscope.cn/models/{MODEL_ID}/resolve/master']
for f in need:
    out = f'{model_dir}/{f}'
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print(f'  ✓ 已有 {f}', flush=True); continue
    for b in bases:
        subprocess.run(['curl','-L','-s','--max-time','1800','-o',out, f'{b}/{f}'])
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f'  ✓ 下载 {f} ({os.path.getsize(out)/1e6:.1f} MB)', flush=True); break
    else:
        if f == 'model.safetensors': raise SystemExit('!!! 权重下载失败 !!!')
        print(f'  ✗ {f} 失败(可忽略)', flush=True)

import torch, numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

print('=== [3/7] 加载模型 ===', flush=True)
tok = AutoTokenizer.from_pretrained(model_dir)
if tok.pad_token is None: tok.pad_token = '<|endoftext|>'
tok.padding_side = 'left'          # ★ 生成必须左 padding
print(f'  pad={tok.pad_token}({tok.pad_token_id}) eos={tok.eos_token}({tok.eos_token_id})', flush=True)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16, device_map={'':0})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map={'':0})
model.eval()
print(f'  ✓ 模型就绪  显存 {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

# ---- chat template ----
def build_prompt(text, variant):
    if variant == 'zero':
        msgs = [{'role':'user','content': f'{INSTR}\n评论：{text}'}]
    elif variant == 'zero_fmt':
        msgs = [{'role':'user','content': f'{INSTR_STRICT}\n评论：{text}'}]
    elif variant == 'fewshot4':
        msgs = [{'role':'user','content': f'{INSTR}\n评论：{FS[0]["text"]}'}]
        msgs.append({'role':'assistant','content': str(int(FS[0]['label']))})
        for ex in FS[1:]:
            msgs.append({'role':'user','content': f'评论：{ex["text"]}'})
            msgs.append({'role':'assistant','content': str(int(ex['label']))})
        msgs.append({'role':'user','content': f'评论：{text}'})
    else:
        raise ValueError(variant)
    # ★ Qwen3 默认开 thinking 模式，必须关掉，否则先吐一堆思考过程
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except Exception as e:
        print(f'  ⚠ enable_thinking 不被支持({e})，退回默认模板', flush=True)
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# 验证一下模板里没有 thinking 残留
_p = build_prompt('测试评论', 'zero')
print('  --- 提示模板示例 ---')
for _l in _p.split(chr(10))[:8]: print('   |', _l[:110])
if '<think>' in _p: print('  ⚠⚠ 模板里出现 <think>，输出解析会受影响', flush=True)

# ==================== 4. 生成 + 解析 ====================
def parse_label(s):
    s = (s or '').strip()
    if not s: return -1
    m = re.search(r'[01]', s)
    if m: return int(m.group(0))
    if '负面' in s: return 0
    if '正面' in s: return 1
    return -1

def generate_all(texts, variant, batch=GEN_BATCH):
    outs, ptoks = [], []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        prompts = [build_prompt(t, variant) for t in chunk]
        enc = tok(prompts, return_tensors='pt', padding=True, truncation=True,
                  max_length=2048, add_special_tokens=False).to('cuda')
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                               temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.pad_token_id)
        new = g[:, enc['input_ids'].shape[1]:]
        dec = tok.batch_decode(new, skip_special_tokens=True)
        outs += [d.strip() for d in dec]
        ptoks += [int(x) for x in enc['attention_mask'].sum(-1).tolist()]
        if (i // batch) % 20 == 0:
            print(f'    [{variant}] {i+len(chunk)}/{len(texts)}  '
                  f'{time.time()-t0:.0f}s', flush=True)
    return outs, ptoks, time.time() - t0

# ==================== 5. 跑三种变体 ====================
TAG = 'SMOKE' if SMOKE else 'FULL'
OUT = f'/kaggle/working/results_PROMPT_{TAG}'
os.makedirs(OUT, exist_ok=True)
VARIANTS = ['zero', 'zero_fmt', 'fewshot4']
ALL = {}

for v in VARIANTS:
    print(f'\n=== [4/7] 变体 {v} ===', flush=True)
    outs, ptoks, el = generate_all(te_texts, v)
    preds = [parse_label(o) for o in outs]
    unparsed = sum(1 for p in preds if p == -1)
    # 解析失败的按「多数类」兜底，同时单独统计
    maj = 1 if sum(te_labels) > len(te_labels)/2 else 0
    final = [maj if p == -1 else p for p in preds]
    acc = sum(1 for a,b in zip(te_labels, final) if a==b)/len(te_labels)
    tcnt = {}
    for a in te_labels: tcnt[a]=tcnt.get(a,0)+1
    f1s=[]
    for c_ in range(NUM_LABELS):
        tp=sum(1 for a,b in zip(te_labels,final) if a==c_ and b==c_)
        fp=sum(1 for a,b in zip(te_labels,final) if a!=c_ and b==c_)
        fn=sum(1 for a,b in zip(te_labels,final) if a==c_ and b!=c_)
        p_=tp/(tp+fp) if tp+fp else 0.0; r_=tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p_*r_/(p_+r_) if p_+r_ else 0.0)
    mf1=sum(f1s)/NUM_LABELS
    wf1=sum(f1s[i]*tcnt.get(i,0) for i in range(NUM_LABELS))/max(len(te_labels),1)
    follow = 1 - unparsed/len(outs)

    print(f'\n  ===== {v} =====')
    print(f'  acc={acc:.4f}  macroF1={mf1:.4f}  weightedF1={wf1:.4f}')
    print(f'  指令遵循率={follow*100:.2f}%  (解析失败 {unparsed}/{len(outs)})')
    print(f'  平均提示 token={sum(ptoks)/len(ptoks):.1f}  总耗时={el:.1f}s  '
          f'均耗时={el/len(texts)*1000:.0f}ms/条', flush=True)

    pd.DataFrame({'sample_id':te_ids,'text':te_texts,
                  'true_label':[CLS[i] for i in te_labels],
                  'pred_label':[CLS[i] for i in final],
                  'raw_output':outs,'parsed':preds,
                  'prompt_tokens':ptoks,
                  'correct':[a==b for a,b in zip(te_labels,final)]}).to_csv(
        f'{OUT}/predictions_{v}.csv', index=False, encoding='utf-8-sig')
    errs=[{'true':CLS[a],'pred':CLS[b],'raw_output':o,'prompt_tokens':p,'text':t}
          for a,b,o,p,t in zip(te_labels,final,outs,ptoks,te_texts) if a!=b]
    pd.DataFrame(errs, columns=['true','pred','raw_output','prompt_tokens','text']).to_csv(
        f'{OUT}/errors_{v}.csv', index=False, encoding='utf-8-sig')

    ALL[v] = dict(variant=v, test_acc=round(acc,4), macro_f1=round(mf1,4),
                  weighted_f1=round(wf1,4),
                  instruction_follow_rate=round(follow,4),
                  unparsed=unparsed, n=len(te_labels),
                  avg_prompt_tokens=round(sum(ptoks)/len(ptoks),1),
                  total_seconds=round(el,1),
                  ms_per_sample=round(el/len(texts)*1000,1))
    del outs, ptoks
    gc.collect(); torch.cuda.empty_cache()

# ==================== 6. 延迟计时（batch=1，预热20条） ====================
print('\n=== [5/7] 延迟计时 ===', flush=True)
LAT = []
if LAT_IDS:
    lat_idx = {sid:i for i,sid in enumerate(te_ids)}
    lat_texts = [te_texts[lat_idx[sid]] for sid in LAT_IDS if sid in lat_idx]
    for v in VARIANTS:
        for t in lat_texts[:20]:               # 预热
            p = build_prompt(t, v)
            e = tok(p, return_tensors='pt', add_special_tokens=False).to('cuda')
            with torch.no_grad():
                model.generate(**e, max_new_tokens=MAX_NEW, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        lats, pt = [], []
        for t in lat_texts:
            p = build_prompt(t, v)
            e = tok(p, return_tensors='pt', add_special_tokens=False).to('cuda')
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                model.generate(**e, max_new_tokens=MAX_NEW, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            lats.append((time.time()-t0)*1000)
            pt.append(int(e['attention_mask'].sum().item()))
        lats.sort()
        row = dict(variant=v, n=len(lats),
                   mean_ms=round(sum(lats)/len(lats),1),
                   p50_ms=round(lats[len(lats)//2],1),
                   p95_ms=round(lats[int(len(lats)*0.95)],1),
                   max_ms=round(lats[-1],1),
                   avg_prompt_tokens=round(sum(pt)/len(pt),1),
                   # 本地 T4 免费；换成 API 计费时这个数直接乘单价
                   est_cost_per_1k_usd=round(sum(pt)/len(pt)/1000*0.00015, 6))
        LAT.append(row)
        print(f'  [{v}] mean={row["mean_ms"]}ms p50={row["p50_ms"]}ms '
              f'p95={row["p95_ms"]}ms  prompt≈{row["avg_prompt_tokens"]}tok', flush=True)
    pd.DataFrame(LAT).to_csv(f'{OUT}/latency.csv', index=False, encoding='utf-8-sig')

# ==================== 7. 汇总 + 存档 ====================
print('\n=== [6/7] 汇总 ===', flush=True)
ab = pd.DataFrame(ALL.values())
ab.to_csv(f'{OUT}/prompt_variants.csv', index=False, encoding='utf-8-sig')
print(ab.to_string(index=False), flush=True)
with open(f'{OUT}/prompt_report.txt','w',encoding='utf-8') as fh:
    fh.write(f'{MODEL_ID} 提示方案对比  [{TAG}]\n')
    fh.write(f'底座: {MODEL_ID}（与 LoRA 臂同一个模型）\n')
    fh.write(f'生成: greedy(do_sample=False) max_new_tokens={MAX_NEW}\n')
    fh.write(f'指令: {INSTR}\n')
    fh.write(f'少样本示例: {len(FS)} 条, 标签顺序 {[int(r["label"]) for r in FS]}\n')
    fh.write(f'测试集 n={len(te_labels)}\n\n')
    fh.write(ab.to_string(index=False)); fh.write('\n\n')
    if LAT:
        fh.write('延迟 (batch=1, 预热20条):\n')
        fh.write(pd.DataFrame(LAT).to_string(index=False)); fh.write('\n')

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(ab))
    ax[0].bar(x, ab['test_acc'], color='crimson', alpha=.85)
    ax[0].set_xticks(x); ax[0].set_xticklabels(ab['variant'])
    ax[0].set_ylabel('Test Accuracy'); ax[0].set_ylim(0,1)
    ax[0].set_title('Prompting vs Accuracy'); ax[0].grid(alpha=.3, axis='y')
    for i,v in enumerate(ab['test_acc']): ax[0].text(i, v+.02, f'{v:.3f}', ha='center', fontsize=9)
    ax[1].bar(x, ab['macro_f1'], color='steelblue', alpha=.85)
    ax[1].set_xticks(x); ax[1].set_xticklabels(ab['variant'])
    ax[1].set_ylabel('Macro F1'); ax[1].set_ylim(0,1)
    ax[1].set_title('Prompting vs Macro F1'); ax[1].grid(alpha=.3, axis='y')
    for i,v in enumerate(ab['macro_f1']): ax[1].text(i, v+.02, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{OUT}/prompt_compare.png', dpi=150); plt.close()
    print('  ✓ prompt_compare.png', flush=True)
except Exception as e:
    print(f'  画图跳过: {e}', flush=True)

print('\n=== [7/7] 存档 ===', flush=True)
zp = f'/kaggle/working/results_PROMPT_{TAG}.zip'
subprocess.run(['zip','-r','-q',zp, f'results_PROMPT_{TAG}'], cwd='/kaggle/working')
print('\n' + '='*66)
for v, r in ALL.items():
    print(f"★ {v:>10}: acc={r['test_acc']}  macroF1={r['macro_f1']}  "
          f"遵循率={r['instruction_follow_rate']}  {r['ms_per_sample']}ms/条")
print(f"★ 结果包 → {zp}  ({os.path.getsize(zp)/1e6:.1f} MB)")
print('='*66)
print('\n📁 右侧文件浏览器 → working → 下载 results_PROMPT_' + TAG + '.zip', flush=True)
print('\n✅ 冒烟通过！把 SMOKE 改成 False 再跑正式' if SMOKE else 'DONE', flush=True)
""")
```

---

## 📍 数据放哪

`Add Input` → `Upload` → `chunksenti_v1-20261023.zip`。**这个 zip 里已经含 `fewshot_ids.json` / `latency_ids.json`**，脚本直接读，不用自己抽。

---

## 操作

1. `Restart & Clear Output`
2. `SMOKE=True` 跑一次（6 分钟）→ 确认最后打印出三个变体的 acc
3. `SMOKE=False` → `Restart` → 跑 25 分钟
4. 右侧文件浏览器下载 `results_PROMPT_FULL.zip`

---

## 产出对照

| 文件 | 报告哪里 |
|---|---|
| `prompt_variants.csv` | **⑤评估 - 提示 vs 微调对比表**（和 BERT/LoRA 并排）|
| `latency.csv` | **⑦工程与伦理 - 成本与延迟**（任务书点名的加分项）|
| `prompt_report.txt` | ⑤评估 |
| `predictions_zero.csv` / `_zero_fmt.csv` / `_fewshot4.csv` | ⑥错误分析（**列结构与 BERT/LoRA 对齐**）|
| `errors_*.csv` | ⑥≥3 类错误归因 |
| `prompt_compare.png` | ⑤图表 |

---

## 🔑 三个坑

1. **Qwen3 默认开 thinking 模式** → 必须 `enable_thinking=False`，否则模型先吐一长串思考，解析全废。脚本里已经带 try/except 兜底。
2. **生成必须左 padding** → `tok.padding_side = 'left'`。用右边 padding 会让生成从 pad 后面开始，输出乱码。
3. **解析要容错** → 模型可能回"0"、也可能回"负面"、也可能回一段解释。脚本先找第一个 0/1，再找"负面/正面"，都不中记为解析失败并统计**指令遵循率**（这本身就是个好指标）。

---

冒烟过了告诉我。三条臂（BERT / LoRA / 提示）的脚本我都备齐了，接下来帮你写报告。
