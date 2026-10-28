
import os, gc, subprocess, json, math, types, time

# ★★★ 开关 ★★★
SMOKE = True
RUN_RANK_SWEEP = True
MODEL_ID = 'Qwen/Qwen3-1.7B'
SEED = 42
NUM_LABELS = 2
CLS = ['negative', 'positive']
EPOCHS, LR, BATCH, ACCUM, MAXLEN = 3, 1e-4, 4, 4, 512
INSTRUCTION = '判断评论整体情感，仅输出0表示负面、1表示正面。'
USE_INSTRUCTION = True     # 与提示臂公平对比：LoRA 也要看到任务说明
BASE_R = 8

# ==================== 0. 清理显存 ====================
print('=== [0/8] 清理显存 ===', flush=True)
try:
    import torch
    _keep = {'os','gc','subprocess','json','math','types','torch','__builtins__','exec'}
    for _k in list(globals().keys()):
        if _k in _keep or _k.startswith('_'): continue
        try:
            if torch.is_tensor(globals()[_k]) or 'Model' in type(globals()[_k]).__name__:
                globals().pop(_k, None); print(f'  释放: {_k}', flush=True)
        except Exception: pass
    gc.collect(); torch.cuda.empty_cache()
    print(f'  当前显存 {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)
except Exception as e:
    print(f'  {e}', flush=True)

# ==================== 1. torchao 绕过 ====================
print('=== [1/8] 处理 torchao ===', flush=True)
try:
    r = subprocess.run(['pip','install','-q','-U','torchao','--index-url','https://pypi.org/simple/'],
                       capture_output=True, text=True, timeout=300)
    print(f'  pip rc={r.returncode}', flush=True)
except Exception as e:
    print(f'  pip 失败(可忽略): {e}', flush=True)
try:
    import peft.import_utils as P
    P.is_torchao_available = lambda: False
    P.TORCHAO_MINIMUM_VERSION = '0.0.0'
    print('  peft 版本检查已绕过', flush=True)
except Exception as e:
    print(f'  {e}', flush=True)

# ==================== 2. transformers 版本（需要 Qwen3） ====================
print('=== [2/8] 检查 transformers ===', flush=True)
try:
    from transformers import Qwen3ForSequenceClassification
    print('  ✓ Qwen3ForSequenceClassification 可用', flush=True)
except ImportError:
    print('  升级 transformers ...', flush=True)
    subprocess.run(['pip','install','-q','-U','transformers>=4.51'], timeout=600)
    from transformers import Qwen3ForSequenceClassification
    print('  ✓ 升级完成', flush=True)

# ==================== 3. 数据集 ====================
print('=== [3/8] 找数据集 ===', flush=True)
def find_data_dir():
    here = os.getcwd()
    fixed = [os.path.join(here,'data','processed','v1-20261023'),
             '/kaggle/input/chunksenti-v1-20261023',
             '/kaggle/input/chunksenti_v1_20261023']
    for c in fixed:
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
                    except Exception as e:
                        print(f'  zip 失败 {f}: {e}', flush=True)
    return None

DATA = find_data_dir()
if DATA is None:
    raise SystemExit('!!! 找不到数据集。Kaggle 请 Add Input → Upload 上传 '
                     'chunksenti_v1-20261023.zip')
print('  ✓ 数据目录:', DATA, flush=True)

def load_jsonl(p):
    out = []
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

tr_rows = load_jsonl(os.path.join(DATA,'train.jsonl'))
va_rows = load_jsonl(os.path.join(DATA,'validation.jsonl'))
te_rows = load_jsonl(os.path.join(DATA,'test.jsonl'))
def prep(rows):
    ts = [(INSTRUCTION + '\n' + r['text']) if USE_INSTRUCTION else r['text'] for r in rows]
    return ts, [int(r['label']) for r in rows]
tr_texts, tr_labels = prep(tr_rows)
va_texts, va_labels = prep(va_rows)
te_texts, te_labels = prep(te_rows)
print(f'  train {len(tr_texts)} val {len(va_texts)} test {len(te_texts)}', flush=True)

# ==================== 4. 模型文件 ====================
print('=== [4/8] 准备模型文件 ===', flush=True)
model_dir = '/kaggle/working/qwen3_local'
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

cfg_path = f'{model_dir}/config.json'
with open(cfg_path) as fh: cfg = json.load(fh)
cfg['num_labels'] = NUM_LABELS
cfg['problem_type'] = 'single_label_classification'
with open(cfg_path,'w') as fh: json.dump(cfg, fh, indent=2)
print('  ✓ config 就绪', flush=True)

# ==================== 5. 加载 ====================
import torch, numpy as np, random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

print('=== [5/8] 准备 tokenizer ===', flush=True)
tok = AutoTokenizer.from_pretrained(model_dir)
if tok.pad_token is None:
    tok.pad_token = '<|endoftext|>'
tok.padding_side = 'right'
print(f'  tokenizer pad_token={tok.pad_token} id={tok.pad_token_id}', flush=True)

KW = dict(num_labels=NUM_LABELS, ignore_mismatched_sizes=True, low_cpu_mem_usage=True)

# ★ Qwen3 的 config.json 是「扁平」的，没有 text_config 嵌套，而且 pad_token_id 根本没设。
#   Qwen3ForSequenceClassification 取最后一个非 pad token 做池化时读的是
#   config.get_text_config().pad_token_id —— 不设的话会静默取错特征。
def set_pad(obj, pid):
    if hasattr(obj, 'pad_token_id') and obj.pad_token_id is None:
        obj.pad_token_id = pid
    for a in ['text_config', 'sub_config', 'decoder']:
        if hasattr(obj, a):
            set_pad(getattr(obj, a), pid)

n_total_base = 0   # 在 run_lora 里赋值

# ==================== 6. 训练函数 ====================
print('=== [6/8] 定义训练函数 ===', flush=True)

class DS(Dataset):
    def __init__(s, t, y): s.t=t; s.y=y
    def __len__(s): return len(s.t)
    def __getitem__(s,i): return s.t[i], s.y[i]

def make_col(maxlen):
    def col(b):
        ts = [x[0] for x in b]
        ys = torch.tensor([x[1] for x in b], dtype=torch.long)
        e = tok(ts, padding=True, truncation=True, max_length=maxlen, return_tensors='pt')
        return e.input_ids, e.attention_mask, ys
    return col

def run_lora(r, save_adapter=False):
    global n_total_base
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    t0 = time.time()
    col = make_col(MAXLEN)
    tr = DataLoader(DS(tr_texts,tr_labels), batch_size=BATCH, shuffle=True, collate_fn=col)
    va = DataLoader(DS(va_texts,va_labels), batch_size=8, collate_fn=col)
    te = DataLoader(DS(te_texts,te_labels), batch_size=8, collate_fn=col)

    try:
        base = AutoModelForSequenceClassification.from_pretrained(
            model_dir, dtype=torch.float16, device_map={'':0}, **KW)
    except TypeError:
        base = AutoModelForSequenceClassification.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map={'':0}, **KW)
    set_pad(base.config, tok.pad_token_id)
    base.config.pad_token_id = tok.pad_token_id
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    base.enable_input_require_grads()
    n_total_base = sum(p.numel() for p in base.parameters())
    print(f'  底座参数 {n_total_base:,}  config.pad_token_id={base.config.pad_token_id}  '
          f'显存 {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=2*r,
                             lora_dropout=0.05,
                             target_modules=['q_proj','k_proj','v_proj','o_proj'])
    model = get_peft_model(base, lora_config)

    # ★★ 核心修复：可训练参数 fp16 -> fp32（AdamW eps=1e-8 在 fp16 下溢 → NaN）
    n32 = 0
    for n_, p_ in model.named_parameters():
        if p_.requires_grad:
            p_.data = p_.data.float(); n32 += p_.numel()
    # 分类头输入也转 fp32
    try:
        sc = model.base_model.model.score
        tgt = sc.modules_to_save['default'] if hasattr(sc,'modules_to_save') else sc
        _orig = tgt.forward
        tgt.forward = types.MethodType(lambda s,x,*a,**kw: _orig(x.float(),*a,**kw), tgt)
    except Exception as e:
        print(f'  分类头 patch 跳过: {e}', flush=True)
    print(f'  ✓ r={r} 已转 fp32 参数 {n32:,} '
          f'({n32/n_total_base*100:.4f}%)  显存 {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    total = math.ceil(len(tr)/ACCUM)*EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(total*0.1), total)

    LOSS_LOG, VAL_LOG, nan_cnt, oom_cnt = [], [], 0, 0
    for ep in range(EPOCHS):
        model.train(); ls=[]; opt.zero_grad(set_to_none=True)
        for i,(ids,msk,y) in enumerate(tr):
            ids,msk,y = ids.to('cuda'), msk.to('cuda'), y.to('cuda')
            try:
                o = model(input_ids=ids, attention_mask=msk, labels=y)
                loss = o.loss.item()
                if math.isnan(loss) or math.isinf(loss):
                    nan_cnt += 1; opt.zero_grad(set_to_none=True); continue
                (o.loss/ACCUM).backward()
            except torch.cuda.OutOfMemoryError:
                # 显存不够就把这批丢掉，别让整轮训练崩掉
                oom_cnt += 1; opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f'  ⚠ OOM 跳过一批 (累计 {oom_cnt})，'
                      f'显存={torch.cuda.memory_allocated()/1e9:.1f}GB', flush=True)
                continue
            if (i+1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            ls.append(loss)
            LOSS_LOG.append({'epoch':ep+1,'step':i,'loss':loss})
            if i % 300 == 0:
                print(f'  [r={r}] Ep{ep+1} i={i}/{len(tr)} loss={round(loss,4)} '
                      f'显存={torch.cuda.memory_allocated()/1e9:.1f}GB', flush=True)
        model.eval(); c=t=0
        with torch.no_grad():
            for ids,msk,y in va:
                ids,msk = ids.to('cuda'), msk.to('cuda')
                o = model(input_ids=ids, attention_mask=msk)
                c += (o.logits.argmax(-1)==y).sum().item(); t += y.size(0)
        vacc = c/t
        VAL_LOG.append({'epoch':ep+1,'train_loss':sum(ls)/max(len(ls),1),'val_acc':vacc})
        print(f'  >>> [r={r}] Epoch {ep+1} loss={round(sum(ls)/max(len(ls),1),4)} '
              f'val_acc={round(vacc,4)} 跳过NaN={nan_cnt}', flush=True)

    # 测试
    model.eval(); preds=[]; confs=[]; ytrue=[]; ttext=[]; tlen=[]
    with torch.no_grad():
        for ids,msk,y in te:
            ids,msk = ids.to('cuda'), msk.to('cuda')
            o = model(input_ids=ids, attention_mask=msk)
            pr = torch.softmax(o.logits.float(),-1).cpu()
            preds.extend(pr.argmax(-1).tolist()); confs.extend(pr.max(-1).values.tolist())
            ytrue.extend(y.tolist())
            for j in range(ids.shape[0]):
                ttext.append(tok.decode(ids[j], skip_special_tokens=True))
                tlen.append(int(msk[j].sum().item()))

    acc = sum(1 for a,b in zip(ytrue,preds) if a==b)/len(ytrue)
    f1s=[]
    for c_ in range(NUM_LABELS):
        tp=sum(1 for a,b in zip(ytrue,preds) if a==c_ and b==c_)
        fp=sum(1 for a,b in zip(ytrue,preds) if a!=c_ and b==c_)
        fn=sum(1 for a,b in zip(ytrue,preds) if a==c_ and b!=c_)
        p_=tp/(tp+fp) if tp+fp else 0.0; r_=tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p_*r_/(p_+r_) if p_+r_ else 0.0)
    mf1 = sum(f1s)/NUM_LABELS
    # ★ weighted F1 按「测试集真实标签数」加权，和 sklearn average='weighted' 一致
    tcnt = {}
    for y_ in ytrue: tcnt[y_] = tcnt.get(y_,0)+1
    wf1 = sum(f1s[i]*tcnt.get(i,0) for i in range(NUM_LABELS))/max(len(ytrue),1)

    print(f'\n  ===== LoRA r={r} =====')
    print(f'  acc={acc:.4f} macroF1={mf1:.4f} weightedF1={wf1:.4f} '
          f'训练参数={n32:,} ({n32/n_total_base*100:.4f}%) 用时={time.time()-t0:.0f}s', flush=True)

    res = dict(r=r, alpha=2*r, lr=LR, epochs=EPOCHS, maxlen=MAXLEN,
               trainable_params=n32, total_params=n_total_base,
               trainable_pct=round(n32/n_total_base*100, 4),
               test_acc=round(acc,4), macro_f1=round(mf1,4), weighted_f1=round(wf1,4),
               # ★ 验证集准确率：代表秩按它来选（与 TF-IDF / MacBERT / 提示臂同一口径）
               #   取**最后一个 epoch** 的 val_acc，和 representative.py 的口径一致。
               #   注意 VAL_LOG 是全局累积的，所以只能取末尾那条，不能取 max。
               val_acc=round(float(VAL_LOG[-1]['val_acc']), 4) if VAL_LOG else None,
               nan_skipped=nan_cnt, oom_skipped=oom_cnt, seconds=round(time.time()-t0))
    for i,cname in enumerate(CLS): res[f'f1_{cname}']=round(f1s[i],4)

    if save_adapter:
        ck = f'/kaggle/working/qwen3_lora_{TAG}'
        model.save_pretrained(ck)
        _sz = sum(os.path.getsize(os.path.join(r_,f)) for r_,_,fs in os.walk(ck) for f in fs)
        print(f'  ✓ LoRA adapter 已存 {ck}  ({_sz/1e6:.1f} MB)', flush=True)

    del model, base, opt, tr, va, te
    gc.collect(); torch.cuda.empty_cache()
    return res, LOSS_LOG, VAL_LOG, preds, ytrue, confs, ttext, tlen, n32

# ==================== 7. 跑 ====================
TAG = 'SMOKE' if SMOKE else 'FULL'
OUT = f'/kaggle/working/results_QWEN_{TAG}'
os.makedirs(OUT, exist_ok=True)

if SMOKE:
    tr_texts = tr_texts[:400]; tr_labels = tr_labels[:400]
    va_texts = va_texts[:200]; va_labels = va_labels[:200]
    te_texts = te_texts[:200]; te_labels = te_labels[:200]
    EPOCHS = 1
    print('  ★ 冒烟：400 条 / 1 epoch', flush=True)

print('\n=== [7/8] 开始训练 ===', flush=True)
res, LOSS, VAL, preds, ytrue, confs, ttext, tlen, n32 = run_lora(BASE_R, save_adapter=True)

pd.DataFrame(LOSS).to_csv(f'{OUT}/train_log.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(VAL).to_csv(f'{OUT}/val_log.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'text':ttext,'true_label':[CLS[i] for i in ytrue],
              'pred_label':[CLS[i] for i in preds],'confidence':confs,
              'token_len':tlen,'correct':[a==b for a,b in zip(ytrue,preds)]}).to_csv(
    f'{OUT}/predictions.csv', index=False, encoding='utf-8-sig')
errs=[{'true':CLS[a],'pred':CLS[b],'confidence':round(c,4),'token_len':tl,'text':tt}
      for a,b,c,tl,tt in zip(ytrue,preds,confs,tlen,ttext) if a!=b]
pd.DataFrame(errs).sort_values('confidence', ascending=False).to_csv(
    f'{OUT}/errors_by_class.csv', index=False, encoding='utf-8-sig')
print(f'  ✓ predictions.csv ({len(preds)}) errors_by_class.csv ({len(errs)})', flush=True)

# ---- LoRA 秩对比 ----
RANK_RESULTS = [res]
if RUN_RANK_SWEEP and not SMOKE:
    for r_ in [4, 16, 32]:
        print(f'\n{"="*66}\n  LoRA 秩对比: r={r_}\n{"="*66}', flush=True)
        rr = run_lora(r_)[0]
        RANK_RESULTS.append(rr)
        pd.DataFrame(RANK_RESULTS).to_csv(f'{OUT}/lora_rank_sweep.csv',
                                          index=False, encoding='utf-8-sig')
    print('\n===== LoRA 秩对比 =====')
    print(pd.DataFrame(RANK_RESULTS).to_string(index=False), flush=True)

# ==================== 8. 报告 + 画图 + 存档 ====================
print('\n=== [8/8] 报告 / 画图 / 存档 ===', flush=True)
from sklearn.metrics import classification_report, confusion_matrix
rep = classification_report(ytrue, preds, target_names=CLS, digits=4)
cm  = confusion_matrix(ytrue, preds)
with open(f'{OUT}/lora_report.txt','w',encoding='utf-8') as fh:
    fh.write(f'{MODEL_ID} + LoRA  [{TAG}]\n')
    fh.write(f'LoRA: r={BASE_R} alpha={2*BASE_R} dropout=0.05 target=[q_proj,k_proj,v_proj,o_proj]\n')
    fh.write(f'超参: lr={LR} epochs={EPOCHS} batch={BATCH}x{ACCUM} maxlen={MAXLEN} seed={SEED}\n')
    fh.write(f'instruction 前置: {USE_INSTRUCTION}\n')
    fh.write(f'数据: train={len(tr_texts)} val={len(va_texts)} test={len(te_texts)}\n')
    fh.write(f'训练参数: {n32:,} / {n_total_base:,} ({n32/n_total_base*100:.4f}%)\n')
    fh.write(f'测试集 accuracy={res["test_acc"]} macroF1={res["macro_f1"]} '
             f'weightedF1={res["weighted_f1"]}\n')
    fh.write(f'跳过NaN步数={res["nan_skipped"]}\n\n')
    fh.write(rep)
    fh.write('\n混淆矩阵 (行=真实, 列=预测)\n')
    fh.write('        '+' '.join(f'{c:>9}' for c in CLS)+'\n')
    for i,c in enumerate(CLS):
        fh.write(f'{c:>8} '+' '.join(f'{v:>9}' for v in cm[i])+'\n')
    if len(RANK_RESULTS) > 1:
        fh.write('\nLoRA 秩对比:\n')
        fh.write(pd.DataFrame(RANK_RESULTS).to_string(index=False))

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = pd.DataFrame(LOSS); vd = pd.DataFrame(VAL)
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(df['step'], df['loss'], alpha=.55, lw=.7, label='batch loss')
    ax[0].plot(df['step'], df['loss'].rolling(50, min_periods=1).mean(),
               lw=2.2, color='crimson', label='smoothed(50)')
    ax[0].set_xlabel('Step'); ax[0].set_ylabel('Training Loss')
    ax[0].set_title('Qwen3-1.7B + LoRA  Training Loss'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(vd['epoch'], vd['val_acc'], 'o-', lw=2.2, color='crimson')
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Validation Accuracy')
    ax[1].set_ylim(0,1); ax[1].set_title('Validation Accuracy'); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/loss_curve.png', dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6.2,5.6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(NUM_LABELS)); ax.set_yticks(range(NUM_LABELS))
    ax.set_xticklabels(['Negative','Positive'], rotation=20, ha='right')
    ax.set_yticklabels(['Negative','Positive'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Qwen3-1.7B + LoRA Confusion Matrix (acc={res["test_acc"]:.4f})')
    for i in range(NUM_LABELS):
        for j in range(NUM_LABELS):
            ax.text(j,i,cm[i,j],ha='center',va='center',
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.colorbar(im); plt.tight_layout(); plt.savefig(f'{OUT}/confusion_matrix.png', dpi=150); plt.close()

    if len(RANK_RESULTS) > 1:
        rd = pd.DataFrame(RANK_RESULTS)
        fig, ax = plt.subplots(1,2, figsize=(11,4.2))
        ax[0].plot(rd['r'], rd['test_acc'], 'o-', lw=2, color='crimson')
        ax[0].set_xlabel('LoRA rank r'); ax[0].set_ylabel('Test Accuracy')
        ax[0].set_title('Accuracy vs LoRA Rank'); ax[0].grid(alpha=.3)
        ax[1].plot(rd['r'], rd['trainable_params']/1e6, 's-', lw=2, color='steelblue')
        ax[1].set_xlabel('LoRA rank r'); ax[1].set_ylabel('Trainable Params (M)')
        ax[1].set_title('Trainable Params vs LoRA Rank'); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f'{OUT}/lora_rank_sweep.png', dpi=150); plt.close()
    print('  ✓ 图已保存', flush=True)
except Exception as e:
    print(f'  画图跳过: {e}', flush=True)

# LoRA adapter 已在 run_lora(BASE_R, save_adapter=True) 里存到
# /kaggle/working/qwen3_lora_{TAG}/

zp = f'/kaggle/working/results_QWEN_{TAG}.zip'
subprocess.run(['zip','-r','-q',zp, f'results_QWEN_{TAG}'], cwd='/kaggle/working')

print('\n' + '='*66)
print(f'★ 测试集 accuracy = {res["test_acc"]}  macroF1 = {res["macro_f1"]}')
print(f'★ 训练参数 = {n32:,} ({res["trainable_pct"]}%)')
print(f'★ 跳过 NaN 步数 = {res["nan_skipped"]}')
print(f'★ 结果包 → {zp}  ({os.path.getsize(zp)/1e6:.1f} MB)')
print('='*66)
print('\n📁 右侧文件浏览器 → working → 下载 results_QWEN_' + TAG + '.zip', flush=True)
print('\n✅ 冒烟通过！把 SMOKE 改成 False 再跑正式' if SMOKE else 'DONE', flush=True)
