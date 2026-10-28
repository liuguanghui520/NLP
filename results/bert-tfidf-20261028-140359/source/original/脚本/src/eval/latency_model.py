# -*- coding: utf-8 -*-
"""
单个「分类模型」的单条推理延迟测量（LoRA 臂用；也可测任意 AutoModelForSequenceClassification）。

为什么要有这个脚本
------------------
- `latency_baseline.py` 只认死了 TF-IDF 和 MacBERT 两臂；
- 提示臂是**生成**任务（`model.generate`），延迟脚本已经在 `qwen_prompt.py` 里内置了；
- **LoRA 臂是分类任务**（`AutoModelForSequenceClassification` 的一次 forward），
  测法跟提示臂完全不同、跟 MacBERT 一样 —— 但没有现成脚本。

所以这个脚本按 MacBERT 的测法做，但把模型、adapter、序列长度全部参数化，
让 LoRA 臂能在**同一套协议**下测出可比的数字。

协议（与基线两臂完全一致）
--------------------------
    - 样本：data/processed/<版本>/latency_ids.json 里冻结的 200 条
    - batch = 1，逐条推理
    - 预热 20 次，不计入统计
    - 统计 mean / P50 / P95 / max

⚠️ **必须在机器空闲时测**。训练进程占满 CPU 时测出来的数字虚高一倍多，
   基线那边已经踩过这个坑（219.8 ms/条 vs 空闲时 103.5 ms/条）。

用法
----
    # LoRA 臂（Qwen3 + PEFT adapter）
    python src/eval/latency_model.py \
        --model-path /kaggle/working/qwen3_gen_local \
        --lora-path  /kaggle/working/lora_r8 \
        --variant-name "Qwen3 + LoRA" \
        --out src/results/results_QWEN_FULL/latency.csv

    # 不带 adapter 的基座分类模型
    python src/eval/latency_model.py --model-path <dir> --variant-name "Qwen3 基座"
"""
import argparse
import json
import os
import statistics
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

DATA_CANDIDATES = [
    'data/processed/v1-20261023',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                 'data', 'processed', 'v1-20261023'),
]
WARMUP = 20


def find_data_dir():
    for c in DATA_CANDIDATES:
        if os.path.isdir(c):
            return c
    return None


def load_jsonl(path):
    out = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_latency_texts(data_dir):
    lat_path = os.path.join(data_dir, 'latency_ids.json')
    te = load_jsonl(os.path.join(data_dir, 'test.jsonl'))
    by_id = {r['sample_id']: r['text'] for r in te}
    ids = json.load(open(lat_path, encoding='utf-8'))
    ids = [i for i in ids if i in by_id]
    return ids, [by_id[i] for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True, help='模型目录（含 config.json / tokenizer）')
    ap.add_argument('--lora-path', default=None, help='PEFT adapter 目录（可选）')
    ap.add_argument('--ckpt', default=None, help='额外的 .pt 权重（可选）')
    ap.add_argument('--variant-name', default='model', help='写进 CSV 的 variant 名')
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--num-labels', type=int, default=2)
    ap.add_argument('--out', default='src/results/results_QWEN_FULL/latency.csv')
    args = ap.parse_args()

    data_dir = find_data_dir()
    if not data_dir:
        sys.exit('找不到数据目录')
    ids, texts = get_latency_texts(data_dir)
    print(f'✓ 延迟子集：{len(ids)} 条（latency_ids.json 冻结）')
    print(f'  协议：batch=1，预热 {WARMUP} 次\n')

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        sys.exit(f'torch/transformers 不可用：{type(e).__name__}: {e}')

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=args.num_labels)

    if args.lora_path:
        try:
            from peft import PeftModel
        except Exception as e:
            sys.exit(f'需要 peft 才能加载 adapter：{type(e).__name__}: {e}')
        model = PeftModel.from_pretrained(model, args.lora_path)
        print(f'  ✓ 已加载 adapter：{args.lora_path}')

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu')
        missing = model.load_state_dict(sd, strict=False)
        print(f'  ✓ 已加载权重 {args.ckpt}（缺失 {len(missing.missing_keys)} 个键）')

    model.eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)

    def one(t):
        e = tok([t], padding=True, truncation=True,
                max_length=args.max_length, return_tensors='pt').to(dev)
        with torch.no_grad():
            return int(model(**e).logits.argmax(-1).item())

    for t in texts[:WARMUP]:
        one(t)

    lats = []
    for t in texts:
        if dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        one(t)
        if dev == 'cuda':
            torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)

    lats.sort()
    n = len(lats)
    row = dict(variant=args.variant_name, n=n,
               mean_ms=round(statistics.fmean(lats), 3),
               p50_ms=round(lats[n // 2], 3),
               p95_ms=round(lats[min(int(n * 0.95), n - 1)], 3),
               max_ms=round(lats[-1], 3),
               batch=1, warmup=WARMUP, device=dev,
               max_length=args.max_length)

    print(f'  mean={row["mean_ms"]}ms  p50={row["p50_ms"]}ms  '
          f'p95={row["p95_ms"]}ms  max={row["max_ms"]}ms  [{dev}]')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    import pandas as pd
    pd.DataFrame([row]).to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'\n✓ 已生成 {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
