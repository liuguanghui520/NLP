"""
基线两臂的延迟测量（TF-IDF + MacBERT 全参）。

为什么要有这个
--------------
报告的「表 1 · 四条路线主对比」里有一列「平均延迟」，
目前只有提示臂有真实数字，TF-IDF / MacBERT / LoRA 三行都写死成 `—`。
任务书把"说清成本与延迟"列为三个拉开分差的地方之一，所以这一列必须补齐。

**协议必须和提示臂完全一致**，否则四行数字不能并排：
    - 样本：data/processed/<版本>/latency_ids.json 里**冻结的 200 条**
    - batch = 1，逐条处理
    - 预热 20 次，不计入统计
    - 统计 mean / P50 / P95 / max

用法
----
    python src/eval/latency_baseline.py
    python src/eval/latency_baseline.py --skip-tfidf --skip-bert
"""
import argparse
import json
import os
import statistics
import sys
import time

DATA_CANDIDATES = [
    'data/processed/v1-20261023',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                 'data', 'processed', 'v1-20261023'),
]
MODEL_DIRS = ['macbert_base', os.path.join('src', 'models', 'macbert_base')]
CKPT_CANDIDATES = ['bert_FULL.pt',
                   os.path.join('src', 'results', 'bert_FULL.pt')]

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
    """取冻结的 200 条延迟子集（按 latency_ids.json 的顺序）。"""
    lat_path = os.path.join(data_dir, 'latency_ids.json')
    te = load_jsonl(os.path.join(data_dir, 'test.jsonl'))
    by_id = {r['sample_id']: r['text'] for r in te}
    ids = json.load(open(lat_path, encoding='utf-8'))
    ids = [i for i in ids if i in by_id]
    texts = [by_id[i] for i in ids]
    return ids, texts


def summarize(name, lats, extra=None):
    lats = sorted(lats)
    n = len(lats)
    row = dict(variant=name, n=n,
               mean_ms=round(statistics.fmean(lats), 3),
               p50_ms=round(lats[n // 2], 3),
               p95_ms=round(lats[min(int(n * 0.95), n - 1)], 3),
               max_ms=round(lats[-1], 3))
    if extra:
        row.update(extra)
    return row


def bench_tfidf(texts):
    """TF-IDF + LogisticRegression 单条推理延迟。"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except Exception as e:
        print(f'  ⚠ sklearn 不可用（{type(e).__name__}: {str(e)[:100]}），跳过 TF-IDF 臂')
        print('    → 本机被 Windows 应用控制策略拦了 sklearn.metrics 的 C 扩展；')
        print('    → 在正常机器或 WSL/Kaggle 上重跑本脚本即可')
        return None

    from importlib import import_module
    # 复用基线脚本里已经定好的参数（C=4.0, char 1-3 gram, min_df=2, sublinear_tf）
    train = load_jsonl(os.path.join(find_data_dir(), 'train.jsonl'))
    tr_texts = [r['text'] for r in train]
    tr_labels = [int(r['label']) for r in train]
    pipe = make_pipeline(
        TfidfVectorizer(analyzer='char', ngram_range=(1, 3), min_df=2,
                        sublinear_tf=True, max_features=200000),
        LogisticRegression(C=4.0, max_iter=2000))
    t0 = time.time()
    pipe.fit(tr_texts, tr_labels)
    print(f'  ✓ TF-IDF 拟合完成（{time.time()-t0:.1f}s）')

    for t in texts[:WARMUP]:
        pipe.predict([t])

    lats = []
    for t in texts:
        t0 = time.perf_counter()
        pipe.predict([t])
        lats.append((time.perf_counter() - t0) * 1000)
    return summarize('TF-IDF + LR', lats, extra={'batch': 1, 'warmup': WARMUP})


def bench_bert(texts):
    """MacBERT 全参单条推理延迟（需要 bert_FULL.pt）。"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        print(f'  ⚠ torch/transformers 不可用（{type(e).__name__}），跳过 MacBERT 臂')
        return None

    model_dir = next((d for d in MODEL_DIRS if os.path.isdir(d)), None)
    ckpt = next((c for c in CKPT_CANDIDATES if os.path.exists(c)), None)
    if not model_dir or not ckpt:
        print('  ⚠ 找不到 macbert_base/ 或 bert_FULL.pt，跳过 MacBERT 臂')
        print('    → bert_FULL.pt 在 main 组训练结束后才会生成')
        return None

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    sd = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(sd)
    model.eval()

    def one(t):
        e = tok([t], padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            return int(model(**e).logits.argmax(-1).item())

    for t in texts[:WARMUP]:
        one(t)

    lats = []
    for t in texts:
        t0 = time.perf_counter()
        one(t)
        lats.append((time.perf_counter() - t0) * 1000)
    return summarize('MacBERT 全参', lats, extra={'batch': 1, 'warmup': WARMUP})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-tfidf', action='store_true')
    ap.add_argument('--skip-bert', action='store_true')
    ap.add_argument('--out', default='src/results/results_BERT_FULL/latency_baseline.csv')
    args = ap.parse_args()

    data_dir = find_data_dir()
    if not data_dir:
        sys.exit('找不到数据目录')
    ids, texts = get_latency_texts(data_dir)
    print(f'✓ 延迟子集：{len(ids)} 条（latency_ids.json 冻结）')
    print(f'  协议：batch=1，预热 {WARMUP} 次\n')

    rows = []
    if not args.skip_tfidf:
        print('=== TF-IDF + LR ===')
        r = bench_tfidf(texts)
        if r:
            rows.append(r)
            print(f'  mean={r["mean_ms"]}ms p50={r["p50_ms"]}ms '
                  f'p95={r["p95_ms"]}ms max={r["max_ms"]}ms')
    if not args.skip_bert:
        print('\n=== MacBERT 全参 ===')
        r = bench_bert(texts)
        if r:
            rows.append(r)
            print(f'  mean={r["mean_ms"]}ms p50={r["p50_ms"]}ms '
                  f'p95={r["p95_ms"]}ms max={r["max_ms"]}ms')

    if not rows:
        print('\n⚠ 两臂都没测成（缺 sklearn 或缺权重），没有输出')
        return 1

    try:
        import pandas as pd
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False, encoding='utf-8-sig')
        print(f'\n✓ 已生成 {args.out}')
    except Exception as e:
        print(f'(pandas 不可用，无法写 CSV: {e})')
        for r in rows:
            print(r)
    return 0


if __name__ == '__main__':
    sys.exit(main())
