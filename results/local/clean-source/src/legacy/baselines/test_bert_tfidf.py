"""Evaluate only frozen winners; separate from all search code."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from run_bert_tfidf import write, append, sha, metrics, read_rows, predictions, word_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--method', choices=['bert', 'tfidf'], required=True)
    ap.add_argument('--output-name', default='test')
    args = ap.parse_args()
    torch.set_num_threads(4)
    root, data = Path(args.root), Path(args.data)
    frozen_hashes = json.loads((root/'data_hashes.json').read_text())
    assert all(sha(data/name) == digest for name, digest in frozen_hashes.items())
    winner = json.loads((root/'frozen_winner.json').read_text())
    assert json.loads((root/'status.json').read_text())['status'] == 'training_complete'
    test = read_rows(data/'test.jsonl')
    assert len(test) == 777
    ids = json.loads((data/'split_ids.json').read_text())['test']
    assert [r['sample_id'] for r in test] == ids
    lookup = {r['sample_id']: r for r in test}
    for split in ['train', 'validation']:
        lookup.update({r['sample_id']: r for r in read_rows(data/f'{split}.jsonl')})
    latency = [lookup[s] for s in json.loads((data/'latency_ids.json').read_text())]
    warmup = [lookup[s] for s in json.loads((data/'smoke_ids.json').read_text())]
    assert len(latency) == 200 and len(warmup) == 20
    allscores = []
    for seed in [42, 43, 44]:
        path = root/f'{args.method}-{winner["config"]["config_id"]}-seed{seed}'
        cfg = json.loads((path/'config.json').read_text())
        selection = json.loads((path/'selection.json').read_text())
        out = path/args.output_name
        out.mkdir(exist_ok=False)
        if args.method == 'tfidf':
            pipe = joblib.load(selection['model'])
            def predict(rows):
                return pipe.predict([r['text'] for r in rows]).tolist()
            sync = lambda: None
            lengths = None
            latency_lengths = [None] * len(latency)
        else:
            tok = AutoTokenizer.from_pretrained(selection['model'], local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(selection['model'], local_files_only=True).cuda().eval()
            def predict(rows):
                enc = tok([r['text'] for r in rows], padding=True, truncation=True,
                          max_length=cfg['maxlen'], return_tensors='pt').to('cuda')
                with torch.no_grad():
                    return model(**enc).logits.argmax(-1).cpu().tolist()
            sync = torch.cuda.synchronize
            lengths = [len(x) for x in tok([r['text'] for r in test], truncation=False)['input_ids']]
            latency_lengths = [len(x) for x in tok([r['text'] for r in latency], truncation=False)['input_ids']]
        p = []
        for i in range(0, len(test), 32):
            p.extend(predict(test[i:i+32]))
        predictions(out/'predictions.jsonl', test, p, cfg, selection['epoch'], 'test', lengths)
        saved = read_rows(out/'predictions.jsonl')
        assert [r['sample_id'] for r in saved] == ids
        assert [r['true_label'] for r in saved] == [int(r['label']) for r in test]
        score = metrics([r['true_label'] for r in saved], [r['parsed_label'] for r in saved])
        score.update(seed=seed, n=777)
        write(out/'metrics.json', score)
        write(out/'errors.json', [r for r in saved if r['parsed_label'] != r['true_label']])
        for row in warmup:
            predict([row])
        times = []
        for order, row in enumerate(latency):
            sync()
            t = time.perf_counter()
            label = predict([row])[0]
            sync()
            ms = (time.perf_counter()-t)*1000
            times.append(ms)
            append(out/'latency.jsonl', dict(sample_id=row['sample_id'], order=order, latency_ms=ms,
                   status='ok', parsed_label=label, config_id=cfg['config_id'], seed=seed,
                   checkpoint_sha256=selection['model_sha256'], input_length=len(row['text']),
                   token_length=latency_lengths[order],
                   truncated=bool(latency_lengths[order] > cfg['maxlen']) if latency_lengths[order] else False,
                   timing_scope='tokenize_transform_transfer_inference_parse_batch1_serial'))
        write(out/'latency_summary.json', dict(mean_ms=float(np.mean(times)),
              p50_ms=float(np.percentile(times, 50, method='linear')),
              p95_ms=float(np.percentile(times, 95, method='linear')), max_ms=max(times),
              throughput=200/(sum(times)/1000), numpy=np.__version__, percentile_method='linear',
              warmup_ids=[r['sample_id'] for r in warmup], device='cuda' if args.method=='bert' else 'cpu',
              threads=4, batch=1, model_sha256=selection['model_sha256']))
        allscores.append(score)
        write(out/'storage.json', dict(deployment_bytes=sum(p.stat().st_size for p in Path(selection['model']).rglob('*') if p.is_file())
              if Path(selection['model']).is_dir() else Path(selection['model']).stat().st_size))
    summary = {k: dict(mean=float(np.mean([s[k] for s in allscores])),
                      sample_std=float(np.std([s[k] for s in allscores], ddof=1)))
               for k in ['macro_f1', 'accuracy', 'weighted_f1']}
    write(root/'three_seed_test.json', dict(individual=allscores, summary=summary))


if __name__ == '__main__':
    main()
