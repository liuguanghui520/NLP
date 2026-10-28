"""Local validation-only search. Test is a separate, explicitly frozen stage."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import joblib
import jieba
import numpy as np
import psutil
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def append(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(obj, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x]


def metrics(y, p):
    report = classification_report(y, p, labels=[0, 1], output_dict=True, zero_division=0)
    return dict(macro_f1=report['macro avg']['f1-score'], accuracy=float(np.mean(np.array(y) == p)),
                weighted_f1=report['weighted avg']['f1-score'], report=report,
                confusion_matrix=confusion_matrix(y, p, labels=[0, 1]).tolist())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def word_tokens(text):
    return jieba.lcut(text, HMM=True)


def configs(method):
    if method == 'tfidf':
        return [dict(config_id=n, analyzer=a, ngram_range=g, C=4.0, min_df=2,
                     sublinear_tf=True, max_features=200000, max_iter=2000,
                     solver='lbfgs', class_weight=None, jieba_HMM=True)
                for n, a, g in [('char_2_4', 'char', (2, 4)), ('char_1_3', 'char', (1, 3)),
                                ('char_2_5', 'char', (2, 5)), ('word_1_2', 'word', (1, 2))]]
    return [dict(config_id=n, lr=lr, epochs=ep, maxlen=ml, freeze_lower=fr, weighted=w,
                 batch=8 if ml == 512 else 16, eval_batch=32, device='cuda', precision='fp32',
                 weight_decay=0.01, warmup_ratio=0.1, grad_clip=1.0, accumulation=1,
                 length_group_mega=50, optimizer='AdamW', scheduler='linear')
            for n, lr, ep, ml, fr, w in [
                ('main', 2e-5, 3, 256, 0, False), ('partial', 3e-5, 3, 256, 6, False),
                ('weighted', 2e-5, 3, 256, 0, True), ('lr_5e5', 5e-5, 3, 256, 0, False),
                ('len128', 2e-5, 3, 128, 0, False), ('len512', 2e-5, 3, 512, 0, False),
                ('lr_1e5', 1e-5, 3, 256, 0, False), ('ep5', 2e-5, 5, 256, 0, False)]]


def resource(path, phase, start):
    row = dict(phase=phase, elapsed_s=time.perf_counter()-start,
               rss_bytes=psutil.Process().memory_info().rss)
    if torch.cuda.is_initialized():
        row.update(allocated_peak=torch.cuda.max_memory_allocated(), reserved_peak=torch.cuda.max_memory_reserved())
    append(path / 'resources/samples.jsonl', row)


def predictions(path, rows, pred, cfg, epoch, split, lengths=None, elapsed=None):
    assert len(rows) == len(pred) and len({r['sample_id'] for r in rows}) == len(rows)
    for i, (row, p) in enumerate(zip(rows, pred)):
        append(path, dict(run_id=cfg['run_id'], method=cfg['method'], config_id=cfg['config_id'],
                          seed=cfg['seed'], sample_id=row['sample_id'], split=split,
                          true_label=int(row['label']), parsed_label=int(p), raw_output=str(int(p)),
                          status='ok', checkpoint_id=f'epoch-{epoch}', input_length=len(row['text']),
                          token_length=lengths[i] if lengths is not None else None,
                          truncated=bool(lengths[i] > cfg['maxlen']) if lengths is not None else False,
                          elapsed_ms=elapsed, timing_scope='whole_split_batch' if elapsed else 'not_measured'))


def tfidf(path, cfg, tr, va):
    kw = {k: cfg[k] for k in ['analyzer', 'ngram_range', 'min_df', 'sublinear_tf', 'max_features']}
    if cfg['analyzer'] == 'word':
        kw.update(tokenizer=word_tokens, token_pattern=None)
    pipe = Pipeline([('tfidf', TfidfVectorizer(**kw)), ('classifier', LogisticRegression(
        C=cfg['C'], max_iter=cfg['max_iter'], solver=cfg['solver'], random_state=cfg['seed']))])
    x, y = [r['text'] for r in tr], [int(r['label']) for r in tr]
    xv, yv = [r['text'] for r in va], [int(r['label']) for r in va]
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pipe.fit(x, y)
    fit_s = time.perf_counter()-start
    p = pipe.predict(xv)
    score = metrics(yv, p)
    score.update(epoch=1, fit_s=fit_s, warnings=[str(w.message) for w in caught],
                 iterations=pipe[-1].n_iter_.tolist(), features=len(pipe[0].vocabulary_),
                 train_log_loss=log_loss(y, pipe.predict_proba(x)),
                 validation_log_loss=log_loss(yv, pipe.predict_proba(xv)),
                 trainable_parameters=int(pipe[-1].coef_.size + pipe[-1].intercept_.size))
    predictions(path/'validation/epoch-1/predictions.jsonl', va, p, cfg, 1, 'validation')
    append(path/'epoch_metrics.jsonl', score)
    modelpath = path/'checkpoints/best/pipeline.joblib'
    modelpath.parent.mkdir(parents=True)
    joblib.dump(pipe, modelpath)
    assert np.array_equal(joblib.load(modelpath).predict(xv), p)
    score.update(model=str(modelpath), model_sha256=sha(modelpath), config_id=cfg['config_id'])
    write(path/'selection.json', score)
    return score


def bert(path, cfg, tr, va, modeldir, smoke=False):
    assert torch.cuda.is_available(), 'Frozen device cuda unavailable; no CPU fallback'
    torch.cuda.reset_peak_memory_stats()
    seed_all(cfg['seed'])
    tok = AutoTokenizer.from_pretrained(modeldir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(modeldir, num_labels=2, local_files_only=True)
    if cfg['freeze_lower']:
        for p in model.bert.embeddings.parameters():
            p.requires_grad = False
        for layer in model.bert.encoder.layer[:cfg['freeze_lower']]:
            for p in layer.parameters():
                p.requires_grad = False
    model.cuda()
    cfg.update(total_parameters=sum(p.numel() for p in model.parameters()),
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    write(path/'config.json', cfg)
    raw = tok([r['text'] for r in tr], truncation=True, max_length=cfg['maxlen'])
    lens = np.array([len(x) for x in raw['input_ids']])
    def batch(rows):
        return {k: v.cuda() for k, v in tok([r['text'] for r in rows], padding=True,
                truncation=True, max_length=cfg['maxlen'], return_tensors='pt').items()}
    def evaluate():
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(va), cfg['eval_batch']):
                out.extend(model(**batch(va[i:i+cfg['eval_batch']])).logits.argmax(-1).cpu().tolist())
        return out
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=.01)
    nsteps = math.ceil(len(tr)/cfg['batch']) * cfg['epochs']
    sched = get_linear_schedule_with_warmup(opt, int(nsteps*.1), nsteps)
    counts = np.bincount([int(r['label']) for r in tr], minlength=2)
    cw = torch.tensor(len(tr)/(2*counts), dtype=torch.float32, device='cuda') if cfg['weighted'] else None
    best, step = -1., 0
    for ep in range(cfg['epochs']):
        start = time.perf_counter()
        g = np.random.RandomState(cfg['seed']+ep)
        idx = g.permutation(len(tr))
        batches = []
        for i in range(0, len(idx), cfg['batch']*50):
            chunk = idx[i:i+cfg['batch']*50]
            chunk = chunk[np.argsort(lens[chunk], kind='stable')]
            batches.extend([chunk[j:j+cfg['batch']] for j in range(0, len(chunk), cfg['batch'])])
        g.shuffle(batches)
        model.train()
        losses = []
        for ids in batches:
            t = time.perf_counter()
            opt.zero_grad()
            logits = model(**batch([tr[i] for i in ids])).logits
            y = torch.tensor([int(tr[i]['label']) for i in ids], device='cuda')
            loss = torch.nn.functional.cross_entropy(logits, y, weight=cw)
            assert torch.isfinite(loss), 'Nonfinite loss'
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            append(path/'train_steps.jsonl', dict(step=step, epoch=ep+1, loss=loss.item(), lr=sched.get_last_lr()[0], seconds=time.perf_counter()-t))
            if step % 50 == 0:
                print(cfg['run_id'], ep+1, step, loss.item(), flush=True)
                resource(path, 'train', start)
        train_s = time.perf_counter()-start
        t = time.perf_counter()
        p = evaluate()
        val_s = time.perf_counter()-t
        score = metrics([int(r['label']) for r in va], p)
        score.update(epoch=ep+1, train_loss=float(np.mean(losses)), train_s=train_s, validation_s=val_s,
                     trainable_parameters=cfg['trainable_parameters'], config_id=cfg['config_id'])
        predictions(path/f'validation/epoch-{ep+1}/predictions.jsonl', va, p, cfg, ep+1, 'validation',
                    [len(x) for x in tok([r['text'] for r in va], truncation=False)['input_ids']], val_s*1000)
        append(path/'epoch_metrics.jsonl', score)
        last = path/'checkpoints/last'
        last.mkdir(parents=True, exist_ok=True)
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     epoch=ep+1, step=step, random=random.getstate(), numpy=np.random.get_state(),
                     torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
        torch.save(state, last/'state.tmp')
        (last/'state.tmp').replace(last/'state.pt')
        # Exercise epoch-boundary state restoration in smoke, including optimizer and RNG.
        if smoke:
            restored = torch.load(last/'state.pt', weights_only=False)
            model.load_state_dict(restored['model'])
            opt.load_state_dict(restored['optimizer'])
            sched.load_state_dict(restored['scheduler'])
            random.setstate(restored['random'])
            np.random.set_state(restored['numpy'])
            torch.set_rng_state(restored['torch'])
            torch.cuda.set_rng_state_all(restored['cuda'])
            assert evaluate() == p
            write(path/'restore_check.json', dict(epoch_boundary=True, predictions_equal=True))
        if score['macro_f1'] > best:
            best = score['macro_f1']
            dest = path/'checkpoints/best'
            model.save_pretrained(dest)
            tok.save_pretrained(dest)
            loaded = AutoModelForSequenceClassification.from_pretrained(dest, local_files_only=True).cuda().eval()
            with torch.no_grad():
                q = []
                for i in range(0, len(va), cfg['eval_batch']):
                    q.extend(loaded(**batch(va[i:i+cfg['eval_batch']])).logits.argmax(-1).cpu().tolist())
            assert q == p
            del loaded
            score.update(model=str(dest), model_sha256=sha(dest/'model.safetensors'))
            write(path/'selection.json', score)
        print(cfg['run_id'], score, flush=True)
        resource(path, 'epoch', start)
    return json.loads((path/'selection.json').read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--method', choices=['bert', 'tfidf'], required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    root, data = Path(args.root), Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    # Search never reads test text. Hashing it does not perform test evaluation.
    manifest = {p.name: sha(p) for p in data.glob('*') if p.is_file()}
    write(root/'data_hashes.json', manifest)
    tr, va = read_rows(data/'train.jsonl'), read_rows(data/'validation.jsonl')
    split_ids = json.loads((data/'split_ids.json').read_text())
    for name, rows, count in [('train', tr, 6210), ('validation', va, 776)]:
        assert len(rows) == count
        assert [r['sample_id'] for r in rows] == split_ids[name]
    assert not ({r['sample_id'] for r in tr} & {r['sample_id'] for r in va})
    env = dict(python=sys.version, platform=platform.platform(), threads=4,
               versions={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'scikit-learn', 'numpy', 'jieba', 'joblib']},
               source_sha256=sha(__file__), command=sys.argv, test_loaded=False)
    write(root/'environment.json', env)
    if args.method == 'bert':
        write(root/'base_model_hashes.json', {p.name: sha(p) for p in Path(args.model).glob('*') if p.is_file()})
    matrix = configs(args.method)
    write(root/'matrix.json', matrix)
    if args.smoke:
        def smoke_subset(rows, per_class):
            return [r for label in [0, 1] for r in [x for x in rows if int(x['label']) == label][:per_class]]
        tr, va = smoke_subset(tr, 16), smoke_subset(va, 8)
        matrix = [dict(c, epochs=2) for c in matrix if c['config_id'] in ['main', 'len512']] if args.method == 'bert' else [matrix[0]]
    results = []
    def run(c, seed):
        run_id = f'{args.method}-{c["config_id"]}-seed{seed}'
        path = root/run_id
        path.mkdir(exist_ok=False)
        c = dict(c, seed=seed, method=args.method, run_id=run_id, stage='smoke' if args.smoke else 'formal',
                 environment=env, model_source=args.model, data_manifest_sha256=sha(data/'manifest.json'))
        write(path/'config.json', c)
        write(path/'manifest.json', manifest)
        write(path/'status.json', dict(status='running'))
        start = time.perf_counter()
        try:
            with (path/'stdout.log').open('w') as out, (path/'stderr.log').open('w') as err:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    score = tfidf(path, c, tr, va) if args.method == 'tfidf' else bert(path, c, tr, va, args.model, args.smoke)
            resource(path, 'complete', start)
            write(path/'status.json', dict(status='completed', elapsed_s=time.perf_counter()-start))
            print(run_id, score['macro_f1'], flush=True)
            return score
        except BaseException:
            write(path/'status.json', dict(status='failed', traceback=traceback.format_exc()))
            raise
    for c in matrix:
        results.append(run(c, 42))
        write(root/'validation_comparison.json', results)
    eligible = [r for r in results if r['config_id'] != 'partial']
    selected = sorted(eligible, key=lambda r: (-r['macro_f1'], r['trainable_parameters'],
                                               [c['config_id'] for c in matrix].index(r['config_id'])))[0]
    winner = next(c for c in matrix if c['config_id'] == selected['config_id'])
    write(root/'frozen_winner.json', dict(config=winner, config_sha256=hashlib.sha256(json.dumps(winner, sort_keys=True).encode()).hexdigest(), selected=selected))
    if not args.smoke:
        for seed in [43, 44]:
            run(winner, seed)
    write(root/'status.json', dict(status='training_complete', test_stage='pending_separate_execution'))


if __name__ == '__main__':
    main()
