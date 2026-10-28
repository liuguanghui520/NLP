from collections import defaultdict
from itertools import combinations
from pathlib import Path
import statistics
from src.common.io import load, rows, save, save_rows


def summarize(root):
    index = rows(root/'results/run_index.jsonl')
    grouped = defaultdict(list)
    for r in index:
        grouped[(r['method'], r['split'])].append(r)
    summary = []
    for (method, split), group in grouped.items():
        values = {key: [r['metrics'][key] for r in group] for key in ['macro_f1', 'accuracy', 'weighted_f1', 'invalid_rate']}
        summary.append(dict(method=method, split=split, runs=[r['run_id'] for r in group],
            seeds=[r['seed'] for r in group], n=group[0]['n'],
            metrics={k:dict(mean=statistics.mean(v), sample_std=statistics.stdev(v) if len(v)>1 else None) for k,v in values.items()}))
    save(root/'results/summaries/methods.json', summary)
    text = ['# 统一结果（程序重算）', '', '|方法|划分|运行数|Macro-F1均值|样本标准差|Accuracy均值|', '|---|---|---:|---:|---:|---:|']
    for r in summary:
        m = r['metrics']
        text.append(f"|{r['method']}|{r['split']}|{len(r['runs'])}|{m['macro_f1']['mean']:.6f}|{m['macro_f1']['sample_std']}|{m['accuracy']['mean']:.6f}|")
    (root/'results/summaries/methods.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return summary


def errors(root):
    index = rows(root/'results/run_index.jsonl')
    review = []
    for r in index:
        for item in rows(root/'results/predictions'/f'{r["run_id"]}.jsonl'):
            if item['parsed_label'] != item['true_label']:
                review.append(dict(run_id=r['run_id'], method=r['method'], sample_id=item['sample_id'],
                    split=item['split'], true_label=item['true_label'], parsed_label=item['parsed_label'],
                    input_trace=dict(data_key='data', split=item['split'], sample_id=item['sample_id']),
                    primary_cause=None, secondary_cause=None, evidence=None, limitations=None,
                    improvement=None, retest=None, human_review='pending'))
    save_rows(root/'results/errors/review.jsonl', review)
    pairs = []
    from scipy.stats import binomtest
    for a, b in combinations(index, 2):
        if a['split'] != b['split'] or a['method'] == b['method']:
            continue
        # Compare matching training seeds; deterministic prompts have no seed.
        if a['seed'] is not None and b['seed'] is not None and a['seed'] != b['seed']:
            continue
        ar = rows(root/'results/predictions'/f'{a["run_id"]}.jsonl')
        br = rows(root/'results/predictions'/f'{b["run_id"]}.jsonl')
        assert [(r['sample_id'],r['true_label']) for r in ar] == [(r['sample_id'],r['true_label']) for r in br]
        counts = defaultdict(int)
        details = []
        for x,y in zip(ar,br):
            ac, bc = x['parsed_label']==x['true_label'], y['parsed_label']==y['true_label']
            category = 'both_correct' if ac and bc else 'both_failed' if not ac and not bc else 'B_corrects_A' if bc else 'B_regresses_A'
            counts[category] += 1
            if category != 'both_correct':
                details.append(dict(sample_id=x['sample_id'], true_label=x['true_label'], a=x['parsed_label'], b=y['parsed_label'], category=category))
        n = counts['B_corrects_A']+counts['B_regresses_A']
        pairs.append(dict(a=a['run_id'], b=b['run_id'], split=a['split'], counts=dict(counts),
            direction='B minus A', accuracy_difference=(b['metrics']['accuracy']-a['metrics']['accuracy']),
            exact_mcnemar_p=binomtest(counts['B_corrects_A'], n, .5).pvalue if n else 1.,
            note='Exploratory paired exact McNemar; Holm correction over this exported family; nonsignificance is not equivalence'))
        save(root/'results/errors'/f'{a["run_id"]}--{b["run_id"]}.json', details)
    ordered = sorted(range(len(pairs)), key=lambda i:pairs[i]['exact_mcnemar_p'])
    running = 0.
    for pos, i in enumerate(ordered):
        running = max(running, min(1., (len(pairs)-pos)*pairs[i]['exact_mcnemar_p']))
        pairs[i]['holm_adjusted_p'] = running
    save(root/'results/summaries/paired.json', pairs)


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target = root/'reports/figures'
    target.mkdir(parents=True, exist_ok=True)
    index = rows(root/'results/run_index.jsonl')
    for r in index:
        fig, ax = plt.subplots(figsize=(5,3))
        matrix = r['metrics']['confusion_matrix']
        ax.imshow(matrix, cmap='Blues')
        ax.set_xticks([0,1,2], ['0','1','invalid'])
        ax.set_yticks([0,1], ['0','1'])
        ax.set_xlabel('Prediction'); ax.set_ylabel('Truth')
        for i in range(2):
            for j in range(3):
                ax.text(j,i,str(matrix[i][j]),ha='center',va='center')
        ax.set_title(f'{r["method"]} {r["split"]} seed={r["seed"]}')
        fig.tight_layout(); fig.savefig(target/f'{r["run_id"]}-confusion.png', dpi=140); plt.close(fig)
    tests = [r for r in index if r['split']=='test']
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar([r['method']+' '+str(r['seed']) for r in tests], [r['metrics']['macro_f1'] for r in tests])
    ax.set_ylabel('Test Macro-F1'); ax.tick_params(axis='x', rotation=30)
    fig.tight_layout(); fig.savefig(target/'test-methods.png', dpi=140); plt.close(fig)
    benchmark_path=root/'results/benchmarks/integration.json'
    if benchmark_path.exists():
        benchmarks=load(benchmark_path)
        summary=load(root/'results/summaries/methods.json')
        fig,ax=plt.subplots(figsize=(7,4))
        for r in summary:
            if r['split']!='test':
                continue
            latency=[b['benchmark']['mean_ms'] for b in benchmarks if b['method']==r['method']]
            if latency:
                x=statistics.mean(latency); y=r['metrics']['macro_f1']['mean']
                ax.scatter(x,y); ax.annotate(r['method'],(x,y),xytext=(-5,6) if x>60 else (4,6),
                                            textcoords='offset points',ha='right' if x>60 else 'left')
        ax.margins(x=.12,y=.15)
        ax.set_xlabel('Mean serial end-to-end latency (ms), actual CPU/GPU routes')
        ax.set_ylabel('Test Macro-F1 mean'); fig.tight_layout()
        fig.savefig(target/'effect-latency.png',dpi=140); plt.close(fig)
