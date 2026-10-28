import argparse
from pathlib import Path
import subprocess
import sys

from src.common.io import local_paths, load, rows, save
from src.evaluation.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description='Unified sentiment project')
    p.add_argument('--paths', default=str(ROOT/'configs/paths.local.json'))
    p.add_argument('command', choices=['check','import','evaluate','summarize','errors','plot','predict','benchmark','train'])
    p.add_argument('--method', choices=['bert','tfidf','qwen_zero_shot','qwen_few_shot','qwen_lora'])
    p.add_argument('--output')
    p.add_argument('--split', choices=['validation','test'], default='test')
    p.add_argument('--seed', type=int, choices=[42,43,44], default=42)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    paths = local_paths(args.paths)
    if args.command=='check':
        manifest = load(Path(paths['data'])/'manifest.json')
        from src.common.io import digest
        for name in ['train.jsonl','validation.jsonl','test.jsonl']:
            assert digest(Path(paths['data'])/name)==manifest['files'][name]
        print('Configured resources exist; frozen data hashes match')
    elif args.command=='import':
        from src.adapters.import_runs import import_all
        print('Imported and audited', len(import_all(ROOT, paths)), 'runs')
    elif args.command=='evaluate':
        for item in rows(ROOT/'results/run_index.jsonl'):
            result = evaluate(rows(ROOT/'results/predictions'/f'{item["run_id"]}.jsonl'),
                              rows(Path(paths['data'])/f'{item["split"]}.jsonl'))
            save(ROOT/'results/metrics'/f'{item["run_id"]}.json', result)
            if not result['complete']:
                raise ValueError(result['errors'])
        print('All imported predictions audited')
    elif args.command in ['summarize','errors','plot']:
        from src.analysis import report
        getattr(report, args.command)(ROOT)
    elif args.command=='train':
        if args.method not in ['bert','tfidf'] or not args.output:
            p.error('train requires --method bert/tfidf and new --output directory')
        if args.method=='bert' and (not paths.get('macbert_model') or not Path(paths['macbert_model']).exists()):
            p.error('Configure existing macbert_model resource in --paths')
        cmd = [sys.executable, str(ROOT/'src/legacy/baselines/run_bert_tfidf.py'),
               '--root', args.output, '--data', paths['data'], '--model', paths.get('macbert_model',''), '--method',args.method]
        if args.smoke:
            cmd.append('--smoke')
        subprocess.run(cmd, check=True)
    else:
        if not args.method or not args.output:
            p.error('predict/benchmark requires --method and new --output')
        from src.adapters.inference import run
        run(ROOT, paths, args)


if __name__=='__main__':
    main()
