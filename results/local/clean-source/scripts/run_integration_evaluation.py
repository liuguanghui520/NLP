"""Serial integration smoke, frozen tests and inference-resource probes."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common.io import load, save, rows, digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    queue=root/'results/local'/args.run_id
    queue.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for method in ['qwen_few_shot','qwen_lora','bert','tfidf']:
        jobs.append((f'smoke-{method}', ['predict','--method',method,'--smoke']))
    jobs.extend([(f'test-{method}', ['predict','--method',method,'--split','test'])
                 for method in ['qwen_zero_shot','qwen_few_shot']])
    jobs.extend([(f'test-lora-seed{seed}', ['predict','--method','qwen_lora','--seed',str(seed),'--split','test'])
                 for seed in [42,43,44]])
    jobs.extend([(f'probe-{method}-seed{seed}', ['benchmark','--method',method,'--seed',str(seed)])
                 for method in ['bert','tfidf'] for seed in [42,43,44]])
    completed=[]
    try:
        for name, command in jobs:
            output=queue/name
            save(queue/'status.json',dict(status='running',current=name,completed=completed,total=len(jobs)))
            with (queue/f'{name}.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-m','src.cli',*command,'--output',str(output)],cwd=root,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            completed.append(name)
            print(name,'completed',flush=True)
        for command in ['evaluate','summarize','errors','plot']:
            subprocess.run([sys.executable,'-m','src.cli',command],cwd=root,check=True)
        benchmark=[]
        for name in completed:
            output=queue/name
            if not name.startswith('smoke'):
                config=load(output/'config.json')
                benchmark.append(dict(run_id=config['run_id'],method=config['method'],seed=config['seed'],
                     benchmark=load(output/'benchmark.json'),resources=load(output/'resources.json')))
        save(root/'results/benchmarks/integration.json',benchmark)
        save(root/'artifacts/final-acceptance.json',dict(status='code_and_existing_artifact_inference_audited',
             finished_utc=datetime.now(timezone.utc).isoformat(),queue_id=args.run_id,
             runs=[{k:v for k,v in r.items() if k!='metrics'} for r in rows(root/'results/run_index.jsonl')],
             freeze_sha256=digest(root/'artifacts/evaluation-freeze.json'),
             new_machine_reproduction='not_executed',human_error_analysis='pending_student_review',
             public_artifact_downloads='not_yet_uploaded; local checkpoints verified'))
        save(queue/'status.json',dict(status='completed',completed=completed))
    except BaseException:
        save(queue/'status.json',dict(status='failed',completed=completed,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
