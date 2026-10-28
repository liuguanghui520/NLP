"""Build public provenance/cost summaries without publishing local paths or text."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common.io import load, rows, save, digest, save_rows, local_paths


def main():
    root=Path(__file__).resolve().parents[1]
    paths=local_paths(root/'configs/paths.local.json')
    inventory=[]
    for key, base in [('prompt_delivery',Path(paths['prompt_delivery'])),
                      ('baseline_batch',Path(paths['baseline_results']).parent)]:
        for p in sorted(base.rglob('*')):
            if p.is_file():
                inventory.append(dict(source_key=key,relative_path=p.relative_to(base).as_posix(),
                                      bytes=p.stat().st_size,sha256=digest(p)))
    save(root/'results/imported/source-artifacts.json',inventory)
    costs=[]
    lifecycle=[]
    for method,key in [('bert','bert_results'),('tfidf','baseline_results')]:
        base=Path(paths[key])
        for run in sorted(base.glob(f'{method}-*')):
            if not (run/'config.json').exists():
                continue
            config=load(run/'config.json')
            state=load(run/'status.json')
            epochs=rows(run/'epoch_metrics.jsonl')
            selection=load(run/'selection.json')
            if method=='bert':
                train_s=sum(e['train_s'] for e in epochs)
                validation_s=sum(e['validation_s'] for e in epochs)
                parameters=config['total_parameters']
                trainable=config['trainable_parameters']
                features=None
            else:
                train_s=epochs[0]['fit_s']
                validation_s=None
                parameters=None
                trainable=epochs[0]['trainable_parameters']
                features=epochs[0]['features']
            costs.append(dict(run_id=run.name,method=method,seed=config['seed'],
                phase='search' if config['seed']==42 else 'seed_repeat',
                end_to_end_s=state['elapsed_s'],train_s=train_s,validation_s=validation_s,
                separate_model_load_s=None,separate_save_s=None,separate_test_s=None,
                missing_timing_reason='Historical run did not separately time loading/saves/test; no inferred exact values',
                total_parameters=parameters,trainable_parameters=trainable,
                trainable_fraction=trainable/parameters if parameters else None,tfidf_features=features,
                memory_scope='BERT CUDA peaks may include extra checkpoint reload verification; device samples omit initial interval',
                cuda_peak_samples=rows(run/'resources/samples.jsonl') if (run/'resources/samples.jsonl').exists() else [],
                best_epoch=selection['epoch'],checkpoint_sha256=selection['model_sha256']))
            lifecycle.append(dict(run_id=run.name,method=method,seed=config['seed'],stage=costs[-1]['phase'],
                              status=state['status'],source_key=key,relative_path=run.relative_to(base).as_posix()))
    private=load(Path(paths['lora_freeze'])/'local_manifest.json')
    for adapter in private['adapters']:
        run=Path(paths['lora_adapters'][str(adapter['seed'])]).parents[1]
        costs.append(dict(run_id=run.name,method='qwen_lora',seed=adapter['seed'],
                         original_timing=load(run/'timing.json'),
                         memory_scope='Original training segments reset CUDA peaks after loading and validations'))
    for run in sorted((Path(paths['prompt_delivery'])/'run_evidence').iterdir()):
        if run.is_dir():
            manifest=load(run/'run_manifest.json') if (run/'run_manifest.json').exists() else {}
            lifecycle.append(dict(run_id=run.name,method=manifest.get('method'),stage=manifest.get('purpose'),
                                  status=manifest.get('status'),source_key='prompt_delivery',relative_path='run_evidence/'+run.name))
    # Retain failed baseline smoke and retry statuses in the source batch index.
    for p in sorted(Path(paths['baseline_results']).parent.glob('tfidf-smoke*/**/status.json')):
        lifecycle.append(dict(source_key='baseline_batch',relative_path=p.relative_to(Path(paths['baseline_results']).parent).as_posix(),
                              stage='smoke',status=load(p).get('status')))
    save(root/'results/summaries/training-costs.json',costs)
    save_rows(root/'results/imported/lifecycle.jsonl',lifecycle)
    # Curves already generated from raw logs are copied, not redrawn from rounded tables.
    import shutil
    fig=root/'reports/figures'
    fig.mkdir(parents=True,exist_ok=True)
    for p in Path(paths['baseline_results']).parent.glob('*.png'):
        shutil.copyfile(p,fig/p.name)
    save(root/'reports/audits/integration-scope.json',dict(
        baseline_runs=16,prompt_validation_runs=2,lora_validation_runs=3,
        source_inventory_count=len(inventory),
        historical_extensions='Retained in original source inventory; not retrained',
        privacy='Review text, recovered prompts, raw outputs and local paths excluded from public git',
        incomplete=['Human error analysis','Clean-machine actual inference','Public artifact upload/access verification'],
        corrections=['Old BERT representative errors:60, not245',
                     'Equal accuracy does not imply identical sample predictions',
                     'Nonsignificance does not establish equivalence',
                     'Historical and current selection/test protocols kept distinct']))


if __name__=='__main__':
    main()
