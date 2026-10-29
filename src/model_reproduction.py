"""Serial, full-test inference using published artifacts and the original adapters.

Run only when explicitly selecting --models. No training or test-based selection.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from uuid import uuid4
import zipfile

from src.common.io import load, rows, save, save_rows, digest
from src.common.downloads import download_verified, quarantine, safe_path


def jobs():
    return [(method,seed) for method in ('tfidf','bert','qwen_lora') for seed in (42,43,44)] + [
        ('qwen_zero_shot',None),('qwen_few_shot',None)]


def prepare(root, specification, download_prefix=""):
    """Create portable selection metadata without editing original model files."""
    from scripts.package_trained_artifacts import verify_archive
    qwen=Path(specification['qwen_model']).expanduser().resolve()
    factory=Path(specification['factory_root']).expanduser().resolve()
    if not qwen.is_dir() or not (factory/'src/llamafactory').is_dir():
        raise ValueError('Set qwen_model and factory_root to your existing model/tool directories')
    for name,expected in load(root/'artifacts/lora-lock.json')['base_model_files'].items():
        if not (qwen/name).is_file() or digest(qwen/name)!=expected:
            raise ValueError(f'Frozen Qwen file mismatch: {name}; do not replace the model revision')
    assets=Path(specification.get('trained_artifacts_root') or root/'resources/trained').expanduser().resolve()
    assets.mkdir(parents=True,exist_ok=True)
    catalog=load(root/'artifacts/reproduction-downloads.json')['trained_artifacts']
    models={}
    for asset in catalog:
        artifact=asset['archive'][:-4]
        directory=assets/artifact
        if directory.exists():
            try:
                verify_extracted(directory, asset)
            except (OSError, ValueError, KeyError):
                failed=quarantine(directory)
                print('Retained incomplete/corrupt extracted artifact:',failed,flush=True)
        if not directory.exists():
            archive=assets/asset['archive']
            print('Verifying/downloading trained artifact:',asset['archive'],flush=True)
            download_verified(asset['download_url'],archive,asset['archive_sha256'],
                              expected_bytes=asset['archive_bytes'],prefix=download_prefix,
                              log=root/'results/model-reproduction/download-events.jsonl')
            verify_archive(archive,asset['archive_sha256'])
            staging=assets/('.extract-'+artifact+'-'+uuid4().hex[:8])
            staging.mkdir()
            with zipfile.ZipFile(archive) as package:
                package.extractall(staging)
            verify_extracted(staging/artifact,asset)
            (staging/artifact).rename(directory)
            staging.rmdir()
        verify_extracted(directory,asset)
        models[(asset['method'],asset['seed'])]=directory/'model'
    base=root/'resources/selections'
    for method,config_id in [('tfidf','char_1_3'),('bert','len512')]:
        target=base/method
        save(target/'frozen_winner.json',dict(config=dict(config_id=config_id,maxlen=512)))
        for seed in (42,43,44):
            asset=next(x for x in catalog if x['method']==method and x['seed']==seed)
            model=models[(method,seed)]
            save(target/f'{method}-{config_id}-seed{seed}/selection.json',
                 dict(model=str(model/'pipeline.joblib' if method=='tfidf' else model),model_sha256=asset['checkpoint_sha256']))
    prompt=root/'resources/prompt'
    prompt.mkdir(parents=True,exist_ok=True)
    data=root/'data/processed/v1-20261023'
    shutil.copyfile(root/'data/manifests/manifest.json',data/'manifest.json')
    paths=dict(data=str(data),baseline_results=str(base/'tfidf'),bert_results=str(base/'bert'),
               prompt_delivery=str(prompt),lora_freeze=str(assets),qwen_model=str(qwen),
               macbert_model=str(models[('bert',42)]),factory_root=str(factory),
               portable_reproduction=True,
               lora_adapters={str(seed):str(models[('qwen_lora',seed)]) for seed in (42,43,44)})
    save(root/'configs/paths.local.json',paths)
    return paths


def verify_extracted(directory, asset):
    manifest=load(directory/'manifest.json')
    if manifest['weight_sha256']!=asset['checkpoint_sha256']:
        raise ValueError(f'Checkpoint selection mismatch: {directory.name}')
    for name,expected in manifest['files'].items():
        target=safe_path(directory,name)
        if not target.is_file() or target.stat().st_size!=expected['bytes'] or digest(target)!=expected['sha256']:
            raise ValueError(f'Trained artifact hash mismatch: {directory.name}/{name}')


def summarize(records,contract):
    grouped=defaultdict(list)
    for item in records:
        grouped[item['method']].append(item['metrics'])
    summary=[]
    for method in contract['method_order']:
        values=grouped[method]
        item=dict(method=method,runs=len(values),n=777)
        for key in ('macro_f1','accuracy','weighted_f1','invalid_rate'):
            scores=[r[key] for r in values]
            mean=statistics.mean(scores)
            item[key]=dict(mean=mean,sample_std=statistics.stdev(scores) if len(scores)>1 else None,
                           reference=contract['expected_test_means'][method][key],
                           difference=mean-contract['expected_test_means'][method][key])
        summary.append(item)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification',type=Path,required=True)
    parser.add_argument('--download-prefix',default='')
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[1]
    output=root/'results/model-reproduction'
    output.mkdir(parents=True,exist_ok=False)
    completed=[]
    active_run=None
    save(output/'status.json',dict(status='preparing',inference=True,training=False))
    try:
        from src.model_preflight import inspect_environment
        check=inspect_environment(root,load(args.specification))
        save(output/'preflight.json',check)
        if check['status']!='passed':
            raise RuntimeError('Model environment preflight failed: '+'; '.join(check['errors']))
        prepare(root,load(args.specification),args.download_prefix)
        references={(item['method'],item.get('seed')):item for item in rows(root/'results/run_index.jsonl') if item['split']=='test'}
        prediction_differences=[]
        for method,seed in jobs():
            run_id=f'reproduced-{method}-seed{seed}' if seed is not None else f'reproduced-{method}'
            active_run=run_id
            command=[sys.executable,'-m','src.cli','predict','--method',method,'--seed',str(seed or 42),
                     '--split','test','--output',str(root/'results/local'/run_id)]
            print('Starting:',run_id,flush=True)
            with (output/f'{run_id}.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=load(root/'results/local'/run_id/'metrics.json')
            if not result['complete']:raise ValueError(f'Incomplete prediction: {run_id}')
            reference=references[(method,seed)]
            delta={key:result['metrics'][key]-reference['metrics'][key] for key in ('macro_f1','accuracy','weighted_f1','invalid_rate')}
            completed.append(dict(method=method,seed=seed,run_id=run_id,metrics=result['metrics'],
                                  reference_run_id=reference['run_id'],reference_metrics=reference['metrics'],metric_differences=delta))
            original={item['sample_id']:item for item in rows(root/'results/predictions'/(reference['run_id']+'.jsonl'))}
            for item in rows(root/'results/local'/run_id/'predictions.jsonl'):
                old=original[item['sample_id']]
                if item['parsed_label']!=old['parsed_label'] or item['raw_output']!=old['raw_output']:
                    prediction_differences.append(dict(run_id=run_id,reference_run_id=reference['run_id'],
                        sample_id=item['sample_id'],true_label=item['true_label'],
                        reference_label=old['parsed_label'],new_label=item['parsed_label'],
                        reference_raw_output=old['raw_output'],new_raw_output=item['raw_output']))
            save_rows(output/'prediction-differences.jsonl',prediction_differences)
            save(output/'status.json',dict(status='running',completed=[r['run_id'] for r in completed]))
        summary=summarize(completed,load(root/'configs/reproduction.json'))
        save(output/'metrics.json',dict(runs=completed,methods=summary))
        differences=[abs(item[key]['difference']) for item in summary for key in ('macro_f1','accuracy','weighted_f1','invalid_rate')]
        differences.extend(abs(value) for item in completed for value in item['metric_differences'].values())
        maximum=max(differences)
        save(output/'status.json',dict(status='passed' if maximum<=1e-12 else 'completed_with_differences',
              model_loading=True,
              inference=True,training=False,run_count=len(completed),n_per_run=777,
              maximum_metric_difference=maximum,prediction_difference_count=len(prediction_differences),
              comparison='each seed and method mean; frozen reference is not overwritten'))
        return 0 if maximum<=1e-12 else 1
    except BaseException as error:
        save(output/'status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
              error=str(error),completed=completed,active_run=active_run,
              model_loading=bool(completed) or (active_run is not None and (root/'results/local'/active_run/'config.json').exists()),
              model_loading_attempted=active_run is not None,inference=True,training=False))
        raise


if __name__=='__main__':
    raise SystemExit(main())
