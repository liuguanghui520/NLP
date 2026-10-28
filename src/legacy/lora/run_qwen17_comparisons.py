"""Execute the nine student-defined comparisons, reusing recorded B00 evidence.

Use --check-only for compatibility/hash checks, --prepare-only to freeze files,
or no flag to prepare a fresh batch and execute serially in the user's terminal.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid

from evaluate_qwen06 import digest, metrics, read_validation, write_json


PROJECT = Path(__file__).resolve().parents[1]
BASE_ID = 'qwen17-full-20261025-231706-ea8173'
BASE_INDEX = PROJECT / 'results/qwen17-profile' / BASE_ID
FACTORY = Path('/home/lgh/LLaMA-Factory')


def preflight():
    from llamafactory.hparams import get_train_args

    definition = json.loads((PROJECT / 'configs/lora_comparison_10.json').read_text(encoding='utf-8'))
    reference = json.loads((PROJECT / 'configs/lora_reference.json').read_text(encoding='utf-8'))
    manifest = json.loads((BASE_INDEX / 'manifest.json').read_text(encoding='utf-8'))
    base = json.loads((BASE_INDEX / 'training_config.json').read_text(encoding='utf-8'))
    source = Path(manifest['training_workspace'])
    for name, key in [('profile_qwen17.py', 'script_sha256'), ('evaluate_qwen06.py', 'metric_script_sha256')]:
        if digest(source / name) != manifest[key]:
            raise ValueError(f'Baseline archived source mismatch: {name}')
    for package, version in manifest['packages'].items():
        if metadata.version(package) != version:
            raise ValueError(f'Baseline package changed: {package}')
    revision = subprocess.check_output(['git', '-C', str(FACTORY), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != manifest['factory_commit']:
        raise ValueError('LLaMA-Factory revision differs from baseline.')
    for name, expected in manifest['model_files'].items():
        if digest(Path(base['model_name_or_path']) / name) != expected:
            raise ValueError(f'Model file mismatch: {name}')
    data = Path(base['dataset_dir']).parent
    if digest(data / 'manifest.json') != manifest['data_manifest_sha256']:
        raise ValueError('Frozen manifest differs from baseline.')
    frozen = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    for name in ('train.jsonl', 'validation.jsonl', 'alpaca/train.json', 'alpaca/dataset_info.json'):
        if digest(data / name) != frozen['files'][name]:
            raise ValueError(f'Frozen data mismatch: {name}')
    read_validation(argparse.Namespace(validation=data / 'validation.jsonl'))
    if json.loads((BASE_INDEX / 'status.json').read_text(encoding='utf-8'))['status'] != 'completed':
        raise ValueError('Baseline is not completed.')
    configs = []
    signatures = set()
    factors = reference['comparison_rule']['factor_groups']
    for item in definition['configurations']:
        effective = dict(reference['reference'], **item['overrides'])
        signature = json.dumps(effective, sort_keys=True)
        if signature in signatures:
            raise ValueError('Duplicate configuration.')
        signatures.add(signature)
        changed = {key for key, value in effective.items() if value != reference['reference'][key]}
        allowed = set() if item['id'] == 'B00' else set(factors[item['group']])
        if not changed.issubset(allowed) or (item['id'] != 'B00' and not changed):
            raise ValueError(f'Unexpected factor changes: {item["id"]}')
        if effective['lora_alpha'] != 2 * effective['lora_rank']:
            raise ValueError('alpha must equal 2r.')
        config = dict(base, **effective)
        config['output_dir'] = '/tmp/qwen17-comparison-check-not-created/' + item['id']
        get_train_args(config)
        configs.append((item, config))
    if len(configs) != 10 or len(signatures) != 10:
        raise ValueError('Expected 10 unique configurations including B00.')
    print('CHECK PASSED: nine new configurations; same baseline model/data/source/versions; SwanLab enabled. No training started.', flush=True)
    return definition, manifest, source, data, configs


def prepare(definition, baseline_manifest, source, data, configs):
    run_id = 'qwen17-comparisons-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    work = FACTORY / 'project-workspaces/qwen17-comparisons' / run_id
    index = PROJECT / 'results/qwen17-comparisons' / run_id
    work.mkdir(parents=True, exist_ok=False)
    index.mkdir(parents=True, exist_ok=False)
    for name in ('profile_qwen17.py', 'evaluate_qwen06.py'):
        shutil.copyfile(source / name, work / name)
    shutil.copyfile(Path(__file__), work / 'run_qwen17_comparisons.py')
    shutil.copyfile(Path(__file__).parent / 'record_run_resources.py', work / 'record_run_resources.py')
    write_json(work / 'comparison_definition.json', definition)
    jobs = []
    for item, config in configs:
        if item['id'] == 'B00':
            continue
        directory = work / item['id']
        directory.mkdir()
        config.update(output_dir=str(directory / 'adapter'), swanlab_project='nlp-qwen17',
                      swanlab_run_name=run_id + '-' + item['id'],
                      swanlab_logdir=str(directory / 'swanlog'),
                      use_swanlab=True, report_to='none', swanlab_mode='cloud')
        write_json(directory / 'train.json', config)
        jobs.append(dict(id=item['id'], state='pending', training_directory=str(directory),
                         config_sha256=digest(directory / 'train.json'),
                         swanlab_experiment=config['swanlab_run_name']))
    plan = dict(run_id=run_id, status='prepared', project_index=str(index),
                training_workspace=str(work), data_directory=str(data),
                reference_run_id=BASE_ID, baseline_retrained=False, jobs=jobs,
                scripts={name: digest(work / name) for name in (
                    'profile_qwen17.py', 'evaluate_qwen06.py', 'record_run_resources.py', 'run_qwen17_comparisons.py')},
                baseline_model_files=baseline_manifest['model_files'],
                packages=baseline_manifest['packages'], factory_commit=baseline_manifest['factory_commit'],
                ai_assistance='Engineering implementation of user-defined nine comparisons, with unchanged baseline training/validation worker; user runs training and interprets results.')
    write_json(work / 'batch.json', plan)
    write_json(index / 'batch.json', plan)
    return work, index, plan


def execute(work, index, plan):
    plan.update(status='running', started_utc=datetime.now(timezone.utc).isoformat())

    def persist():
        write_json(work / 'batch.json', plan)
        write_json(index / 'batch.json', plan)

    persist()
    try:
        for job in plan['jobs']:
            directory = Path(job['training_directory'])
            if digest(directory / 'train.json') != job['config_sha256']:
                raise ValueError('Prepared configuration changed.')
            job.update(state='running', started_utc=datetime.now(timezone.utc).isoformat())
            persist()
            print(f'\nSTART {job["id"]}: {job["swanlab_experiment"]}', flush=True)
            command = [sys.executable, '-u', str(work / 'profile_qwen17.py'),
                       '--worker', str(directory / 'train.json'), '--data', plan['data_directory']]
            try:
                code = subprocess.call([sys.executable, '-u', str(work / 'record_run_resources.py'),
                                        '--output', str(directory / 'resource-record'), '--'] + command, cwd=FACTORY)
                if code:
                    raise RuntimeError(f'Training worker exited with code {code}')
                history = json.loads((directory / 'validation_history.json').read_text(encoding='utf-8'))
                if len(history) != 3 or history[-1]['epoch'] != 3:
                    raise ValueError('Expected three completed epoch validations.')
                for record in history:
                    predictions = directory / f'validation-step-{record["global_step"]}' / 'predictions.jsonl'
                    rows = [json.loads(line) for line in predictions.read_text(encoding='utf-8').splitlines()]
                    if len(rows) != 776 or len({r['sample_id'] for r in rows}) != 776:
                        raise ValueError('Incomplete or repeated validation IDs.')
                    if metrics(rows)['macro_f1'] != record['macro_f1']:
                        raise ValueError('Saved and recomputed metrics differ.')
                best = json.loads((directory / 'best_checkpoint.json').read_text(encoding='utf-8'))
                adapter = Path(best['checkpoint'])
                job.update(state='completed', best_checkpoint=str(adapter),
                           validation_accuracy=best['accuracy'], validation_macro_f1=best['macro_f1'],
                           invalid_rate=best['invalid_rate'],
                           best_adapter_sha256=digest(adapter / 'adapter_model.safetensors'))
            except Exception as error:
                job.update(state='failed', error=repr(error))
                (directory / 'batch_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                print(f'FAILED {job["id"]}; evidence retained; continuing next configured job.', flush=True)
            finally:
                job['ended_utc'] = datetime.now(timezone.utc).isoformat()
                local_index = index / job['id']
                local_index.mkdir(exist_ok=True)
                for name in ('train.json', 'validation_history.json', 'best_checkpoint.json', 'timing.json', 'parameters.json'):
                    if (directory / name).exists():
                        shutil.copyfile(directory / name, local_index / name)
                if (directory / 'resource-record/status.json').exists():
                    shutil.copyfile(directory / 'resource-record/status.json', local_index / 'resources_summary.json')
                persist()
        plan['status'] = 'completed' if all(j['state'] == 'completed' for j in plan['jobs']) else 'incomplete'
    except BaseException as error:
        plan.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        for job in plan['jobs']:
            if job['state'] == 'running':
                job['state'] = 'interrupted'
        raise
    finally:
        plan['ended_utc'] = datetime.now(timezone.utc).isoformat()
        persist()
    print('BATCH FINISHED:', plan['status'], '\nINDEX:', index, flush=True)
    if plan['status'] != 'completed':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', type=Path)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    definition, manifest, source, data, configs = preflight()
    if args.check_only:
        return
    lock_path = FACTORY / 'project-workspaces/qwen17-comparisons.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another comparison batch is already running.')
        free = shutil.disk_usage(FACTORY).free
        if free < 8 * 1024 ** 3:
            raise RuntimeError('Less than 8GiB free in training workspace.')
        if not args.prepare_only:
            used = float(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                                                  '--format=csv,noheader,nounits'], text=True).strip())
            if used > 2048:
                raise RuntimeError(f'GPU already uses {used}MiB; unload other models before starting.')
        if args.run_prepared:
            if args.prepare_only:
                parser.error('--prepare-only and --run-prepared cannot be combined.')
            work = args.run_prepared.resolve()
            plan = json.loads((work / 'batch.json').read_text(encoding='utf-8'))
            index = Path(plan['project_index'])
            if plan['status'] != 'prepared' or any(j['state'] != 'pending' for j in plan['jobs']):
                raise RuntimeError('This batch has already started; never overwrite or silently retry it.')
            if plan['reference_run_id'] != BASE_ID:
                raise ValueError('Prepared batch reference differs.')
            for name, expected in plan['scripts'].items():
                if digest(work / name) != expected:
                    raise ValueError(f'Prepared source changed: {name}')
        else:
            work, index, plan = prepare(definition, manifest, source, data, configs)
        print('BATCH:', plan['run_id'], '\nTRAINING:', work, '\nINDEX:', index, flush=True)
        if not args.prepare_only:
            execute(work, index, plan)


if __name__ == '__main__':
    main()
