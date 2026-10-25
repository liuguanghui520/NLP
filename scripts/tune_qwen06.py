"""Student-confirmed two-candidate Optuna exercise; launch manually in WSL."""
import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

from evaluate_qwen06 import digest, read_validation, write_json


PROJECT = Path(__file__).resolve().parents[1]
FACTORY = Path('/home/lgh/LLaMA-Factory')
SMOKE = FACTORY / 'project-workspaces/qwen06-experience/qwen06-smoke-20261023-235652/data'


def train_config(args, rank, output, name):
    return dict(
        stage='sft', do_train=True, model_name_or_path=str(args.model),
        preprocessing_num_workers=1, finetuning_type='lora', template='qwen3',
        flash_attn='auto', dataset_dir=str(SMOKE), dataset='sentiment_smoke',
        cutoff_len=2048, learning_rate=0.0001, num_train_epochs=3.0,
        max_samples=20, seed=42, data_seed=42, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, lr_scheduler_type='linear',
        max_grad_norm=1.0, logging_steps=1, save_steps=10, warmup_steps=2,
        packing=False, enable_thinking=False, train_on_prompt=False,
        report_to='none', use_swanlab=True, output_dir=str(output),
        bf16=True, plot_loss=True, trust_remote_code=False,
        include_num_input_tokens_seen=True, optim='adamw_torch',
        max_steps=20, gradient_checkpointing=True, dataloader_num_workers=0,
        save_total_limit=2, run_name=name, lora_rank=rank, lora_alpha=2 * rank,
        lora_dropout=0.05, lora_target='q_proj,v_proj', weight_decay=0.0,
        swanlab_project='nlp-qwen06-experience', swanlab_run_name=name,
        swanlab_mode=args.swanlab_mode, swanlab_logdir=str(output.parent / 'swanlog'),
        overwrite_output_dir=False,
    )


def run_logged(command, log_path, cwd=None):
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f'Subprocess exit {code}; retained log: {log_path}')


def preflight(args):
    import yaml
    from llamafactory.hparams import get_train_args

    read_validation(args)
    frozen = PROJECT / 'data/processed/v1-20261023'
    manifest = json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))
    if digest(frozen / 'smoke_ids.json') != manifest['files']['smoke_ids.json']:
        raise ValueError('Frozen smoke IDs changed.')
    ids = json.loads((frozen / 'smoke_ids.json').read_text(encoding='utf-8'))
    if json.loads((SMOKE / 'smoke_ids.json').read_text(encoding='utf-8')) != ids:
        raise ValueError('Workspace smoke IDs differ from frozen IDs.')
    train = {row['sample_id']: row for row in map(json.loads, (frozen / 'train.jsonl').read_text(encoding='utf-8').splitlines())}
    if digest(frozen / 'train.jsonl') != manifest['files']['train.jsonl']:
        raise ValueError('Frozen training data changed.')
    instruction = manifest['config']['instruction']
    expected = [dict(instruction=instruction, input=train[item]['text'], output=str(train[item]['label'])) for item in ids]
    if json.loads((SMOKE / 'smoke.json').read_text(encoding='utf-8')) != expected:
        raise ValueError('Smoke contents differ from frozen training subset.')
    registration = json.loads((SMOKE / 'dataset_info.json').read_text(encoding='utf-8'))
    if registration['sentiment_smoke'] != {'file_name': 'smoke.json', 'columns': {'prompt': 'instruction', 'query': 'input', 'response': 'output'}}:
        raise ValueError('Unexpected smoke dataset registration.')
    if not (args.model / 'config.json').is_file():
        raise FileNotFoundError(args.model / 'config.json')
    for rank in (8, 16):
        config = train_config(args, rank, args.workspace / 'CHECK-ONLY' / str(rank), 'check-only')
        get_train_args(config)
        yaml.safe_dump(config, allow_unicode=True)
    import optuna
    import swanlab
    print(f'CHECK PASSED: ranks 8/16, alpha 16/32, 20 frozen training rows, 776 validation rows; Optuna {optuna.__version__}, SwanLab {swanlab.__version__}. No training started.', flush=True)


def summary_worker(args):
    import swanlab

    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    swanlab.init(project='nlp-qwen06-experience', experiment_name=summary['run_id'] + '-summary',
                 mode=args.swanlab_mode, logdir=str(args.summary.parent / 'swanlog-summary'),
                 config={'purpose': '0.6B Optuna exercise', 'objective': 'validation_macro_f1',
                         'ranks': [8, 16], 'alpha_rule': '2 * rank', 'training_steps': 20})
    try:
        for row in summary['trials']:
            if row['state'] == 'COMPLETE':
                swanlab.log({'validation/macro_f1': row['macro_f1'],
                             'validation/accuracy': row['accuracy'],
                             'validation/invalid_rate': row['invalid_rate'],
                             'lora/rank': row['rank']}, step=row['number'] + 1)
    finally:
        swanlab.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/home/lgh/models/Qwen3-0.6B'))
    parser.add_argument('--validation', type=Path, default=PROJECT / 'data/processed/v1-20261023/validation.jsonl')
    parser.add_argument('--workspace', type=Path, default=FACTORY / 'project-workspaces/qwen06-optuna')
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'results/qwen06-optuna')
    parser.add_argument('--swanlab-mode', choices=['cloud', 'offline'], default='cloud')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--summary', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.summary:
        summary_worker(args)
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    preflight(args)
    if args.check_only:
        return
    import optuna
    import yaml

    run_id = 'qwen06-optuna-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    root = args.output_root / run_id
    work = args.workspace / run_id
    root.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    state = dict(run_id=run_id, status='running', started_utc=datetime.now(timezone.utc).isoformat())
    write_json(root / 'status.json', state)
    print('RESULT DIRECTORY:', root, '\nTRAINING DIRECTORY:', work, flush=True)
    try:
        manifest = dict(run_id=run_id, training_workspace=str(work), command=sys.argv,
                        purpose='Student-confirmed 0.6B tool exercise; not formal 1.7B experiments',
                        confirmation='2026-10-25 user: rank 8/16, 2 trials, alpha=2*rank',
                        search={'lora_rank': [8, 16]}, objective='maximize validation macro_f1',
                        fixed={'max_steps': 20, 'learning_rate': 0.0001, 'seed': 42,
                               'smoke_count': 20, 'validation_count': 776},
                        script_sha256=digest(Path(__file__)),
                        evaluator_sha256=digest(PROJECT / 'scripts/evaluate_qwen06.py'),
                        validation_sha256=digest(args.validation),
                        smoke_sha256=digest(SMOKE / 'smoke.json'),
                        packages={name: metadata.version(name) for name in ('torch', 'transformers', 'peft', 'llamafactory', 'swanlab', 'optuna')},
                        model_files={p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()},
                        model_revision=None, model_revision_note='Downloaded revision unknown; SHA256 records local model',
                        factory_commit=subprocess.check_output(['git', '-C', str(FACTORY), 'rev-parse', 'HEAD'], text=True).strip(),
                        hardware=subprocess.check_output(['nvidia-smi'], text=True),
                        ai_assistance='AI engineering implementation and checks; user chose search scope, executes and interprets results.')
        write_json(root / 'manifest.json', manifest)
        study = optuna.create_study(study_name=run_id, direction='maximize',
                                   storage='sqlite:///' + str(root / 'study.db'),
                                   sampler=optuna.samplers.GridSampler({'lora_rank': [8, 16]}, seed=42))
        study.enqueue_trial({'lora_rank': 8})
        study.enqueue_trial({'lora_rank': 16})

        def objective(trial):
            rank = trial.suggest_categorical('lora_rank', [8, 16])
            name = f'{run_id}-trial-{trial.number:03d}-r{rank}'
            local = root / f'trial-{trial.number:03d}'
            training = work / f'trial-{trial.number:03d}'
            local.mkdir()
            training.mkdir()
            trial.set_user_attr('run_id', name)
            trial.set_user_attr('result_directory', str(local))
            config = train_config(args, rank, training / 'adapter', name)
            config_path = training / 'train.yaml'
            config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
            write_json(local / 'training_config.json', config)
            trial_state = dict(run_id=name, status='training', rank=rank,
                               started_utc=datetime.now(timezone.utc).isoformat())
            write_json(local / 'status.json', trial_state)
            try:
                run_logged([str(Path(sys.executable).parent / 'llamafactory-cli'), 'train', str(config_path)], training / 'train.log', FACTORY)
                adapter = training / 'adapter'
                trainer = json.loads((adapter / 'trainer_state.json').read_text(encoding='utf-8'))
                if trainer['global_step'] != 20:
                    raise RuntimeError('Training did not complete 20 steps.')
                write_json(local / 'adapter_manifest.json', {name: digest(adapter / name) for name in ('adapter_model.safetensors', 'adapter_config.json')})
                evaluation = local / 'evaluation'
                evaluation.mkdir()
                trial_state['status'] = 'evaluating'
                write_json(local / 'status.json', trial_state)
                run_logged([sys.executable, '-u', str(PROJECT / 'scripts/evaluate_qwen06.py'),
                            '--worker', 'lora', '--run-dir', str(evaluation), '--adapter', str(adapter),
                            '--model', str(args.model), '--validation', str(args.validation)], local / 'evaluation.log', FACTORY)
                result = json.loads((evaluation / 'lora/metrics.json').read_text(encoding='utf-8'))
                if result['count'] != 776:
                    raise RuntimeError('Validation incomplete.')
                for key in ('accuracy', 'invalid_rate'):
                    trial.set_user_attr(key, result[key])
                trial_state['status'] = 'completed'
                return result['macro_f1']
            except BaseException as error:
                trial_state.update(status='failed', error=repr(error))
                (local / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                raise
            finally:
                trial_state['ended_utc'] = datetime.now(timezone.utc).isoformat()
                write_json(local / 'status.json', trial_state)

        # Failed trials stay FAIL, never receive a made-up score; continue other candidate.
        study.optimize(objective, n_trials=2, n_jobs=1, catch=(Exception,))
        trials = [dict(number=t.number, state=t.state.name, rank=t.params.get('lora_rank'),
                       macro_f1=t.value, **t.user_attrs) for t in study.trials]
        complete = [t for t in trials if t['state'] == 'COMPLETE']
        best = sorted(complete, key=lambda t: (-t['macro_f1'], t['rank']))[0] if complete else None
        summary = dict(run_id=run_id, trials=trials, best=best,
                       selection_rule='Higher validation Macro-F1; ties prefer smaller rank',
                       complete_comparison=(len(complete) == 2))
        write_json(root / 'summary.json', summary)
        print('\nrank\tstate\tmacro_f1\taccuracy\tinvalid_rate', flush=True)
        for t in trials:
            print(f"{t['rank']}\t{t['state']}\t{t['macro_f1']}\t{t.get('accuracy')}\t{t.get('invalid_rate')}", flush=True)
        if complete:
            try:
                run_logged([sys.executable, '-u', str(Path(__file__).resolve()), '--summary', str(root / 'summary.json'),
                            '--swanlab-mode', args.swanlab_mode], root / 'swanlab-summary.log')
                state['summary_monitoring'] = 'finished'
            except Exception as error:
                state['summary_monitoring'] = repr(error)
                print('SwanLab summary recording failed; local metrics and Optuna database are preserved.', flush=True)
        state['status'] = 'completed' if len(complete) == 2 else 'incomplete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        (root / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        state['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(root / 'status.json', state)
    if state['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
