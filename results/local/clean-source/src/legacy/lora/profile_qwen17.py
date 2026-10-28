"""Prepare/run original confirmed 1.7B LoRA configuration with cost records.

Manual entry: --check-only parses; --smoke executes one training step; default
executes the unchanged full 3-epoch configuration with classification validation.
"""
import argparse
from datetime import datetime
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from evaluate_qwen06 import INSTRUCTION, digest, metrics, read_validation, write_json


PROJECT = Path(__file__).resolve().parents[1]
FACTORY = Path('/home/lgh/LLaMA-Factory')


def make_config(args, output):
    data = args.data / 'alpaca'
    if args.smoke:
        data = FACTORY / 'project-workspaces/qwen06-experience/qwen06-smoke-20261023-235652/data'
    config = dict(model_name_or_path=str(args.model), stage='sft', do_train=True,
                  finetuning_type='lora', template='qwen3', enable_thinking=False,
                  trust_remote_code=False, dataset_dir=str(data),
                  dataset='sentiment_smoke' if args.smoke else 'sentiment_train',
                  cutoff_len=2048, train_on_prompt=False, packing=False,
                  preprocessing_num_workers=1, dataloader_num_workers=0,
                  lora_rank=8, lora_alpha=16, lora_dropout=0.05,
                  lora_target='q_proj,v_proj', learning_rate=0.0001,
                  weight_decay=0.0, optim='adamw_torch', lr_scheduler_type='linear',
                  warmup_ratio=0.1, num_train_epochs=3.0, seed=42, data_seed=42,
                  per_device_train_batch_size=1,
                  gradient_accumulation_steps=1 if args.smoke else 16,
                  max_grad_norm=1.0, gradient_checkpointing=True, bf16=True,
                  save_strategy='epoch', logging_steps=1 if args.smoke else 10,
                  output_dir=str(output), overwrite_output_dir=False,
                  report_to='none', use_swanlab=True, swanlab_mode='cloud',
                  swanlab_project='nlp-qwen17', swanlab_run_name=output.parent.name,
                  swanlab_logdir=str(output.parent / 'swanlog'), plot_loss=True)
    if args.smoke:
        config['max_steps'] = 1
    return config


def train_worker(args):
    import torch
    from transformers import TrainerCallback
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.train.tuner import run_exp

    config = json.loads(args.worker.read_text(encoding='utf-8'))
    model_args, data_args, *_ = get_train_args(config)
    tokenizer = load_tokenizer(model_args)['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    root = args.worker.parent
    rows, _ = read_validation(args)
    history = []

    class ClassificationValidation(TrainerCallback):
        def on_train_begin(self, train_args, state, control, model=None, **kwargs):
            self.started = time.perf_counter()
            self.validation_seconds = 0.0
            write_json(root / 'parameters.json', dict(
                total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            torch.cuda.reset_peak_memory_stats()

        def on_save(self, train_args, state, control, model=None, **kwargs):
            checkpoint = Path(train_args.output_dir) / f'checkpoint-{state.global_step}'
            record = dict(epoch=state.epoch, global_step=state.global_step,
                          checkpoint=str(checkpoint),
                          preceding_train_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          preceding_train_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if args.smoke:
                write_json(root / 'smoke_check.json', record)
                return
            directory = root / f'validation-step-{state.global_step}'
            directory.mkdir(exist_ok=False)
            was_training = model.training
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            predictions = []
            try:
                with (directory / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
                    for index, row in enumerate(rows, 1):
                        body = row['text']
                        while True:
                            prompt = INSTRUCTION + '\n' + body
                            ids, _ = template.encode_oneturn(tokenizer, [
                                {'role': 'user', 'content': prompt},
                                {'role': 'assistant', 'content': ''}], None, None)
                            if len(ids) <= 2048:
                                break
                            if not body:
                                raise ValueError('Instruction exceeds input budget.')
                            body = body[:-1]
                        inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
                        torch.cuda.synchronize()
                        item_started = time.perf_counter()
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                            generated = model.generate(
                                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                do_sample=False, max_new_tokens=8, repetition_penalty=1.0,
                                use_cache=True, eos_token_id=template.get_stop_token_ids(tokenizer),
                                pad_token_id=tokenizer.pad_token_id)
                        torch.cuda.synchronize()
                        output_ids = generated[0, len(ids):]
                        raw = tokenizer.decode(output_ids, skip_special_tokens=True)
                        pred = int(raw.strip()) if raw.strip() in ('0', '1') else None
                        item = dict(sample_id=row['sample_id'], label=row['label'],
                                    prediction=pred, raw_output=raw, prompt=prompt,
                                    status='ok' if pred is not None else 'invalid',
                                    input_tokens=len(ids), output_tokens=len(output_ids),
                                    body_token_budget_truncated=(body != row['text']),
                                    generation_seconds=time.perf_counter() - item_started)
                        stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                        stream.flush()
                        predictions.append(item)
                        if index % 100 == 0 or index == len(rows):
                            print(f'Epoch {state.epoch}: validation {index}/{len(rows)}', flush=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                self.validation_seconds += elapsed
                result = metrics(predictions)
                record.update(result, validation_seconds=elapsed,
                              validation_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              validation_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                write_json(directory / 'metrics.json', record)
                history.append(record)
                write_json(root / 'validation_history.json', history)
                best = max(history, key=lambda item: item['macro_f1'])
                write_json(root / 'best_checkpoint.json', best)
                state.log_history.append(dict(step=state.global_step, epoch=state.epoch,
                                              validation_macro_f1=result['macro_f1'],
                                              validation_accuracy=result['accuracy']))
                import swanlab
                swanlab.log({'classification/macro_f1': result['macro_f1'],
                             'classification/accuracy': result['accuracy'],
                             'classification/invalid_rate': result['invalid_rate'],
                             'classification/validation_seconds': elapsed})
            except BaseException:
                (directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                raise
            finally:
                model.train(was_training)
                torch.cuda.reset_peak_memory_stats()

        def on_train_end(self, train_args, state, control, **kwargs):
            elapsed = time.perf_counter() - self.started
            write_json(root / 'timing.json', dict(
                trainer_interval_seconds=elapsed, classification_validation_seconds=self.validation_seconds,
                trainer_interval_minus_classification_validation_seconds=elapsed - self.validation_seconds,
                boundary='Trainer begin through end, includes checkpoint saves and logging; excludes initial model load. Subtracted classification intervals include generation, input encoding and prediction IO; not a pure CUDA kernel timing.',
                memory_boundary='CUDA peaks reset after model load and each validation; per segment, includes allocator reservations.',
                global_step=state.global_step, epoch=state.epoch))

    run_exp(config, callbacks=[ClassificationValidation()])
    if not args.smoke and len(history) != 3:
        raise RuntimeError(f'Expected 3 completed epoch validations, got {len(history)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/home/lgh/models/Qwen3-1.7B'))
    parser.add_argument('--data', type=Path, default=PROJECT / 'data/processed/v1-20261023')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.validation = args.data / 'validation.jsonl'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if args.worker:
        train_worker(args)
        return
    from llamafactory.hparams import get_train_args
    read_validation(args)
    frozen = json.loads((args.data / 'manifest.json').read_text(encoding='utf-8'))
    for name in ('train.jsonl', 'alpaca/train.json', 'alpaca/dataset_info.json'):
        if digest(args.data / name) != frozen['files'][name]:
            raise ValueError(f'Frozen data hash mismatch: {name}')
    if args.smoke:
        ids = json.loads((args.data / 'smoke_ids.json').read_text(encoding='utf-8'))
        if digest(args.data / 'smoke_ids.json') != frozen['files']['smoke_ids.json']:
            raise ValueError('Frozen smoke IDs changed.')
        train = {r['sample_id']: r for r in map(json.loads, (args.data / 'train.jsonl').read_text(encoding='utf-8').splitlines())}
        expected = [dict(instruction=INSTRUCTION, input=train[item]['text'], output=str(train[item]['label'])) for item in ids]
        workspace = FACTORY / 'project-workspaces/qwen06-experience/qwen06-smoke-20261023-235652/data'
        if json.loads((workspace / 'smoke.json').read_text(encoding='utf-8')) != expected:
            raise ValueError('Workspace smoke data differs from frozen training subset.')
    if not (args.model / 'config.json').is_file():
        raise FileNotFoundError(args.model)
    get_train_args(make_config(args, Path('/tmp/check-qwen17-not-created/adapter')))
    if args.check_only:
        print('CHECK PASSED. Config and frozen hashes only; no model loaded or training started.')
        return
    purpose = 'smoke' if args.smoke else 'full'
    run_id = f'qwen17-{purpose}-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    work = FACTORY / 'project-workspaces/qwen17-profile' / run_id
    results = PROJECT / 'results/qwen17-profile' / run_id
    work.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    write_json(results / 'status.json', dict(status='preparing', run_id=run_id,
                                            training_workspace=str(work)))
    config = make_config(args, work / 'adapter')
    write_json(work / 'train.json', config)
    write_json(results / 'training_config.json', config)
    # Archive the actual training entry and its metric implementation in tool workspace.
    for name in ('profile_qwen17.py', 'evaluate_qwen06.py'):
        shutil.copyfile(Path(__file__).parent / name, work / name)
    write_json(results / 'manifest.json', dict(
        run_id=run_id, purpose=purpose, training_workspace=str(work),
        config_sha256=digest(work / 'train.json'),
        script_sha256=digest(work / 'profile_qwen17.py'),
        metric_script_sha256=digest(work / 'evaluate_qwen06.py'),
        validation_sha256=digest(args.validation), data_manifest_sha256=digest(args.data / 'manifest.json'),
        model_files={p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()},
        downloaded_model_revision=None,
        factory_commit=subprocess.check_output(['git', '-C', str(FACTORY), 'rev-parse', 'HEAD'], text=True).strip(),
        packages={n: metadata.version(n) for n in ('torch', 'transformers', 'peft', 'llamafactory', 'swanlab')},
        ai_assistance='AI engineering implementation; original confirmed rank8 baseline, not a new four-parameter design. Student runs and interprets results.'))
    command = [sys.executable, '-u', str(work / 'profile_qwen17.py'), '--worker', str(work / 'train.json'),
               '--data', str(args.data), '--model', str(args.model)]
    if args.smoke:
        command.append('--smoke')
    print('TRAINING DIRECTORY:', work, '\nRESULT INDEX:', results, flush=True)
    try:
        code = subprocess.call([sys.executable, '-u', str(Path(__file__).parent / 'record_run_resources.py'),
                                '--output', str(work / 'resource-record'), '--'] + command, cwd=FACTORY)
    except BaseException as error:
        write_json(results / 'status.json', dict(status='failed', error=repr(error),
                                                training_workspace=str(work)))
        raise
    write_json(results / 'status.json', dict(status='completed' if code == 0 else 'failed',
                                            exit_code=code, training_workspace=str(work)))
    for name in ('timing.json', 'parameters.json', 'validation_history.json', 'best_checkpoint.json', 'smoke_check.json'):
        if (work / name).exists():
            shutil.copyfile(work / name, results / name)
    if (work / 'resource-record/status.json').exists():
        shutil.copyfile(work / 'resource-record/status.json', results / 'resources_summary.json')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
