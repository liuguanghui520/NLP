"""Compare the original and smoke-trained Qwen on frozen validation only.

Run manually in WSL. Workers run serially and never train or overwrite a run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid


INSTRUCTION = '判断评论整体情感，仅输出0表示负面、1表示正面。'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def metrics(records):
    """Invalid predictions count as false negatives, with their own matrix column."""
    matrix = [[0, 0, 0], [0, 0, 0]]
    for row in records:
        pred = row['prediction']
        matrix[row['label']][pred if pred in (0, 1) else 2] += 1
    classes = {}
    for label in (0, 1):
        tp = matrix[label][label]
        fp = matrix[1 - label][label]
        fn = sum(matrix[label]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        classes[str(label)] = dict(precision=precision, recall=recall, f1=f1,
                                   support=sum(matrix[label]))
    count = len(records)
    return dict(count=count, accuracy=(matrix[0][0] + matrix[1][1]) / count,
                macro_f1=sum(item['f1'] for item in classes.values()) / 2,
                invalid_rate=(matrix[0][2] + matrix[1][2]) / count,
                per_class=classes, confusion_matrix=matrix,
                matrix_rows=[0, 1], matrix_columns=[0, 1, 'invalid'])


def infer_config(args, variant):
    config = dict(model_name_or_path=str(args.model), template='qwen3',
                  finetuning_type='lora', infer_backend='huggingface',
                  infer_dtype='bfloat16', enable_thinking=False,
                  trust_remote_code=False, do_sample=False, max_new_tokens=8,
                  repetition_penalty=1.0)
    if variant == 'lora':
        config['adapter_name_or_path'] = str(args.adapter)
    return config


def read_validation(args):
    rows = [json.loads(line) for line in args.validation.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 776 or len({row['sample_id'] for row in rows}) != 776:
        raise ValueError('Expected exactly 776 unique frozen validation records.')
    if any(row['split'] != 'validation' or row['label'] not in (0, 1)
           or not isinstance(row['text'], str) for row in rows):
        raise ValueError('Invalid validation data or split.')
    manifest = json.loads((args.validation.parent / 'manifest.json').read_text(encoding='utf-8'))
    if digest(args.validation) != manifest['files']['validation.jsonl']:
        raise ValueError('Validation SHA256 differs from frozen manifest.')
    return rows, manifest


def worker(args):
    import torch
    from llamafactory.chat import ChatModel

    root = args.run_dir / args.worker
    root.mkdir(exist_ok=False)
    state = dict(status='running', variant=args.worker, started_utc=datetime.now(timezone.utc).isoformat())
    write_json(root / 'status.json', state)
    model = None
    try:
        rows, _ = read_validation(args)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('This exercise requires a CUDA GPU with BF16 support.')
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = infer_config(args, args.worker)
        write_json(root / 'inference_config.json', config)
        model = ChatModel(config)
        if next(model.engine.model.parameters()).device.type != 'cuda':
            raise RuntimeError('Model is not on CUDA; refusing an unrecorded CPU fallback.')
        torch.cuda.reset_peak_memory_stats()
        records = []
        with (root / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
            for index, row in enumerate(rows, 1):
                body = row['text']
                # Encode with the installed Factory template, preserving instruction.
                while True:
                    prompt = INSTRUCTION + '\n' + body
                    messages = [{'role': 'user', 'content': prompt}]
                    prompt_ids, _ = model.engine.template.encode_oneturn(
                        model.engine.tokenizer,
                        messages + [{'role': 'assistant', 'content': ''}], None, None)
                    if len(prompt_ids) <= 2048:
                        break
                    if not body:
                        raise ValueError('Instruction alone exceeds input token budget.')
                    body = body[:-1]
                torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    response, = model.chat(messages, do_sample=False, max_new_tokens=8,
                                           repetition_penalty=1.0)
                    torch.cuda.synchronize()
                    raw = response.response_text
                    parsed = int(raw.strip()) if raw.strip() in ('0', '1') else None
                    record = dict(sample_id=row['sample_id'], label=row['label'],
                                  prompt=prompt, raw_output=raw, prediction=parsed,
                                  status='ok' if parsed is not None else 'invalid',
                                  input_tokens=response.prompt_length,
                                  output_tokens=response.response_length,
                                  finish_reason=response.finish_reason,
                                  body_token_budget_truncated=(body != row['text']),
                                  generation_call_seconds=time.perf_counter() - started)
                except BaseException as error:
                    stream.write(json.dumps(dict(sample_id=row['sample_id'], label=row['label'],
                                                 prompt=prompt, status='failed',
                                                 error=repr(error)), ensure_ascii=False) + '\n')
                    stream.flush()
                    raise
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                records.append(record)
                if index % 25 == 0 or index == len(rows):
                    print(f'{args.worker}: {index}/{len(rows)}', flush=True)
        result = metrics(records)
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      memory_boundary='After model load; generation only, not training peak',
                      timing_boundary='Per model.chat call with CUDA sync; excludes manual prompt encoding; not formal latency benchmark')
        write_json(root / 'metrics.json', result)
        state.update(status='completed', count=len(records))
        print(json.dumps({key: result[key] for key in ('count', 'accuracy', 'macro_f1', 'invalid_rate')}, indent=2), flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        (root / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        state['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(root / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/home/lgh/models/Qwen3-0.6B'))
    parser.add_argument('--adapter', type=Path, default=Path('/home/lgh/LLaMA-Factory/saves/Qwen3-0.6B-Base/lora/train_2026-10-24-00-00-09'))
    parser.add_argument('--validation', type=Path, default=Path(__file__).resolve().parents[1] / 'data/processed/v1-20261023/validation.jsonl')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[1] / 'results/qwen06-validation')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', choices=['base', 'lora'], help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if args.worker:
        worker(args)
        return
    rows, _ = read_validation(args)
    from llamafactory.hparams import get_infer_args
    for variant in ('base', 'lora'):
        get_infer_args(infer_config(args, variant))
    for path in (args.model / 'config.json', args.adapter / 'adapter_config.json',
                 args.adapter / 'adapter_model.safetensors'):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.check_only:
        print('CHECK PASSED: 776 validation records; both inference configurations parse. No weights loaded or predictions generated.')
        return
    run_id = 'qwen06-validation-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    root = args.output_root / run_id
    root.mkdir(parents=True, exist_ok=False)
    state = dict(run_id=run_id, status='running', purpose='0.6B tool exercise; not formal 1.7B results',
                 started_utc=datetime.now(timezone.utc).isoformat())
    write_json(root / 'status.json', state)
    print('RUN DIRECTORY:', root, flush=True)
    try:
        evidence = dict(script_sha256=digest(Path(__file__)), validation_sha256=digest(args.validation),
                        data_manifest_sha256=digest(args.validation.parent / 'manifest.json'),
                        sample_ids=[row['sample_id'] for row in rows],
                        instruction=INSTRUCTION, input_budget=2048, batch_size=1, seed=42,
                        packages={name: metadata.version(name) for name in ('torch', 'transformers', 'peft', 'llamafactory')},
                        command=sys.argv, model_revision=None,
                        model_revision_note='Downloaded revision unknown; local files pinned by SHA256',
                        model_files={path.name: digest(path) for path in sorted(args.model.iterdir()) if path.is_file()},
                        adapter_files={name: digest(args.adapter / name) for name in ('adapter_config.json', 'adapter_model.safetensors')},
                        ai_assistance='AI implemented and checked engineering evaluation; student executes and interprets results.')
        evidence['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
        write_json(root / 'manifest.json', evidence)
        for variant in ('base', 'lora'):
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', variant,
                       '--run-dir', str(root), '--model', str(args.model), '--adapter', str(args.adapter),
                       '--validation', str(args.validation)]
            with (root / f'{variant}.log').open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace')
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
                raise RuntimeError(f'{variant} exited with code {code}; logs preserved. Retry creates a new run.')
        results = {variant: json.loads((root / variant / 'metrics.json').read_text(encoding='utf-8')) for variant in ('base', 'lora')}
        write_json(root / 'comparison.json', results)
        print('\nvariant\tcount\taccuracy\tmacro_f1\tinvalid_rate', flush=True)
        for variant, result in results.items():
            print(f"{variant}\t{result['count']}\t{result['accuracy']:.6f}\t{result['macro_f1']:.6f}\t{result['invalid_rate']:.6f}", flush=True)
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        (root / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        state['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(root / 'status.json', state)


if __name__ == '__main__':
    main()
