"""Explicit FP32 GPU validation probe; no training or formal result replacement."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    from src.common.io import load, save, digest, rows
    from src.common.runtime import runtime_snapshot
    from src.demo.backend import prompt_registry
    from src.adapters.prompts import render, text_sha
    from src.evaluation.metrics import parse
    from scripts.diagnose_gpu_reproduction import choose_samples
    status = dict(status='running', precision='float32', split='validation', training=False,
                  formal_results_modified=False, cpu_offload=False, quantization=False, predictions_completed=0)
    save(output / 'status.json', status)
    started = time.perf_counter()
    records = []
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        paths = load(root / 'configs/paths.local.json')
        data = Path(paths['data'])
        freeze = load(root / 'artifacts/evaluation-freeze.json')['data_files']
        for name in ('train.jsonl', 'validation.jsonl'):
            if digest(data / name) != freeze[name]:
                raise ValueError('Frozen data mismatch: ' + name)
        lock = load(root / 'artifacts/lora-lock.json')['base_model_files']
        for name, expected in lock.items():
            print('Verifying:', name, flush=True)
            if digest(Path(paths['qwen_model']) / name) != expected:
                raise ValueError('Frozen model mismatch: ' + name)
        if str(torch.__version__) != '2.14.1+cu130' or not torch.cuda.is_available():
            raise ValueError('Use the verified torch2.14.1+cu130 CUDA environment')
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        before = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.used,memory.total',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        save(output / 'before-loading.json', dict(nvidia_smi=before.stdout, runtime=runtime_snapshot(torch)))
        tokenizer = AutoTokenizer.from_pretrained(paths['qwen_model'], local_files_only=True)
        registry = prompt_registry(root, data, tokenizer)
        samples = choose_samples(rows(data / 'validation.jsonl'))
        save(output / 'sample-ids.json', [r['sample_id'] for r in samples])
        torch.cuda.reset_peak_memory_stats()
        model_started = time.perf_counter()
        print('Loading FP32 directly onto CUDA; no CPU offload', flush=True)
        model = AutoModelForCausalLM.from_pretrained(paths['qwen_model'], local_files_only=True,
            dtype=torch.float32, device_map={'': 'cuda:0'}, low_cpu_mem_usage=True,
            attn_implementation='sdpa').eval()
        torch.cuda.synchronize()
        before_cleanup = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                              free=torch.cuda.mem_get_info()[0])
        # Streaming BF16->FP32 loading can retain large, unused staging blocks.
        # Release allocator cache, never parameters or model precision.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        save(output / 'allocation-cleanup.json', dict(before=before_cleanup,
            after=dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                       free=torch.cuda.mem_get_info()[0])))
        load_seconds = time.perf_counter() - model_started
        if not all(p.dtype == torch.float32 and p.device.type == 'cuda' for p in model.parameters()):
            raise ValueError('FP32/CUDA-only invariant failed')
        save(output / 'config.json', dict(runtime=runtime_snapshot(torch, model), precision='float32',
            load_seconds=load_seconds, total_parameters=sum(p.numel() for p in model.parameters()),
            parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
            source_sha256=digest(__file__), model_files=lock, selection_seed=67,
            do_sample=False, max_new_tokens=8, tf32=False, cpu_offload=False,
            diagnostic_only=True, formal_results_modified=False))
        for method in ('qwen_zero_shot', 'qwen_few_shot'):
            for row in samples:
                prompt, body, count = render(tokenizer, registry['modes'][method], row['text'])
                torch.cuda.synchronize()
                begin = time.perf_counter()
                encoded = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    generated = model.generate(**encoded, do_sample=False, num_beams=1,
                        max_new_tokens=8, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                ids = generated[0, count:].tolist()
                raw = tokenizer.decode(ids, skip_special_tokens=True)
                record = dict(method=method, sample_id=row['sample_id'], true_label=row['label'],
                    raw_output=raw, parsed_label=parse(raw), generated_ids=ids,
                    input_tokens=count, rendered_input_sha256=text_sha(prompt),
                    elapsed_ms=elapsed * 1000, split='validation', precision='float32')
                records.append(record)
                with (output / 'predictions.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(method, len(records), '/16', flush=True)
        status.update(status='completed', predictions_completed=len(records), load_seconds=load_seconds,
                      wall_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      timing_scope='validation probe including cold predictions; not formal 20/200 benchmark')
        save(output / 'status.json', status)
        return 0
    except BaseException as error:
        status.update(status='oom' if 'OutOfMemory' in type(error).__name__ else 'failed',
                      error=str(error), traceback=traceback.format_exc(),
                      predictions_completed=len(records), wall_seconds=time.perf_counter() - started)
        if 'torch' in locals() and torch.cuda.is_available():
            status.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(output / 'status.json', status)
        print(status['traceback'], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
