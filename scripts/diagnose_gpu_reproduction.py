"""Small validation-only cross-GPU diagnostic; never edits formal results."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

PROFILES = ('original', 'controlled_flash', 'controlled_math')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def choose_samples(rows):
    """Four examples of each class, independent of prediction correctness."""
    generator = random.Random(67)
    chosen = []
    for label in (0, 1):
        candidates = sorted((r for r in rows if r['label'] == label), key=lambda r: r['sample_id'])
        chosen.extend(generator.sample(candidates, 4))
    return chosen


def compare(reference, current):
    result = dict(status='compared', profiles={}, diagnostic_only=True,
                  formal_acceptance_changed=False)
    for profile in PROFILES:
        left, right = reference / (profile + '.json'), current / (profile + '.json')
        if not left.exists() or not right.exists():
            result['profiles'][profile] = dict(status='missing_profile')
            continue
        a, b = load(left), load(right)
        if a.get('status') != 'passed' or b.get('status') != 'passed':
            result['profiles'][profile] = dict(status='profile_failed')
            continue
        ar = {(x['method'], x['sample_id']): x for x in a['samples']}
        br = {(x['method'], x['sample_id']): x for x in b['samples']}
        if set(ar) != set(br):
            result['profiles'][profile] = dict(status='sample_sets_differ')
            continue
        differences = []
        for key in ar:
            x, y = ar[key], br[key]
            same_input = x['input_ids'] == y['input_ids']
            first_difference = next((i for i, (u, v) in enumerate(zip(x['generated_ids'], y['generated_ids']))
                                     if u != v), None)
            if first_difference is None and len(x['generated_ids']) != len(y['generated_ids']):
                first_difference = min(len(x['generated_ids']), len(y['generated_ids']))
            differences.append(dict(method=key[0], sample_id=key[1], same_input_ids=same_input,
                same_generated_ids=x['generated_ids'] == y['generated_ids'],
                first_different_generated_step=first_difference,
                same_first_step_scores=x['steps'][0]['scores_sha256'] == y['steps'][0]['scores_sha256'],
                reference_first_step=x['steps'][0], current_first_step=y['steps'][0]))
        result['profiles'][profile] = dict(status='compared', n=len(ar),
            input_differences=sum(not r['same_input_ids'] for r in differences),
            output_differences=sum(not r['same_generated_ids'] for r in differences),
            first_step_score_differences=sum(not r['same_first_step_scores'] for r in differences),
            same_model_files=a['model_files'] == b['model_files'],
            same_buffers=a['runtime']['buffer_fingerprints'] == b['runtime']['buffer_fingerprints'],
            reference_runtime=a['runtime'], current_runtime=b['runtime'], samples=differences)
    return result


def child(args):
    root = args.work_root.resolve()
    sys.path.insert(0, str(root))
    # Works both as a repository script and as a standalone handoff bundle.
    try:
        from runtime_fingerprint import runtime_snapshot, tensor_sha256
    except ImportError:
        # Load our helper even when the old working tree predates its addition.
        import importlib.util
        helper = Path(__file__).resolve().parents[1] / 'src/common/runtime.py'
        spec = importlib.util.spec_from_file_location('_diagnostic_runtime', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runtime_snapshot, tensor_sha256 = module.runtime_snapshot, module.tensor_sha256
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.demo.backend import prompt_registry
    from src.adapters.prompts import render
    from src.evaluation.metrics import parse
    paths = load(args.paths or root / 'configs/paths.local.json')
    data = Path(paths['data'])
    frozen = load(root / 'artifacts/evaluation-freeze.json')['data_files']
    for name in ('train.jsonl', 'validation.jsonl'):
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != frozen[name]:
            raise ValueError('Frozen validation/training input mismatch: ' + name)
    lock = load(root / 'artifacts/lora-lock.json')['base_model_files']
    for name, digest in lock.items():
        print('Checking locked model file:', name, flush=True)
        if hashlib.sha256((Path(paths['qwen_model']) / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Model mismatch: ' + name)
    if str(torch.__version__) != '2.14.1+cu130' or not torch.cuda.is_bf16_supported():
        raise ValueError('Use the validated torch2.14.1+cu130/BF16 environment')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.child != 'original':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(paths['qwen_model'], local_files_only=True)
    registry = prompt_registry(root, data, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(paths['qwen_model'], local_files_only=True,
        dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation='sdpa').cuda().eval()
    print('Model loaded; collecting validation evidence', flush=True)
    records = choose_samples([json.loads(line) for line in (data / 'validation.jsonl').read_text(encoding='utf-8').splitlines()])
    from torch.nn.attention import sdpa_kernel, SDPBackend
    kernel = {'controlled_flash': SDPBackend.FLASH_ATTENTION, 'controlled_math': SDPBackend.MATH}
    context = sdpa_kernel([kernel[args.child]]) if args.child in kernel else nullcontext()
    results = []
    with context:
        runtime = runtime_snapshot(torch, model)
        for method in ('qwen_zero_shot', 'qwen_few_shot'):
            for row in records:
                print('Starting sample:', method, row['sample_id'], flush=True)
                prompt, body, count = render(tokenizer, registry['modes'][method], row['text'])
                encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
                with torch.inference_mode():
                    output = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=8,
                        use_cache=True, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        return_dict_in_generate=True, output_scores=True)
                torch.cuda.synchronize()
                ids = output.sequences[0, count:].tolist()
                steps = []
                for index, score in enumerate(output.scores):
                    values, tokens = torch.topk(score[0].float(), 10)
                    steps.append(dict(step=index, scores_sha256=tensor_sha256(score, torch),
                        score_dtype=str(score.dtype), selected_token=ids[index],
                        top10_ids=tokens.tolist(), top10_scores=values.tolist(),
                        top1_top2_margin=float(values[0] - values[1])))
                raw = tokenizer.decode(ids, skip_special_tokens=True)
                results.append(dict(method=method, sample_id=row['sample_id'], true_label=row['label'],
                    input_ids=encoded.input_ids[0].tolist(), rendered_input_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    generated_ids=ids, raw_output=raw, parsed_label=parse(raw), steps=steps))
                print(args.child, method, len(results), flush=True)
    save(args.output / (args.child + '.json'), dict(status='passed', profile=args.child,
        model_files=lock, runtime=runtime, samples=results, split='validation', selection_seed=67,
        diagnostic_only=True, precision='bfloat16', formal_results_modified=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path)
    parser.add_argument('--paths', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=['original'],
                        help='Default original only; controlled configurations are optional interventions')
    parser.add_argument('--profile-timeout', type=int, default=600,
                        help='Seconds per subprocess, including file verification and model loading')
    parser.add_argument('--child', choices=PROFILES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.work_root or not args.output:
        parser.error('--work-root and a NEW --output directory are required')
    if args.profile_timeout <= 0:
        parser.error('--profile-timeout must be positive')
    args.output = args.output.resolve()
    if args.child:
        try:
            child(args)
        except BaseException:
            save(args.output / (args.child + '.json'), dict(status='failed', traceback=traceback.format_exc()))
            raise
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for profile in args.profiles:
        env = os.environ.copy()
        if profile != 'original':
            env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        command = [sys.executable, str(Path(__file__).resolve()), '--work-root', str(args.work_root.resolve()),
                   '--output', str(args.output), '--child', profile]
        if args.paths:
            command.extend(['--paths', str(args.paths.resolve())])
        with (args.output / (profile + '.log')).open('w', encoding='utf-8') as log:
            try:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.profile_timeout)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 124
                prior = args.output / (profile + '.json')
                if prior.exists():
                    prior.rename(args.output / (profile + '.before-timeout.json'))
                save(prior, dict(status='timeout', timeout_seconds=args.profile_timeout,
                                 diagnostic_only=True))
        results.append(dict(profile=profile, returncode=code, command=command))
        print(profile, 'completed' if code == 0 else 'failed; see log', flush=True)
    save(args.output / 'status.json', dict(status='completed' if all(r['returncode'] == 0 for r in results) else 'failed',
        timestamp=datetime.now(timezone.utc).isoformat(), runs=results, diagnostic_only=True,
        training=False, formal_inference=False, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if args.reference:
        save(args.output / 'cross-gpu-comparison.json', compare(args.reference, args.output))
    return 0 if all(r['returncode'] == 0 for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
