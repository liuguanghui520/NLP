import math
from pathlib import Path
import statistics

from src.common.io import load, rows, save, save_rows, digest
from src.adapters.predictions import adapt
from src.evaluation.metrics import evaluate


def import_one(root, paths, source, metadata, expected, source_key, relative_source):
    records = adapt(source, **metadata)
    result = evaluate(records, expected)
    if not result['complete']:
        raise ValueError(f'{metadata["run_id"]}: {result["errors"]}')
    run_id = metadata['run_id']
    save_rows(root/'results/predictions'/f'{run_id}.jsonl', records)
    save(root/'results/metrics'/f'{run_id}.json', result)
    return dict(**metadata, source_key=source_key, relative_source=str(relative_source),
                source_sha256=digest(source), phase='final_test' if metadata['split']=='test' else 'formal_validation',
                status='audited', n=len(records), metrics=result['metrics'])


def import_all(root, paths):
    data = Path(paths['data'])
    expected = {s: rows(data/f'{s}.jsonl') for s in ['train', 'validation', 'test']}
    ids = load(data/'split_ids.json')
    manifest = load(data/'manifest.json')
    for s, n in [('train', 6210), ('validation', 776), ('test', 777)]:
        assert len(expected[s]) == n
        assert [r['sample_id'] for r in expected[s]] == ids[s]
        assert digest(data/f'{s}.jsonl') == manifest['files'][f'{s}.jsonl']
    prompt = Path(paths['prompt_delivery'])
    assert load(prompt/'data_manifest/split_ids.json') == ids
    # Different JSON serialization is allowed; reconstruct the exact prompt-side
    # row content hashes from its per-sample rendered-input hashes during restore.
    index, audit, benchmarks, searches = [], [], [], []
    for method, key in [('bert', 'bert_results'), ('tfidf', 'baseline_results')]:
        base = Path(paths[key])
        winner = load(base/'frozen_winner.json')
        config_id = winner['config']['config_id']
        comparison = load(base/'validation_comparison.json')
        assert len(comparison) == (8 if method=='bert' else 4)
        config_order = [r['config_id'] for r in comparison]
        chosen = sorted([r for r in comparison if r['config_id'] != 'partial'],
                        key=lambda r: (-r['macro_f1'], r['trainable_parameters'], config_order.index(r['config_id'])))[0]
        assert chosen['config_id'] == config_id
        for c in comparison:
            run = base/f'{method}-{c["config_id"]}-seed42'
            epochs = [evaluate(adapt(p, method, run.name, c['config_id'], 42, 'validation',
                                    'MacBERT' if method=='bert' else 'TF-IDF', None), expected['validation'])
                      for p in sorted(run.glob('validation/epoch-*/predictions.jsonl'))]
            assert all(e['complete'] for e in epochs)
            actual = rows(run/'epoch_metrics.jsonl')
            assert len(epochs) == len(actual)
            for e, record in zip(epochs, actual):
                assert math.isclose(e['metrics']['macro_f1'], record['macro_f1'], abs_tol=1e-12)
            best_epoch = max(range(len(actual)), key=lambda i: actual[i]['macro_f1'])+1
            assert load(run/'selection.json')['epoch'] == best_epoch
            searches.append(dict(method=method, **c))
        for seed in [42, 43, 44]:
            run_id = f'{method}-{config_id}-seed{seed}'
            run = base/run_id
            config = load(run/'config.json')
            for k, v in winner['config'].items():
                assert config[k] == v
            selection = load(run/'selection.json')
            model = Path(selection['model'])
            weight = model/'model.safetensors' if model.is_dir() else model
            assert digest(weight) == selection['model_sha256']
            source = run/'test-isolated/predictions.jsonl'
            metadata = dict(method=method, run_id=run_id, config_id=config_id, seed=seed, split='test',
                            model_id='hfl/chinese-macbert-base' if method=='bert' else 'TF-IDF+LogisticRegression',
                            checkpoint_id=selection['model_sha256'])
            record = import_one(root, paths, source, metadata, expected['test'], key, source.relative_to(base))
            original = load(run/'test-isolated/metrics.json')
            assert all(math.isclose(record['metrics'][k], original[k], abs_tol=1e-12)
                       for k in ['macro_f1', 'accuracy', 'weighted_f1'])
            latency = rows(run/'test-isolated/latency.jsonl')
            latency_summary = load(run/'test-isolated/latency_summary.json')
            assert [r['sample_id'] for r in latency] == load(data/'latency_ids.json')
            assert latency_summary['warmup_ids'] == load(data/'smoke_ids.json')
            assert len(latency)==200 and all(r['checkpoint_sha256']==selection['model_sha256'] for r in latency)
            import numpy as np
            times = [r['latency_ms'] for r in latency]
            assert math.isclose(float(np.mean(times)), latency_summary['mean_ms'], abs_tol=1e-9)
            for p in [50, 95]:
                assert math.isclose(float(np.percentile(times, p, method='linear')), latency_summary[f'p{p}_ms'], abs_tol=1e-9)
            benchmarks.append(dict(run_id=run_id, method=method, seed=seed, **latency_summary,
                                   inference_peak_memory=None, missing_reason='Not sampled in original baseline test',
                                   storage=load(run/'test-isolated/storage.json')))
            index.append(record)
            audit.append(dict(run_id=run_id, model_hash_verified=True, truth_ids_verified=True,
                              metrics_recomputed=True, latency_ids_verified=True, selected_epoch=selection['epoch']))
    for mode in ['qwen_zero_shot', 'qwen_few_shot']:
        run = prompt/'run_evidence'/f'C_{mode}_fixed_prompt_v1_noseed_20261024T154625Z'
        metadata = dict(method=mode, run_id=run.name, config_id='fixed_prompt_v1', seed=None, split='validation',
                        model_id='Qwen/Qwen3-1.7B', checkpoint_id=None)
        index.append(import_one(root, paths, run/'predictions_validation.jsonl', metadata,
                                expected['validation'], 'prompt_delivery', (run/'predictions_validation.jsonl').relative_to(prompt)))
    # Keep original LoRA lifecycle records intact. Final tests receive new IDs.
    freeze = load(Path(paths['lora_freeze'])/'protocol.json')
    private = load(Path(paths['lora_freeze'])/'local_manifest.json')
    modeldir = Path(paths['qwen_model'])
    for name, expected_sha in private['base_model_files'].items():
        assert digest(modeldir/name) == expected_sha
    for item in private['adapters']:
        adapter = Path(paths['lora_adapters'][str(item['seed'])])
        assert digest(adapter/'adapter_model.safetensors') == item['adapter_sha256']
        assert digest(adapter/'adapter_config.json') == item['adapter_config_sha256']
        source = adapter.parents[1]/'validation-step-1167/predictions.jsonl'
        metadata = dict(method='qwen_lora',run_id=f'qwen_lora-T01-seed{item["seed"]}-validation',
                        config_id='T01',seed=item['seed'],split='validation',model_id='Qwen/Qwen3-1.7B',
                        checkpoint_id=item['adapter_sha256'])
        index.append(import_one(root, paths, source, metadata, expected['validation'],
                                f'lora_adapter_{item["seed"]}', '../../validation-step-1167/predictions.jsonl'))
    save(root/'artifacts/lora-lock.json', dict(protocol_sha256=digest(Path(paths['lora_freeze'])/'protocol.json'),
        base_model_files=private['base_model_files'], adapters=[{k:v for k,v in x.items() if k!='checkpoint'} for x in private['adapters']]))
    existing = rows(root/'results/run_index.jsonl') if (root/'results/run_index.jsonl').exists() else []
    names = {r['run_id'] for r in index}
    index.extend(r for r in existing if r['run_id'] not in names)
    save_rows(root/'results/run_index.jsonl', index)
    save(root/'reports/audits/baselines.json', audit)
    save(root/'results/summaries/validation_searches.json', searches)
    save(root/'results/benchmarks/baselines.json', benchmarks)
    save(root/'artifacts/final-acceptance.json', dict(status='baseline_audited_qwen_pending',
         data_files=manifest['files'], baseline_runs=audit, prompt_formats='Preserved complete-method comparison',
         lora_protocol_sha256=digest(Path(paths['lora_freeze'])/'protocol.json'),
         source_history_note='Historical train-stage pending statuses preserved'))
    return index
