"""Freeze the already selected T01 adapters and validated inference protocol.

Hashes existing files only. Does not load model weights or evaluate test data.
"""
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import uuid

from evaluate_qwen06 import INSTRUCTION, digest, write_json


PROJECT = Path(__file__).resolve().parents[1]


def main():
    batch_path = PROJECT / 'results/qwen17-comparisons/qwen17-comparisons-20261026-200544-750ecb/batch.json'
    batch = json.loads(batch_path.read_text(encoding='utf-8'))
    if batch['status'] != 'completed':
        raise ValueError('Seed repeats are not completed.')
    data = Path(batch['data_directory'])
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    entries = [dict(batch['existing_seed42'], seed=42)] + batch['jobs']
    selected = []
    original = None
    ignore = {'seed', 'data_seed', 'output_dir', 'swanlab_run_name', 'swanlab_logdir'}
    for entry in entries:
        root = Path(entry['training_directory'])
        config = json.loads((root / 'train.json').read_text(encoding='utf-8'))
        if digest(root / 'train.json') != entry['config_sha256']:
            raise ValueError('Recorded configuration changed.')
        if original is None:
            original = config
        elif {k:v for k,v in config.items() if k not in ignore} != {k:v for k,v in original.items() if k not in ignore}:
            raise ValueError('Three seeds use different hyperparameters.')
        checkpoint = Path(entry['best_checkpoint'])
        adapter_hash = digest(checkpoint / 'adapter_model.safetensors')
        if adapter_hash != entry['best_adapter_sha256']:
            raise ValueError('Adapter changed since validation audit.')
        selected.append(dict(seed=entry['seed'], checkpoint=str(checkpoint),
                             adapter_sha256=adapter_hash,
                             adapter_config_sha256=digest(checkpoint / 'adapter_config.json'),
                             source_config_sha256=entry['config_sha256']))
    model = Path(original['model_name_or_path'])
    for name, expected in batch['baseline_model_files'].items():
        if digest(model / name) != expected:
            raise ValueError(f'Base model changed: {name}')
    for package, version in batch['packages'].items():
        if metadata.version(package) != version:
            raise ValueError(f'Environment changed: {package}')
    protocol = dict(
        model_repository='Qwen/Qwen3-1.7B', selected_configuration='T01',
        selection_metric='validation_macro_f1', seeds=[42,43,44],
        seed_policy='Evaluate all three seeds separately; report individual scores, mean and sample standard deviation. Do not select a seed by score.',
        adapter=dict(lora_rank=8,lora_alpha=16,lora_target='q_proj,k_proj,v_proj,o_proj',lora_dropout=0.05),
        training=dict(learning_rate=0.0001,warmup_ratio=0.1,lr_scheduler_type='linear',num_train_epochs=3),
        inference=dict(template='qwen3',enable_thinking=False,infer_dtype='bfloat16',quantized=False,
                       do_sample=False,max_new_tokens=8,repetition_penalty=1.0,batch_size=1,
                       instruction=INSTRUCTION,message_format='instruction + newline + comment',
                       history=False,few_shot_examples=0,input_token_budget=2048,
                       truncation='Use frozen 512-character prefix; if encoded prompt exceeds2048 tokens remove comment characters from tail, preserving instruction.',
                       decoding='Decode generated continuation only, skip_special_tokens=True; retain raw decoded output before strip.',
                       stop_tokens='Installed Factory qwen3 template.get_stop_token_ids(tokenizer)'),
        metrics=dict(labels={'0':'negative','1':'positive'},
                     parser='Only raw_output.strip() exactly0 or1 is valid.',
                     invalid='Count as incorrect, include in denominator and true-class FN; report invalid rate and invalid confusion column.',
                     primary='macro_f1',additional=['accuracy','per_class_precision_recall_f1','confusion_matrix','invalid_rate'],
                     zero_division=0,std='sample, denominator n-1'),
        data=dict(version='v1-20261023',train_count=6210,validation_count=776,test_count=777,
                  file_hashes={name:manifest['files'][name] for name in ('train.jsonl','validation.jsonl','test.jsonl','latency_ids.json')}),
        evidence=dict(search_batch=batch['parent_batch_id'],repeat_batch=batch['run_id'],
                      failed_repeat_batch=batch.get('retry_of')),
        boundaries=dict(test_executed_by_this_script=False,other_routes_frozen=False,
                        model_revision='Original download revision unknown; local base files pinned by SHA256 in private manifest.'),
        ai_assistance='Engineering transcription of already confirmed T01 and existing validation inference rules; no new design, model inference or result analysis.')
    freeze_id='qwen17-lora-freeze-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
    output=PROJECT / 'results/qwen17-freeze' / freeze_id
    output.mkdir(parents=True,exist_ok=False)
    write_json(output / 'protocol.json',protocol)
    write_json(output / 'local_manifest.json',dict(
        freeze_id=freeze_id,created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(output / 'protocol.json'),base_model_path=str(model),
        base_model_files=batch['baseline_model_files'],adapters=selected,packages=batch['packages'],
        factory_commit=batch['factory_commit'],validated_training_entry_sha256=batch['scripts']['profile_qwen17.py'],
        data_manifest_sha256=digest(data / 'manifest.json'),script_sha256=digest(Path(__file__))))
    print('FROZEN:',output)
    print('SEEDS:',[item['seed'] for item in selected])
    print('Protocol SHA256:',digest(output / 'protocol.json'))
    print('No model loaded; no test data read or inference performed.')


if __name__=='__main__':
    main()
