"""Prepare an isolated 20-record LoRA software smoke run; never starts training."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess

import yaml


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--factory', required=True, type=Path)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    args = parser.parse_args()
    run_id = 'qwen06-smoke-' + datetime.now().strftime('%Y%m%d-%H%M%S')
    root = args.workspace / run_id
    root.mkdir(parents=True, exist_ok=False)
    data = root / 'data'
    data.mkdir()
    ids = json.loads((args.data / 'smoke_ids.json').read_text(encoding='utf-8'))
    rows = {r['sample_id']: r for r in map(json.loads, (args.data / 'train.jsonl').read_text(encoding='utf-8').splitlines())}
    selected = [rows[i] for i in ids]
    assert len(ids) == len(set(ids)) == 20
    assert Counter(r['label'] for r in selected) == {0: 10, 1: 10}
    instruction = '判断评论整体情感，仅输出0表示负面、1表示正面。'
    examples = [dict(instruction=instruction, input=r['text'], output=str(r['label'])) for r in selected]
    (data / 'smoke.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (data / 'dataset_info.json').write_text(json.dumps({'sentiment_smoke': {'file_name': 'smoke.json', 'columns': {'prompt': 'instruction', 'query': 'input', 'response': 'output'}}}, indent=2) + '\n', encoding='utf-8')
    (data / 'smoke_ids.json').write_text(json.dumps(ids, indent=2) + '\n', encoding='utf-8')
    config = dict(model_name_or_path=str(args.model.resolve()), trust_remote_code=False,
                  stage='sft', do_train=True, finetuning_type='lora', lora_rank=8,
                  lora_alpha=16, lora_dropout=.05, lora_target='q_proj,v_proj',
                  dataset_dir=str(data), dataset='sentiment_smoke', template='qwen3',
                  enable_thinking=False, cutoff_len=2048, train_on_prompt=False,
                  packing=False, preprocessing_num_workers=1, dataloader_num_workers=0,
                  output_dir=str(root / 'output'), logging_steps=1, save_strategy='steps',
                  save_steps=10, save_total_limit=2, plot_loss=True,
                  overwrite_output_dir=False, report_to=['swanlab'], run_name=run_id,
                  per_device_train_batch_size=1, gradient_accumulation_steps=1,
                  learning_rate=1e-4, max_steps=20, lr_scheduler_type='linear',
                  warmup_ratio=.1, bf16=True, gradient_checkpointing=True,
                  optim='adamw_torch', weight_decay=0.0, seed=42, data_seed=42)
    path = root / 'train.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    # Only parse and validate configuration; no trainer or model construction.
    from llamafactory.hparams import get_train_args
    get_train_args(config)
    script = f'''#!/usr/bin/env bash
set -uo pipefail
cd "{args.factory}"
source /home/lgh/.venvs/llamafactory/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SWANLAB_MODE=offline
export SWANLAB_PROJECT=nlp-qwen06-experience
export SWANLAB_LOG_DIR="{root / 'swanlog'}"
if [[ -e "{root / 'STARTED'}" ]]; then
  echo "This run has already started; prepare a new run to preserve logs."
  exit 2
fi
date -Iseconds > "{root / 'STARTED'}"
llamafactory-cli train "{path}" 2>&1 | tee "{root / 'train.log'}"
result=${{PIPESTATUS[0]}}
printf '%s\\n' "$result" > "{root / 'exit_code.txt'}"
date -Iseconds > "{root / 'ENDED'}"
exit "$result"
'''
    (root / 'start.sh').write_text(script, encoding='utf-8')
    audit = dict(run_id=run_id, purpose='20-record software smoke; not formal course results',
                 model_repository='https://modelscope.cn/models/Qwen/Qwen3-0.6B',
                 downloaded_revision=None, model_files={p.name: sha256(p) for p in sorted(args.model.iterdir()) if p.is_file()},
                 dataset_manifest_sha256=sha256(args.data / 'manifest.json'),
                 sample_ids=ids, config_sha256=sha256(path),
                 packages={p: metadata.version(p) for p in ['torch','transformers','peft','llamafactory','swanlab','optuna']},
                 factory_commit=subprocess.check_output(['git','-C',str(args.factory),'rev-parse','HEAD'],text=True).strip(),
                 distinctions='0.6B,20 samples,20 steps,accumulation=1,no validation; engineering exercise only; formal experiment unchanged')
    (root / 'preparation.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('PREPARED:', root)
    print('START MANUALLY: bash ' + str(root / 'start.sh'))


if __name__ == '__main__':
    main()
