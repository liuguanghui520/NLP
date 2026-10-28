"""Audit completed local comparison predictions and save factual result tables."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from evaluate_qwen06 import digest, metrics, write_json


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    batch = json.loads(args.batch.read_text(encoding='utf-8'))
    if batch['status'] != 'completed' or any(j['state'] != 'completed' for j in batch['jobs']):
        raise ValueError('Batch is not fully completed; retain partial results without full comparison.')
    workspace = Path(batch['training_workspace'])
    for name, expected in batch['scripts'].items():
        if digest(workspace / name) != expected:
            raise ValueError(f'Archived source mismatch: {name}')
    truth = {r['sample_id']: r['label'] for r in map(json.loads,
             (Path(batch['data_directory']) / 'validation.jsonl').read_text(encoding='utf-8').splitlines())}
    baseline = PROJECT / 'results/qwen17-profile' / batch['reference_run_id']
    baseline_manifest = json.loads((baseline / 'manifest.json').read_text(encoding='utf-8'))
    original = dict(id='B00', training_directory=baseline_manifest['training_workspace'])
    summaries = []
    epochs = []
    for job in [original] + batch['jobs']:
        root = Path(job['training_directory'])
        config = json.loads((root / 'train.json').read_text(encoding='utf-8'))
        if 'config_sha256' in job and digest(root / 'train.json') != job['config_sha256']:
            raise ValueError('Training config hash mismatch.')
        history = json.loads((root / 'validation_history.json').read_text(encoding='utf-8'))
        if len(history) != 3:
            raise ValueError('Expected three epoch records.')
        for saved in history:
            path = root / f'validation-step-{saved["global_step"]}' / 'predictions.jsonl'
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if len(records) != 776 or len({r['sample_id'] for r in records}) != 776:
                raise ValueError(f'Incomplete predictions: {job["id"]}')
            if {r['sample_id']: r['label'] for r in records} != truth:
                raise ValueError('Validation IDs or labels mismatch.')
            result = metrics(records)
            for key in ('accuracy', 'macro_f1', 'invalid_rate', 'confusion_matrix'):
                if result[key] != saved[key]:
                    raise ValueError(f'Recomputed metric mismatch: {job["id"]} {key}')
            epochs.append(dict(id=job['id'], epoch=saved['epoch'], step=saved['global_step'],
                               prediction_sha256=digest(path), **result))
        best = max(history, key=lambda r: r['macro_f1'])
        declared = json.loads((root / 'best_checkpoint.json').read_text(encoding='utf-8'))
        if best['checkpoint'] != declared['checkpoint']:
            raise ValueError('Best checkpoint selection mismatch.')
        adapter = Path(best['checkpoint']) / 'adapter_model.safetensors'
        adapter_hash = digest(adapter)
        if 'best_adapter_sha256' in job and job['best_adapter_sha256'] != adapter_hash:
            raise ValueError('Best adapter hash mismatch.')
        resource = json.loads((root / 'resource-record/status.json').read_text(encoding='utf-8'))
        if resource['status'] != 'completed' or resource['exit_code'] != 0:
            raise ValueError('Resource worker did not complete.')
        timing = json.loads((root / 'timing.json').read_text(encoding='utf-8'))
        params = json.loads((root / 'parameters.json').read_text(encoding='utf-8'))
        utc_seconds = (datetime.fromisoformat(resource['ended_utc']) - datetime.fromisoformat(resource['started_utc'])).total_seconds()
        summaries.append(dict(
            id=job['id'], rank=config['lora_rank'], alpha=config['lora_alpha'],
            target=config['lora_target'], dropout=config['lora_dropout'],
            learning_rate=config['learning_rate'], warmup_ratio=config['warmup_ratio'],
            scheduler=config['lr_scheduler_type'], seed=config['seed'],
            best_epoch=best['epoch'], best_step=best['global_step'],
            accuracy=best['accuracy'], macro_f1=best['macro_f1'], invalid_rate=best['invalid_rate'],
            best_checkpoint=best['checkpoint'], best_adapter_sha256=adapter_hash,
            trainable_parameters=params['trainable'], adapter_bytes=adapter.stat().st_size,
            perf_counter_seconds=resource['wall_seconds'], utc_span_seconds=utc_seconds,
            clock_difference_seconds=utc_seconds-resource['wall_seconds'],
            classification_validation_seconds=timing['classification_validation_seconds'],
            device_sample_peak_mib=resource['sampled_peaks']['device_memory_mib'],
            train_peak_allocated_bytes=max(r['preceding_train_peak_allocated_bytes'] for r in history),
            train_peak_reserved_bytes=max(r['preceding_train_peak_reserved_bytes'] for r in history),
            recorded_sampling_errors=resource['sampling_errors'],
            swanlab_experiment=job.get('swanlab_experiment',batch['reference_run_id'])))
    winner = max(summaries, key=lambda r: r['macro_f1'])
    root = PROJECT / 'reports' / batch['run_id'] / ('audit-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'summary.json', dict(
        batch_id=batch['run_id'], audited_configurations=10, audited_epoch_predictions=len(epochs),
        predictions_per_epoch=776, test_evaluated=False, seed=42, results=summaries,
        selected_by_validation=winner['id'],
        new_nine_perf_counter_seconds=sum(r['perf_counter_seconds'] for r in summaries[1:]),
        new_nine_utc_span_seconds=sum(r['utc_span_seconds'] for r in summaries[1:]),
        note='Factual audit only; one seed per configuration, validation selection, no student result interpretation.'))
    write_json(root / 'epoch_metrics.json', epochs)
    lines = ['# LoRA十套配置结果核验', '', '核验日期：' + datetime.now().strftime('%Y-%m-%d'), '',
             '全部为seed42验证集结果。10套各3轮，每轮776个唯一ID与真值已核对，指标已从逐条预测重算；最佳适配器SHA256已核对。未运行测试集。', '',
             '| 配置 | 最佳轮次 | Accuracy | Macro-F1 | 非法率 | 程序分钟 | UTC跨度分钟 | 设备采样峰值MiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['id']} | {row['best_epoch']:g} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['invalid_rate']:.6f} | {row['perf_counter_seconds']/60:.2f} | {row['utc_span_seconds']/60:.2f} | {row['device_sample_peak_mib']:g} |")
    lines += ['', f"按预定验证Macro-F1规则，当前选中{winner['id']}，检查点：{winner['best_checkpoint']}。", '',
              '成本须区分perf_counter、UTC跨度、分类验证、加载与保存；基准两种计时差异已另行记录。设备显存采样非精确进程峰值，采样错误保留在summary.json中。单种子结果不代表多种子稳定性。', '',
              '本文件由AI辅助核对工程证据与生成事实表，不提供课程结果解释、错误归因或未经学生确定的新实验设计。']
    (root / '核验汇总.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('AUDIT:',root)
    print('ID accuracy macro_f1 epoch minutes device_peak_mib')
    for row in summaries:
        print(row['id'],f"{row['accuracy']:.6f}",f"{row['macro_f1']:.6f}",row['best_epoch'],round(row['perf_counter_seconds']/60,2),row['device_sample_peak_mib'])
    print('SELECTED:',winner['id'])


if __name__ == '__main__':
    main()
