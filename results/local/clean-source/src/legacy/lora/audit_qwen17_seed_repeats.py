"""Verify the selected configuration's three validation seeds; factual audit only."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics

from evaluate_qwen06 import digest, metrics, write_json


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    args = parser.parse_args()
    batch = json.loads(args.batch.read_text(encoding='utf-8'))
    if batch['status'] != 'completed' or len(batch['jobs']) != 2:
        raise ValueError('Both seed repeat jobs must be completed.')
    if any(j['state'] != 'completed' for j in batch['jobs']):
        raise ValueError('Incomplete seed repeat.')
    parent_file = PROJECT / 'results/qwen17-comparisons' / batch['parent_batch_id'] / 'batch.json'
    parent = json.loads(parent_file.read_text(encoding='utf-8'))
    if batch['baseline_model_files'] != parent['baseline_model_files'] or batch['packages'] != parent['packages'] or batch['factory_commit'] != parent['factory_commit']:
        raise ValueError('Baseline model/source/version metadata differs.')
    root = Path(batch['training_workspace'])
    for name, expected in batch['scripts'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Archived script changed: {name}')
    for name in ('profile_qwen17.py', 'evaluate_qwen06.py'):
        if batch['scripts'][name] != parent['scripts'][name]:
            raise ValueError('Training or evaluation entry changed between seeds.')
    data = Path(batch['data_directory'])
    frozen = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    if digest(data / 'validation.jsonl') != frozen['files']['validation.jsonl']:
        raise ValueError('Frozen validation changed.')
    truth = {r['sample_id']: r['label'] for r in map(json.loads, (data / 'validation.jsonl').read_text(encoding='utf-8').splitlines())}
    initial = batch['existing_seed42']
    initial_config = json.loads((Path(initial['training_directory']) / 'train.json').read_text(encoding='utf-8'))
    # These are output/log identity fields, not model/training hyperparameters.
    ignored = {'seed', 'data_seed', 'output_dir', 'swanlab_run_name', 'swanlab_logdir'}
    rows = []
    epoch_results = []
    for job in [dict(initial, seed=42)] + batch['jobs']:
        directory = Path(job['training_directory'])
        config = json.loads((directory / 'train.json').read_text(encoding='utf-8'))
        if digest(directory / 'train.json') != job['config_sha256']:
            raise ValueError('Training config hash changed.')
        if config['seed'] != job['seed'] or config['data_seed'] != job['seed']:
            raise ValueError('Wrong training/shuffle seed.')
        if {k: v for k, v in config.items() if k not in ignored} != {k: v for k, v in initial_config.items() if k not in ignored}:
            raise ValueError('A hyperparameter other than seed changed.')
        history = json.loads((directory / 'validation_history.json').read_text(encoding='utf-8'))
        if len(history) != 3 or [r['epoch'] for r in history] != [1, 2, 3]:
            raise ValueError('Expected three full epoch validations.')
        for epoch in history:
            path = directory / f'validation-step-{epoch["global_step"]}' / 'predictions.jsonl'
            predictions = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if len(predictions) != 776 or len({p['sample_id'] for p in predictions}) != 776 or {p['sample_id']: p['label'] for p in predictions} != truth:
                raise ValueError('Predictions incomplete or labels/IDs mismatched.')
            result = metrics(predictions)
            for key in ('accuracy', 'macro_f1', 'invalid_rate', 'confusion_matrix'):
                if result[key] != epoch[key]:
                    raise ValueError('Recomputed metrics mismatch.')
            epoch_results.append(dict(seed=job['seed'], epoch=epoch['epoch'],
                                      step=epoch['global_step'], prediction_sha256=digest(path), **result))
        best = max(history, key=lambda item: item['macro_f1'])
        declared = json.loads((directory / 'best_checkpoint.json').read_text(encoding='utf-8'))
        if best != declared or best['checkpoint'] != job['best_checkpoint']:
            raise ValueError('Best checkpoint disagrees with validation selection.')
        adapter = Path(best['checkpoint']) / 'adapter_model.safetensors'
        if digest(adapter) != job['best_adapter_sha256']:
            raise ValueError('Adapter SHA256 mismatch.')
        trainer = json.loads((directory / 'adapter/trainer_state.json').read_text(encoding='utf-8'))
        if trainer['global_step'] != 1167 or trainer['epoch'] != 3:
            raise ValueError('Training did not complete original three epochs.')
        resource = json.loads((directory / 'resource-record/status.json').read_text(encoding='utf-8'))
        if resource['status'] != 'completed' or resource['exit_code'] != 0:
            raise ValueError('Resource wrapper indicates failure.')
        utc_seconds = (datetime.fromisoformat(resource['ended_utc']) - datetime.fromisoformat(resource['started_utc'])).total_seconds()
        rows.append(dict(seed=job['seed'], accuracy=best['accuracy'], macro_f1=best['macro_f1'],
                         invalid_rate=best['invalid_rate'], best_epoch=best['epoch'],
                         best_checkpoint=best['checkpoint'], adapter_sha256=job['best_adapter_sha256'],
                         perf_counter_seconds=resource['wall_seconds'], utc_span_seconds=utc_seconds,
                         clock_difference_seconds=utc_seconds-resource['wall_seconds'],
                         device_sample_peak_mib=resource['sampled_peaks']['device_memory_mib'],
                         power_samples=resource['power_samples'],
                         sampling_errors=resource['sampling_errors']))
    if [r['seed'] for r in rows] != [42, 43, 44]:
        raise ValueError('Expected seeds42/43/44 in fixed order.')
    aggregate = {key: dict(mean=statistics.mean(r[key] for r in rows),
                           sample_std=statistics.stdev(r[key] for r in rows))
                 for key in ('accuracy', 'macro_f1', 'invalid_rate')}
    report = PROJECT / 'reports' / batch['run_id'] / ('audit-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    report.mkdir(parents=True, exist_ok=False)
    write_json(report / 'summary.json', dict(configuration='T01', batch_id=batch['run_id'],
               results=rows, aggregate=aggregate, std_definition='sample standard deviation, n-1 denominator',
               audited_epoch_records=9, audited_predictions=9*776, test_evaluated=False,
               retry_of=batch.get('retry_of'), note='Validation scores only; do not choose a seed by its score. Factual audit, no student result interpretation.'))
    write_json(report / 'epoch_metrics.json', epoch_results)
    lines=['# T01三个种子验证成果核验', '', '核验日期：'+datetime.now().strftime('%Y-%m-%d'), '',
           '两次新增运行均完成3轮、1167更新步；与原seed42配置相比，仅随机种子和输出/日志标识不同。9轮共6984条验证预测ID/真值和重算指标一致，最佳检查点和适配器哈希通过。', '',
           '| 种子 | 最佳轮次 | Accuracy | Macro-F1 | 非法率 | 程序分钟 | UTC分钟 | 设备采样峰值MiB |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['best_epoch']:g} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['invalid_rate']:.6f} | {row['perf_counter_seconds']/60:.2f} | {row['utc_span_seconds']/60:.2f} | {row['device_sample_peak_mib']:g} |")
    lines += ['', '样本标准差使用n−1分母，各指标保持0—1尺度：', '',
              '| 指标 | 均值 | 样本标准差 |', '|---|---:|---:|']
    for key, value in aggregate.items():
        lines.append(f"| {key} | {value['mean']:.8f} | {value['sample_std']:.8f} |")
    lines += ['', '以上为验证结果，不是测试成绩；不按验证分数挑选种子。原中断批次及其成本保留，retry_of见summary.json。', '',
              '资源口径、时钟差异和采样错误见summary.json；设备总显存采样不是精确进程峰值。下一步仍需冻结最终评估配置，并按项目整体流程完成其他模型与最终测试。', '',
              'AI辅助工程核验与均值/标准差计算；学生负责课程结果解释与错误归因。未启动新训练或推理。']
    (report / '核验汇总.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('AUDIT:',report)
    print(json.dumps(dict(results=rows, aggregate=aggregate), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
