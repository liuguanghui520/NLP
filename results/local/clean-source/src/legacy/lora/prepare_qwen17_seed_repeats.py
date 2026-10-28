"""Prepare (never start) seed43/44 repeats of the validation-selected T01."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

from evaluate_qwen06 import digest, write_json
from run_qwen17_comparisons import FACTORY, PROJECT, preflight, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--retry-of', help='Previous interrupted repeat batch ID to preserve as provenance')
    args = parser.parse_args()
    _, manifest, source, data, _ = preflight()
    parent_id = 'qwen17-comparisons-20261026-012117-9e8e51'
    parent = json.loads((PROJECT / 'results/qwen17-comparisons' / parent_id / 'batch.json').read_text(encoding='utf-8'))
    if parent['status'] != 'completed':
        raise ValueError('Original search is not completed.')
    # Current recorded baseline is lower; verify selection against all ten configs.
    baseline_best = json.loads((PROJECT / 'results/qwen17-profile' / parent['reference_run_id'] / 'best_checkpoint.json').read_text(encoding='utf-8'))
    selected = max(parent['jobs'], key=lambda row: row['validation_macro_f1'])
    if selected['id'] != 'T01' or selected['validation_macro_f1'] <= baseline_best['macro_f1']:
        raise ValueError('Recorded selected configuration differs from expected T01.')
    directory = Path(selected['training_directory'])
    if digest(directory / 'train.json') != selected['config_sha256']:
        raise ValueError('Selected training configuration changed.')
    original = json.loads((directory / 'train.json').read_text(encoding='utf-8'))
    from llamafactory.hparams import get_train_args
    configs = []
    for seed in (43, 44):
        item = dict(id=f'T01-S{seed}', group='seed_repeat', seed=seed)
        config = dict(original, seed=seed, data_seed=seed)
        config['output_dir'] = f'/tmp/qwen17-seed-check-not-created/{seed}'
        get_train_args(config)
        changed = {key for key, value in config.items() if value != original.get(key)}
        if changed != {'seed', 'data_seed', 'output_dir'}:
            raise ValueError('Unexpected configuration change in seed repeat.')
        configs.append((item, config))
    print('REPEAT CHECK PASSED: T01 fixed, seed/data_seed43 and44; split unchanged; no training started.', flush=True)
    if args.check_only:
        return
    definition = dict(purpose='Repeat validation-selected T01 using previously confirmed seeds43/44',
                      parent_batch_id=parent_id, selected_configuration='T01',
                      existing_seed42=selected, configurations=[item for item, _ in configs])
    if args.retry_of:
        previous = json.loads((PROJECT / 'results/qwen17-comparisons' / args.retry_of / 'batch.json').read_text(encoding='utf-8'))
        if previous.get('selected_configuration') != 'T01' or previous['status'] not in ('interrupted', 'failed', 'incomplete'):
            raise ValueError('Retry source is not a stopped T01 repeat batch.')
        if any(job['state'] == 'completed' for job in previous['jobs']):
            raise ValueError('Previous batch contains completed jobs; prepare only missing seeds separately.')
        definition['retry_of'] = args.retry_of
        definition['retry_reason'] = 'User reported computer reboot; previous batch stopped, no completed repeat jobs; new run preserves original evidence.'
    with (FACTORY / 'project-workspaces/qwen17-comparisons.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        work, index, plan = prepare(definition, manifest, source, data, configs)
        plan.update(purpose='selected_configuration_seed_repeats',
                    parent_batch_id=parent_id, selected_configuration='T01',
                    existing_seed42=selected,
                    ai_assistance='Engineer prepares previously confirmed seed43/44 repeats; selected settings fixed; user manually runs and interprets.')
        if args.retry_of:
            plan['retry_of'] = args.retry_of
            plan['retry_reason'] = definition['retry_reason']
        shutil.copyfile(__file__, work / 'prepare_qwen17_seed_repeats.py')
        plan['scripts']['prepare_qwen17_seed_repeats.py'] = digest(work / 'prepare_qwen17_seed_repeats.py')
        for job, (item, _) in zip(plan['jobs'], configs):
            job['seed'] = item['seed']
            job['selected_configuration'] = 'T01'
        write_json(work / 'batch.json', plan)
        write_json(index / 'batch.json', plan)
        print('PREPARED:', work, '\nINDEX:', index, flush=True)
        print('START MANUALLY: python -u ' + str(PROJECT / 'scripts/run_qwen17_comparisons.py') + ' --run-prepared ' + str(work), flush=True)


if __name__ == '__main__':
    main()
