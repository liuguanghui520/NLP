"""Execute the student's approved data rules. Never trains a model."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import random
import sys
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def read_source(path):
    checksum = digest(path)
    decoded = path.read_bytes().decode('utf-8-sig', errors='strict')
    reader = csv.DictReader(io.StringIO(decoded, newline=''), strict=True)
    if reader.fieldnames != ['label', 'review']:
        raise ValueError(f'Unexpected fields: {reader.fieldnames}')
    records, issues = [], []
    for index, row in enumerate(reader, 1):
        sample_id = f'{checksum}:{index}'
        if None in row or row['review'] is None or row['label'] not in ('0', '1'):
            issues.append({'sample_id': sample_id, 'source_row': index, 'reason': 'invalid_record', 'record': row})
            continue
        raw = row['review']
        normalized = raw.replace('\r\n', '\n').replace('\r', '\n').strip()
        records.append(dict(sample_id=sample_id, source_sha256=checksum, source_row=index,
                            raw_text=raw, normalized_text=normalized, raw_label=row['label'],
                            label=int(row['label']), raw_char_count=len(raw),
                            normalized_char_count=len(normalized),
                            processing_flags=['minimal_cleaning'] if normalized != raw else []))
    return records, issues, checksum


def prepare(records, limit):
    groups, excluded, conflicts = defaultdict(list), [], []
    for row in records:
        if not row['normalized_text']:
            excluded.append(dict(row, reason='empty_text'))
        else:
            groups[row['normalized_text']].append(row)
    valid = []
    for group in groups.values():
        if len({r['label'] for r in group}) != 1:
            conflicts.extend(dict(r, reason='conflicting_labels') for r in group)
            continue
        valid.append(group[0])
        excluded.extend(dict(r, reason='duplicate_same_label', retained_id=group[0]['sample_id']) for r in group[1:])
    valid.sort(key=lambda r: r['source_row'])
    for row in valid:
        row.update(text=row['normalized_text'][:limit], text_char_count=min(limit, len(row['normalized_text'])),
                   char_truncated=len(row['normalized_text']) > limit)
    return valid, excluded, conflicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=['analyze', 'build'], default='analyze')
    parser.add_argument('--source-evidence', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'status.json', {'status': 'running', 'mode': args.mode})
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        records, issues, checksum = read_source(args.source)
        valid, excluded, conflicts = prepare(records, config['max_chars'])
        write_lines(args.output / 'invalid_records.jsonl', issues)
        write_lines(args.output / 'excluded.jsonl', excluded)
        write_lines(args.output / 'conflicts.jsonl', conflicts)
        prefixes = defaultdict(list)
        for row in valid:
            prefixes[row['text']].append(row['sample_id'])
        collisions = [ids for ids in prefixes.values() if len(ids) > 1]
        stats = dict(source_sha256=checksum, encoding='utf-8-sig (strict)',
                     total_records=len(records) + len(issues), valid_records=len(valid),
                     raw_labels=dict(Counter(r['label'] for r in records)), invalid_records=len(issues),
                     exclusions=dict(Counter(r['reason'] for r in excluded)), conflict_records=len(conflicts),
                     missing_body_records=sum(x['record'].get('review') is None for x in issues),
                     empty_string_records=sum(r['raw_text'] == '' for r in records),
                     truncated_count=sum(r['char_truncated'] for r in valid),
                     truncated_ratio=sum(r['char_truncated'] for r in valid) / len(valid) if valid else 0,
                     prefix_collision_groups=collisions, approximate_deduplication=False)
        lengths = sorted(r['normalized_char_count'] for r in valid)
        stats['length_distribution'] = {str(q): lengths[round((len(lengths)-1)*q)] for q in [0, .25, .5, .75, .95, 1]} if lengths else {}
        write_json(args.output / 'statistics.json', stats)
        if issues or conflicts or checksum != config['expected_sha256']:
            raise ValueError('Blocked: malformed records, label conflicts or source SHA mismatch; inspect reports')
        if args.mode == 'analyze':
            write_json(args.output / 'status.json', {'status': 'analysis_complete', 'frozen': False})
            print(json.dumps(stats, ensure_ascii=False, indent=2))
            return
        evidence = json.loads(args.source_evidence.read_text(encoding='utf-8')) if args.source_evidence else {}
        if not config['decisions_confirmed'] or not evidence.get('label_mapping_verified') or not evidence.get('verified_content_revision'):
            raise ValueError('Blocked: decisions, label evidence or actual download revision unresolved')
        if collisions:
            if config.get('prefix_collision_policy') != 'keep_first_source_row' or not config.get('prefix_confirmation'):
                raise ValueError('Blocked: truncated prefix collisions need student review before freeze')
            retained, retained_text = [], {}
            for row in valid:
                if row['text'] in retained_text:
                    excluded.append(dict(row, reason='duplicate_truncated_prefix', retained_id=retained_text[row['text']]))
                else:
                    retained_text[row['text']] = row['sample_id']
                    retained.append(row)
            valid = retained
            write_lines(args.output / 'excluded.jsonl', excluded)
            stats['valid_records'] = len(valid)
            stats['exclusions'] = dict(Counter(r['reason'] for r in excluded))
            stats['truncated_count'] = sum(r['char_truncated'] for r in valid)
            stats['truncated_ratio'] = stats['truncated_count'] / len(valid)
            stats['prefix_collision_resolution'] = config['prefix_confirmation']
            lengths = sorted(r['normalized_char_count'] for r in valid)
            stats['length_distribution'] = {str(q): lengths[round((len(lengths)-1)*q)] for q in [0, .25, .5, .75, .95, 1]}
        from sklearn.model_selection import train_test_split
        import sklearn
        train, temporary = train_test_split(valid, test_size=.2, random_state=config['split_seed'], stratify=[r['label'] for r in valid])
        validation, test = train_test_split(temporary, test_size=.5, random_state=config['split_seed'], stratify=[r['label'] for r in temporary])
        splits = dict(train=train, validation=validation, test=test)
        for name, rows in splits.items():
            rows.sort(key=lambda r: r['source_row'])
            for row in rows:
                row['split'] = name
        by_id = {r['sample_id']: r for r in valid}
        final_prefixes = defaultdict(list)
        for row in valid:
            final_prefixes[row['text']].append(row['sample_id'])
        remaining_collisions = [ids for ids in final_prefixes.values() if len(ids) > 1]
        cross = [ids for ids in remaining_collisions if len({by_id[i]['split'] for i in ids}) > 1]
        stats['prefix_collision_groups_after_resolution'] = remaining_collisions
        stats['prefix_collisions_cross_split'] = cross
        write_json(args.output / 'statistics.json', stats)
        if remaining_collisions:
            raise ValueError('Blocked: truncated prefix collisions need student review before freeze')
        rng = random.Random(config['smoke_seed'])
        smoke = [r['sample_id'] for label in [0, 1] for r in rng.sample([r for r in train if r['label'] == label], 10)]
        rng = random.Random(config['fewshot_seed'])
        few = {label: rng.sample([r for r in train if r['label'] == label and len(r['normalized_text']) <= 80], 2) for label in [0, 1]}
        fewshot = [few[0][0]['sample_id'], few[1][0]['sample_id'], few[0][1]['sample_id'], few[1][1]['sample_id']]
        latency = test[:] if len(test) <= 200 else train_test_split(test, train_size=200, random_state=config['latency_seed'], stratify=[r['label'] for r in test])[0]
        write_json(args.output / 'split_ids.json', {s: [r['sample_id'] for r in rows] for s, rows in splits.items()})
        for name, ids in [('smoke', smoke), ('fewshot', fewshot), ('latency', [r['sample_id'] for r in latency])]:
            write_json(args.output / f'{name}_ids.json', ids)
        alpaca = args.output / 'alpaca'
        alpaca.mkdir()
        for name, rows in splits.items():
            write_lines(args.output / f'{name}.jsonl', rows)
            write_json(alpaca / f'{name}.json', [dict(instruction=config['instruction'], input=r['text'], output=str(r['label'])) for r in rows])
            write_lines(alpaca / f'{name}_index.jsonl', [dict(index=i, sample_id=r['sample_id']) for i, r in enumerate(rows)])
        write_json(alpaca / 'dataset_info.json', {f'sentiment_{s}': dict(file_name=f'{s}.json', columns=dict(prompt='instruction', query='input', response='output')) for s in ['train', 'validation']})
        # Re-read all written content and check independent invariants.
        seen, full_texts = set(), set()
        for name, rows in splits.items():
            restored = [json.loads(line) for line in (args.output / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]
            assert restored == rows
            examples = json.loads((alpaca / f'{name}.json').read_text(encoding='utf-8'))
            indices = [json.loads(line) for line in (alpaca / f'{name}_index.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(examples) == len(rows) == len(indices)
            for i, (row, example, index) in enumerate(zip(rows, examples, indices)):
                assert row['sample_id'] not in seen and row['normalized_text'] not in full_texts
                seen.add(row['sample_id']); full_texts.add(row['normalized_text'])
                assert row['text'] == row['normalized_text'][:512]
                assert example == dict(instruction=config['instruction'], input=row['text'], output=str(row['label']))
                assert index == dict(index=i, sample_id=row['sample_id'])
        assert len(seen) == len(valid)
        assert len(records) == len(valid) + len(excluded) + len(conflicts)
        assert set(smoke + fewshot) <= {r['sample_id'] for r in train}
        assert Counter(by_id[i]['label'] for i in smoke) == {0: 10, 1: 10}
        assert [by_id[i]['label'] for i in fewshot] == [0, 1, 0, 1]
        assert all(len(by_id[i]['normalized_text']) <= 80 for i in fewshot)
        assert {r['sample_id'] for r in latency} <= {r['sample_id'] for r in test}
        assert digest(args.source) == checksum
        stats['splits'] = {s: dict(count=len(rows), labels=dict(Counter(r['label'] for r in rows))) for s, rows in splits.items()}
        write_json(args.output / 'statistics.json', stats)
        write_lines(args.output / 'manual_decisions.jsonl', [{'operator': 'user', 'time': '2026-10-23', 'decision': config[key], 'stage': 'before_freeze'} for key in ['confirmation', 'prefix_confirmation'] if key in config])
        manifest = dict(source_path=str(args.source), source_size=args.source.stat().st_size, source_sha256=checksum,
                        config=config, evidence=evidence, sklearn_version=sklearn.__version__, python_version=sys.version,
                        script_sha256=digest(Path(__file__)), config_sha256=digest(args.config),
                        packages={p: importlib.metadata.version(p) for p in ['scikit-learn','numpy','scipy']},
                        split_method='20% temporary then 50/50 validation/test; both stratified seed 67',
                        subset_method='Python Random.sample per class in source_row order; latency sklearn stratified selection',
                        validation='all assertions passed', statistics=stats,
                        files={str(p.relative_to(args.output)): digest(p) for p in sorted(args.output.rglob('*')) if p.is_file() and p.name != 'status.json'})
        write_json(args.output / 'manifest.json', manifest)
        write_json(args.output / 'status.json', {'status': 'frozen', 'redistribution_verified': config['redistribution_verified']})
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    except Exception:
        write_json(args.output / 'status.json', {'status': 'failed_or_blocked', 'frozen': False, 'reason': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
