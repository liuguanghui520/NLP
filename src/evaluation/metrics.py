from collections import Counter

REQUIRED = {'run_id', 'method', 'config_id', 'seed', 'sample_id', 'split', 'true_label',
            'parsed_label', 'raw_output', 'status', 'model_id', 'checkpoint_id', 'source_artifact_sha256'}
FAILURES = {'oom', 'timeout', 'runtime_error', 'failed', 'missing_prediction'}


def parse(raw):
    return int(raw.strip()) if isinstance(raw, str) and raw.strip() in ('0', '1') else None


def evaluate(records, expected):
    errors = []
    ids = [r.get('sample_id') for r in records]
    expected_ids = [r['sample_id'] for r in expected]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Duplicate IDs in frozen data')
    if len(set(ids)) != len(ids):
        errors.append('duplicate_id')
    if ids != expected_ids:
        errors.append('id_order_or_set_mismatch')
    truth = {r['sample_id']: r for r in expected}
    for r in records:
        if REQUIRED - r.keys():
            errors.append(f'{r.get("sample_id")}: missing_fields')
        t = truth.get(r.get('sample_id'))
        if t and (r.get('true_label') != t['label'] or r.get('split') != t['split']):
            errors.append(f'{r.get("sample_id")}: truth_or_split_conflict')
        label = r.get('parsed_label')
        if label is not None and (type(label) is not int or label not in (0, 1)):
            errors.append('invalid_label')
        status = r.get('status')
        if status == 'ok':
            if label is None or parse(r.get('raw_output')) != label:
                errors.append('status_raw_label_conflict')
        elif status == 'invalid_format':
            if label is not None or parse(r.get('raw_output')) is not None:
                errors.append('invalid_format_conflict')
        elif status in FAILURES:
            errors.append('infrastructure_failure')
        else:
            errors.append('unknown_status')
    if errors or not expected:
        return dict(complete=False, errors=sorted(set(errors)), metrics=None)
    matrix = [[0, 0, 0], [0, 0, 0]]
    for r in records:
        matrix[r['true_label']][r['parsed_label'] if r['status'] == 'ok' else 2] += 1
    classes = {}
    n = len(expected)
    for label in [0, 1]:
        tp, fp = matrix[label][label], matrix[1-label][label]
        support = sum(matrix[label])
        fn = support - tp
        classes[str(label)] = dict(precision=tp/(tp+fp) if tp+fp else 0.,
            recall=tp/support if support else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., support=support)
    return dict(complete=True, errors=[], metrics=dict(n=n,
        accuracy=(matrix[0][0]+matrix[1][1])/n,
        macro_f1=sum(c['f1'] for c in classes.values())/2,
        weighted_f1=sum(c['f1']*c['support'] for c in classes.values())/n,
        invalid_rate=sum(row[2] for row in matrix)/n, per_class=classes,
        confusion_matrix=matrix, matrix_rows=[0, 1], matrix_columns=[0, 1, 'invalid'],
        status_counts=dict(Counter(r['status'] for r in records))))
