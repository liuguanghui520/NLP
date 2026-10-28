from src.common.io import digest, rows


def label(value):
    if value in ('negative', 'positive'):
        return {'negative': 0, 'positive': 1}[value]
    if value is None:
        return None
    if type(value) is int and value in (0, 1):
        return value
    raise ValueError(f'Unsupported label: {value!r}')


def adapt(path, method, run_id, config_id, seed, split, model_id, checkpoint_id):
    source_sha = digest(path)
    out = []
    for r in rows(path):
        truth = label(r.get('true_label', r.get('label')))
        pred = label(r.get('parsed_label', r.get('prediction', r.get('pred_label'))))
        status = {'invalid': 'invalid_format', 'failed': 'runtime_error'}.get(r.get('status'), r.get('status'))
        if status is None:
            status = 'ok' if pred is not None else 'invalid_format'
        raw = r.get('raw_output')
        # Classification IDs are the actual model decision, not generated text.
        if 'raw_output' not in r and method in ('bert', 'tfidf') and pred is not None:
            raw = str(pred)
        item = dict(run_id=run_id, method=method, config_id=config_id, seed=seed,
            sample_id=r['sample_id'], split=split, true_label=truth, parsed_label=pred,
            raw_output=raw, status=status, model_id=model_id, checkpoint_id=checkpoint_id,
            source_artifact_sha256=source_sha, input_chars=r.get('input_chars', r.get('input_length')),
            input_tokens=r.get('input_tokens'), pretruncation_tokens=r.get('token_length'),
            char_truncated=r.get('prefix_truncated'), token_truncated=r.get('token_truncated', r.get('truncated')),
            output_tokens=r.get('output_tokens'), prompt_id=r.get('prompt_id'),
            prompt_sha256=r.get('rendered_input_sha256'), latency_ms=r.get('latency_ms'),
            timing_scope=r.get('timing_scope', 'historical_unavailable'),
            missing_metadata_reason='Fields absent from historical source remain null',
            source_row_index=len(out))
        if item['timing_scope'] == 'whole_split_batch':
            item['latency_ms'] = None
            item['timing_scope'] = 'not_individually_measured'
        out.append(item)
    return out
