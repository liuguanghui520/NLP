import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_rows(path, records):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')


def local_paths(path):
    result = load(path)
    for key in ['data', 'baseline_results', 'bert_results', 'prompt_delivery', 'lora_freeze', 'qwen_model', 'factory_root']:
        if not result.get(key) or not Path(result[key]).exists():
            raise ValueError(f'Missing local resource: {key}; configure --paths')
    return result
