import copy
import hashlib
import json
from pathlib import Path
from src.common.io import load, rows, save


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def render(tokenizer, mode, text):
    def candidate(body):
        messages = [dict(role=m['role'], content=m['content'].replace('{{comment}}', body)) for m in mode['messages_template']]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return prompt, len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
    # Preserve the member runner's binary prefix search exactly.
    low, high, best = 0, len(text), None
    while low <= high:
        mid = (low+high)//2
        prompt, count = candidate(text[:mid])
        if count <= 2048:
            best = (prompt, text[:mid], count)
            low = mid+1
        else:
            high = mid-1
    if best is None:
        raise ValueError('Frozen instruction/examples exceed 2048 tokens')
    return best


def restore(root, paths, tokenizer):
    registry = copy.deepcopy(load(root/'configs/methods/prompt_registry.json'))
    assert text_sha(tokenizer.chat_template or '') == registry['chat_template_sha256']
    train = {r['sample_id']: r for r in rows(Path(paths['data'])/'train.jsonl')}
    examples = load(root/'configs/methods/few_shot_examples.json')['examples']
    zero = registry['modes']['qwen_zero_shot']['messages_template'][0]['content']
    few = registry['modes']['qwen_few_shot']
    assert few['example_ids'] == [e['sample_id'] for e in examples]
    for i, e in enumerate(examples):
        r = train[e['sample_id']]
        assert r['label'] == e['label']
        assert text_sha(r['normalized_text']) == e['original_text_sha256']
        content = zero.replace('{{comment}}', r['normalized_text'])
        expected_sha = few['messages_template'][i*2]['content'].split('original_content_sha256=')[1]
        assert text_sha(content) == expected_sha
        few['messages_template'][i*2]['content'] = content
    for name, mode in registry['modes'].items():
        value = mode['messages_template'] if name=='qwen_zero_shot' else dict(messages=mode['messages_template'], example_ids=mode['example_ids'])
        assert text_sha(json.dumps(value, ensure_ascii=False, sort_keys=True)) == mode['template_sha256']
        mode['rendered_template'] = tokenizer.apply_chat_template(mode['messages_template'], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    # Check all existing validation input hashes, without running the model.
    va = {r['sample_id']:r for r in rows(Path(paths['data'])/'validation.jsonl')}
    checked = {}
    for name, mode in registry['modes'].items():
        source = Path(paths['prompt_delivery'])/'run_evidence'/f'C_{name}_fixed_prompt_v1_noseed_20261024T154625Z/predictions_validation.jsonl'
        for r in rows(source):
            prompt, _, _ = render(tokenizer, mode, va[r['sample_id']]['text'])
            assert text_sha(prompt) == r['rendered_input_sha256'], r['sample_id']
        checked[name] = len(rows(source))
    registry.pop('package_sanitization', None)
    save(root/'results/local/restored-prompt-registry.json', registry)
    save(root/'reports/audits/prompt-restoration.json', dict(examples=4, validation_input_hashes=checked,
         message_template_hashes_verified=True, tokenizer_template_verified=True,
         raw_vs_sanitized_manifest='Original hashes used only after content reconstruction; sanitized file has separate provenance hash'))
    return registry
