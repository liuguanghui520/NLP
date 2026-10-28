"""Acceptance checks on source immutability and source-separated annotation export."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'results/error-review/review-formal-20261028-01'


def rows(name):
    return [json.loads(x) for x in (REVIEW/name).read_text(encoding='utf-8').splitlines()]


class FormalAnnotationTests(unittest.TestCase):
    def test_frozen_inputs_unchanged(self):
        audit=json.loads((REVIEW/'annotation-audit.json').read_text(encoding='utf-8'))
        for rel,sha in audit['input_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),sha,rel)

    def test_human_confirmations_are_not_replaced_by_ai(self):
        human=rows('pilot-human-confirmed.jsonl')
        ai=rows('ai-remaining382.jsonl')
        merged=rows('samples-all402.jsonl')
        self.assertEqual((len(human),len(ai),len(merged)),(20,382,402))
        self.assertEqual(human+ai,merged)
        self.assertEqual(len({s['sample_id'] for s in merged}),402)
        self.assertEqual({s['annotation_origin'] for s in human},{'human_confirmed_in_design_chat'})
        self.assertEqual({s['annotation_origin'] for s in ai},{'ai_assisted_under_confirmed_rules'})
        self.assertEqual(human[7]['assessed_label'],0)
        self.assertEqual(human[7]['source_label'],1)
        self.assertEqual(human[7]['review_status'],'label_disagreement_pending')

    def test_disagreement_does_not_become_confirmed_semantic_model_failure(self):
        samples={s['sample_id']:s for s in rows('samples-all402.jsonl')}
        for e in rows('run-error-attributions.jsonl'):
            s=samples[e['sample_id']]
            if s['review_status'] in ('label_disagreement_pending','pending_review'):
                self.assertIn(e['error_type_v1'],('T5','T6'))
            if e['status']=='invalid_format':
                self.assertEqual(e['error_type_v1'],'T5')
                self.assertNotIn(e['raw_output'].strip(),('0','1'))
            self.assertFalse(e['diagnostic_used_for_metrics'])

    def test_all_and_only_official_errors_are_present(self):
        audit=json.loads((REVIEW/'annotation-audit.json').read_text(encoding='utf-8'))
        expected=set()
        for rel in audit['input_sha256']:
            if rel.startswith('results/predictions/'):
                for line in (ROOT/rel).read_text(encoding='utf-8').splitlines():
                    p=json.loads(line)
                    if p['status']!='ok' or p['parsed_label']!=p['true_label']:
                        expected.add((p['sample_id'],p['run_id']))
        errors=rows('run-error-attributions.jsonl')
        self.assertEqual(len(errors),len(expected))
        self.assertEqual({(e['sample_id'],e['run_id']) for e in errors},expected)
