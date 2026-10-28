import json
from pathlib import Path
import tempfile
import unittest
from src.adapters.predictions import adapt
from src.evaluation.metrics import evaluate


class AdapterTests(unittest.TestCase):
    def test_three_sources_keep_raw_and_source_immutable(self):
        truth=[dict(sample_id='x',split='validation',label=0)]
        inputs=[('bert',dict(sample_id='x',true_label=0,parsed_label=0,raw_output='0',status='ok',timing_scope='whole_split_batch',elapsed_ms=100)),
                ('qwen_zero_shot',dict(sample_id='x',true_label=0,parsed_label=0,raw_output=' 0\n',status='ok')),
                ('qwen_lora',dict(sample_id='x',label=0,prediction=None,raw_output='答案是0',status='invalid'))]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.jsonl'
            for method,item in inputs:
                original=json.dumps(item,ensure_ascii=False)+'\n'
                p.write_text(original,encoding='utf-8')
                records=adapt(p,method,'run','config',42,'validation','model',None)
                self.assertTrue(evaluate(records,truth)['complete'])
                self.assertEqual(records[0]['raw_output'],item['raw_output'])
                self.assertEqual(p.read_text(encoding='utf-8'),original)
                if method=='bert':
                    self.assertIsNone(records[0]['latency_ms'])


if __name__=='__main__':
    unittest.main()
