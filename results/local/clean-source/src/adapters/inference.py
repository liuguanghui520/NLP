"""Real inference for frozen artifacts; never trains or selects on test."""
from pathlib import Path
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback

from src.common.io import load, rows, save, save_rows, digest
from src.evaluation.metrics import parse, evaluate
from src.adapters.prompts import render, restore, text_sha


def run(root, paths, args):
    import numpy as np
    import psutil
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
    torch.set_num_threads(4)
    method = args.method
    split = 'validation' if args.smoke else args.split
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out/'status.json', dict(status='running', method=method, seed=args.seed, split=split, smoke=args.smoke))
    data = Path(paths['data'])
    dataset = rows(data/f'{split}.jsonl')
    assert [r['sample_id'] for r in dataset] == load(data/'split_ids.json')[split]
    if args.smoke:
        dataset = dataset[:4]
    freeze = load(root/'artifacts/evaluation-freeze.json')
    assert freeze['status']=='frozen'
    assert digest(data/f'{split}.jsonl') == freeze['data_files'][f'{split}.jsonl']
    started = time.perf_counter()
    checkpoint_id, prompt_id, prompt_hash = None, None, None
    torch.manual_seed(42)
    state = dict(method=method, split=split, seed=args.seed if method in ['bert','tfidf','qwen_lora'] else None,
                 run_id=out.name, batch=1, threads=4, precision='fp32' if method in ['bert','tfidf'] else 'bfloat16',
                 data_sha256=digest(data/f'{split}.jsonl'), freeze_sha256=digest(root/'artifacts/evaluation-freeze.json'),
                 git_commit=subprocess.check_output(['git','-c',f'safe.directory={root}','-C',str(root),'rev-parse','HEAD'], text=True).strip(),
                 git_dirty=bool(subprocess.check_output(['git','-c',f'safe.directory={root}','-C',str(root),'status','--porcelain'], text=True).strip()),
                 source_sha256=digest(__file__), versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','scikit-learn']})
    try:
        if method.startswith('qwen'):
            for name, expected_sha in load(root/'artifacts/lora-lock.json')['base_model_files'].items():
                assert digest(Path(paths['qwen_model'])/name)==expected_sha, name
        if method in ['bert','tfidf']:
            import joblib
            # Legacy Pipeline serialized the tokenizer under __main__.
            from src.legacy.baselines.run_bert_tfidf import word_tokens
            sys.modules['__main__'].word_tokens = word_tokens
            base = Path(paths['bert_results' if method=='bert' else 'baseline_results'])
            winner = load(base/'frozen_winner.json')['config']
            config_id = winner['config_id']
            selection = load(base/f'{method}-{config_id}-seed{args.seed}'/'selection.json')
            artifact = Path(selection['model'])
            checkpoint_id = selection['model_sha256']
            assert digest(artifact/'model.safetensors' if artifact.is_dir() else artifact)==checkpoint_id
            model_id = 'hfl/chinese-macbert-base' if method=='bert' else 'TF-IDF+LogisticRegression'
            if method=='tfidf':
                model = joblib.load(artifact)
                def predict(row):
                    raw = str(int(model.predict([row['text']])[0]))
                    return dict(raw_output=raw, input_chars=len(row['text']), input_tokens=None,
                                pretruncation_tokens=None, token_truncated=False, output_tokens=None)
            else:
                tok = AutoTokenizer.from_pretrained(artifact,local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(artifact,local_files_only=True).cuda().eval()
                def predict(row):
                    text = row['text']
                    encoded = tok(text, truncation=True, max_length=winner['maxlen'], return_tensors='pt').to('cuda')
                    with torch.inference_mode():
                        raw = str(model(**encoded).logits.argmax(-1).item())
                    return dict(raw_output=raw, input_chars=len(text), input_tokens=encoded.input_ids.shape[1],
                                pretruncation_tokens=None, token_truncated=None, output_tokens=None)
        elif method in ['qwen_zero_shot','qwen_few_shot']:
            model_id, config_id = 'Qwen/Qwen3-1.7B', 'fixed_prompt_v1'
            tok = AutoTokenizer.from_pretrained(paths['qwen_model'], local_files_only=True)
            registry = restore(root, paths, tok)
            mode = registry['modes'][method]
            prompt_id, prompt_hash = mode['prompt_id'], mode['template_sha256']
            model = AutoModelForCausalLM.from_pretrained(paths['qwen_model'], local_files_only=True,
                      dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation='sdpa').cuda().eval()
            def predict(row):
                prompt, body, count = render(tok, mode, row['text'])
                encoded = tok(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
                with torch.inference_mode():
                    output = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=8,
                                            use_cache=True, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                continuation = output[0, encoded.input_ids.shape[1]:]
                return dict(raw_output=tok.decode(continuation,skip_special_tokens=True), input_chars=len(body),
                            input_tokens=count, pretruncation_tokens=None, token_truncated=len(body)<len(row['text']),
                            output_tokens=len(continuation), rendered_input_sha256=text_sha(prompt))
        else:
            sys.path.insert(0, paths['factory_root']+'/src')
            from llamafactory.chat import ChatModel
            model_id, config_id = 'Qwen/Qwen3-1.7B', 'T01'
            adapter = Path(paths['lora_adapters'][str(args.seed)])
            lock = next(x for x in load(root/'artifacts/lora-lock.json')['adapters'] if x['seed']==args.seed)
            checkpoint_id = digest(adapter/'adapter_model.safetensors')
            assert checkpoint_id == lock['adapter_sha256']
            model = ChatModel(dict(model_name_or_path=paths['qwen_model'], adapter_name_or_path=str(adapter),
                template='qwen3', finetuning_type='lora', infer_backend='huggingface', infer_dtype='bfloat16',
                enable_thinking=False, trust_remote_code=False, do_sample=False, max_new_tokens=8, repetition_penalty=1.0))
            tok = model.engine.tokenizer
            instruction = load(root/'configs/methods/lora-frozen.json')['inference']['instruction']
            prompt_id, prompt_hash = 'lora-frozen-single-newline', text_sha(instruction+'\n{{comment}}')
            def predict(row):
                body = row['text']
                while True:
                    prompt = instruction+'\n'+body
                    messages = [dict(role='user',content=prompt)]
                    ids, _ = model.engine.template.encode_oneturn(tok,messages+[dict(role='assistant',content='')],None,None)
                    if len(ids)<=2048:
                        break
                    if not body:
                        raise ValueError('Instruction exceeds input budget')
                    body = body[:-1]
                response, = model.chat(messages, do_sample=False,max_new_tokens=8,repetition_penalty=1.0)
                return dict(raw_output=response.response_text, input_chars=len(body), input_tokens=response.prompt_length,
                            pretruncation_tokens=None, token_truncated=body!=row['text'], output_tokens=response.response_length,
                            rendered_input_sha256=text_sha(prompt))
        if method!='tfidf':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        sync = torch.cuda.synchronize if method!='tfidf' else lambda: None
        state.update(model_load_and_setup_s=time.perf_counter()-started, config_id=config_id,
                     model_id=model_id, checkpoint_id=checkpoint_id, prompt_id=prompt_id, prompt_sha256=prompt_hash)
        actual_model = model.engine.model if method=='qwen_lora' else model
        if hasattr(actual_model,'parameters'):
            state['total_parameters'] = sum(p.numel() for p in actual_model.parameters())
        if hasattr(actual_model,'generation_config'):
            state['model_generation_defaults'] = actual_model.generation_config.to_dict()
            state['explicit_call_overrides'] = dict(do_sample=False,max_new_tokens=8,repetition_penalty=1.0) if method=='qwen_lora' else dict(do_sample=False,num_beams=1,max_new_tokens=8,use_cache=True,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if method=='qwen_lora':
            state['stop_token_ids'] = model.engine.template.get_stop_token_ids(tok)
        save(out/'config.json',state)
        result_rows = []
        if args.command=='predict':
            for pos,row in enumerate(dataset):
                sync(); t=time.perf_counter()
                value=predict(row)
                sync(); ms=(time.perf_counter()-t)*1000
                pred=parse(value['raw_output'])
                value.update(run_id=out.name, method=method,config_id=config_id, seed=state['seed'],
                    sample_id=row['sample_id'],split=split,true_label=row['label'],parsed_label=pred,
                    status='ok' if pred is not None else 'invalid_format', model_id=model_id,checkpoint_id=checkpoint_id,
                    prompt_id=prompt_id,prompt_sha256=prompt_hash, latency_ms=ms,
                    char_truncated=row.get('char_truncated'), timing_scope='preprocess_transfer_inference_parse_batch1')
                value['missing_metadata_reason'] = 'Pretruncation token count not separately measured; baseline truncation flag may be unavailable in this extra inference probe'
                result_rows.append(value)
                with (out/'raw.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(value,ensure_ascii=False)+'\n')
                if (pos+1)%100==0:
                    print(out.name,pos+1,len(dataset),flush=True)
            source_sha=digest(out/'raw.jsonl')
            for r in result_rows:
                r['source_artifact_sha256']=source_sha
            save_rows(out/'predictions.jsonl',result_rows)
            result=evaluate(result_rows,dataset)
            assert result['complete'],result
            save(out/'metrics.json',result)
        all_data={r['sample_id']:r for s in ['train','validation','test'] for r in rows(data/f'{s}.jsonl')}
        warmup=[all_data[s] for s in load(data/'smoke_ids.json')]
        latency=[all_data[s] for s in load(data/'latency_ids.json')]
        if args.smoke:
            warmup,latency=warmup[:2],warmup[2:5]
        for r in warmup:
            predict(r)
        times,detail=[],[]
        for order,r in enumerate(latency):
            sync(); t=time.perf_counter(); value=predict(r); parsed=parse(value['raw_output']); sync()
            ms=(time.perf_counter()-t)*1000
            times.append(ms)
            detail.append(dict(sample_id=r['sample_id'],order=order,latency_ms=ms,parsed_label=parsed,
                status='ok' if parsed is not None else 'invalid_format',input_chars=value['input_chars'],
                input_tokens=value['input_tokens'],token_truncated=value['token_truncated'],
                checkpoint_id=checkpoint_id,config_id=config_id,run_id=out.name,prompt_sha256=prompt_hash))
        save_rows(out/'latency.jsonl',detail)
        save(out/'benchmark.json', dict(mean_ms=float(np.mean(times)),p50_ms=float(np.percentile(times,50,method='linear')),
             p95_ms=float(np.percentile(times,95,method='linear')),max_ms=max(times),throughput=len(times)/(sum(times)/1000),
             warmup_ids=[r['sample_id'] for r in warmup],n=len(times),batch=1,numpy=np.__version__,percentile_method='linear',
             device='cpu' if method=='tfidf' else torch.cuda.get_device_name(0),threads=4,
             timing_scope='preprocess_transfer_inference_parse; excludes disk/model loading; GPU synchronized'))
        save(out/'resources.json', dict(rss_bytes_at_end=psutil.Process().memory_info().rss,
             inference_allocated_peak=torch.cuda.max_memory_allocated() if method!='tfidf' else None,
             inference_reserved_peak=torch.cuda.max_memory_reserved() if method!='tfidf' else None,
             scope='after model load through test/warmup/latency; RSS at end is not sampled peak',
             end_to_end_s=time.perf_counter()-started))
        if args.command=='predict' and not args.smoke:
            save_rows(root/'results/predictions'/f'{out.name}.jsonl',result_rows)
            save(root/'results/metrics'/f'{out.name}.json',result)
            index=rows(root/'results/run_index.jsonl')
            assert out.name not in [r['run_id'] for r in index]
            index.append(dict(run_id=out.name,method=method,config_id=config_id,seed=state['seed'],split=split,
                 model_id=model_id,checkpoint_id=checkpoint_id,source_sha256=digest(out/'raw.jsonl'),
                 source_key='integration_local',relative_source=str(out.relative_to(root))+'/raw.jsonl',
                 phase='final_test',status='audited',n=len(result_rows),metrics=result['metrics']))
            save_rows(root/'results/run_index.jsonl',index)
        save(out/'status.json',dict(status='completed',method=method,split=split,smoke=args.smoke))
    except BaseException:
        save(out/'status.json',dict(status='failed',method=method,traceback=traceback.format_exc()))
        raise
