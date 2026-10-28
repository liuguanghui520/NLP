"""Materialize individually read semantic decisions and confirmed pilot; never infer polarity by keywords."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/error-review/review-20261028-v1'
HANDOFF = 'docs/协作与融合交接/正式对比分析与错例标注执行交接.md'
DECISIONS = 'results/error-review/semantic-decisions-20261028.tsv'
ORIGIN_H = 'human_confirmed_in_design_chat'
ORIGIN_AI = 'ai_assisted_under_confirmed_rules'
PILOT_LINES = [5546,3631,6353,5580,1307,7015,1153,647,3340,6865,5790,6197,2568,6948,4086,6133,1260,6957,905,5773]
PILOT_LABELS = [0,1,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0]
PILOT_QUOTES = ['很差，相当差','实在很划得来','总体感觉比较脏','服务也很差','绝对性价比一流','不值啊','总的来说比较满意','房间设施很陈旧','其它还好','服务员的素质太差','大失所望','早餐感觉也没有以前好了','宾馆总体给人感觉还不错','这不是宰客人吗','其余的还可以吧','决心不再入住','总体还可以','推荐大家移步去那里','299的价格比较实惠','客人不满意']


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pilot_type(i, run):
    if run['status'] == 'invalid_format':
        return 'T5'
    if i == 8:
        return 'T6'
    return {1:'T1',3:'T2',4:'T2',6:'T1',9:'T2',10:'T3',11:'T1',12:'T4',13:'T2',14:'T1',15:'T2',16:'T1',17:'T2',18:'T3',19:'T2'}[i]


def build(output):
    assert not output.exists(), 'Use a new batch; never overwrite annotations'
    started = datetime.now().astimezone().isoformat()
    pack = json.loads((BASE/'input_manifest.json').read_text(encoding='utf-8'))
    sources = dict(pack['source_manifest'])
    for rel, sha in sources.items():
        assert digest(ROOT/rel) == sha, rel
    for p in [BASE/'candidates.jsonl', BASE/'pilot_ids.json', ROOT/HANDOFF, ROOT/DECISIONS, Path(__file__)]:
        sources[p.relative_to(ROOT).as_posix()] = digest(p)
    candidates = rows(BASE/'candidates.jsonl')
    pilot_ids = json.loads((BASE/'pilot_ids.json').read_text(encoding='utf-8'))['sample_ids']
    assert [r['sample_id'] for r in candidates[:20]] == pilot_ids
    assert [int(x.rsplit(':',1)[1]) for x in pilot_ids] == PILOT_LINES
    with (ROOT/DECISIONS).open(encoding='utf-8',newline='') as handle:
        decisions = list(csv.DictReader(handle, delimiter='\t'))
    assert [int(d['queue_index']) for d in decisions] == list(range(21,403))
    assert len(candidates) == len({r['sample_id'] for r in candidates}) == 402
    originals = {r['sample_id']:r for r in rows(ROOT/'data/processed/v1-20261023/test.jsonl')}
    predictions = {}
    for rel in pack['source_manifest']:
        if '/predictions/' in rel:
            rr = rows(ROOT/rel)
            predictions[rr[0]['run_id']] = {p['sample_id']:p for p in rr}
    samples, errors, pending = [], [], []
    for i, item in enumerate(candidates,1):
        source = originals[item['sample_id']]
        assert item['true_label'] == source['label']
        assert item['normalized_text'] == source['normalized_text'] and item['text'] == source['text']
        origin = ORIGIN_H if i <=20 else ORIGIN_AI
        if i<=20:
            label, quotes = PILOT_LABELS[i-1], [PILOT_QUOTES[i-1]]
            context = None
            phenomena = {2:['正负混合','转折'],3:['正负混合'],4:['正负混合'],5:['正负混合','酒店回复'],6:['种子不一致'],7:['源文本不完整'],8:['标签分歧'],9:['正负混合'],10:['多评价对象','比较'],12:['酒店回复','正负混合'],13:['正负混合'],14:['反问'],15:['正负混合','弱肯定'],16:['叙事较长'],17:['正负混合'],18:['多评价对象','比较'],19:['正负混合'],20:['多评价对象']}.get(i,[])
        else:
            d = decisions[i-21]
            label = int(d['assessed_label']) if d['assessed_label'] in ('0','1') else 'uncertain'
            quotes = d['evidence'].split('|')
            context = d['context_type']
            phenomena = d['phenomena'].split(',')
        for q in quotes:
            assert q and q in item['normalized_text'], (i,q)
        if item['char_truncated']:
            phenomena = sorted(set(phenomena+['字符截断']))
        status = 'pending_review' if label == 'uncertain' else ('label_disagreement_pending' if label != item['true_label'] else ('human_confirmed' if i<=20 else 'ai_assisted_completed'))
        sample = dict(review_id=output.name,queue_index=i,sample_id=item['sample_id'],source_label=item['true_label'],
            assessed_label=label,evidence_quotes=quotes,phenomena=phenomena,target_entities=['目标酒店'] + (['其他酒店/平台/旅游项目（见原文）'] if context=='T3' or '多评价对象' in phenomena else []),
            speakers=['住客','酒店回复方'] if '酒店回复' in phenomena else ['住客'],annotation_origin=origin,
            reviewer_id='user_with_ai_category_confirmation' if i<=20 else 'Codex_assistant',
            annotation_rule_version='error_type_v1-T1-T6',review_status=status,ai_tool='Codex',ai_model_if_known=None,
            annotated_at=started,source_sha256=sources['data/processed/v1-20261023/test.jsonl'],
            normalized_text=item['normalized_text'],visible_prefix=item['text'],char_truncated=item['char_truncated'],
            quote_visible_in_common_prefix=[q in item['text'] for q in quotes],
            sentiment_scope='Full normalized text independently read before consulting source label; prefix visibility checked separately' if i>20 else 'User confirmation transcribed from handoff section 5',
            source_confirmation=HANDOFF+'#5' if i<=20 else DECISIONS,
            mechanism_status='unverified; descriptive context only')
        samples.append(sample)
        reasons = []
        if status in ('pending_review','label_disagreement_pending'):
            reasons.append('整体倾向证据不足/正文异常' if label=='uncertain' else '情感判断与冻结标签不一致；请复核，正式标签未改')
        if not all(sample['quote_visible_in_common_prefix']):
            reasons.append('情感证据部分超出共同512字符前缀；不能直接用全文判断解释不可见输入')
        for run in item['runs']:
            p = predictions[run['run_id']][item['sample_id']]
            assert all(run[k] == p.get(k) for k in ['raw_output','parsed_label','status','seed','method'])
            assert run['error'] == (p['status']!='ok' or p['parsed_label']!=source['label'])
            if not run['error']:
                continue
            if i<=20:
                etype = pilot_type(i,run)
            elif run['status']=='invalid_format':
                etype = 'T5'
            elif status in ('pending_review','label_disagreement_pending') or not all(sample['quote_visible_in_common_prefix']):
                etype = 'T6'
            else:
                etype = context
                # A reply outside the common input cannot describe a model's visible reply context.
                if etype=='T4' and '宾馆反馈' not in item['text']:
                    etype = 'T6'
                    reasons.append('酒店回复在共同前缀之外；运行语义分类需复核')
            rstatus = 'format_verified' if etype=='T5' else (status if etype!='T6' else ('label_disagreement_pending' if status=='label_disagreement_pending' else 'pending_review'))
            errors.append(dict(sample_id=item['sample_id'],run_id=run['run_id'],method=run['method'],seed=run['seed'],
                source_label=item['true_label'],raw_output=run['raw_output'],parsed_label=run['parsed_label'],status=run['status'],
                official_error=True,error_type_v1=etype,secondary_phenomena=phenomena,
                attribution_evidence=dict(text_quotes=quotes,common_prefix_visible=sample['quote_visible_in_common_prefix'],
                    observation='strip后非精确0/1' if etype=='T5' else '合法输出与源标签相反；情境来自逐条语义阅读' if etype!='T6' else '源标签下为错误；语义分歧/可见性/文本问题未解决'),
                confidence='high' if etype=='T5' else 'low' if etype=='T6' else 'medium',
                mechanism_status='unverified; no causal ablation',review_status=rstatus,annotation_origin=origin,
                char_truncated=run['char_truncated'],token_truncated=run['token_truncated'],
                source_artifact_sha256=p['source_artifact_sha256'],
                leading_label_diagnostic=int(run['raw_output'].strip()[0]) if run['raw_output'].strip()[:1] in ('0','1') else None,
                semantic_polarity_diagnostic=label,diagnostic_used_for_metrics=False))
        if reasons:
            pending.append(dict(sample_id=item['sample_id'],queue_index=i,source_label=item['true_label'],assessed_label=label,
                evidence_quotes=quotes,annotation_origin=origin,review_status=status if status in ('pending_review','label_disagreement_pending') else 'pending_visibility_review',
                questions=list(dict.fromkeys(reasons)),user_resolution=None))
    actual_errors = {(p['sample_id'],p['run_id']) for rr in predictions.values() for p in rr.values() if p['status']!='ok' or p['parsed_label']!=p['true_label']}
    assert {(e['sample_id'],e['run_id']) for e in errors} == actual_errors
    assert len(errors)==len(actual_errors)
    assert Counter(s['annotation_origin'] for s in samples)=={ORIGIN_H:20,ORIGIN_AI:382}
    assert samples[7]['review_status']=='label_disagreement_pending'
    # Pilot mapping is checked against explicit complete expected erroneous run sets.
    expected = {1:['Z'],2:['Z'],3:['F42','F43','F44','Z'],4:['F42','F43','F44','Z'],5:['Z'],6:['B44'],7:['Z'],8:['B42','B44','Z','S','L42','L43','L44'],9:['Z','S'],10:['B42','B44','F42','F43','F44','Z','L42','L43','L44'],11:['Z'],12:['B42','B43','B44','F42','F43','F44','Z','S','L42','L43','L44'],13:['S'],14:['Z'],15:['Z','S'],16:['Z'],17:['Z','S'],18:['F42','F43','F44'],19:['S'],20:['Z']}
    short = {**{f'B{s}':f'bert-len512-seed{s}' for s in [42,43,44]},**{f'F{s}':f'tfidf-char_1_3-seed{s}' for s in [42,43,44]},**{f'L{s}':f'test-lora-seed{s}' for s in [42,43,44]},'Z':'test-qwen_zero_shot','S':'test-qwen_few_shot'}
    for i, names in expected.items():
        assert {e['run_id'] for e in errors if e['sample_id']==pilot_ids[i-1]} == {short[n] for n in names}
    for rel, sha in sources.items():
        assert digest(ROOT/rel)==sha
    output.mkdir(parents=True)
    def save(name,data):
        (output/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf-8')
    save('pilot-human-confirmed.jsonl',samples[:20])
    save('ai-remaining382.jsonl',samples[20:])
    save('samples-all402.jsonl',samples)
    save('run-error-attributions.jsonl',errors)
    save('needs-review.jsonl',pending)
    audit = dict(status='coverage_and_provenance_verified',sample_count=402,human_confirmed=20,ai_assisted=382,
        unique_error_runs=len(errors),origin_counts=dict(Counter(s['annotation_origin'] for s in samples)),
        sample_review_status=dict(Counter(s['review_status'] for s in samples)),pending_unique_samples=len(pending),
        run_type_counts=dict(Counter(e['error_type_v1'] for e in errors)),pilot_full_ids_and_error_sets_checked=True,
        source_truth_unchanged=True,all_quotes_exact_substrings=True,all_11_prediction_sources_unchanged=True,
        input_sha256=sources,ai_model_version=None,started_at=started,completed_at=datetime.now().astimezone().isoformat(),
        elapsed_semantic_review_seconds=None,elapsed_note='逐批会话语义阅读已发生；没有精确逐条计时，不用脚本耗时冒充标注耗时',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        annotation_method='AI逐条阅读全文，先形成独立情感和证据，再核对源标签/前缀/各运行；脚本只转录、展开和校验。非关键词情感分类器。',
        human_agreement_with_source='19/20 purposive pilot; not inter-rater agreement or accuracy',
        output_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output/'annotation-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'README.md').write_text('# 正式错例辅助标注\n\n20条来源于交接第5节本人确认，382条为本次AI逐条辅助判断；二者不是双人独立标注。T1—T6是描述性情境，不是模型内部原因。\n\n'
        '全部402条均有判断或明确待复核，原始777条标签和11份预测未改。T5为格式事实；分歧与不确定语义归T6。全文证据超出共同前缀的运行另列可见性复核，token可见文本未知时不猜测。\n\n'
        '六类材料：pilot-human-confirmed.jsonl、ai-remaining382.jsonl、samples-all402.jsonl、run-error-attributions.jsonl、needs-review.jsonl、annotation-audit.json。\n\n'
        '复跑（必须新目录）：`python scripts/build_formal_annotations.py --output results/error-review/<new-id>`。决策TSV来自会话逐条阅读，复跑是校验和转录，不会重新发起AI标注。\n',encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['sample_count','unique_error_runs','sample_review_status','pending_unique_samples','run_type_counts']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=(ROOT/args.output).resolve()
    output.relative_to(ROOT)
    build(output)
