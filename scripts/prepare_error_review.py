"""Prepare a standalone human-review page from frozen predictions; no model execution."""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import evaluate  # noqa: E402


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>评论错例人工标注</title>
<style>
*{box-sizing:border-box}body{font:16px/1.65 system-ui,sans-serif;color:#223042;background:#f3f6fa;margin:0}
header{background:#163349;color:white;padding:18px 4vw}h1{font-size:24px;margin:0}header p{margin:6px 0}
main{max-width:1250px;margin:auto;padding:20px}section{background:white;border:1px solid #dce3eb;border-radius:10px;padding:20px;margin:16px 0}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center}button,select,input{font:inherit;padding:8px;border:1px solid #9aabba;border-radius:5px}button{cursor:pointer;background:#edf4fa}button.primary{background:#176477;color:white}
label{display:block;margin:12px 0}textarea{width:100%;min-height:85px;font:inherit;padding:10px;border:1px solid #9aabba;border-radius:5px}
.text{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fb;padding:16px;border-left:4px solid #176477}.muted{color:#526779;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #dce3eb;padding:9px;vertical-align:top}th{background:#edf3f8}.scroll{overflow:auto}.error{color:#a13422;font-weight:600}.badge{background:#edf4fa;padding:4px 8px;border-radius:5px}#notice{color:#ffefbc}h2{font-size:20px}h3{font-size:17px}summary{cursor:pointer}
</style>
<header><h1>评论错例 · 人工阅读与归因</h1><p>先试标队列前20条，确认规则后再决定后续覆盖。先独立判断文本，再查看模型分歧。所有人工结论初始留空。</p><div id="notice">本页只读取已有预测；填写保存在本浏览器。请每次结束导出JSON，跨电脑不会自动同步。</div></header>
<main><section><div class="bar">
<label>标注员 <input id="reviewer" placeholder="例如成员A" maxlength="60"></label>
<label>阅读视图 <select id="view"><option value="blind">独立判断（隐藏真值和预测）</option><option value="analysis">对比分析（显示真值和预测）</option></select></label>
<button id="export" class="primary">导出本人的标注JSON</button><label>导入本人的备份 <input id="import" type="file" accept=".json"></label>
</div><p id="summary" class="muted"></p><p class="muted">当前队列来自至少一个正式运行出错的评论，属于经过预测筛选的队列。隐藏字段只是阅读辅助，不是严格随机盲评。无需上传到外部网站。</p>
<div class="bar"><button id="prev">上一条</button><button id="next">下一条</button><label>跳到第 <input id="jump" type="number" min="1" style="width:85px"> 条</label><button id="pending">下一条未填写</button><span id="position"></span></div>
</section>
<section><h2 id="title"></h2><p id="metadata" class="muted"></p><h3>共同输入：规范化后的前512个字符</h3><div id="prefix" class="text"></div>
<details><summary>展开规范化全文及被字符截断的部分</summary><div id="full" class="text"></div></details>
<p class="muted">共同字符前缀不一定等于模型最终token输入。精确token截断文本缺失时，本页不会猜测补全。</p>
<label>你判断的整体情感 <select id="sentiment"><option value="">尚未判断</option><option value="0">0 负面</option><option value="1">1 正面</option><option value="uncertain">有歧义／信息不足</option></select></label>
<label>文本现象（可写多项，例如否定、转折、混合评价、反讽）<input id="phenomena" style="width:100%"></label>
<label>支持情感判断的原文片段与说明<textarea id="evidence" placeholder="引用具体片段，说明整体评价对象与情感。"></textarea></label>
<label>备注／疑似原标签问题（复核后再确认）<textarea id="notes"></textarea></label></section>
<section id="analysis" hidden><h2>各方案预测与运行级归因</h2><p id="truth"></p><div class="scroll"><table><thead><tr><th>方案／种子</th><th>原始输出</th><th>解析／状态</th><th>是否出错</th><th>字符／token截断</th><th>token记录</th></tr></thead><tbody id="runs"></tbody></table></div>
<p class="muted">未知≠未截断。模型错误类型请逐运行填写；仅有关键词或截断标记不足以证明原因。对所有种子同因的情况，可人工使用复制功能。</p><div id="causes"></div></section>
<section><h2>归因速查</h2><p>E1 否定／范围／程度／比较；E2 转折／混合评价／对象变化；E3 隐含情感／反讽；E4 关键信息截断；E5 非法输出格式；E6 标签歧义／数据问题；E7 领域语境／上下文不足；E8 其他／无法归因。</p><p>标记“否定”应指出否定作用范围；标记“截断”须指出不可见的关键证据；疑似标签问题不修改正式测试真值。主因单选、次因可多写，保留不确定性。</p><p class="muted">两人协作时使用不同标注员名称并独立导出；第三人复核后另存裁定文件，不能覆盖两人的原始判断。本页不计算或宣称人工归因已经完成。</p></section></main>
<script id="dataset" type="application/json">__DATA__</script>
<script>
const pack=JSON.parse(document.getElementById('dataset').textContent),rows=pack.samples,$=id=>document.getElementById(id);
let index=0,annotations={};const causes=['','E1','E2','E3','E4','E5','E6','E7','E8'];
const key=()=>pack.review_id+':'+$('reviewer').value.trim();
const message=text=>{$('notice').textContent=text;};
const state=()=>annotations[rows[index].sample_id]||(annotations[rows[index].sample_id]={sentiment:'',phenomena:'',evidence:'',notes:'',attributions:{}});
function persist(){if(!$('reviewer').value.trim()){message('请先填写标注员名称，再开始填写。');return false;}try{localStorage.setItem(key(),JSON.stringify(annotations));message('已暂存本浏览器；结束前请导出JSON备份。');}catch(e){message('浏览器暂存不可用，请立即导出JSON保存。');}return true;}
function load(){try{annotations=JSON.parse(localStorage.getItem(key())||'{}');}catch(e){annotations={};message('无法读取浏览器暂存，请导入已导出的JSON。');}render();}
function update(){if(!$('reviewer').value.trim()){message('请先填写标注员名称。');return;}const s=state();for(const k of ['sentiment','phenomena','evidence','notes'])s[k]=$(k).value;s.updated_at=new Date().toISOString();persist();updateSummary();}
function updateSummary(){const done=Object.values(annotations).filter(s=>s.sentiment&&s.evidence.trim()).length;$('summary').textContent=`${pack.review_id} ｜ 候选评论 ${rows.length} 条 ｜ 原始测试 ${pack.test_count} 条 ｜ 完成独立判断 ${done} 条（归因另行填写）`;}
const show=v=>v===null||v===undefined?'未知':v===true?'是':v===false?'否':String(v);
function render(){const r=rows[index],s=annotations[r.sample_id]||{};$('title').textContent=`评论 ${index+1}`;$('metadata').textContent=`ID：${r.sample_id} ｜ 全文 ${r.normalized_char_count} 字符 ｜ 前缀 ${r.text.length} 字符 ｜ 字符截断：${show(r.char_truncated)}`;$('prefix').textContent=r.text;$('full').textContent=r.normalized_text;$('position').textContent=`${index+1} / ${rows.length}`;$('jump').value=index+1;$('jump').max=rows.length;
for(const k of ['sentiment','phenomena','evidence','notes']){$(k).value=s[k]||'';$(k).disabled=!$('reviewer').value.trim();}
$('analysis').hidden=$('view').value!=='analysis';$('truth').textContent=`源标签：${r.true_label}（0负面／1正面）；仅分析参考，不直接改标签。`;
$('runs').replaceChildren();$('causes').replaceChildren();for(const run of r.runs){const tr=document.createElement('tr');for(const val of [`${run.display_name} / ${run.seed??'无种子'}`,run.raw_output,`${run.parsed_label??'非法'} / ${run.status}`,run.error?'错':'对',`${show(run.char_truncated)} / ${show(run.token_truncated)}`,`输入 ${show(run.input_tokens)}；截断前 ${show(run.pretruncation_tokens)}；输出 ${show(run.output_tokens)}`]){const td=document.createElement('td');td.textContent=String(val??'');tr.append(td);}if(run.error)tr.className='error';$('runs').append(tr);if(!run.error)continue;
const block=document.createElement('details'),summary=document.createElement('summary');summary.textContent=`${run.display_name} / ${run.seed??'无种子'}：填写本次错误原因`;block.append(summary);const previous=(s.attributions||{})[run.run_id]||{};
for(const [field,title] of [['primary_cause','主因'],['secondary_cause','次因（可多项）'],['evidence','归因证据（具体片段、可见信息与预测）'],['confidence','可信度（高／中／低）'],['improvement','可能改进与尚未验证内容']]){const label=document.createElement('label');label.textContent=title;const el=field==='primary_cause'?document.createElement('select'):document.createElement('textarea');if(field==='primary_cause')for(const c of causes){const option=document.createElement('option');option.value=c;option.textContent=c||'尚未归因';el.append(option);}el.value=previous[field]||'';el.disabled=!$('reviewer').value.trim();el.addEventListener('change',()=>{const st=state();st.attributions[run.run_id]=st.attributions[run.run_id]||{};st.attributions[run.run_id][field]=el.value;st.updated_at=new Date().toISOString();persist();});label.append(el);block.append(label);}const copy=document.createElement('button');copy.textContent='将这条归因复制到本评论同方案的其他错误种子';copy.disabled=!$('reviewer').value.trim();copy.onclick=()=>{const st=state(),a=st.attributions[run.run_id];if(!a||!a.primary_cause){message('请先填写该运行主因，再复制。');return;}for(const other of r.runs)if(other.error&&other.method===run.method&&other.run_id!==run.run_id){if(st.attributions[other.run_id]&&Object.values(st.attributions[other.run_id]).some(Boolean)){message('目标已有内容，未覆盖；请人工核对。');continue;}st.attributions[other.run_id]={...a};}persist();render();};block.append(copy);$('causes').append(block);}
updateSummary();}
for(const k of ['sentiment','phenomena','evidence','notes'])$(k).addEventListener('change',update);
$('reviewer').addEventListener('change',load);$('view').addEventListener('change',render);
$('prev').onclick=()=>{index=Math.max(0,index-1);render();};$('next').onclick=()=>{index=Math.min(rows.length-1,index+1);render();};$('jump').onchange=()=>{index=Math.max(0,Math.min(rows.length-1,Number($('jump').value)-1||0));render();};
$('pending').onclick=()=>{for(let step=1;step<=rows.length;step++){const j=(index+step)%rows.length,s=annotations[rows[j].sample_id];if(!s||!s.sentiment||!s.evidence?.trim()){index=j;render();return;}}message('独立判断均已填写，请检查运行级归因和导出。');};
$('export').onclick=()=>{const reviewer=$('reviewer').value.trim();if(!reviewer){message('请填写标注员名称。');return;}document.activeElement.blur();const out={schema_version:1,review_id:pack.review_id,source_manifest:pack.source_manifest,reviewer_id:reviewer,exported_at:new Date().toISOString(),annotation_rule:'docs/对比实验与错误分析方案.md',annotations};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));a.download=pack.review_id+'-'+reviewer.replace(/[^\p{L}\p{N}_-]/gu,'_')+'-'+Date.now()+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
$('import').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{const d=JSON.parse(await file.text());if(d.review_id!==pack.review_id||d.reviewer_id!==$('reviewer').value.trim()||JSON.stringify(d.source_manifest)!==JSON.stringify(pack.source_manifest))throw Error('批次、标注员或来源不一致，请使用对应页面与名称');const valid=new Set(rows.map(r=>r.sample_id));for(const [id,annotation] of Object.entries(d.annotations||{})){if(!valid.has(id))throw Error('备份含未知样本');if(annotations[id]&&JSON.stringify(annotations[id])!==JSON.stringify(annotation))throw Error('存在内容冲突，未合并；请保留两个导出文件人工裁定');}annotations={...annotations,...d.annotations};persist();render();}catch(e){message('导入失败：'+e.message);}};
render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError("Use a new output directory; existing annotations must not be overwritten")
    data_path = ROOT / "data/processed/v1-20261023/test.jsonl"
    data = read_rows(data_path)
    assert len(data) == 777
    files = [(f"bert-len512-seed{s}.jsonl", "MacBERT") for s in [42, 43, 44]]
    files += [(f"tfidf-char_1_3-seed{s}.jsonl", "TF-IDF") for s in [42, 43, 44]]
    files += [("test-qwen_zero_shot.jsonl", "Qwen零样本"), ("test-qwen_few_shot.jsonl", "Qwen少样本")]
    files += [(f"test-lora-seed{s}.jsonl", "LoRA T01") for s in [42, 43, 44]]
    sources = {str(data_path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(data_path.read_bytes()).hexdigest()}
    runs = []
    for filename, name in files:
        path = ROOT / "results/predictions" / filename
        rows = read_rows(path)
        result = evaluate(rows, data)
        assert result["complete"], (filename, result["errors"])
        sources[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        runs.append((name, {r["sample_id"]: r for r in rows}))
    samples = []
    for row in data:
        predictions = []
        for name, lookup in runs:
            p = lookup[row["sample_id"]]
            pred = {k: p.get(k) for k in ["run_id", "method", "seed", "raw_output", "parsed_label", "status", "char_truncated", "token_truncated", "input_tokens", "pretruncation_tokens", "output_tokens", "rendered_input_sha256", "prompt_sha256", "checkpoint_id"]}
            pred.update(display_name=name, error=p["status"] != "ok" or p["parsed_label"] != row["label"])
            predictions.append(pred)
        if not any(p["error"] for p in predictions):
            continue
        samples.append(dict(sample_id=row["sample_id"], true_label=row["label"], raw_text=row["raw_text"],
                            normalized_text=row["normalized_text"], text=row["text"],
                            normalized_char_count=row["normalized_char_count"], char_truncated=row["char_truncated"], runs=predictions))
    random.Random(67).shuffle(samples)
    assert len({s["sample_id"] for s in samples}) == len(samples)
    pack = dict(schema_version=1, review_id=output.name, generated_at=datetime.now().astimezone().isoformat(),
                test_count=len(data), run_count=len(runs), shuffle_seed=67, source_manifest=sources, samples=samples)
    output.mkdir(parents=True)
    (output / "candidates.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    (output / "input_manifest.json").write_text(json.dumps({k: v for k, v in pack.items() if k != "samples"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "pilot_ids.json").write_text(json.dumps(dict(review_id=output.name, selection="first20_of_seed67_shuffled_error_union", sample_ids=[s["sample_id"] for s in samples[:20]], note="Purposive error-review pilot, not a representative accuracy sample"), ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.dumps(pack, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = PAGE.replace("__DATA__", payload)
    (output / "review.html").write_text(page, encoding="utf-8")
    assert "__DATA__" not in page
    reread = read_rows(output / "candidates.jsonl")
    assert reread == samples
    error_counts = {next(iter(lookup.values()))["run_id"]: sum(p["status"] != "ok" or p["parsed_label"] != p["true_label"] for p in lookup.values()) for _, lookup in runs}
    (output / "README.md").write_text(
        f"# 错例人工阅读材料\n\n批次：{output.name}\n\n"
        f"已核对11份完整测试预测，每份777条。任一运行出错的评论按ID去重后共{len(samples)}条；未替人填写归因。\n\n"
        "本人已确认先试标队列前20条，ID见pilot_ids.json；规则确认后再决定后续覆盖。试标集不是随机全测试集，不用于估计总体表现。\n\n"
        "1. 用浏览器打开review.html，填写自己的标注员名称。\n"
        "2. 默认独立判断视图隐藏源标签和各模型预测，先判断情感并填写证据。\n"
        "3. 切换对比分析查看各运行结果，逐运行填写主因、次因和依据。\n"
        "4. 每次结束导出JSON并备份；网页缓存不跨设备同步。导入只接受同批次、同标注员、同来源且无冲突的文件。\n"
        "5. 第二人使用不同标注员名称独立填写；裁定另存，不覆盖原始标注。\n\n"
        "材料覆盖全量错误候选，但不要求一次标完。尚未确定最终人工覆盖数量及双人分工。\n"
        "候选是按模型错误筛选的，不能从错误内部占比推断所有测试评论某现象的错误率。\n"
        "共同512字符前缀与全文均可阅读；缺失的模型token输入与截断标记保留未知，不自行重建。\n"
        "页面不联网、不运行模型、不改变正式指标。人工内容只进入导出文件或浏览器暂存。\n\n"
        "## 各运行错误数\n\n" + "\n".join(f"- {k}: {v}" for k, v in error_counts.items()) + "\n",
        encoding="utf-8")
    print(json.dumps(dict(output=str(output), candidates=len(samples), runs=len(runs), errors=error_counts, human_causes_filled=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
