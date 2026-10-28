"""Formal offline extensions: provenance-aware annotations, sensitivity, curves and evidence."""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import binomtest

from scripts.build_comparison_report import ROOT, ORDER, NAMES, COLORS, macro_f1, paired_bootstrap, holm
from scripts.case_analysis import DEFAULT_CASE_NOTES, enrich_case, render_casebook


def table(headers, rows):
    esc = lambda x: str(x).replace('|', '\\|').replace('\n', '<br>')
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',
                      *['|'+'|'.join(esc(c) for c in row)+'|' for row in rows]])


def fmt(v, scale=1):
    if v is None:
        return '未知/不适用'
    v *= scale
    return f'{v:.3e}' if 0 < abs(v) < .0001 else f'{v:.4f}'


def seed_sensitivity(r):
    results=[]
    for seed in [43,44]:
        models={}
        for m in r.models:
            if m=='majority':
                models[m]=r.models[m]
            else:
                run=next(x for x in r.index if x['method']==m and x['seed'] in (seed,None))
                models[m]=[np.array([p['parsed_label'] if p['status']=='ok' else -1 for p in r.records[run['run_id']]])]
        samples=paired_bootstrap(r.y,models,seed=r.config['analysis_seed'],resamples=r.config['bootstrap_resamples'],batch=r.config['bootstrap_batch'])
        group=[]
        for a,b in r.config['comparison_pairs']:
            pa,pb=models[a][0],models[b][0]
            ca,cb=pa==r.y,pb==r.y
            gain,loss=int(np.sum(ca&~cb)),int(np.sum(~ca&cb))
            group.append(dict(seed=seed,a=a,b=b,n=777,macro_f1_delta=float(macro_f1(r.y,pa)-macro_f1(r.y,pb)),
                bootstrap_95_percentile=np.quantile(samples[a]-samples[b],[.025,.975],method='linear').tolist(),
                A_correct_B_wrong=gain,A_wrong_B_correct=loss,both_wrong=int(np.sum(~ca&~cb)),
                exact_mcnemar_p=float(binomtest(gain,gain+loss,.5).pvalue) if gain+loss else 1.,
                scope='Sensitivity only; same 777 comments, fixed trained seed; not extra independent observations'))
        for row,p in zip(group,holm([x['exact_mcnemar_p'] for x in group])):
            row['holm_adjusted_p_within_seed_seven_pairs']=p
        results+=group
    r.write('seed43-44-sensitivity.json',results)
    return results


def annotations(r):
    root=Path(r.config['annotations'])
    audit=r.read(root/'annotation-audit.json')
    for rel,sha in audit['input_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
    for filename,sha in audit['output_sha256'].items():
        assert hashlib.sha256((ROOT/root/filename).read_bytes()).hexdigest()==sha,filename
    samples=r.read(root/'samples-all402.jsonl',lines=True)
    errors=r.read(root/'run-error-attributions.jsonl',lines=True)
    pending=r.read(root/'needs-review.jsonl',lines=True)
    assert len(samples)==len({x['sample_id'] for x in samples})==402
    expected={(p['sample_id'],run) for run,rr in r.records.items() for p in rr if p['status']!='ok' or p['parsed_label']!=p['true_label']}
    assert len(errors)==len(expected) and {(e['sample_id'],e['run_id']) for e in errors}==expected
    assert Counter(s['annotation_origin'] for s in samples)=={'human_confirmed_in_design_chat':20,'ai_assisted_under_confirmed_rules':382}
    source={x['sample_id']:x for x in r.data}
    for s in samples:
        assert s['source_label']==source[s['sample_id']]['label']
        assert all(q in source[s['sample_id']]['normalized_text'] for q in s['evidence_quotes'])
    primary=[]
    for method in ORDER:
        rr=[e for e in errors if e['run_id']==r.reps[method]['run_id']]
        count=Counter(e['error_type_v1'] for e in rr)
        primary.append(dict(method=method,unit='one erroneous comment per method, seed42 or deterministic prompt',
            denominator=len(rr),type_counts={f'T{i}':count[f'T{i}'] for i in range(1,7)},
            classified_non_T6=len(rr)-count['T6'],classified_fraction=(len(rr)-count['T6'])/len(rr),
            origin_counts=dict(Counter(e['annotation_origin'] for e in rr)),
            scope='402-comment error union only; AI labels are not independent human truth'))
    r.write('annotation-statistics.json',dict(audit=audit,representative_error_types=primary,
        all_seed_type_counts=dict(Counter(e['error_type_v1'] for e in errors)),
        pending_unique_samples=len(pending),denominator_warning='Do not infer error rates for all comments containing a phenomenon from error-only annotations'))
    r.annotation_counts=dict(human_confirmed_samples=20,ai_assisted_samples=382,pending_unique_samples=len(pending),run_attributions=len(errors))
    return samples,errors,pending,primary


def parameters(r):
    evidence=Path(r.config['training_log_evidence'])
    sizes=r.read(r.config['candidate_sizes'])['records']
    manifest=r.read(evidence/'manifest.json')
    for x in manifest['files']:
        if x['exists']:
            payload=(ROOT/evidence/x['copied_to']).read_bytes()
            assert hashlib.sha256(payload).hexdigest()==x['sha256']
    configs={}
    epochs={}
    for path in sorted((ROOT/evidence/'bert').glob('*/config.json')):
        cfg=r.read(path)
        configs[cfg['run_id']]=cfg
        ep=r.read(path.parent/'epoch_metrics.jsonl',lines=True)
        epochs[cfg['run_id']]=ep
        best=sorted(ep,key=lambda e:(-e['macro_f1'],e['epoch']))[0]
        if cfg['seed']==42:
            original=next(x for x in r.validation if x['method']=='bert' and x['config_id']==cfg['config_id'])
            assert best['epoch']==original['epoch'] and abs(best['macro_f1']-original['macro_f1'])<1e-12
    anchor=configs['bert-main-seed42']
    varying={'lr','config_id','run_id'}
    keys=['maxlen','batch','eval_batch','epochs','freeze_lower','weighted','precision','optimizer','weight_decay','warmup_ratio','grad_clip','accumulation','scheduler','seed','length_group_mega']
    for name in ['lr_1e5','lr_5e5']:
        c=configs[f'bert-{name}-seed42']
        assert all(c[k]==anchor[k] for k in keys)
    comparisons=[]
    for x in r.validation:
        cost=next(t for t in r.training if t['run_id']==f'{x["method"]}-{x["config_id"]}-seed42')
        cfg=configs[cost['run_id']] if x['method']=='bert' else r.read(f'results/bert-tfidf-20261028-140359/tfidf-formal/{cost["run_id"]}/config.json')
        comparisons.append(dict(method=x['method'],config_id=x['config_id'],seed=42,split='validation',n=776,
            macro_f1=x['macro_f1'],best_epoch=x['epoch'],config=cfg,
            features=x.get('features'),iterations=x.get('iterations'),fit_s=x.get('fit_s'),train_log_loss=x.get('train_log_loss'),
            validation_log_loss=x.get('validation_log_loss'),trainable_parameters=x['trainable_parameters'],
            end_to_end_s=cost['end_to_end_s'],train_s=cost['train_s'],validation_s=cost['validation_s'],
            checkpoint_or_pipeline_bytes=next(z['bytes'] for z in sizes if z['run_id']==cost['run_id']),
            missing_size_reason=None,artifact_size_scope='best checkpoint model+tokenizer or Pipeline; optimizer/environment excluded'))
    lora=[]
    base=next(x for x in r.lora_audit['results'] if x['id']=='B00')
    for x in r.lora_audit['results']:
        assert x['alpha']==2*x['rank'] and x['seed']==42
        rel='results/qwen17-profile/qwen17-full-20261025-231706-ea8173/validation_history.json' if x['id']=='B00' else f'results/qwen17-comparisons/qwen17-comparisons-20261026-012117-9e8e51/{x["id"]}/validation_history.json'
        history=r.read(rel)
        best=sorted(history,key=lambda e:(-e['macro_f1'],e['epoch']))[0]
        assert best['epoch']==x['best_epoch'] and abs(best['macro_f1']-x['macro_f1'])<1e-12
        lora.append(dict(x,delta_vs_B00=x['macro_f1']-base['macro_f1'],comparison='each versus same B00, not cumulative',split='validation',n=776))
    payload=dict(baselines=comparisons,lora=lora,bert_learning_rate_controlled_keys=keys,
        boundary='BERT len512 changes batch too; partial changes LR/freezing; LoRA module/rank changes parameter count; config searches only seed42')
    r.write('formal-parameter-tables.json',payload)
    r.write('bert-epoch-metrics.json',epochs)
    base_parameters=r.read('results/local/integration-20261028-v1/test-qwen_zero_shot/config.json')['total_parameters']
    model_parameters=[dict(method='bert',base_or_total_parameters=anchor['total_parameters'],trainable_parameters=anchor['trainable_parameters'],trainable_fraction=1.,meaning='MacBERT classification model all parameters trained')]
    model_parameters += [dict(method='qwen_lora',config_id=x['id'],base_parameters=base_parameters,
        total_with_adapter=base_parameters+x['trainable_parameters'],trainable_parameters=x['trainable_parameters'],
        trainable_fraction=x['trainable_parameters']/(base_parameters+x['trainable_parameters']),adapter_bytes=x['adapter_bytes'],
        meaning='base configuration count plus additional LoRA matrices; training uses adapter matrices only') for x in lora]
    model_parameters += [dict(method=x['method'],config_id=x['config_id'],features=x['features'],
        fitted_coefficients=x['features']+1,meaning='binary logistic coefficients plus intercept; not neural parameter count') for x in comparisons if x['method']=='tfidf']
    r.write('model-parameter-counts.json',model_parameters)
    return evidence,configs,epochs,comparisons,lora


def formal_costs(r):
    inference=[]
    for x in r.latency_stats:
        resources=x['resources']
        inference.append(dict(run_id=x['run_id'],method=x['method'],seed=x['seed'],
            loaded_process_end_to_end_s=resources.get('end_to_end_s'),
            benchmark_measured_seconds=float(r.latency[x['run_id']].sum()/1000),benchmark_n=200,
            scope='Process resource timer and 200-item latency timer are nested; never add them together. Baseline probes are benchmark-only, Qwen test jobs include full test/warmup/benchmark.',
            full_test_only_s=None,missing_reason='Full-test-only interval not separately retained',resources=resources))
    lifecycle=r.read('results/imported/lifecycle.jsonl',lines=True)
    failures=r.read('reports/audits/failures.json')
    payload=dict(training_unique_runs=r.ledger,training_run_count=len(r.ledger),original_training_resource_records=r.training,
        winners_seed42_already_in_search=True,LoRA_B00_counted_once=True,
        inference_and_benchmarks=inference,historical_lifecycle=lifecycle,failures=failures,
        failures_seconds=None,interpretation='Known recorded cost lower bound only; failures/preparation/timing gaps unknown. No energy/currency estimate. Baseline original test-only interval unavailable.',
        totals_by_method={m:sum(x['seconds'] for x in r.ledger if x['method']==m) for m in ['tfidf','bert','qwen_lora']},
        phase_units='Successful search includes winner42; repeat phase only43/44. Do not add winner42 again.')
    r.write('formal-cost-ledger.json',payload)
    return payload


def curves(r,evidence,configs):
    data={'bert':{},'lora':{}}
    for seed in [42,43,44]:
        name=f'bert-len512-seed{seed}'
        steps=r.read(evidence/'bert'/name/'train_steps.jsonl',lines=True)
        assert all(x['step']==i+1 for i,x in enumerate(steps))
        assert all(x['seconds']>=0 and np.isfinite(x['loss']) for x in steps)
        data['bert'][str(seed)]=dict(steps=[x['step'] for x in steps],loss=[x['loss'] for x in steps],
            measured_step_seconds_cumulative=np.cumsum([x['seconds'] for x in steps]).tolist(),
            scope='Sum of recorded batch processing intervals; excludes epoch validation/checkpoint I/O and inter-step gaps; NOT total elapsed wall time')
        key='T01' if seed==42 else f'T01-S{seed}'
        log=r.read(evidence/'lora'/key/'trainer_log.jsonl',lines=True)
        state=r.read(evidence/'lora'/key/'trainer_state.json')
        points=[x for x in log if x.get('loss') is not None]
        original={x['step']:x['loss'] for x in state['log_history'] if 'loss' in x}
        assert all(abs(x['loss']-original[x['current_steps']])<1e-12 for x in points)
        elapsed=lambda s: sum(float(x)*60**i for i,x in enumerate(reversed(s.split(':'))))
        data['lora'][str(seed)]=dict(steps=[x['current_steps'] for x in points],loss=[x['loss'] for x in points],
            elapsed_seconds=[elapsed(x['elapsed_time']) for x in points],
            scope='Original Trainer elapsed_time, integer-second resolution; logged answer-token loss every10steps; includes trainer overhead')
    r.write('training-curves.json',data)
    plt=r.plt
    for method,number,ylabel in [('bert','15','分类交叉熵（每batch）'),('lora','16','答案token loss（日志区间均值）')]:
        fig,axes=plt.subplots(1,2,figsize=(12,5.2))
        for seed in [42,43,44]:
            d=data[method][str(seed)]
            t=d['measured_step_seconds_cumulative'] if method=='bert' else d['elapsed_seconds']
            axes[0].plot(d['steps'],d['loss'],linewidth=.8,alpha=.8,label=f'seed{seed}')
            axes[1].plot(np.array(t)/60,d['loss'],linewidth=.8,alpha=.8,label=f'seed{seed}')
        axes[0].set(xlabel='优化步',ylabel=ylabel)
        axes[1].set(xlabel='累计实测batch计时（分钟，非完整墙钟）' if method=='bert' else '原Trainer实际经过时间（分钟）',ylabel=ylabel)
        for ax in axes: ax.legend()
        r.figure(fig,f'{number}-{method}-training-loss',f'{"MacBERT len512" if method=="bert" else "LoRA T01"}：原始训练loss（分路线展示）',
            '训练6210条、seed42/43/44；原始日志，无平滑/插值；不同目标loss不可比较大小。\n'+('右图仅累计实测batch区间，不含验证/保存及间隙。' if method=='bert' else '右图为原Trainer计时；记录每10步区间均值，精度为秒。'), 'training-curves.json',role='附录')
    return data


def extra_figures(r,primary):
    plt=r.plt
    fig,ax=plt.subplots(figsize=(11,5.3))
    bottom=np.zeros(5)
    palette=['#477da5','#65a397','#c49b55','#967cab','#c96555','#c2c7cd']
    labels=['T1 明确误判','T2 混合评价','T3 多对象/比较','T4 评论＋回复','T5 格式不合规','T6 未解决']
    for i in range(6):
        values=np.array([p['type_counts'][f'T{i+1}']/p['denominator']*100 for p in primary])
        ax.bar(range(5),values,bottom=bottom,color=palette[i],label=labels[i]);bottom+=values
    ax.set(xticks=range(5),xticklabels=[NAMES[p['method']]+'\n错误N='+str(p['denominator']) for p in primary],ylabel='该方法错误评论内比例（%）',ylim=(0,100))
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.2),fontsize=9)
    r.figure(fig,'14-error-types','错误主类：本人确认20＋AI辅助382（错误队列402条）','固定seed42＋确定性提示；每错误评论每方法一次，互斥主类；T6含未归因/标签分歧。\n这是描述性辅助标注，不是双人独立人工真值；分母不是全部777条。','annotation-statistics.json')
    fig,axes=plt.subplots(1,2,figsize=(13,5.2))
    for i,p in enumerate(r.pairs):
        axes[0].barh(i,p['A_correct_B_wrong'],color='#519685');axes[0].barh(i,-p['A_wrong_B_correct'],color='#be7669')
        axes[1].barh(i,p['both_wrong'],color='#708199');axes[1].text(p['both_wrong']+1,i,str(p['both_wrong']),va='center',fontsize=9)
    labels=[NAMES[p['a']]+' − '+NAMES[p['b']] for p in r.pairs]
    for ax in axes:
        ax.set_yticks(range(7),labels);ax.invert_yaxis()
    axes[0].axvline(0,color='#999');axes[0].set(xlabel='A退化 ← 评论数 → A改善')
    axes[1].set(xlabel='共同失败评论数（包括非法输出）');axes[1].margins(x=.2)
    r.figure(fig,'17-common-failure','改善、退化与共同失败','测试777条、固定seed42/确定性提示；同一评论配对；不把不同种子当新增测试样本。','paired-comparisons.json')
    fig,ax=plt.subplots(figsize=(10,4.9))
    for i,m in enumerate(ORDER):
        s=next(x for x in r.summary if x['method']==m)
        ax.bar(i,s['invalid_rate']['mean']*100,color=COLORS[m]);ax.text(i,s['invalid_rate']['mean']*100+.5,fmt(s['invalid_rate']['mean'],100),ha='center')
    ax.set(xticks=range(5),xticklabels=[NAMES[m] for m in ORDER],ylabel='非法输出占全部测试评论（%）');ax.margins(y=.2)
    r.figure(fig,'18-invalid-output','严格输出政策下的格式失败','测试777条；训练路线3种子均值，提示单次；strip后必须精确0/1，解释或情感词不重新解析。','methods.json')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    x=np.arange(5)
    for j,key in enumerate(['inference_allocated_peak','inference_reserved_peak']):
        vals=[max((p['resources'][key]/2**20 for p in r.latency_stats if p['method']==m and p['resources'][key] is not None),default=np.nan) for m in ORDER]
        axes[0].bar(x+(j-.5)*.34,vals,width=.34,label='allocated' if j==0 else 'reserved')
    axes[0].legend();axes[0].set(ylabel='CUDA峰值 MiB（各运行最大值）')
    rss=[statistics.mean(p['resources']['rss_bytes_at_end']/2**20 for p in r.latency_stats if p['method']==m) for m in ORDER]
    axes[1].bar(x,rss,color=[COLORS[m] for m in ORDER]);axes[1].set(ylabel='进程期末RSS MiB（非峰值；各运行均值）')
    for ax in axes: ax.set_xticks(x,[NAMES[m] for m in ORDER],rotation=18)
    r.figure(fig,'19-memory-scopes','内存口径：CUDA峰值与期末RSS分开展示','加载后测试/预热/延迟全过程；TF-IDF在CPU、CUDA不适用。\n整设备采样属于历史训练日志，不与allocated/reserved或RSS混作同一指标。','latency.json',role='附录')
    fig,ax=plt.subplots(figsize=(10,5.1))
    for m in ORDER:
        g=[next(x for x in r.groups if x['method']==m and x['group']==f'char_truncated_{v}') for v in [False,True]]
        ax.plot([0,1],[x['mean_error_rate']*100 if x['n'] else np.nan for x in g],'o-',label=NAMES[m],color=COLORS[m])
    n=[next(x for x in r.groups if x['group']==f'char_truncated_{v}')['n'] for v in [False,True]]
    ax.set(xticks=[0,1],xticklabels=[f'未字符截断 N={n[0]}',f'共同前512字符截断 N={n[1]}'],ylabel='错误率（%）');ax.legend()
    r.figure(fig,'20-char-truncation','共同字符截断子群（相关性描述）','测试777条；训练路线3种子错误率均值；两组类别组成见数据JSON；N<30仅描述，不据此判断截断因果。','length-and-truncation.json',role='附录')
    fig,ax=plt.subplots(figsize=(10,4.9))
    ep=r.read('results/analysis/'+r.id+'/bert-epoch-metrics.json')
    for seed in [42,43,44]:
        points=ep[f'bert-len512-seed{seed}'];ax.plot([x['epoch'] for x in points],[x['macro_f1']*100 for x in points],'o-',label=f'seed{seed}')
    ax.set(xlabel='训练轮次',ylabel='验证 Macro-F1（%）',xticks=[1,2,3]);ax.legend()
    r.figure(fig,'21-bert-validation','MacBERT len512：逐轮验证（非测试）','验证776条；seed42/43/44；最佳轮次只根据验证Macro-F1，同分取较早轮次。','bert-epoch-metrics.json',role='附录')
    token=r.read('results/analysis/'+r.id+'/token-truncation.json')
    fig,axes=plt.subplots(1,2,figsize=(13,5.8))
    names=[]
    for i,run in enumerate(r.index):
        names.append(NAMES[run['method']]+'\n'+str(run['seed']))
        bottom=0
        for flag,color,label in [(False,'#70a99b','记录未截断'),(True,'#c38d56','记录截断'),(None,'#b8c0ca','未知')]:
            g=next(x for x in token if x['run_id']==run['run_id'] and x['token_truncated'] is flag)
            axes[0].bar(i,g['n'],bottom=bottom,color=color,label=label if i==0 else None);bottom+=g['n']
            if g['n']:
                axes[1].scatter(i,g['errors']/g['n']*100,color=color,marker='o',label=label if i==0 else None)
    for ax in axes: ax.set_xticks(range(len(names)),names,rotation=55,ha='right',fontsize=8)
    axes[0].set(ylabel='评论数（每运行总计777）');axes[0].legend(fontsize=8)
    axes[1].set(ylabel='该路线token子群错误率（%）')
    r.figure(fig,'22-token-truncation','各路线token截断：未知与实测分开展示','测试777条/运行；flag来自原始记录，未知不当未截断；N/类别组成见JSON。\n截断子群N<30或单类仅描述；不同路线子群可能不同，不能据错误率推断截断因果。','token-truncation.json',role='附录')


def documents(r,samples,errors,pending,primary,sensitivity,baselines,lora,costs):
    summary={x['method']:x for x in r.summary}
    main=table(['方法/代表配置','N/种子','Macro-F1 %','样本SD pp','Accuracy %','Weighted-F1 %','负面召回 %','非法率 %'],
        [[NAMES[x['method']]+({'tfidf':' char_1_3','bert':' len512全参','qwen_lora':' T01 r8 q/k/v/o'}.get(x['method'],'')),
          '777 / '+str(x['seeds'] or '参考单次'),fmt(x['macro_f1']['mean'],100),fmt(x['macro_f1']['sample_std'],100),fmt(x['accuracy']['mean'],100),fmt(x['weighted_f1']['mean'],100),fmt(x['per_class']['0']['recall'],100),fmt(x['invalid_rate']['mean'],100)] for x in r.summary])
    pairs=table(['A−B','Macro-F1差 pp','条件95%区间 pp','A对B错/A错B对','共同失败','精确McNemar p','七组Holm p'],
        [[NAMES[x['a']]+'−'+NAMES[x['b']],fmt(x['macro_f1_mean_delta'],100),str([round(v*100,4) for v in x['bootstrap_95_percentile']]),f'{x["A_correct_B_wrong"]}/{x["A_wrong_B_correct"]}',x['both_wrong'],fmt(x['exact_mcnemar_p']),fmt(x['holm_adjusted_p_seven_pairs'])] for x in r.pairs])
    observations=['# 对比分析数据观察稿（待本人确认解释）','',f'批次：{r.id}。已有冻结记录的事后探索性分析；没有新训练、模型加载或新增模型推理。','',main,'',
        '分类别P/R/F1/support、每种子完整指标和均值/样本SD见methods.json与分类别指标表.md。数据v1-20261023：train6210、val776、test777；划分种子67。0负面、1正面。MacBERT为fp32全参，Qwen提示与普通SFT LoRA为bf16；TF-IDF为CPU sklearn Pipeline。',
        '',f'在此固定划分和配置下，MacBERT与LoRA测试Macro-F1均值相差{(summary["bert"]["macro_f1"]["mean"]-summary["qwen_lora"]["macro_f1"]["mean"])*100:.4f}个百分点。它们的架构、规模、精度与输入格式不同，不能将全部差异归因为训练方式。',
        '',f'固定4-shot相对零样本的Macro-F1变化为{(summary["qwen_few_shot"]["macro_f1"]["mean"]-summary["qwen_zero_shot"]["macro_f1"]["mean"])*100:.4f}个百分点。仅适用于这组冻结训练示例及提示。',
        '',f'零样本非法率{summary["qwen_zero_shot"]["invalid_rate"]["mean"]*100:.4f}%，少样本非法率{summary["qwen_few_shot"]["invalid_rate"]["mean"]*100:.4f}%；非法输出计入对应真类FN，不能按首数字纠正正式成绩。',
        '',pairs,'','区间为seed67、5000次分层配对bootstrap的逐项百分位区间，固定已训练模型和观察到的类别比例，非训练总体区间，也非七组同时覆盖。McNemar固定seed42/确定性提示，检验正误而不是F1；不显著不等于等效。43/44敏感性见附录，不扩张777条的N。','',
        '## 验证参数观察','',table(['BERT配置','验证F1 %','最佳轮次','运行秒'],[[x['config_id'],fmt(x['macro_f1'],100),x['best_epoch'],fmt(x['end_to_end_s'])] for x in baselines if x['method']=='bert']),
        '',table(['LoRA配置','r/alpha/模块','验证F1 %','相对B00 pp','最佳轮/步','运行秒','可训练参数','适配器字节'],[[x['id'],f'{x["rank"]}/{x["alpha"]}/{x["target"]}',fmt(x['macro_f1'],100),fmt(x['delta_vs_B00'],100),f'{x["best_epoch"]}/{x["best_step"]}',fmt(x['perf_counter_seconds']),x['trainable_parameters'],x['adapter_bytes']] for x in lora]),
        '', 'BERT学习率三组已逐字段核验长度256、batch16、最多3轮等一致，仅seed42。len512同时改变batch，partial同时改变学习率/冻结；LoRA每组对同一B00而非累积改变，alpha/r=2。模块和秩改变也改变参数量。TF-IDF四组的实际特征维度、迭代数、拟合秒数与训练/验证log-loss见参数总表.md。','',
        '## 成本和资源观察','',table(['方法','已记录成功搜索＋补种子秒数（去重）'],[[NAMES[m],fmt(s)] for m,s in costs['totals_by_method'].items()]),
        '',table(['路线/种子','均值ms','P50ms','P95ms','吞吐条/秒','设备'],[[NAMES[x['method']]+'/'+str(x['seed']),fmt(x['mean_ms']),fmt(x['p50_ms']),fmt(x['p95_ms']),fmt(x['throughput']),x['device']] for x in r.latency_stats]),
        '', '11份延迟均为相同200条、同顺序、训练集20条预热、batch1、串行。计时含分词/传输/推理/解析、GPU同步，不含加载和读盘。吞吐为200/总秒数；各运行P95不合并后冒称总体P95。CPU和GPU资源不同。完整部署大小见deployment-storage.json：LoRA包含基座与适配器；参数/比例与TF-IDF特征/系数见原成本表。',
        '', '成功搜索已包含seed42优胜模型，不再次累计；B00只计一次，补43/44另列。训练、验证、保存计时边界不同，不叠加嵌套区间。失败与准备成本时间未知，已知合计仅为已记录成本下界。BERT历史显存峰值可能含检查点重载，整设备采样漏初始区间；推理峰值覆盖加载后测试/预热/延迟，RSS仅期末值。无能耗或电价换算。','',
        '## 错例覆盖与局限','',f'402个去重错例：20条本人情感判断＋AI建议归类后本人确认，382条由AI按确认规则辅助标注。{len(pending)}条需复核（原因可能重叠），其中源标签分歧{sum(s["review_status"]=="label_disagreement_pending" for s in samples)}条、情感不确定{sum(s["assessed_label"]=="uncertain" for s in samples)}条；不是402条人工或双人标注。',
        '',table(['方法','代表运行错误评论数','T1','T2','T3','T4','T5','T6','非T6完成率'],[[NAMES[p['method']],p['denominator'],*[p['type_counts'][f'T{i}'] for i in range(1,7)],fmt(p['classified_fraction'],100)+'%'] for p in primary]),
        '', '比例的单位为每方法seed42/确定性提示的一条错误评论。三种子明细留附录，不重复扩大评论样本数。T6包含分歧/不确定/可见性问题；T5可独立确定。只标错例不能计算所有含否定、酒店回复等评论的错误率。',
        '', '酒店单领域、单固定划分、类别不均衡、共同512字符限制、token可见性部分未知、单组few-shot、部分搜索单种子、不同基座和提示格式、笔记本硬件和历史成本缺项、已查看测试、AI标注未逐条人工确认，均限制推广。没有跨领域实测或归因干预，不能声称跨域下降或特定内部机制。错误案例提出的改进使用了测试反馈，未经新实验不能声称已改善。','']
    (r.report/'对比分析观察稿.md').write_text('\n'.join(observations),encoding='utf-8')
    classes=[]
    for x in r.summary:
        for seed in x['per_seed']:
            for c,metrics in seed['per_class'].items():
                classes.append([NAMES[x['method']],seed['seed'],c,metrics['support'],fmt(metrics['precision'],100),fmt(metrics['recall'],100),fmt(metrics['f1'],100),fmt(seed['weighted_f1'],100),fmt(seed['invalid_rate'],100)])
    (r.report/'分类别指标表.md').write_text('# 测试分类别指标（%）\n\n'+table(['方法','种子','类','support','P','R','F1','Weighted-F1','非法率'],classes),encoding='utf-8')
    (r.report/'参数总表.md').write_text('# 完整验证参数表\n\n仅validation776条，seed42。LoRA十组另见观察稿；BERT全部见formal-parameter-tables.json。\n\n'+table(['TF-IDF配置','F1 %','特征维度','迭代数','拟合秒','训练log-loss','验证log-loss','Pipeline字节'],[[x['config_id'],fmt(x['macro_f1'],100),x['features'],x['iterations'],fmt(x['fit_s']),fmt(x['train_log_loss']),fmt(x['validation_log_loss']),x['checkpoint_or_pipeline_bytes']] for x in baselines if x['method']=='tfidf'])+'\n\n'+table(['BERT配置','最佳完整部署字节','训练/可训练参数'],[[x['config_id'],x['checkpoint_or_pipeline_bytes'],x['trainable_parameters']] for x in baselines if x['method']=='bert'])+'\n\n所有候选体积来自只读文件大小查询，不加载权重；包含最佳模型与tokenizer或Pipeline，不含优化器/环境。完整Qwen部署见deployment-storage.json；LoRA增量见十组表。总参数、可训练量和比例见model-parameter-counts.json；TF-IDF特征与系数有不同含义，不混称神经网络参数。',encoding='utf-8')
    (r.report/'种子敏感性.md').write_text('# 43/44配对敏感性\n\n同777条；每行固定已训练种子；提示不复制成新的训练重复。每种子内七组Holm是敏感性展示，不新增确认性比较族。\n\n'+table(['种子','A−B','F1差 pp','95%条件区间 pp','A对B错','A错B对','共同失败','Holm p'],[[x['seed'],NAMES[x['a']]+'−'+NAMES[x['b']],fmt(x['macro_f1_delta'],100),str([round(z*100,4) for z in x['bootstrap_95_percentile']]),x['A_correct_B_wrong'],x['A_wrong_B_correct'],x['both_wrong'],fmt(x['holm_adjusted_p_within_seed_seven_pairs'])] for x in sensitivity]),encoding='utf-8')
    cases=[]
    case_notes=r.read(r.config.get('case_analysis_notes',DEFAULT_CASE_NOTES))
    by_id={s['sample_id']:s for s in samples}
    chosen=[1,3,10,12,18,20,8,15,6]
    for i in chosen:
        s=next(x for x in samples if x['queue_index']==i)
        sid=s['sample_id'];allruns=[p for rr in r.records.values() for p in rr if p['sample_id']==sid]
        representative=[p for p in allruns if p['seed'] in (42,None)]
        e=[x for x in errors if x['sample_id']==sid]
        case=dict(sample=s,all_runs=allruns,error_attributions=e,selection='confirmed pilot exemplars; not representative sample')
        cases.append(enrich_case(case,case_notes))
    r.write('casebook.json',cases)
    (r.report/'真实错误案例与局限.md').write_text(render_casebook(cases),encoding='utf-8')
    (r.report/'待复核清单.md').write_text('# 集中待复核清单\n\n无需现在逐条回复也可使用已完成分析；正式真值不变。请后续按ID裁定，另存人工复核记录，不覆盖AI初判。\n\n'+table(['完整ID','队列序号','源标签','判断','来源','证据片段','需决定问题'],[[p['sample_id'],p['queue_index'],p['source_label'],p['assessed_label'],p['annotation_origin'],'；'.join(p['evidence_quotes']),'；'.join(p['questions'])] for p in pending]),encoding='utf-8')
    claims=[]
    def claim(cid,observation,files,figures,scope,limits,pair=None):
        sources=[]
        for rel in files:
            p=ROOT/rel
            sources.append(dict(path=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        is_validation='parameter' in cid
        claims.append(dict(claim_id=cid,observation=observation,split='validation' if is_validation else 'test',
            runs_or_seeds=([dict(method=x['method'],config_id=x['config_id'],seed=42) for x in baselines]+[dict(method='qwen_lora',config_id=x['id'],seed=42) for x in lora]) if is_validation else [dict(run_id=x['run_id'],seed=x['seed']) for x in r.index],
            metric_or_difference=pair if pair else observation,interval_and_test=pair if pair else None,
            figures=figures,sources=sources,applicable_scope=scope,limitations=limits,confirmation_status='AI数据观察稿；学生最终解释待确认'))
    out='results/analysis/'+r.id+'/'
    claim('effect-five','五方案和多数类主表；逐种子先算再平均',[out+'methods.json'],['01-main-effect','02-seed-stability'],'固定测试777条','不同基座/输入/精度，不能全归因微调方法')
    claim('negative-miss','负面召回、混淆矩阵与非法FN',[out+'methods.json'],['03-confusion'],'源标签0负面245条','非法与合法误判分开；源标签未按AI改写')
    claim('format','严格0/1输出的非法率',[out+'methods.json',out+'annotation-statistics.json'],['18-invalid-output'],'全部777条为分母','不能用首数字诊断替代正式解析')
    claim('seed-stability','种子波动和每评论0/1/2/3次失败',[out+'seed-stability.json',out+'seed43-44-sensitivity.json'],['02-seed-stability'],'固定模型42/43/44','不是扩大N或训练总体推断')
    claim('parameter-validation','BERT学习率与LoRA模块/秩验证表',[out+'formal-parameter-tables.json'],['04-bert-learning-rate','05-lora-parameters'],'validation776 seed42','len512/batch及partial/LR组合变化')
    claim('cost-tradeoff','延迟/已记录投入/完整部署体积',[out+'latency.json',out+'formal-cost-ledger.json','results/summaries/deployment-storage.json'],['07-effect-latency','08-resources-storage','12-effect-recorded-training-time'],'本机实测部署路线','CPU/GPU不同；历史成本为下界，时间边界不同')
    claim('shared-failures','改善、退化和共同失败',[out+'paired-comparisons.json',out+'annotation-statistics.json'],['14-error-types','17-common-failure'],'代表seed42和确定性提示','20本人确认＋382AI辅助；未复核不作真值')
    for i,pair in enumerate(r.pairs,1):
        claim(f'pair-{i}',NAMES[pair['a']]+'−'+NAMES[pair['b']],[out+'paired-comparisons.json'],['10-paired-f1-difference'],'固定模型条件差异','事后探索性，逐项CI；正误McNemar非F1检验',pair)
    (r.report/'结论证据台账.json').write_text(json.dumps(claims,ensure_ascii=False,indent=2),encoding='utf-8')
    core={'01-main-effect','03-confusion','04-bert-learning-rate','05-lora-parameters','06-latency-ecdf','07-effect-latency','10-paired-f1-difference','14-error-types','17-common-failure','18-invalid-output'}
    for f in r.figure_index:
        f['role']='正文候选（待学生选用）' if f['id'] in core else '附录'
    r.write('figure-index.json',r.figure_index)
    cards=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>正式离线对比</title><style>body{font:16px/1.7 system-ui;max-width:1300px;margin:auto;padding:24px;color:#263649}img{width:100%}section{padding:20px;border-bottom:1px solid #ddd}</style><h1>正式离线对比图表</h1><p>20本人确认＋382 AI辅助；待复核'+str(len(pending))+'条；无新增模型实验。学生最终解释待确认。</p>']
    for f in r.figure_index:
        cards.append(f'<section><h2>{html.escape(f["id"]+" · "+f["title"])}</h2><p>{f["role"]}</p><img src="../../figures/{r.id}/{f["png"]}"><p>{html.escape(f["caption"])}</p><p>数据：{html.escape(f["data"])} ｜ <a href="../../figures/{r.id}/{f["pdf"]}">PDF</a></p></section>')
    (r.report/'图表总览.html').write_text(''.join(cards)+'</html>',encoding='utf-8')
    (r.report/'AI使用记录.md').write_text('# AI使用记录\n\n工具：Codex；精确模型版本未知。时间见status.json、annotation-audit.json及证据采集manifest。382条逐条读取全文后形成情感、情境与原文证据，再核对冻结标签、共同前缀和运行输出；没有用关键词/正则代替语义审阅。TSV保存逐条判断，脚本只展开、转录与校验。未记录精确逐条耗时，不伪造标注工时。\n\n20条为交接第5节本人判断＋AI建议分类后本人确认；382条为本轮授权下AI辅助，不能称人工逐条验收或双人标注。试标19/20与源标签一致仅是这组定向错例的描述，不是kappa或总体质量估计。\n\nAI完成：代码核查、离线重算、辅助标注、图表、真实案例、带证据的数据观察草稿。学生负责最终解释与课程报告；建议来自测试反馈，没有新实验验证效果。没有修改原始标签/预测、提示/模型或依赖版本。\n',encoding='utf-8')
    (r.report/'验收清单.md').write_text('# 执行验收\n\n- [x] 20条完整ID、情感与逐运行类别转录；第8条分歧保留。\n- [x] 382条逐条AI辅助，402条全覆盖；1118条错误运行一一对应。\n- [x] 11份777条预测、真值、顺序、状态不变；指标重算误差≤1e-12。\n- [x] 五方案＋多数类完整分类别指标、逐种子/均值/样本SD/非法率。\n- [x] 七组5000次条件bootstrap、精确McNemar/Holm；43/44敏感性。\n- [x] BERT学习率、LoRA模块/秩，TF-IDF四组与LoRA十组表。\n- [x] T1—T6来源/分母/待复核明确，9个真实案例含相反方向和分歧。\n- [x] 同200条延迟、训练集20条预热、吞吐/分位重算；成本去重、内存/体积边界。\n- [x] 原始训练日志只读补取54份并校验；真实loss分路线重绘。\n- [x] 16个基线候选的完整最佳模型/Pipeline文件大小只读补取。\n- [ ] 学生裁定待复核样本、确认最终解释与课程报告。\n- [ ] 新机真实模型复现、模型外链发布（本次范围之外）。\n\n执行完成指工程派生材料生成并核验，不表示学生已验收解释。视觉审查实测记录另见图表审查清单.md。\n\n缺项：BERT完整逐步墙钟含保存的时间线未记录，图用累计batch实测时间并明确排除区间；LoRA有Trainer实际耗时。失败工时、能耗、阶段RSS峰值未知。不因缺项重训。\n',encoding='utf-8')
    command=f'python scripts/build_comparison_report.py --config {r.config_path} --run-id <全新批次ID>'
    (r.report/'README.md').write_text('# 正式对比交付索引\n\n复跑命令（仅离线，目录必须未存在）：\n\n```powershell\n'+command+'\n```\n\n配置固定bootstrap种子67、5000次、batch250、linear分位。使用现有依赖，无模型目录依赖。标注重放见'+r.config['annotations']+'/README.md；无需再次调用AI。\n\n- 对比分析观察稿.md：主表、参数、区间、成本和局限（解释待本人确认）。\n- 真实错误案例与局限.md / casebook.json：全文、前缀、所有模型与种子输出。\n- 待复核清单.md：集中124条，另存人工裁定，不修改正式真值。\n- 分类别指标表.md、参数总表.md、种子敏感性.md：明细。\n- 图表总览.html：PNG/PDF与数据来源；10组正文候选，其余附录。\n- 结论证据台账.json、AI使用记录.md、验收清单.md。\n- results/analysis/'+r.id+'/status.json：环境、命令、输入输出SHA256及Git状态。\n\n图表审查清单.md和软件检查记录用于工程验收；不替代学生结果解释。\n',encoding='utf-8')


def finalize_formal(r):
    sensitivity=seed_sensitivity(r)
    samples,errors,pending,primary=annotations(r)
    evidence,configs,epochs,baselines,lora=parameters(r)
    costs=formal_costs(r)
    curves(r,evidence,configs)
    extra_figures(r,primary)
    documents(r,samples,errors,pending,primary,sensitivity,baselines,lora,costs)
    for path in [Path(__file__),ROOT/'scripts/case_analysis.py']:
        r.sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    r.note(f'Formal annotation coverage verified:20 human-confirmed+382 AI, {len(errors)} error runs, {len(pending)} pending samples; {len(r.figure_index)} figure pairs.')
