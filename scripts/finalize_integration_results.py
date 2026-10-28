"""Derive reviewable tables and normalized latency evidence from immutable runs."""
from pathlib import Path
import sys
from collections import defaultdict
import statistics

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common.io import load, rows, save, save_rows, digest, local_paths


def main():
    root=Path(__file__).resolve().parents[1]
    paths=local_paths(root/'configs/paths.local.json')
    queue=root/'results/local/integration-20261028-v1'
    assert load(queue/'status.json')['status']=='completed'
    bench=load(root/'results/benchmarks/integration.json')
    frozen=load(Path(paths['data'])/'latency_ids.json')
    warmup=load(Path(paths['data'])/'smoke_ids.json')
    for b in bench:
        source=queue/b['run_id']/'latency.jsonl'
        details=rows(source)
        assert [r['sample_id'] for r in details]==frozen
        assert b['benchmark']['warmup_ids']==warmup
        assert len(details)==200
        config=load(queue/b['run_id']/'config.json')
        for r in details:
            r.update(method=b['method'],seed=b['seed'],model_id=config['model_id'],
                     timing_scope=b['benchmark']['timing_scope'],source_artifact_sha256=digest(source))
        save_rows(root/'results/benchmarks/details'/f'{b["run_id"]}.jsonl',details)
    lines=['# 同机推理成本', '',
           '均值为各训练种子的单模型均值再平均；零/少样本各一次。实际CPU/GPU路线分别标注。', '',
           '|方法|运行数|设备|端到端均值ms|P95均值ms|allocated峰值最大MiB|',
           '|---|---:|---|---:|---:|---:|']
    groups=defaultdict(list)
    for b in bench:
        groups[b['method']].append(b)
    for method, group in groups.items():
        peak=[g['resources']['inference_allocated_peak'] for g in group if g['resources']['inference_allocated_peak'] is not None]
        lines.append('|'+ '|'.join([method,str(len(group)),group[0]['benchmark']['device'],
                     f"{statistics.mean(g['benchmark']['mean_ms'] for g in group):.3f}",
                     f"{statistics.mean(g['benchmark']['p95_ms'] for g in group):.3f}",
                     f'{max(peak)/2**20:.1f}' if peak else '不适用'])+'|')
    lines+=['', '峰值范围为模型加载后测试/预热/延迟全过程；RSS为期末采样，不是采样峰值。',
            '训练分项与累计成本见training-costs.json；不同时间边界不叠加为虚构总时长。']
    (root/'results/summaries/costs.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    accepted=[]
    for r in rows(root/'results/run_index.jsonl'):
        p=root/'results/predictions'/f'{r["run_id"]}.jsonl'
        result=load(root/'results/metrics'/f'{r["run_id"]}.json')
        assert result['complete']
        accepted.append(dict(run_id=r['run_id'],split=r['split'],n=result['metrics']['n'],
                            derived_predictions_sha256=digest(p),source_sha256=r['source_sha256']))
    save(root/'reports/audits/final-predictions.json',accepted)
    save(root/'reports/audits/failures.json',[
        dict(run_id='smoke-zero-20261028',stage='integration_smoke',reason='WSL Git ownership metadata read failed before inference',
             retry='smoke-zero-retry1-20261028',retry_status='completed'),
        dict(stage='baseline_smoke',reason='Initial single-class subset; retry used both classes; original retained'),
        dict(stage='baseline_test_setup',reason='Frozen warmup IDs came from non-test splits; original retained; fixed ID lookup'),
        dict(stage='baseline_latency',reason='Initial TF-IDF latency overlapped training; replaced by test-isolated')])
    print('Audited final prediction sets',len(accepted),'normalized benchmarks',len(bench))


if __name__=='__main__':
    main()
