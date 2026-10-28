"""Plot measured memory, complete deployment and LoRA incremental storage separately."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHODS = ['tfidf', 'bert', 'qwen_zero_shot', 'qwen_few_shot', 'qwen_lora']
LABELS = ['TF-IDF\nCPU', 'MacBERT\nGPU', 'Qwen零样本\nGPU', 'Qwen少样本\nGPU', 'LoRA\nGPU']


def resource_breakdown(latency, storage, adapter_weight_bytes):
    rows = []
    for method in METHODS:
        bench = [r for r in latency if r['method'] == method]
        deployed = [r['deployment_bytes'] for r in storage if r['method'] == method]
        if not bench or not deployed or len(set(deployed)) != 1:
            raise ValueError('Expected one frozen deployment size per method across seeds')
        peaks = [r['resources']['inference_allocated_peak'] for r in bench if r['resources']['inference_allocated_peak'] is not None]
        rows.append(dict(method=method, allocated_peak_MiB=max(peaks)/2**20 if peaks else None,
                         deployment_bytes=deployed[0], deployment_MB=deployed[0]/1e6))
    lookup = {r['method']: r for r in rows}
    base = lookup['qwen_zero_shot']['deployment_bytes']
    if base != lookup['qwen_few_shot']['deployment_bytes']:
        raise ValueError('Prompt methods must use the same frozen Qwen base')
    incremental = lookup['qwen_lora']['deployment_bytes'] - base
    if not 0 < adapter_weight_bytes <= incremental:
        raise ValueError('Adapter weights must fit inside the measured full deployment increment')
    return dict(methods=rows, base_deployment_bytes=base,
                adapter_weight_bytes=adapter_weight_bytes, adapter_weight_MB=adapter_weight_bytes/1e6,
                adapter_config_and_other_bytes=incremental-adapter_weight_bytes,
                full_increment_bytes=incremental, full_increment_MB=incremental/1e6,
                memory_scope='after model load through test/warmup/latency; maximum across seed runs',
                storage_scope='decimal MB; full deployment includes tokenizer and base; excludes optimizer/environment',
                adapter_scope='T01 ordinary LoRA weight file only; full increment also includes adapter configuration')


def draw_resource_figure(data, plt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    colors = ['#526b8b', '#167685', '#b58437', '#9470a4', '#c35959']
    rows = data['methods']
    for i, row in enumerate(rows):
        if row['allocated_peak_MiB'] is not None:
            axes[0].bar(i, row['allocated_peak_MiB'], color=colors[i])
            axes[0].text(i, row['allocated_peak_MiB']+65, f"{row['allocated_peak_MiB']:.1f}", ha='center', fontsize=8)
    axes[0].text(0, 100, '不适用', ha='center', fontsize=9)
    axes[0].set_ylabel('CUDA allocated峰值（MiB）')
    axes[0].set_ylim(0, 3900)
    for i, row in enumerate(rows):
        axes[1].bar(i, row['deployment_MB'], color=colors[i])
        axes[1].text(i, row['deployment_MB']+80, f"{row['deployment_MB']:.2f}", ha='center', fontsize=8)
    axes[1].set_ylabel('完整部署文件（MB，十进制）')
    axes[1].set_ylim(0, 4700)
    for ax in axes[:2]:
        ax.set_xticks(range(5), LABELS, rotation=15)
    axes[0].set_title('A  推理资源记录')
    axes[1].set_title('B  独立部署所需完整文件')
    axes[2].bar([0], [data['adapter_weight_MB']], color=colors[-1], width=.45)
    axes[2].text(0, data['adapter_weight_MB']+.35, f"{data['adapter_weight_MB']:.2f} MB", ha='center', fontweight='bold', fontsize=12)
    axes[2].set(xticks=[0], xticklabels=['T01新增适配器权重'], ylabel='新增权重（MB，十进制）', ylim=(0,16))
    axes[2].set_title('C  LoRA增量，单独尺度')
    axes[2].text(.5, .15, f"含配置的完整增量：{data['full_increment_MB']:.2f} MB\n"
                 f"基座仍需：{data['base_deployment_bytes']/1e6:.2f} MB\n适配器无法脱离基座独立推理", transform=axes[2].transAxes,
                 ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', edgecolor='#d7dfe5', boxstyle='round,pad=.5'))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = (ROOT/args.output).resolve()
    target.relative_to(ROOT)
    if target.exists():
        raise FileExistsError('Use a new output directory')
    sources = {}
    def read(rel):
        p = ROOT/rel
        sources[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text(encoding='utf-8'))
    latency = read('results/analysis/comparison-formal-20261028-03/latency.json')
    storage = read('results/summaries/deployment-storage.json')
    parameters = read('results/analysis/comparison-formal-20261028-03/model-parameter-counts.json')
    adapter = next(r for r in parameters if r.get('config_id')=='T01' and r['method']=='qwen_lora')
    data = resource_breakdown(latency, storage, adapter['adapter_bytes'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in ['Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei'] if name in available), 'DejaVu Sans')
    plt.rcParams.update({'font.family':font, 'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False})
    fig = draw_resource_figure(data, plt)
    fig.suptitle('显存、完整部署与LoRA新增适配器', fontsize=15, fontweight='bold')
    caption = '环境E01：i7-14650HX、16GiB、RTX 4070 Laptop GPU；MacBERT FP32，Qwen BF16；推理batch1。\n峰值范围为加载后测试/预热/计时；存储不含优化器和环境。三个面板采用不同单位/尺度，LoRA完整部署仍包含基座。'
    fig.text(.5,.015,caption,ha='center',va='bottom',fontsize=8,color='#46596b')
    fig.tight_layout(rect=(0,.13,1,.91))
    target.mkdir(parents=True)
    for ext in ['png','pdf']:
        fig.savefig(target/f'resources-with-adapter.{ext}',dpi=175)
    plt.close(fig)
    (target/'resource-breakdown.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (target/'README.md').write_text('# 资源图：补充LoRA新增适配器\n\nA为CUDA allocated峰值；B为完整部署大小；C单独展示T01新增权重。\n\n'
        f"适配器权重{data['adapter_weight_MB']:.5f}MB；含配置的完整增量{data['full_increment_MB']:.6f}MB；完整LoRA部署{data['methods'][-1]['deployment_MB']:.6f}MB。\n\n"
        'MB按1000000字节，MiB按1048576字节。环境说明见docs/environment/实验运行环境与延迟口径.md。原正式批次及原图保持不变，后续报告使用本补充图。\n\n'
        '复跑：python scripts/resource_figure.py --output reports/figures/<new-resource-figure-id>\n\n本轮AI整理已有测量与绘图，未加载模型、重新测量或修改正式数据。\n',encoding='utf-8')
    for rel in ['scripts/resource_figure.py','docs/environment/实验运行环境与延迟口径.md']:
        sources[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    manifest=dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),input_sha256=sources,
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file()},
        plot_font=font,matplotlib=matplotlib.__version__,training=False,model_loading=False,inference=False)
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(target),adapter_weight_MB=data['adapter_weight_MB'],full_increment_MB=data['full_increment_MB']),ensure_ascii=False))


if __name__=='__main__':
    main()
