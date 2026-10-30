"""Plot verified cross-machine measurements without running any models."""
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    root = Path(__file__).resolve().parents[1]
    font = Path('/mnt/c/Windows/Fonts/simhei.ttf')
    if not font.exists():
        font = Path('C:/Windows/Fonts/simhei.ttf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'SimHei', 'font.size': 11, 'axes.unicode_minus': False})
    rows = json.loads((root / 'reports/report-crossgpu-20261030-01/chart-data.json').read_text(encoding='utf-8'))
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.3), gridspec_kw={'width_ratios': [1.7, 1]})
    y = list(range(len(rows)))
    left.barh([i - .17 for i in y], [r['reference_macro_f1'] * 100 for r in rows],
              height=.30, color='#245e87', label='4070 原结果')
    left.barh([i + .17 for i in y], [r['reproduced_macro_f1'] * 100 for r in rows],
              height=.30, color='#d8963f', label='4060 完整复验')
    for i, row in enumerate(rows):
        left.text(row['reference_macro_f1'] * 100 + .6, i - .17,
                  f"{row['reference_macro_f1'] * 100:.2f}", va='center', fontsize=9)
        left.text(row['reproduced_macro_f1'] * 100 + .6, i + .17,
                  f"{row['reproduced_macro_f1'] * 100:.2f}", va='center', fontsize=9)
    left.set(yticks=y, yticklabels=[r['label'] for r in rows], xlim=(0, 103), xlabel='Macro-F1（%）',
             title='相同测试集与权重的两机完整推理')
    left.invert_yaxis()
    left.legend(loc='upper left', bbox_to_anchor=(0, -.13), ncol=2, frameon=False, fontsize=10)
    right.barh(y, [r['delta_pp'] for r in rows], color=['#6c8791' if r['delta_pp'] == 0 else '#d8963f' for r in rows], height=.45)
    right.axvline(0, color='#58636b', linewidth=.8)
    for i, row in enumerate(rows):
        value = row['delta_pp']
        right.text(value + .05 if value >= 0 else value - .05, i,
                   f'{value:+.3f}', va='center', ha='left' if value >= 0 else 'right', fontsize=10)
    right.set(yticks=y, yticklabels=[], xlim=(-.65, 2.35), xlabel='4060 - 4070（百分点）',
              title='同机重复不代表跨机一致')
    right.invert_yaxis()
    for axis in (left, right):
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
    fig.text(.5, .025, '训练路线各3种子取均值；提示各1次，每次777条。原始指标保留，不混合两机结果。',
             ha='center', fontsize=10, color='#4f5b63')
    fig.tight_layout(rect=(0, .16, 1, 1))
    output = root / 'docs/最终提交/20261029/assets/cross-machine-results.png'
    fig.savefig(output, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
