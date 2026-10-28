"""把 train_*.md 里的脚本转成可以直接 Import 到 Kaggle 的 .ipynb。

用法：
    python _make_notebooks.py

产出（notebooks/ 目录）：
    01_TFIDF基线.ipynb
    02_MacBERT全参微调.ipynb
    03_Qwen3提示.ipynb
    04_Qwen3_LoRA.ipynb

每个 notebook 的结构：
    [md]  标题 + 怎么跑
    [code] 配置（SMOKE_OVERRIDE）+ 依赖自检 + GPU 自检
    [code] 主脚本（从 .md 抽出，已去掉 exec 包装）
    [code] 产物清单（打印输出文件与大小，方便下载）
"""
import os, re, json, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'notebooks')
os.makedirs(OUT, exist_ok=True)

# (源 md, 输出 ipynb, 标题, 预计耗时说明, 产物目录名, 权重名)
JOBS = [
    ('train_tfidf.md',  '01_TFIDF基线.ipynb',      '① TF-IDF + LogisticRegression 基线',
     '约 1 分钟（CPU 就够，不需要 GPU）', 'results_TFIDF', None),
    ('train_bert.md',   '02_MacBERT全参微调.ipynb', '② MacBERT-base 全参微调（8 组超参 + 数据量曲线）',
     '冒烟约 3 分钟 / 正式约 45 分钟（需要 GPU）', 'results_BERT_FULL', 'bert_FULL.pt'),
    ('train_prompt.md', '03_Qwen3提示.ipynb',      '③ Qwen3-1.7B 零样本 / 少样本提示',
     '冒烟约 6 分钟 / 正式约 25 分钟（需要 GPU）', 'results_PROMPT_FULL', None),
    ('train_lora.md',   '04_Qwen3_LoRA.ipynb',     '④ Qwen3-1.7B + LoRA 轻量微调',
     '冒烟约 5 分钟 / 正式约 50 分钟（需要 GPU）', 'results_QWEN_FULL', None),
]


def extract_code(md_path):
    """从 .md 里抽出主代码块的内容（去掉 exec(r\"\"\" ... \"\"\") 包装）。"""
    with open(md_path, encoding='utf-8') as fh:
        text = fh.read()
    blocks = re.findall(r'```python\n(.*?)```', text, flags=re.S)
    if not blocks:
        raise SystemExit(f'!! {md_path} 里没有 ```python 代码块')
    main = None
    for b in blocks:
        if 'exec(r"""' in b:
            main = b
            break
    if main is None:
        main = max(blocks, key=len)
    # 去掉包装
    m = re.search(r'exec\(r"""(.*)"""\)', main, flags=re.S)
    if m:
        code = m.group(1)
    else:
        code = main
    # 去掉首尾空行
    code = code.strip('\n')
    # 把 SMOKE 开关改成可被上一个单元格覆盖
    code, n = re.subn(r'(?m)^(SMOKE\s*=\s*)True(\s*#.*)?$',
                      r'\1bool(globals().get("SMOKE_OVERRIDE", True))'
                      r'   # <- switch lives in the config cell above', code)
    if n == 0:
        print(f'  ⚠ {md_path}: 没找到 SMOKE 开关行')
    return code, n


def cells_md(lines):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': lines}


def cells_code(src):
    if not src.endswith('\n'):
        src += '\n'
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': src.splitlines(keepends=True)}


CONFIG_CELL = '''# ============================================================
# ① 配置区：只需要改这一行
#    True  = 冒烟（跑几分钟，验证代码能跑通）
#    False = 正式（跑完整实验，出报告要用的数字）
# ============================================================
SMOKE_OVERRIDE = True

# ============================================================
# ② 依赖自检（只在该升级时才升级，不会乱动 torch）
# ============================================================
import importlib.metadata as _md, subprocess, sys

def _ver(p):
    try:
        return _md.version(p)
    except Exception:
        return None

def _tup(s):
    return tuple(int(x) for x in s.split('+')[0].split('.')[:2] if x.isdigit())

NEED = {'transformers': '4.51', 'peft': '0.13', 'accelerate': '0.30'}
_bad = []
for _p, _min in NEED.items():
    _v = _ver(_p)
    if _v is None:
        print(f'✗ {_p}: 未安装'); _bad.append(_p)
    elif _tup(_v) >= _tup(_min):
        print(f'✓ {_p}: {_v}')
    else:
        print(f'⚠ {_p}: {_v} < {_min}，将被升级'); _bad.append(_p)
if _bad:
    print('正在安装/升级:', _bad)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U'] + _bad, check=False)
    print('装完了。下面那个单元格的脚本会用新版本。')

# ============================================================
# ③ GPU 自检
# ============================================================
try:
    import torch
    print('torch', torch.__version__, '| CUDA 可用:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    print('torch 检查失败:', e)
'''

TAIL_CELL = '''# ============================================================
# 产物清单：确认结果都产出了，然后去右侧 Output 面板下载
# ============================================================
import os, glob
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
for d in sorted(glob.glob(os.path.join(WORK, 'results_*'))):
    if not os.path.isdir(d):
        continue
    print('\\n📁', os.path.basename(d))
    for f in sorted(glob.glob(os.path.join(d, '*'))):
        if os.path.isfile(f):
            print(f'   {os.path.basename(f):32s} {os.path.getsize(f)/1024:9.1f} KB')
for z in sorted(glob.glob(os.path.join(WORK, '*.zip'))):
    print(f'\\n📦 {os.path.basename(z):40s} {os.path.getsize(z)/1024/1024:8.2f} MB')
for p in sorted(glob.glob(os.path.join(WORK, '*.pt'))):
    print(f'🧠 {os.path.basename(p):40s} {os.path.getsize(p)/1024/1024:8.2f} MB')
'''


def build():
    for src_md, out_ipynb, title, time_hint, outdir, weight in JOBS:
        code, _ = extract_code(os.path.join(HERE, src_md))
        md_lines = [
            f'# {title}\n', '\n',
            f'**预计耗时**：{time_hint}\n', '\n',
            '## 跑之前确认\n', '\n',
            '1. 右上角 **Settings → Accelerator** 选 `GPU T4 x2`（提示/LoRA 两条臂必须要）\n',
            '2. 右侧 **Add Input** → **Upload** → 选 `chunksenti_v1-20261023.zip`'
            '（**不要建公开 Dataset**，只挂到自己 session）\n',
            '3. 确认 **Internet** 是 `On`（要下载模型权重）\n', '\n',
            '## 跑法\n', '\n',
            '1. 第一个代码单元里把 `SMOKE_OVERRIDE = True` 保持不变，点 **Run All**\n',
            '2. 冒烟通过后，把 `SMOKE_OVERRIDE` 改成 `False`，'
            '**Session → Restart & Clear Output** 后再 Run All\n', '\n',
            f'**产出目录**：`{outdir}/`'
            + (f'　**模型权重**：`{weight}`（这个不要进提交包，只给链接）' if weight else '')
            + '\n',
        ]
        nb = {
            'cells': [
                cells_md(md_lines),
                cells_code(CONFIG_CELL),
                cells_code(code),
                cells_code(TAIL_CELL),
            ],
            'metadata': {
                'kernelspec': {'display_name': 'Python 3',
                               'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python', 'version': '3.11.9'},
            },
            'nbformat': 4, 'nbformat_minor': 0,
        }
        p = os.path.join(OUT, out_ipynb)
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(nb, fh, ensure_ascii=False, indent=1)
        print(f'✓ {out_ipynb}  ({os.path.getsize(p)/1024:.0f} KB, {len(code.splitlines())} 行代码)')


if __name__ == '__main__':
    build()
