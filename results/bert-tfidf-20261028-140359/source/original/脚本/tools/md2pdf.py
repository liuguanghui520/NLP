"""把报告 Markdown 转成 PDF（Markdown → HTML → Chrome headless 打印）。

为什么不用 pandoc / LaTeX
------------------------
本机没有 pandoc、xelatex、wkhtmltopdf，WSL 那边也没有装。
但有 **Chrome**（headless 打印质量好、支持中文字体）和系统 Python 3.11 的
`markdown` 库，够用了。

用法
----
    python tools/md2pdf.py                       # 项目报告_终版.md → 项目报告_终版.pdf
    python tools/md2pdf.py --md 报告.md --pdf 报告.pdf
    python tools/md2pdf.py --check-only           # 只检查引用的图片在不在，不生成

注意
----
* 渲染用 **系统 Python 3.11**（有 markdown 库），不是 D:\\nlp_venv（那个是训练环境，
  装的是 torch 那一套；为了一个 PDF 去污染训练 venv 不值当）。
* 图片用相对路径引用，HTML 写在 Markdown 同目录，Chrome 以 file:// 加载，
  相对路径能解析到。
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 系统 Python 里装了 markdown 库的那个解释器
PY_CANDIDATES = [
    r'C:\Users\Yang\AppData\Local\Programs\Python\Python311\python.exe',
    sys.executable,
]

CHROME_CANDIDATES = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    shutil.which('chrome') or '',
    shutil.which('msedge') or '',
]

CSS = r"""
@page { size: A4; margin: 18mm 15mm 16mm 15mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Microsoft YaHei", "微软雅黑", "DengXian", "等线", "SimSun", sans-serif;
  font-size: 10.5pt; line-height: 1.75; color: #1a1a1a; margin: 0;
}
h1 { font-size: 19pt; margin: 0 0 10pt 0; padding-bottom: 6pt;
     border-bottom: 2.5pt solid #2c3e50; page-break-before: always; page-break-after: avoid; }
body > h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 15pt; margin: 18pt 0 8pt 0; color: #1f3864; page-break-after: avoid; }
h3 { font-size: 12.5pt; margin: 14pt 0 6pt 0; color: #2c3e50; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 12pt 0 5pt 0; color: #34495e; page-break-after: avoid; }
p { margin: 0 0 7pt 0; text-align: justify; }
ul, ol { margin: 0 0 8pt 0; padding-left: 22pt; }
li { margin: 0 0 3pt 0; }
blockquote {
  margin: 9pt 0; padding: 7pt 11pt; background: #f6f8fa;
  border-left: 3.5pt solid #5b7fa6; color: #333;
}
blockquote p:last-child { margin-bottom: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 9pt 0 11pt 0;
  font-size: 9.5pt; page-break-inside: auto;
}
th, td { border: 0.6pt solid #b9c3cc; padding: 4.5pt 6pt; vertical-align: top; }
th { background: #eef2f6; font-weight: 600; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
code {
  font-family: Consolas, "Courier New", monospace; font-size: 9pt;
  background: #f2f4f6; padding: 0.5pt 2.5pt; border-radius: 2pt;
}
pre {
  background: #f7f8fa; border: 0.6pt solid #d5dae0; border-left: 3pt solid #9db2c7;
  padding: 8pt 10pt; overflow-x: auto; font-size: 8.5pt; line-height: 1.5;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8.5pt; }
img { max-width: 100%; height: auto; display: block; margin: 8pt auto 4pt auto;
      page-break-inside: avoid; }
hr { border: none; border-top: 0.8pt solid #ccd3da; margin: 14pt 0; }
a { color: #1f5fa8; text-decoration: none; }
strong { color: #000; }
blockquote strong, table strong { color: #8a1c1c; }
"""


def find_py():
    """找一个能 import markdown 的解释器。"""
    for c in PY_CANDIDATES:
        if not c:
            continue
        p = Path(c)
        if not p.exists():
            continue
        try:
            r = subprocess.run([str(p), '-c', 'import markdown; print(markdown.__version__)'],
                               capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=40)
            if r.returncode == 0:
                return str(p)
        except Exception:
            continue
    return None


def find_chrome():
    for c in CHROME_CANDIDATES:
        if c and Path(c).exists():
            return c
    return None


def md_to_html(md_path: Path, py: str) -> str:
    """调系统 Python 做 md→html（因为 markdown 库只装在那儿）。"""
    script = r'''
import sys, markdown
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding='utf-8').read()
html = markdown.markdown(
    text,
    extensions=['tables', 'fenced_code', 'sane_lists', 'toc', 'attr_list'],
    extension_configs={'toc': {'title': '目录', 'toc_depth': '2-3'}})
open(dst, 'w', encoding='utf-8').write(html)
'''
    tmp = Path(tempfile.gettempdir()) / '_md2html.py'
    tmp.write_text(script, encoding='utf-8')
    out = tmp.with_suffix('.html')
    r = subprocess.run([py, str(tmp), str(md_path), str(out)],
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f'md→html 失败:\n{r.stderr[:2000]}')
    return out.read_text(encoding='utf-8')


def check_images(md_path: Path):
    """报告里引用的图片有没有真的存在。"""
    text = md_path.read_text(encoding='utf-8')
    refs = re.findall(r'!\[[^\]]*\]\(([^)\s]+)', text)
    missing, found = [], []
    base = md_path.parent
    for r in refs:
        if r.startswith(('http://', 'https://')):
            continue
        p = (base / r).resolve()
        (found if p.exists() else missing).append(r)
    return found, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default='项目报告_终版.md')
    ap.add_argument('--pdf', default=None)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()

    md_path = (ROOT / args.md).resolve()
    if not md_path.exists():
        print(f'!! 找不到 Markdown：{md_path}')
        return 1
    pdf_path = Path(args.pdf) if args.pdf else md_path.with_suffix('.pdf')

    found, missing = check_images(md_path)
    print(f'📄 {md_path.name}')
    print(f'   引用的图片：{len(found)} 张存在，{len(missing)} 张缺失')
    for m in missing:
        print(f'   ⚠ 缺图：{m}')
    if missing:
        print('   → 缺图也能生成 PDF（Chrome 会留一个空框），但建议等图生成完再出终版。')

    if args.check_only:
        return 0

    py = find_py()
    if not py:
        print('!! 找不到带 markdown 库的解释器')
        print('   → 试：pip install markdown 后重跑，或手动用 VS Code 导出 PDF')
        return 2
    chrome = find_chrome()
    if not chrome:
        print('!! 找不到 Chrome / Edge')
        return 2

    print(f'   解释器：{py}')
    print(f'   渲染器：{chrome}')

    body = md_to_html(md_path, py)
    html = ('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
            f'<title>{md_path.stem}</title>\n<style>{CSS}</style>\n</head>\n'
            f'<body>\n{body}\n</body>\n</html>\n')
    html_path = md_path.with_name(md_path.stem + '.html')
    html_path.write_text(html, encoding='utf-8')

    profile = Path(tempfile.mkdtemp(prefix='md2pdf_'))
    cmd = [
        chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
        '--disable-extensions', '--disable-dev-shm-usage',
        f'--user-data-dir={profile}',
        '--print-to-pdf=' + str(pdf_path),
        '--no-pdf-header-footer',
        '--run-all-compositor-stages-before-draw',
        '--virtual-time-budget=20000',
        html_path.as_uri(),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
    try:
        shutil.rmtree(profile, ignore_errors=True)
    except Exception:
        pass

    if not pdf_path.exists():
        print('!! Chrome 没生成 PDF')
        print((r.stderr or r.stdout)[-1500:])
        return 3

    size = pdf_path.stat().st_size
    print(f'✓ 已生成 {pdf_path.name}（{size/1024:.0f} KB）')
    if size < 20_000:
        print('   ⚠ 文件偏小，可能没渲染出内容，检查一下')
    # HTML 是中间产物，删掉
    try:
        html_path.unlink()
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
