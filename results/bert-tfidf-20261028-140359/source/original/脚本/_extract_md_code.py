"""从 .md 里抽出 python 代码块，做语法检查 + 可选实跑"""
import sys, re, os

# Windows 控制台默认是 GBK，打印 '✓' / '✗' 会抛 UnicodeEncodeError，
# 一旦在这里崩掉，退出码非 0，run_bert_local.bat 会误判成"抽取失败"而拒绝启动。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

md_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else '_extracted.py'

text = open(md_path, encoding='utf-8').read()
m = re.search(r'exec\(r"""(.*?)"""\)', text, re.S)
if not m:
    print('✗ 没找到 exec(r"""...""") 代码块'); sys.exit(1)
code = m.group(1)

# 去掉缩进（代码块整体缩进 0，但要保证）
lines = code.split('\n')
open(out_path, 'w', encoding='utf-8').write(code)

try:
    compile(code, out_path, 'exec')
    print(f'✓ 语法检查通过 ({len(lines)} 行) -> {out_path}')
except SyntaxError as e:
    print(f'✗ 语法错误 line {e.lineno}: {e.msg}')
    print('   →', (e.text or '').rstrip())
    # 打印上下文
    for i in range(max(0, e.lineno - 4), min(len(lines), e.lineno + 3)):
        mark = '>>' if i == e.lineno - 1 else '  '
        print(f'{mark} {i+1:4d} | {lines[i]}')
    sys.exit(1)
