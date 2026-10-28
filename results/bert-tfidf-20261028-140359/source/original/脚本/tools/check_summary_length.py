# -*- coding: utf-8 -*-
"""检查项目摘要的中文字数（交付要求：正文限 300 字以内）。"""
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, '项目摘要.md')
s = open(p, encoding='utf-8').read()

# 去掉 HTML 注释、提示行、markdown 标记，只数中文
body = re.sub(r'<!--.*?-->', '', s, flags=re.S)
body = '\n'.join(l for l in body.split('\n') if not l.strip().startswith('>'))
body = re.sub(r'[#*`>\-\s]', '', body)
n = len([c for c in body if '\u4e00' <= c <= '\u9fff'])

print(f'中文正文约 {n} 字（限 300 字）')
if n > 300:
    print('  ✗ 超了，需要压缩')
else:
    print(f'  ✓ 还有 {300 - n} 字余量')
print()
print(s)
