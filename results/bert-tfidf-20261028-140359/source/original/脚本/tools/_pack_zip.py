"""用 Python zipfile 重新打包（保证用 '/' 分隔符，Linux/Kaggle 才能正确解压）"""
import zipfile, os, hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / 'data' / 'processed' / 'v1-20261023'
OUT = BASE / 'chunksenti_v1-20261023.zip'

if OUT.exists(): OUT.unlink()

files = sorted([p for p in SRC.rglob('*') if p.is_file()])
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files:
        arc = p.relative_to(SRC).as_posix()      # ★ 正斜杠
        z.write(p, arc)

print(f'打包 {len(files)} 个文件 -> {OUT.name}  ({OUT.stat().st_size/1e6:.2f} MB)\n')
with zipfile.ZipFile(OUT) as z:
    bad = z.testzip()
    print('CRC 校验:', 'OK' if bad is None else f'损坏 {bad}')
    for i in z.infolist():
        print(f'  {i.filename:<34} {i.file_size:>10}')

# 校验能正确解出来
import tempfile, shutil
tmp = Path(tempfile.mkdtemp())
with zipfile.ZipFile(OUT) as z:
    z.extractall(tmp)
need = ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'statistics.json',
        'fewshot_ids.json', 'latency_ids.json']
for n in need:
    p = tmp / n
    print(f'  {"✓" if p.exists() else "✗"} {n}')
print('  alpaca/train.json 存在:', (tmp/'alpaca'/'train.json').exists())
shutil.rmtree(tmp, ignore_errors=True)
