"""Read and copy small existing WSL training logs, never import model libraries."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/evidence/formal-analysis-20261028-01'
assert not OUT.exists(), 'Evidence batch already exists'
OUT.mkdir(parents=True)
manifest=[]
baseline=Path('/home/lgh/experiments/bert-tfidf-20261028-140359/bert-formal')
jobs=[]
for folder in sorted(baseline.glob('bert-*')):
    for name in ['train_steps.jsonl','epoch_metrics.jsonl','config.json','metrics.json']:
        jobs.append((folder/name,Path('bert')/folder.name/name))
factory=Path('/home/lgh/LLaMA-Factory/project-workspaces')
for key in ['B00','L01','L02','L03','L04','L05','R01','D01','D02','T01','T01-S43','T01-S44']:
    if key=='B00':
        folder=factory/'qwen17-profile/qwen17-full-20261025-231706-ea8173'
    elif key.startswith('T01-S'):
        folder=factory/'qwen17-comparisons/qwen17-comparisons-20261026-200544-750ecb'/key
    else:
        folder=factory/'qwen17-comparisons/qwen17-comparisons-20261026-012117-9e8e51'/key
    for name in ['trainer_log.jsonl','trainer_state.json']:
        jobs.append((folder/'adapter'/name,Path('lora')/key/name))
for source,rel in jobs:
    entry=dict(source=str(source),copied_to=rel.as_posix(),exists=source.is_file())
    if source.is_file():
        target=OUT/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        assert source.read_bytes()==target.read_bytes()
        entry.update(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size)
    manifest.append(entry)
(OUT/'manifest.json').write_text(json.dumps(dict(collected_at=datetime.now().astimezone().isoformat(),scope='Existing logs only; no training, inference or model loading',files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(files_copied=sum(x['exists'] for x in manifest),missing=sum(not x['exists'] for x in manifest))))
