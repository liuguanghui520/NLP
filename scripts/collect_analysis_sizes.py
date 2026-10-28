"""Collect existing checkpoint file sizes only; do not deserialize weights."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/evidence/formal-analysis-sizes-20261028-01.json'
assert not OUT.exists()
paths=json.loads((ROOT/'configs/paths.local.json').read_text(encoding='utf-8'))
records=[]
for method,key,prefix in [('bert','bert_results','bert-'),('tfidf','baseline_results','tfidf-')]:
    for folder in sorted(Path(paths[key]).glob(prefix+'*')):
        best=folder/'checkpoints/best'
        files=[p for p in best.rglob('*') if p.is_file()]
        records.append(dict(method=method,run_id=folder.name,scope='Complete best model/tokenizer folder or sklearn Pipeline; no optimizer/environment',
            source=str(best),exists=best.exists(),bytes=sum(p.stat().st_size for p in files) if files else None,
            files=[dict(path=p.relative_to(best).as_posix(),bytes=p.stat().st_size) for p in files]))
OUT.write_text(json.dumps(dict(collected_at=datetime.now().astimezone().isoformat(),records=records),ensure_ascii=False,indent=2),encoding='utf-8')
print(len(records))
