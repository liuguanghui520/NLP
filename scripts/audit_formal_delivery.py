"""Read-only verification of formal outputs; write a new audit and source snapshot."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(batch):
    output=ROOT/'results/analysis'/batch
    report=ROOT/'reports/analysis'/batch
    figures=ROOT/'reports/figures'/batch
    audit_dir=ROOT/'logs/comparison-checks'/('delivery-'+batch)
    assert not audit_dir.exists()
    status=read(output/'status.json')
    assert status['status']=='completed' and not any(status[k] for k in ['training','inference','model_loading'])
    for section in ['input_sha256','output_sha256']:
        for rel,sha in status[section].items():
            assert digest(ROOT/rel)==sha,(section,rel)
    annotation=ROOT/status['config']['annotations']
    annotation_audit=read(annotation/'annotation-audit.json')
    for rel,sha in annotation_audit['input_sha256'].items():
        assert digest(ROOT/rel)==sha,rel
    for name,sha in annotation_audit['output_sha256'].items():
        assert digest(annotation/name)==sha,name
    ids=read(ROOT/'data/manifests/smoke_ids.json')
    train={json.loads(x)['sample_id'] for x in (ROOT/'data/processed/v1-20261023/train.jsonl').read_text(encoding='utf-8').splitlines()}
    assert len(ids)==20 and set(ids)<=train
    index=read(output/'figure-index.json')
    assert len(index)==22 and len({x['id'] for x in index})==22
    images=[]
    for row in index:
        png,pdf=figures/row['png'],figures/row['pdf']
        with Image.open(png) as im:
            im.verify()
        content=pdf.read_bytes()
        assert content.startswith(b'%PDF-') and content.rstrip().endswith(b'%%EOF')
        images.append(dict(id=row['id'],png_sha256=digest(png),pdf_sha256=digest(pdf),data=row['data']))
    unchanged={}
    previous=ROOT/'reports/figures/comparison-formal-20261028-02'
    if previous.exists():
        for p in sorted(previous.glob('*.png')):
            if (figures/p.name).exists():
                unchanged[p.name]=ImageChops.difference(Image.open(p).convert('RGB'),Image.open(figures/p.name).convert('RGB')).getbbox() is None
    majority=read(output/'majority-reference.json')
    assert majority['label']==1 and majority['per_class']['0']['f1']==0
    assert abs(majority['weighted_f1']-majority['per_class']['1']['f1']*532/777)<1e-12
    pairs=read(output/'paired-comparisons.json')
    assert len(pairs)==7 and all(p['both_correct']+p['both_wrong']+p['A_correct_B_wrong']+p['A_wrong_B_correct']==777 for p in pairs)
    sensitivity=read(output/'seed43-44-sensitivity.json')
    assert len(sensitivity)==14
    cost=read(output/'formal-cost-ledger.json')
    assert len(cost['training_unique_runs'])==len({x['identity'] for x in cost['training_unique_runs']})==28
    assert len(read(output/'casebook.json'))>=3
    ledger=read(report/'结论证据台账.json')
    for claim in ledger:
        for source in claim['sources']:
            assert digest(ROOT/source['path'])==source['sha256']
    for rel in ['对比分析观察稿.md','真实错误案例与局限.md','待复核清单.md','验收清单.md','AI使用记录.md','图表审查清单.md','README.md','图表总览.html']:
        assert (report/rel).is_file(),rel
    audit_dir.mkdir(parents=True)
    sources={}
    code_paths=[ROOT/'scripts'/n for n in ['build_comparison_report.py','formal_comparison.py','case_analysis.py','resource_figure.py','build_formal_annotations.py','audit_formal_delivery.py','collect_analysis_logs.py','collect_analysis_sizes.py']]
    from scripts.case_analysis import DEFAULT_CASE_NOTES
    code_paths += [ROOT/status['config'].get('case_analysis_notes',DEFAULT_CASE_NOTES)]
    code_paths += [ROOT/'src/evaluation/metrics.py',ROOT/'tests/test_formal_annotations.py',ROOT/'tests/test_offline_comparison.py',ROOT/'configs/analysis/comparison-formal-v1.json',ROOT/'results/error-review/semantic-decisions-20261028.tsv']
    for path in code_paths:
        dest=audit_dir/'source-snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        assert digest(path)==digest(dest)
        sources[path.relative_to(ROOT).as_posix()]=digest(path)
    manifest={}
    for folder in [output,report,figures,annotation]:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                manifest[path.relative_to(ROOT).as_posix()]=digest(path)
    result=dict(status='passed',analysis_id=batch,checked_at=datetime.now().astimezone().isoformat(),
        checks=[f'{len(status["input_sha256"])} input hashes and generated output hashes','frozen labels and all11 prediction hashes via annotation manifest','20 warmup IDs all in training split','22 PNG/PDF pairs','7 paired comparisons,14 sensitivity rows','28 unique training cost identities','casebook and claim source SHA256','source-separated 20+382 annotation audit'],
        figure_files=images,unchanged_pixels_from_formal02=unchanged,source_snapshot_sha256=sources,
        delivery_sha256=manifest,annotation_counts=status['annotation_counts'],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        limitations='Read-only software/data acceptance; no new-machine model run, student interpretation approval, causal attribution or publication claimed')
    (audit_dir/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',files=len(manifest),figures=22,unchanged_pixels=sum(unchanged.values()),audit=str(audit_dir/'audit.json')),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    main(args.run_id)
