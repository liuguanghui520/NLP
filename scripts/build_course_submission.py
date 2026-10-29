"""Build one dataset-free, weight-free course ZIP from an explicit allowlist.

The archive is a candidate until contributions and final acceptance are checked.
Training assets keep their independent GitHub Release links; they are not put here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/报告初稿/20261029-v6'
HANDOFF = ROOT / 'docs/课程提交/20261029-v1'
NAME = '20245726_刘广徽_中文评论情感分析'
FORBIDDEN_SUFFIXES = {'.mp4', '.mkv', '.safetensors', '.pt', '.pth', '.bin', '.joblib', '.pkl', '.pem', '.key', '.pyc'}
FORBIDDEN_PARTS = {'data', 'results', 'logs', '.git', '.venv', 'venv', '__pycache__', 'node_modules', 'checkpoint', 'checkpoints'}


def package(output):
    output = output.resolve()
    if not output.is_relative_to((ROOT / 'artifacts/packages').resolve()):
        raise ValueError('Output must be below artifacts/packages')
    output.mkdir(parents=True, exist_ok=False)
    stage = output / NAME
    stage.mkdir()
    copied=[]
    def copy(source,relative):
        source=Path(source)
        relative=Path(relative)
        if FORBIDDEN_PARTS.intersection(relative.parts) or source.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f'Forbidden item: {relative}')
        if source.name in {'paths.local.json','.env'} or source.is_symlink():
            raise ValueError(f'Personal file or symlink: {source}')
        target=(stage/relative).resolve()
        if not target.is_relative_to(stage):raise ValueError('Path escapes staging directory')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        copied.append(relative.as_posix())
    # All original integrated source is included. Upstream LLaMA-Factory remains an external dependency.
    for directory in ['src','scripts','tests','configs']:
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc' and path.name!='paths.local.json':
                copy(path,path.relative_to(ROOT))
    for name in ['reproduce.py','reproduce.cmd','submission_reproduce.py','requirements.analysis.lock',
                 'requirements.integration.lock','requirements.integration.txt','verify_llamafactory.py',
                 'verify_tools.py','open-llamafactory.cmd','start-llamafactory.ps1']:
        copy(ROOT/name,name)
    copy(ROOT/'output/pdf/课程项目报告-v6.pdf','报告.pdf')
    copy(REPORT/'项目摘要.md','项目摘要.md')
    for name in ['README.md','任务书逐项核对.md','提交前待办.md']:
        copy(HANDOFF/name,name)
    for path in REPORT.glob('贡献声明_*_*.md'):copy(path,path.name)
    for name in ['项目报告.md','PDF排版稿.md','manifest.json','缺失信息清单.md','修订记录.md','token用量核查.json','评估补充数据.json']:
        copy(REPORT/name,Path('report-source')/name)
    for path in (REPORT/'assets').glob('*'):copy(path,Path('report-source/assets')/path.name)
    for path in (ROOT/'artifacts').glob('*.json'):
        if path.name in {'reproduction-downloads.json','demo-video.json','evaluation-freeze.json','lora-lock.json','qwen-prompt-model-lock.json'}:
            copy(path,Path('artifacts')/path.name)
    # Human-readable engineering docs without copying historical outputs or datasets.
    for directory in ['reproduction','environment','demo']:
        for path in sorted((ROOT/'docs'/directory).rglob('*.md')):
            copy(path,Path('docs')/directory/path.relative_to(ROOT/'docs'/directory))
    # Rebase links in the editable report for the different ZIP layout.
    for name in ['项目报告.md','PDF排版稿.md']:
        target=stage/'report-source'/name
        text=target.read_text(encoding='utf-8')
        def link(match):
            label,destination=match.group(1),match.group(2)
            if destination.startswith(('https://','http://','#')):return match.group(0)
            original=(REPORT/destination).resolve()
            if original.name.startswith('贡献声明_'):
                chosen=stage/original.name
            elif original.is_relative_to(REPORT):
                chosen=stage/'report-source'/original.relative_to(REPORT)
            elif original.is_relative_to(ROOT):
                relative=original.relative_to(ROOT)
                chosen=stage/relative
                if original.suffix.lower() in {'.png','.svg'} and original.is_file() and not chosen.exists():
                    chosen=stage/'report-source/figures'/original.name
                    if not chosen.exists():copy(original,chosen.relative_to(stage))
                if not chosen.is_file():
                    return f'[{label}](https://github.com/liuguanghui520/NLP/blob/106cdbed8db84057dba6b9c980d3a2f914856dd0/{relative.as_posix()})'
            else:raise ValueError(f'Report link escapes project: {destination}')
            if not chosen.is_file():raise ValueError(f'Missing packaged report link: {chosen}')
            return f'[{label}]({Path(os.path.relpath(chosen,target.parent)).as_posix()})'
        text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
        target.write_text(text,encoding='utf-8')
    manifest={
        'created_utc':datetime.now(timezone.utc).isoformat(), 'status':'candidate_pending_personal_confirmations_and_new_machine_model_acceptance',
        'name':NAME+'.zip','single_zip':True,'data_included':False,'weights_included':False,'video_included':False,
        'body_pages':12,'contribution_files':sorted(p.name for p in stage.glob('贡献声明_*.md')),
        'source_files_included':len(copied),'files':{},
        'pending':['Each member completes AI-use details, evidence confirmation, peer confirmation and signature/date',
                   'Final report review and new-machine real-model acceptance; no teacher acceptance fabricated',
                   'Commit/push current changes and freeze final delivery revision before course upload']}
    for path in sorted(stage.rglob('*')):
        if path.is_file():manifest['files'][path.relative_to(stage).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    (stage/'提交清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=output/(NAME+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for path in sorted(stage.rglob('*')):
            if path.is_file():package.write(path,path.relative_to(stage).as_posix())
    with zipfile.ZipFile(archive) as package:
        if package.testzip():raise ValueError('ZIP CRC check failed')
        names=package.namelist()
        if len(set(names))!=len(names):raise ValueError('Duplicate ZIP paths')
        if any(info.file_size>50_000_000 for info in package.infolist()):raise ValueError('An archived file exceeds 50MB')
        for name in names:
            path=Path(name)
            if FORBIDDEN_PARTS.intersection(path.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
                raise ValueError(f'Forbidden archived item: {name}')
    if archive.stat().st_size>50_000_000:raise ValueError('ZIP exceeds 50MB')
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'file_count':len(names),
             'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'crc_check':'passed',
             'dataset_weights_video_excluded':True,'stage':str(stage),'status':manifest['status']}
    (output/'packaging-verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New output directory below artifacts/packages')
    args=parser.parse_args()
    package(args.output)
