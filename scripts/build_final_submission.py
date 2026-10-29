"""Create the exact seven-file/one-source-directory course submission layout."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'docs/最终提交/20261029'
NAME='20245726_刘广徽_中文评论情感分析'
EXCLUDED_SCRIPTS={
    'build_course_submission.py','build_final_submission.py','render_course_report.py',
    'inspect_demo_video.py','upload_trained_release.py','probe_demo.py',
    'audit_data_usage.py','audit_swanlab_records.py','collect_benchmark_environment.ps1',
}
FORBIDDEN={'.mp4','.mkv','.safetensors','.pt','.pth','.bin','.joblib','.pkl','.pyc','.pem','.key'}


def build(output):
    output=output.resolve()
    if not output.is_relative_to((ROOT/'artifacts/packages').resolve()):raise ValueError('Use artifacts/packages')
    output.mkdir(parents=True,exist_ok=False)
    stage=output/NAME
    stage.mkdir()
    for source,target in [(ROOT/'output/pdf/报告-20261029.pdf','报告.pdf'),
                          (FINAL/'项目摘要.md','项目摘要.md'),(FINAL/'README.md','README.md')]:
        shutil.copy2(source,stage/target)
    for path in sorted(FINAL.glob('贡献声明_*_*.md')):shutil.copy2(path,stage/path.name)
    project=stage/'src/project'
    project.mkdir(parents=True)
    for directory in ('src','tests','configs'):
        for source in sorted((ROOT/directory).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts or source.suffix=='.pyc':continue
            if directory=='configs' and (source.name=='paths.local.json' or 'demo' in source.relative_to(ROOT/'configs').parts):
                continue
            target=project/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    for source in sorted((ROOT/'scripts').glob('*')):
        if source.is_file() and source.name not in EXCLUDED_SCRIPTS:
            target=project/'scripts'/source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source,target)
    for name in ('reproduce.py','submission_reproduce.py','requirements.analysis.lock','requirements.integration.lock','requirements.integration.txt'):
        shutil.copy2(ROOT/name,project/name)
    (project/'artifacts').mkdir()
    for name in ('lora-lock.json','evaluation-freeze.json','reproduction-downloads.json','qwen-prompt-model-lock.json'):
        shutil.copy2(ROOT/'artifacts'/name,project/'artifacts'/name)
    (stage/'src/run.py').write_text('''"""One command entry for saved-result or real-model reproduction."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'project'))
from submission_reproduce import main
if __name__=='__main__':
    raise SystemExit(main())
''',encoding='utf-8')
    (stage/'src/一键复现.cmd').write_text('@echo off\r\npython "%~dp0run.py" %*\r\nexit /b %errorlevel%\r\n',encoding='utf-8')
    manifest={
        'date':'2026-10-29','status':'submission_edition','entry':'python src/run.py',
        'model_entry':'python src/run.py --models --paths src/project/configs/reproduce-models.local.json',
        'data_and_weights_and_video_in_zip':False,'files':{},
        'new_machine_reproduction_status':'completed_reported_by_group_leader',
        'signature_representation':'typed member names entered at the group leader request; no simulated handwriting'}
    for path in sorted(stage.rglob('*')):
        if path.is_file():manifest['files'][path.relative_to(stage).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    (stage/'src/提交清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    expected={'报告.pdf','项目摘要.md','README.md','src','贡献声明_20245726_刘广徽.md',
              '贡献声明_20246005_杨函齐.md','贡献声明_20245816_李嘉谊林.md'}
    if {path.name for path in stage.iterdir()}!=expected:raise ValueError('Unexpected submission root')
    archive=output/(NAME+'.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                relative=path.relative_to(stage)
                if path.suffix.lower() in FORBIDDEN or any(part in {'data','results','logs','.git','.venv','node_modules','checkpoint','checkpoints'} for part in relative.parts):
                    raise ValueError(f'Forbidden course item: {relative}')
                package.write(path,relative.as_posix())
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:raise ValueError('CRC failed')
    if archive.stat().st_size>50_000_000:raise ValueError('ZIP exceeds 50MB')
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 root_entries=sorted(expected),crc_check='passed',status='submission_edition',staging_directory=str(stage))
    (output/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.output)
