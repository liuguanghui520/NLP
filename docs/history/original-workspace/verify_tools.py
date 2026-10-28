"""Offline engineering smoke checks; not course experiments."""
import contextlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent / 'logs' / 'tools-install'
ROOT.mkdir(parents=True, exist_ok=True)
with (ROOT / 'verification.log').open('w', encoding='utf-8') as log:
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import torch
        import optuna
        import swanlab
        from swanlab.integration.transformers import SwanLabCallback
        from llamafactory.train.callbacks import LogCallback

        before = json.loads((ROOT / 'packages-before.json').read_text())
        after = {d.metadata['Name']: d.version for d in metadata.distributions() if d.metadata['Name']}
        changes = {k: [v, after.get(k)] for k, v in before.items() if after.get(k) != v}
        print('Existing package version changes:', changes)
        assert not changes, changes
        for p in ['swanlab', 'optuna', 'torch', 'transformers', 'peft', 'llamafactory']:
            print(p, metadata.version(p))
        print('Integration imports: PASS')
        check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
        print('pip check:', check.returncode, check.stdout, check.stderr)
        assert check.returncode == 0
        x = torch.randn(128, 128, device='cuda')
        assert torch.isfinite(x @ x.T).all().item()
        torch.cuda.synchronize()
        print('CUDA smoke: PASS')
        swanlab.init(project='nlp-tool-smoke', experiment_name='software-check', mode='offline',
                     logdir=str(ROOT / 'swanlog'), config={'purpose': 'software smoke only'})
        swanlab.log({'synthetic_check': 1.0})
        swanlab.finish()
        print('SwanLab offline write: PASS')
        study = optuna.create_study(direction='minimize', storage='sqlite:///' + str(ROOT / 'optuna-smoke.db'))
        study.optimize(lambda trial: (trial.suggest_float('x', -1, 1)) ** 2, n_trials=2, n_jobs=1)
        print('Optuna synthetic trials:', len(study.trials))
        (ROOT / 'packages-after.json').write_text(json.dumps(after, indent=2), encoding='utf-8')
        (ROOT / 'resolved-requirements.txt').write_text('\n'.join(f'{k}=={v}' for k, v in sorted(after.items())), encoding='utf-8')
        print('PASS: software checks only; no training or course results')
print((ROOT / 'verification.log').read_text(encoding='utf-8'))
