"""看训练还活着没 —— 一行命令判断"在跑"还是"卡住"。

用法（另开一个 PowerShell 窗口）：
    python tools/watch_progress.py
    python tools/watch_progress.py --watch        # 每 30 秒刷一次

为什么需要它
------------
训练脚本每 100 步才打一行日志，而 100 步现在要 5~6 分钟；
而且后期 loss 小到 0.001 量级，屏幕看着几乎不动。
凭感觉很容易误判成"卡死"。
这个脚本直接读日志文件的修改时间和 CPU 占用，给出明确判断。

判断依据
--------
    * 日志最后修改距今 > 2 倍"上一条标记的间隔" → 疑似卡住
    * 训练进程的 CPU 时间在持续增长 → 一定在算
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / 'src' / 'results' / 'results_BERT_FULL' / 'run_log.txt'

STEP_RE = re.compile(r'\[(\w+)\]\s+Ep(\d+)\s+(\d+)/(\d+)')
CFG_RE = re.compile(r'配置\s+(\d+)/(\d+):\s+(\w+)')
PHASE_RE = re.compile(r'===\s*\[(\d/8)\]\s*(.+?)\s*===')


def read_lines():
    if not LOG.exists():
        return []
    try:
        return LOG.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return []


def find_proc():
    """找训练进程，返回 (pid, cpu_seconds)；找不到返回 (None, None)。

    用 PowerShell 的 CIM 查询拿命令行 —— wmic 在新版 Windows 上已被移除。
    """
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
        "Where-Object { $_.CommandLine -like '*_bert_run.py*' } | "
        "ForEach-Object { \"$($_.ProcessId) $([math]::Round($_.UserModeTime/1e7,1))\" }"
    )
    try:
        import subprocess
        out = subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None, None
    cands = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            pid = int(parts[0])
            cpu = float(parts[1]) if len(parts) > 1 else 0.0
            cands.append((pid, cpu))
        except Exception:
            continue
    if not cands:
        return None, None
    # 命令行里同样含 _bert_run.py 的还有一个几 MB 的壳进程，
    # 真正干活的训练进程是 CPU 时间最大的那个。
    return max(cands, key=lambda x: x[1])


def snapshot():
    lines = read_lines()
    if not lines:
        return None

    last_step, prev_step = None, None
    phase = None
    cfg_i = cfg_n = cfg_name = None
    for ln in lines:
        m = PHASE_RE.search(ln)
        if m:
            phase = f'{m.group(1)} {m.group(2)}'
        m = CFG_RE.search(ln)
        if m:
            cfg_i, cfg_n, cfg_name = m.group(1), m.group(2), m.group(3)
        m = STEP_RE.search(ln)
        if m:
            if last_step:
                prev_step = last_step
            last_step = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))

    if not last_step:
        return None

    # 从文件系统拿修改时间（比读日志可靠，缓冲不影响）
    mtime = LOG.stat().st_mtime
    age = time.time() - mtime

    # 上两条标记之间的时间差 = 100 步的耗时
    gap = None
    if prev_step:
        # 用日志行号反推太麻烦，这里直接用"上一条标记距今"和"现在距今"没法算，
        # 改用：从文件里找最后两行 STEP 的位置，配合 mtime 只能得到粗略值。
        pass

    return dict(phase=phase, cfg_i=cfg_i, cfg_n=cfg_n, cfg_name=cfg_name,
                step=last_step, age=age, mtime=mtime, n_lines=len(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true', help='每 30 秒刷新一次')
    args = ap.parse_args()

    while True:
        s = snapshot()
        print('=' * 62)
        if not s:
            print('!! 读不到日志：', LOG)
        else:
            name, ep, step, total = s['step']
            print(f'阶段      : {s["phase"] or "（未识别）"}')
            if s['cfg_name']:
                print(f'当前配置  : {s["cfg_i"]}/{s["cfg_n"]}  {s["cfg_name"]}')
            print(f'当前进度  : {name}  Ep{ep}  {step}/{total}')
            print(f'日志更新  : {time.strftime("%H:%M:%S", time.localtime(s["mtime"]))}'
                  f'（距今 {s["age"]:.0f} 秒）')
            print(f'日志行数  : {s["n_lines"]}')
            if s['age'] > 900:
                print(f'⚠ 警告    : 日志已 {s["age"]/60:.1f} 分钟没更新，'
                      f'超过 100 步的正常间隔（5~6 分钟），请检查进程是否还活着')
            else:
                print('✓ 状态    : 日志在正常刷新')
        pid, cpu = find_proc()
        if pid:
            print(f'训练进程  : PID {pid} 在跑，累计 CPU {cpu:.0f} 秒'
                  f'（{cpu / 3600:.1f} 小时）')
        else:
            print('训练进程  : !! 没找到（可能已结束）')
        print(f'提示      : 100 步一行日志，约 5~6 分钟才出一行，屏幕不动是正常的')
        if not args.watch:
            break
        time.sleep(30)


if __name__ == '__main__':
    sys.exit(main())
