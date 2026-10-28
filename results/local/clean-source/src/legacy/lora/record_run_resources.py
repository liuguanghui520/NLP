"""Run one command and preserve wall time, process-tree RSS and device samples.

Sampling is not a precise per-process CUDA peak or a power-meter measurement.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

import psutil


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.interval <= 0:
        parser.error('A command and positive sampling interval are required.')
    args.output.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(timezone.utc)
    state = dict(status='running', command=command, sample_interval_seconds=args.interval,
                 started_utc=started_utc.isoformat(),
                 wall_boundary='Legacy wall_seconds uses perf_counter elapsed; includes loading, training, validation and saves; compare utc_elapsed_seconds for clock/suspend discrepancies',
                 rss_boundary='Sampled sum of process-tree RSS; shared pages may be counted more than once',
                 gpu_boundary='Sampled whole-device NVIDIA readings; includes other processes; not exact CUDA process peaks',
                 energy_boundary='Trapezoidal integral of device power samples only; not whole-laptop wall energy or monetary cost')
    write_json(args.output / 'status.json', state)
    process = None
    stop = threading.Event()
    peaks = dict(process_tree_rss_bytes=0, device_memory_mib=0.0)
    energy = [0.0]
    power_samples = [0]
    errors = []
    started = time.perf_counter()

    def sample():
        previous = None
        with (args.output / 'resources.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['utc_timestamp', 'elapsed_s', 'process_tree_rss_bytes', 'device_memory_mib',
                             'device_util_percent', 'device_power_w', 'sample_error'])
            while not stop.is_set():
                rss = 0
                try:
                    parent = psutil.Process(process.pid)
                    for item in [parent] + parent.children(recursive=True):
                        try:
                            rss += item.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except psutil.NoSuchProcess:
                    pass
                memory = utilization = power = ''
                error = ''
                try:
                    output = subprocess.check_output([
                        'nvidia-smi', '--id=0',
                        '--query-gpu=memory.used,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits'], text=True, stderr=subprocess.STDOUT, timeout=5)
                    values = [value.strip() for value in output.strip().split(',')]
                    memory, utilization = float(values[0]), float(values[1])
                    try:
                        power = float(values[2])
                    except ValueError:
                        power = ''
                except Exception as exc:
                    error = repr(exc)
                    if len(errors) < 10:
                        errors.append(error)
                elapsed = time.perf_counter() - started
                peaks['process_tree_rss_bytes'] = max(peaks['process_tree_rss_bytes'], rss)
                if memory != '':
                    peaks['device_memory_mib'] = max(peaks['device_memory_mib'], memory)
                if power != '':
                    power_samples[0] += 1
                    if previous is not None:
                        energy[0] += (power + previous[1]) / 2 * (elapsed - previous[0])
                    previous = elapsed, power
                else:
                    previous = None
                writer.writerow([datetime.now(timezone.utc).isoformat(), elapsed, rss, memory, utilization, power, error])
                stream.flush()
                stop.wait(args.interval)

    sampler = None
    try:
        with (args.output / 'console.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', start_new_session=True)
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            state.update(status='completed' if code == 0 else 'failed', exit_code=code)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        (args.output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if process is not None and process.poll() is None:
            import os
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        if sampler is not None:
            sampler.join(timeout=6)
        ended_utc = datetime.now(timezone.utc)
        utc_elapsed = (ended_utc - started_utc).total_seconds()
        state.update(ended_utc=ended_utc.isoformat(), wall_seconds=elapsed,
                     perf_counter_elapsed_seconds=elapsed, utc_elapsed_seconds=utc_elapsed,
                     clock_difference_seconds=utc_elapsed - elapsed,
                     gpu_hours_boundary='Legacy allocated_gpu_hours uses perf_counter elapsed; not utilization-weighted or verified against suspension',
                     allocated_gpu_hours=elapsed / 3600, sampled_peaks=peaks,
                     power_samples=power_samples[0],
                     sampled_device_energy_wh=energy[0] / 3600 if power_samples[0] >= 2 else None,
                     sampling_errors=errors)
        write_json(args.output / 'status.json', state)
        print('RESOURCE RECORD:', args.output, flush=True)
    if state['status'] != 'completed':
        raise SystemExit(state.get('exit_code', 1) or 1)


if __name__ == '__main__':
    main()
