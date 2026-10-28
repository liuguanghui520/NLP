"""Sample an existing training process, then execute queued isolated evaluations."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
from run_bert_tfidf import write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--bert-root', required=True)
    ap.add_argument('--tfidf-root', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--index', required=True)
    args = ap.parse_args()
    index = Path(args.index)
    p = psutil.Process(args.pid)
    started = p.create_time()
    write(index/'queue_status.json', dict(status='waiting_for_training', pid=args.pid,
          sampling_note='Attached after training started; initial interval is not sampled',
          provisional_latency='First TF-IDF measurement overlapped BERT training; use test-isolated only'))
    with (index/'device_samples.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['unix_time', 'process_rss_bytes', 'whole_gpu_memory_mib', 'whole_gpu_util_percent'])
        while p.is_running() and p.create_time() == started and p.status() != psutil.STATUS_ZOMBIE:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            values = result.stdout.strip().split(',') if result.returncode == 0 else ['', '']
            writer.writerow([time.time(), p.memory_info().rss] + values)
            stream.flush()
            time.sleep(5)
    status = json.loads((Path(args.bert_root)/'status.json').read_text())
    assert status['status'] == 'training_complete', status
    write(index/'queue_status.json', dict(status='isolated_testing'))
    for method, root in [('tfidf', args.tfidf_root), ('bert', args.bert_root)]:
        cmd = [sys.executable, str(Path(__file__).with_name('test_bert_tfidf.py')),
               '--root', root, '--data', args.data, '--method', method, '--output-name', 'test-isolated']
        with (index/f'{method}-isolated-test.log').open('x') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for method, root in [('bert', Path(args.bert_root)), ('tfidf', Path(args.tfidf_root))]:
        table = json.loads((root/'validation_comparison.json').read_text())
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([r['config_id'] for r in table], [r['macro_f1'] for r in table])
        ax.set_ylabel('Validation Macro-F1')
        fig.tight_layout()
        fig.savefig(index/f'{method}-validation.png', dpi=150)
        plt.close(fig)
        write(index/f'{method}-three-seed-test.json', json.loads((root/'three_seed_test.json').read_text()))
    for path in Path(args.bert_root).glob('bert-*'):
        records = [json.loads(x) for x in (path/'train_steps.jsonl').read_text().splitlines()]
        epochs = [json.loads(x) for x in (path/'epoch_metrics.jsonl').read_text().splitlines()]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot([r['step'] for r in records], [r['loss'] for r in records], linewidth=.5)
        ax[0].set_ylabel('Raw training loss')
        for key in ['macro_f1', 'accuracy']:
            ax[1].plot([r['epoch'] for r in epochs], [r[key] for r in epochs], label=key)
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(index/f'{path.name}-curves.png', dpi=150)
        plt.close(fig)
    write(index/'queue_status.json', dict(status='completed', human_error_analysis='pending_student_review'))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
