"""User-scoped completion: finish seven planes, audit and time; never start nine."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from lung_nodule_context_audit_watch import alive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    out = root/'data/lung_nodule_context_20261020_v1'
    launch = json.loads((out/'launch.json').read_text('utf-8-sig'))
    if launch.get('variants') != ['context7']:
        raise ValueError('Main task must be explicitly scoped to context7')
    pid = launch['pid']
    if args.worker_only:
        from lung_nodule_context_eval_worker import evaluate
        evaluate(out, 'context7', pid)
        return
    if (out/'context9').exists():
        raise ValueError('Unexpected nine-plane run folder')
    worker = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker-only'],
        cwd=str(root), creationflags=subprocess.CREATE_NO_WINDOW)
    (out/'worker_supervisor_launch.json').write_text(json.dumps(dict(pid=worker.pid,
        entry='context7 companion only; supervisor itself has no CUDA context'), indent=2), 'utf-8')
    while not (out/'context7/evaluation.json').exists():
        if not alive(pid):
            raise RuntimeError('Main stopped before context7 evaluation')
        if worker.poll() is not None and worker.returncode != 0:
            raise RuntimeError('Seven-plane companion failed')
        time.sleep(3)
    worker.wait(timeout=180)
    if worker.returncode != 0:
        raise RuntimeError('Companion returned failure')
    subprocess.run([sys.executable, '-u', str(root/'python/lung_nodule_context_audit.py'),
        '--variant', 'context7'], cwd=str(root), check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    while alive(pid):
        time.sleep(1)
    subprocess.run([sys.executable, '-u', str(root/'python/lung_nodule_context_timing.py'),
        '--variants', 'near_far5', 'context7'], cwd=str(root), check=True,
        creationflags=subprocess.CREATE_NO_WINDOW)
    if (out/'context9').exists():
        raise AssertionError('Nine-plane task was unexpectedly started')
    print('FINISHED context7 and timing; context9 NOT STARTED', flush=True)


if __name__ == '__main__':
    main()
