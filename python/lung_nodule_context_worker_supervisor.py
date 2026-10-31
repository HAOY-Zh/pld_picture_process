"""Keep no Torch/CUDA context resident while the main process trains."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def alive(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        return code.value == 259
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child-variant', choices=('near_far5', 'context7', 'context9'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    out = root/'data/lung_nodule_context_20261020_v1'
    pid = json.loads((out/'launch.json').read_text('utf-8-sig'))['pid']
    if args.child_variant:
        from lung_nodule_context_eval_worker import evaluate
        evaluate(out, args.child_variant, pid)
        return
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'worker_supervisor_identity.json').write_text(json.dumps(dict(source_sha256=source_sha,
        policy='fresh companion child per completed model; release entire process after inference'), indent=2), 'utf-8')
    for variant in ('near_far5', 'context7', 'context9'):
        while not (out/variant/'model_freeze.json').exists():
            if not alive(pid):
                raise RuntimeError('Main task stopped before '+variant)
            time.sleep(3)
        if not (out/variant/'evaluation.json').exists():
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--child-variant', variant],
                cwd=str(root), check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        print('COMPANION RELEASED', variant, flush=True)


if __name__ == '__main__':
    main()
