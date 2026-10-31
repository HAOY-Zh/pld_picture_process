"""Low-memory supervisor: launch each read-only audit as its evaluation finishes."""
import ctypes
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
    root = Path(__file__).resolve().parent.parent
    out = root/'data/lung_nodule_context_20261020_v1'
    launch = json.loads((out/'launch.json').read_text('utf-8-sig'))
    for variant in ('near_far5', 'context7', 'context9'):
        while not (out/variant/'evaluation.json').exists():
            if not alive(launch['pid']):
                raise RuntimeError('Main task stopped before '+variant+' evaluation')
            time.sleep(3)
        if not (out/variant/'result_brief.json').exists():
            subprocess.run([sys.executable, '-u', str(root/'python/lung_nodule_context_audit.py'),
                '--variant', variant], cwd=str(root), check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        print('AUDITED RESULT READY', variant, flush=True)


if __name__ == '__main__':
    main()
