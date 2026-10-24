"""ROI-enabled board owner, with retained resources after any hardware failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--expected-manifest',required=True);p.add_argument('--port',type=int,default=8766)
    a=p.parse_args();folder=a.folder.resolve();stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    os.chdir(folder);sys.path.insert(0,str(folder))
    started=time.monotonic()
    def record(phase,**extra):
        value=dict(phase=phase,pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            manifest_sha256=a.expected_manifest,elapsed_seconds=time.monotonic()-started,**extra)
        target=folder/'owner_state.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(target)
    try:
        if hashlib.sha256((folder/'platform_manifest.json').read_bytes()).hexdigest()!=a.expected_manifest:
            raise ValueError('Pinned ROI package differs')
        from ct_studio_roi_service import start
        record('starting_actual_selftests',memory=Path('/proc/meminfo').read_text())
        service=start(folder=folder,port=a.port)
        record('ready',health=service.health(),memory=Path('/proc/meminfo').read_text())
        while True:
            stop.wait(1)
            if stop.is_set():
                if service.gate.acquire(False):
                    try:
                        if service.failed or service.hardware.fault:raise RuntimeError('Faulted owner retained')
                        service.hardware._dma_status()
                        service.server.shutdown();service.server.server_close();service.thread.join(timeout=2)
                        if service.thread.is_alive():raise RuntimeError('Server thread retained')
                        service.hardware.release_idle();service.hardware.lock_file.close()
                        record('stopped_idle');return 0
                    finally:service.gate.release()
    except BaseException as exc:
        retained=sys.exc_info();traceback.print_exc();record('failed_owner_retained',error=str(exc))
        while retained:time.sleep(1)


if __name__=='__main__':main()
