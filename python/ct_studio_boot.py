"""systemd owner for an unchanged, hash-pinned CT Studio board package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import traceback


def notify(message):
    address = os.environ.get('NOTIFY_SOCKET')
    if not address:
        return
    if address.startswith('@'):
        address = '\0' + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
        channel.sendto(message.encode(), address)


def validate_identity(folder, expected):
    raw = (folder / 'platform_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Pinned platform manifest differs; package preserved')
    manifest = json.loads(raw)
    if (manifest.get('basic_transport') != 'ct_basic_zlib_cache_v1'
            or manifest.get('ldct', {}).get('clock_mhz') != 100):
        raise ValueError('Expected accepted shared-basic / 100 MHz LDCT package')
    return manifest


def stop_idle(service):
    """Release only a healthy, idle owner; caller retains it on any exception."""
    if not service.gate.acquire(False):
        return False
    try:
        if service.failed or service.hardware.fault:
            raise RuntimeError('Faulted owner retained; no forced stop or reload')
        service.hardware._dma_status()
        service.server.shutdown()
        service.server.server_close()
        service.thread.join(timeout=2)
        if service.thread.is_alive():
            raise RuntimeError('Server thread still active; owner retained')
        service.hardware.release_idle()
        service.hardware.lock_file.close()
        return True
    finally:
        service.gate.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--expected-manifest', required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    records = Path('/var/lib/ct-studio') / boot_id
    records.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    def record(phase, **extra):
        payload = dict(phase=phase, pid=os.getpid(), boot_id=boot_id,
                       package_sha256=args.expected_manifest,
                       elapsed_seconds=time.monotonic() - began, **extra)
        temporary = records / 'startup.tmp'
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(records / 'startup.json')
    stopping = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopping.set())
    try:
        validate_identity(folder, args.expected_manifest)
        os.chdir(folder)
        sys.path.insert(0, str(folder))
        from ct_studio_service import start
        record('starting_actual_selftests')
        notify('STATUS=Validating package and running actual FPGA startup selftests')
        service = start(folder=folder)
        health = service.health()
        if (not health.get('ready') or health.get('busy') or health.get('error')
                or health.get('integration_board_acceptance') != 'startup-selftests-passed'):
            raise RuntimeError('Actual startup acceptance did not pass')
        record('ready', health=health)
        notify('READY=1\nSTATUS=Ready: basic + 100 MHz LDCT + segmentation startup checks PASS')
        while True:
            stopping.wait(1)
            if stopping.is_set():
                if stop_idle(service):
                    record('stopped_idle')
                    return 0
                time.sleep(1)
    except BaseException as exc:
        # Keep traceback frames and partially constructed PYNQ/CMA owners alive.
        retained_failure = sys.exc_info()
        traceback.print_exc()
        record('failed_owner_retained', error=str(exc))
        notify('STATUS=Startup/stop failed; owner retained, no automatic retry or reset')
        while retained_failure:
            time.sleep(1)


if __name__ == '__main__':
    raise SystemExit(main())
