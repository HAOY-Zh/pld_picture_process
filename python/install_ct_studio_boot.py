"""Enable CT Studio on the board's next boot; never stop its current owner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from ct_studio_boot import validate_identity

PYTHON = '/usr/local/share/pynq-venv/bin/python3'
RUNNER = Path('/usr/local/lib/ct-studio/ct_studio_boot.py')
UNIT = Path('/etc/systemd/system/ct-studio.service')


def unit_text(folder, expected):
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', str(folder)):
        raise ValueError('Use an absolute board path without spaces or shell characters')
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Expected manifest must be a SHA256')
    return f'''[Unit]
Description=CT Studio verified FPGA platform
After=network.target bootpy.service
RequiresMountsFor={folder}

[Service]
Type=notify
NotifyAccess=main
User=root
WorkingDirectory={folder}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=XILINX_XRT=/usr
Environment=VIRTUAL_ENV=/usr/local/share/pynq-venv
Environment=PATH=/usr/local/share/pynq-venv/bin:/opt/microblazeel-xilinx-elf/bin:/opt/python3.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart={PYTHON} -B -u {RUNNER} --folder {folder} --expected-manifest {expected}
Restart=no
TimeoutStartSec=infinity
TimeoutStopSec=infinity
SendSIGKILL=no
StateDirectory=ct-studio
UMask=0077
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
'''


def write_once_or_same(path, raw, mode):
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError('Existing configuration preserved: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
    path.chmod(mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--expected-manifest', required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise PermissionError('Install through the board administrator account')
    folder = args.folder.resolve()
    validate_identity(folder, args.expected_manifest)
    sys.path.insert(0, str(folder))
    from ct_studio_service import validate_package
    validate_package(folder)  # Frozen package's complete file/release hash checks.
    runner = Path(__file__).with_name('ct_studio_boot.py').read_bytes()
    unit = unit_text(folder, args.expected_manifest).encode()
    for path, raw in ((RUNNER, runner), (UNIT, unit)):
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError('Existing configuration preserved: ' + str(path))
    write_once_or_same(RUNNER, runner, 0o644)
    write_once_or_same(UNIT, unit, 0o644)
    subprocess.run(['systemd-analyze', 'verify', str(UNIT)], check=True)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', 'enable', 'ct-studio.service'], check=True)
    print(json.dumps(dict(enabled_for_next_boot=True, current_owner_not_stopped=True,
                          package_sha256=args.expected_manifest,
                          runner_sha256=hashlib.sha256(runner).hexdigest()), indent=2), flush=True)


if __name__ == '__main__':
    main()
