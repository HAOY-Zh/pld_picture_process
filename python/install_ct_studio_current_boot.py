"""Install the current dual owner for next boot; never stop or start hardware."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess


def unit_text(folder, expected):
    folder = str(folder)
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', folder) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Need an absolute board path and SHA256')
    return f'''[Unit]
Description=CT Studio current dual-model owner
After=network.target bootpy.service
RequiresMountsFor={folder}
Conflicts=ct-studio.service ct-studio-shared.service ct-studio-shared-batch.service ct-studio-shared-seg.service ct-studio-nodule-full.service ct-studio-nodule-compact.service ct-studio-dual-exact.service ct-studio-dual-exact-r2.service ct-studio-roi-statistics.service
[Service]
Type=simple
User=root
WorkingDirectory={folder}
Environment=PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XILINX_XRT=/usr VIRTUAL_ENV=/usr/local/share/pynq-venv
Environment=PATH=/usr/local/share/pynq-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/usr/local/share/pynq-venv/bin/python3 -B -u {folder}/ct_dual_io_owner.py --folder {folder} --expected-manifest {expected} --port 8766
Restart=no
TimeoutStopSec=infinity
SendSIGKILL=no
UMask=0077
[Install]
WantedBy=multi-user.target
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--unit-file',type=Path,default=Path('/etc/systemd/system/ct-studio-current.service'))
    p.add_argument('--enable-next-boot',action='store_true')
    a=p.parse_args();folder=a.folder.resolve()
    expected=hashlib.sha256((folder/'platform_manifest.json').read_bytes()).hexdigest()
    if not (folder/'ct_dual_io_owner.py').is_file():raise ValueError('Current owner missing')
    text=unit_text(folder,expected)
    if os.name!='posix' or os.geteuid()!=0:raise PermissionError('Run as board administrator')
    sys_path=str(folder)
    import sys
    sys.path.insert(0,sys_path)
    from ct_studio_shared_service import validate_package
    manifest=validate_package(folder)
    if manifest.get('dual_io_interface')!='ct_dual_exact_io_v3':raise ValueError('Wrong current package')
    if a.unit_file.exists() and a.unit_file.read_text()!=text:
        raise FileExistsError('Existing unit differs; preserve it and choose another --unit-file')
    a.unit_file.write_text(text)
    subprocess.run(['systemd-analyze','verify',str(a.unit_file)],check=True)
    subprocess.run(['systemctl','daemon-reload'],check=True)
    if a.enable_next_boot:
        subprocess.run(['systemctl','enable',a.unit_file.name],check=True)
    print('Installed',a.unit_file,'; hardware was NOT stopped or started.')
    print('Disable obsolete owners and obsolete 8080 web autostart before next reboot; see delivery guide.')


if __name__=='__main__':main()
