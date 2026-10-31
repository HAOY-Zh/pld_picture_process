"""Stage/activate the separately versioned board service with explicit SSH auth.

Credentials come only from the caller's process environment. No automatic
password probing, process killing, DMA reset or recovery of faulted owners.
"""
import argparse
import io
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT/'data/CT_Studio_SharedSeg_20261025_01'
BOARD = APP/'data/ct_studio_integration_20261025/ct_studio'
OUT = ROOT/'data/shared_seg_web_upgrade_20261025'
OLD = '/home/xilinx/ct_test/ct_studio_shared_batch_20261024_01'
NEW = '/home/xilinx/ct_test/ct_studio_shared_seg_20261025_01'
UNIT = 'ct-studio-shared-seg.service'
OLD_UNIT = 'ct-studio-shared-batch.service'


def connect():
    sys.path.insert(0, str(ROOT/'data/board_deploy_tools/python'))
    import paramiko
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('192.168.2.99', username='xilinx', password=os.environ['CT_BOARD_SSH_PASSWORD'],
        allow_agent=False, look_for_keys=False, timeout=6, auth_timeout=6, banner_timeout=6)
    return client


def run(client, command, sudo=False):
    if sudo:
        command = "sudo -S -p '' /bin/bash -c "+shlex.quote(command)
    stdin, stdout, stderr = client.exec_command(command, timeout=90)
    if sudo:
        stdin.write(os.environ['CT_BOARD_SSH_PASSWORD']+'\n')
        stdin.flush()
    text = stdout.read().decode('utf8', errors='replace')
    error = stderr.read().decode('utf8', errors='replace')
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError('Remote operation failed: '+error+' '+text)
    return text


def web_state():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    html = opener.open('http://127.0.0.1:8099/', timeout=5).read().decode()
    token = re.search(r'<meta name="ui-token" content="([^"]+)"', html)[1]
    value = json.load(opener.open(urllib.request.Request('http://127.0.0.1:8099/api/state',
        headers={'X-UI-Token': token}), timeout=5))
    if value['phase'] in ('loading', 'connecting', 'computing', 'processing', 'reconstructing'):
        raise RuntimeError('Existing web task is active; keep it running')
    if value.get('ldct', {}).get('running') or (value.get('roi', {}).get('job') or {}).get('running'):
        raise RuntimeError('Existing subtask is active')
    if (value.get('nodule', {}).get('job') or {}).get('status') in ('RUNNING', 'PENDING'):
        raise RuntimeError('Existing nodule job is active')
    (OUT/'pre_activation_state.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    return value


def stage(client):
    archive = OUT/'board_shared_seg_20261025_01.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in BOARD.rglob('*'):
            if path.is_file():
                z.write(path, path.relative_to(BOARD).as_posix())
    with client.open_sftp() as sftp:
        remote_zip = NEW+'.zip'
        for name in (NEW, remote_zip):
            try:
                sftp.stat(name)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError('Refusing existing deployment: '+name)
        sftp.put(str(archive), remote_zip)
    program = f'''import zipfile,pathlib,hashlib,json,ast
p=pathlib.Path({NEW!r});p.mkdir(exist_ok=False)
with zipfile.ZipFile({remote_zip!r}) as z:
 for n in z.namelist():
  q=(p/n).resolve()
  if p not in q.parents:raise ValueError('Unsafe archive')
 z.extractall(p)
m=json.loads((p/'platform_manifest.json').read_text())
for n,h in m['files_sha256'].items():
 if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h:raise ValueError('Changed file '+n)
for n in ('ct_shared_seg_owner.py','ct_shared_seg_service.py','shared_segmentation_contract.py'):
 ast.parse((p/n).read_text())
print('STAGED_BOARD_FILES_HASH_AND_SYNTAX_PASS')
'''
    print(run(client, '/usr/local/share/pynq-venv/bin/python3 -c '+shlex.quote(program)), flush=True)
    original = run(client, 'cat /etc/systemd/system/'+OLD_UNIT)
    provenance = json.loads((APP/'upgrade_provenance.json').read_text())
    new = original.replace(OLD, NEW).replace('ct_studio_shared_owner.py', 'ct_shared_seg_owner.py')
    new = new.replace('87e15b55fe425c02fe6ca3c89e1d099bf937295226b9c33c6acb0c5509d9aca2', provenance['service_bundle_sha256'])
    new = new.replace('Conflicts=', 'Conflicts='+OLD_UNIT+' ')
    (OUT/'old_service.unit').write_text(original)
    (OUT/'new_service.unit').write_text(new)
    with client.open_sftp() as sftp:
        with sftp.open(NEW+'/'+UNIT, 'w') as f:
            f.write(new)
    print(run(client, 'install -m 644 '+shlex.quote(NEW+'/'+UNIT)+' /etc/systemd/system/'+UNIT+
        ' && systemctl daemon-reload', sudo=True), flush=True)
    print('STAGED_ONLY: existing owner is still running', flush=True)


def activate(client):
    web_state()
    sys.path.insert(0, str(APP/'python'))
    from lung_live_web import BoardClient
    package = APP/'data/ct_studio_integration_20261025'
    config = json.loads((package/'host_config.json').read_text())
    current = BoardClient(config['board_url'], (package/'platform_token.txt').read_text().strip())
    health = current.call('/v1/shared/platform', timeout=10)
    if not health.get('ready') or health.get('busy') or health.get('error'):
        raise RuntimeError('Existing owner not ready and idle; kept running')
    print('OLD_OWNER_CONFIRMED_IDLE; graceful stop requested', flush=True)
    run(client, 'systemctl stop '+OLD_UNIT, sudo=True)
    status = run(client, 'systemctl show '+OLD_UNIT+' --property=MainPID,SubState')
    if 'MainPID=0' not in status or 'SubState=dead' not in status:
        raise RuntimeError('Old owner not stopped cleanly; no new owner started')
    print('OLD_OWNER_STOPPED_CLEANLY', flush=True)
    print(run(client, 'systemctl start '+UNIT, sudo=True), flush=True)
    print('NEW_OWNER_STARTED; initializing and checking actual FPGA nodes', flush=True)


def inspect(client):
    code = f'''import json,pathlib
p=pathlib.Path({NEW!r})/'owner_state.json'
if not p.exists():print('NO_OWNER_STATE_YET')
else:
 s=json.loads(p.read_text());print(json.dumps({{k:s.get(k) for k in ('phase','pid','elapsed_seconds','error')}}))
'''
    print(run(client, '/usr/local/share/pynq-venv/bin/python3 -c '+shlex.quote(code), sudo=True), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('stage', 'activate', 'inspect', 'enable'))
    args = p.parse_args()
    client = connect()
    try:
        if args.action == 'stage': stage(client)
        elif args.action == 'activate': activate(client)
        elif args.action == 'inspect': inspect(client)
        else:
            sys.path.insert(0, str(APP/'python'))
            from start_ct_studio_demo import configuration
            configuration(APP/'data/ct_studio_integration_20261025/host_config.json')[0].health()
            print(run(client, 'systemctl disable '+OLD_UNIT+' && systemctl enable '+UNIT, sudo=True))
            print('NEW_ACCEPTED_SERVICE_ENABLED_FOR_BOOT', flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
