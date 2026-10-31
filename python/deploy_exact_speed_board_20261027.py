"""Versioned idle-only deployment; never retries inference or forces DMA stop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_ExactSpeed_20261027_04'
PACKAGE=APP/'data/ct_studio_integration_20261025'
OUT=ROOT/'data/exact_speed_board_acceptance_20261027'
OLD='/home/xilinx/ct_test/ct_studio_dual_exact_20261026_02'
NEW='/home/xilinx/ct_test/ct_studio_exact_speed_20261027_04'
OLD_UNIT='ct-studio-dual-exact-r2.service'
UNIT='ct-studio-exact-speed-r4.service'


def connect():
    sys.path[:0]=[str(ROOT/'python'),str(ROOT/'data/board_deploy_tools/python')]
    import paramiko
    c=paramiko.SSHClient();c.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('192.168.2.99',username='xilinx',password=os.environ['CT_BOARD_SSH_PASSWORD'],
        allow_agent=False,look_for_keys=False,timeout=15,banner_timeout=20,auth_timeout=15)
    return c


def run(c,command,sudo=False):
    from deploy_shared_seg_board import run
    return run(c,command,sudo)


def expected():return hashlib.sha256((PACKAGE/'ct_studio/platform_manifest.json').read_bytes()).hexdigest()


def stage(c):
    proof=json.loads((ROOT/'data/exact_speed_iteration_20261026/candidate04_package_validation.json').read_text())
    archive=ROOT/'data/exact_speed_iteration_20261026/board_exact_speed_candidate_20261027_04.zip'
    if (proof['service_bundle_sha256']!=expected()or
            hashlib.sha256(archive.read_bytes()).hexdigest()!=proof['archive_sha256']):
        raise ValueError('Candidate package identity changed')
    with c.open_sftp()as s:
        for n in (NEW,NEW+'.zip'):
            try:s.stat(n)
            except FileNotFoundError:continue
            raise FileExistsError('Existing staged path preserved '+n)
        s.put(str(archive),NEW+'.zip')
    script=f'''import json,pathlib,zipfile,hashlib
p=pathlib.Path({NEW!r});p.mkdir(exist_ok=False)
with zipfile.ZipFile({(NEW+'.zip')!r})as z:
 for n in z.namelist():
  if p not in (p/n).resolve().parents:raise ValueError('Unsafe path')
 z.extractall(p)
if hashlib.sha256((p/'platform_manifest.json').read_bytes()).hexdigest()!={expected()!r}:raise ValueError('Bundle differs')
m=json.loads((p/'platform_manifest.json').read_text())
for n,h in m['files_sha256'].items():
 if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h:raise ValueError('File differs '+n)
print('STAGED_ALL_PINNED_FILES_EXACT')
'''
    print(run(c,'/usr/local/share/pynq-venv/bin/python3 -c '+shlex.quote(script)),flush=True)
    old=run(c,'systemctl cat '+OLD_UNIT)
    matches=re.findall(r'--expected-manifest ([0-9a-f]{64})',old)
    if len(matches)!=1 or OLD not in old:raise ValueError('Unexpected baseline unit')
    unit=old[old.index('[Unit]'):].replace(OLD,NEW).replace('ct_dual_io_owner.py','ct_ldct_lossless_owner.py')
    unit=unit.replace(matches[0],expected()).replace('Conflicts=','Conflicts='+OLD_UNIT+' ')
    (OUT/'old_service.unit').write_text(old,encoding='utf8');(OUT/'new_service.unit').write_text(unit,encoding='utf8')
    with c.open_sftp()as s:
        with s.open(NEW+'/'+UNIT,'w')as f:f.write(unit)
    run(c,'install -m 644 '+shlex.quote(NEW+'/'+UNIT)+' /etc/systemd/system/'+UNIT+
        ' && systemctl daemon-reload',sudo=True)
    print('STAGED_ONLY_PRODUCTION_UNCHANGED',flush=True)


def activate(c):
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        html=op.open('http://127.0.0.1:8099/',timeout=3).read().decode()
    except urllib.error.URLError as e:
        if not isinstance(e.reason,ConnectionRefusedError):raise
    else:
        token=re.search(r'<meta name="ui-token" content="([^"]+)"',html)[1]
        state=json.load(op.open(urllib.request.Request('http://127.0.0.1:8099/api/state',headers={'X-UI-Token':token}),timeout=6))
        if state['phase']in('loading','computing','processing','reconstructing','connecting'):
            raise RuntimeError('Web active; no owner change')
        (OUT/'pre_activation_web.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
    token=(PACKAGE/'platform_token.txt').read_text().strip()
    health=json.load(op.open(urllib.request.Request('http://192.168.2.99:8766/v1/shared/platform',headers={'Authorization':'Bearer '+token}),timeout=8))
    if (health.get('service_bundle_sha256')!='503b99ab19e5be533d978f36b85c2edb24abd81af6d185a369cd5be159ba585f'
            or not health.get('ready')or health.get('busy')or health.get('error')):
        raise RuntimeError('Baseline owner not matched, ready and idle')
    run(c,'systemctl stop '+OLD_UNIT,sudo=True)
    stopped=run(c,'systemctl show '+OLD_UNIT+' -p MainPID -p SubState')
    if 'MainPID=0'not in stopped or'SubState=dead'not in stopped:raise RuntimeError('Owner did not stop idle')
    run(c,'systemctl start '+UNIT,sudo=True)
    print('R4_STARTING_ACTUAL_SELFTESTS_NOT_YET_BOOT_ENABLED',flush=True)


def enable_accepted(c):
    accepted=json.loads((OUT/'accepted.json').read_text(encoding='utf8'))
    if (accepted.get('status')!='PHYSICAL_FULL_CASE_EXACT_ACCEPTED' or
            accepted.get('service_bundle_sha256')!=expected()):
        raise ValueError('Full three-function physical acceptance is required')
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    token=(PACKAGE/'platform_token.txt').read_text().strip()
    health=json.load(op.open(urllib.request.Request('http://192.168.2.99:8766/v1/shared/platform',
        headers={'Authorization':'Bearer '+token}),timeout=8))
    if (health.get('service_bundle_sha256')!=expected() or not health.get('ready') or
            health.get('busy') or health.get('error')):
        raise RuntimeError('Accepted owner must be matched, ready and idle')
    # Change boot selection only. Never restart or interrupt the accepted owner.
    run(c,'systemctl enable '+UNIT+' && systemctl disable '+OLD_UNIT,sudo=True)
    current=run(c,'systemctl show '+UNIT+' -p UnitFileState -p ActiveState -p MainPID')
    previous=run(c,'systemctl show '+OLD_UNIT+' -p UnitFileState -p ActiveState -p MainPID')
    if 'UnitFileState=enabled'not in current or 'ActiveState=active'not in current or 'UnitFileState=disabled'not in previous:
        raise RuntimeError('Boot selection did not match acceptance; inspect retained services')
    value=dict(status='ACCEPTED_OWNER_ENABLED_WITHOUT_RESTART',service=UNIT,folder=NEW,
        service_bundle_sha256=expected(),current_service=current,previous_service=previous,
        cold_boot_after_this_selection_tested=False)
    (OUT/'boot_configuration.json').write_text(json.dumps(value,indent=2),encoding='utf8')
    print(json.dumps(value,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','activate','inspect','enable-accepted'));a=p.parse_args()
    OUT.mkdir(exist_ok=True);c=connect()
    try:
        if a.action=='stage':stage(c)
        elif a.action=='activate':activate(c)
        elif a.action=='enable-accepted':enable_accepted(c)
        else:
            script=f"import pathlib,json; v=json.loads(pathlib.Path({NEW!r}+'/owner_state.json').read_text()); print(json.dumps({{k:v.get(k)for k in ['phase','error','elapsed_seconds','pid','boot_id']}}))"
            print(run(c,'/usr/local/share/pynq-venv/bin/python3 -c '+shlex.quote(script),sudo=True))
    finally:c.close()


if __name__=='__main__':main()
