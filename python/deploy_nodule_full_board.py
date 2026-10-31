"""Explicit deployment for the prepared canvas+scoring release; no probing.

Set CT_BOARD_SSH_PASSWORD in the invoking process using an authorized credential.
The older deployment is preserved. No activation without confirmed idle health.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile
from urllib.error import URLError

import deploy_shared_seg_board as transport

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_NoduleFull_20261025_01'
PACKAGE=APP/'data/ct_studio_integration_20261025'
BOARD=PACKAGE/'ct_studio'
OUT=ROOT/'data/nodule_full_web_upgrade_20261025'
OLD='/home/xilinx/ct_test/ct_studio_shared_seg_20261025_01'
NEW='/home/xilinx/ct_test/ct_studio_nodule_full_20261025_01'
OLD_UNIT='ct-studio-shared-seg.service'
UNIT='ct-studio-nodule-full.service'


def stage(client):
    receipt=json.loads((OUT/'local_acceptance.json').read_text())
    if receipt['status']!='LOCAL_PASS_BOARD_PENDING':raise ValueError('Local acceptance missing')
    archive=OUT/'ct_studio_nodule_full_20261025_01.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED)as z:
            for p in BOARD.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(BOARD).as_posix())
    with client.open_sftp()as sftp:
        for p in (NEW,NEW+'.zip'):
            try:sftp.stat(p)
            except FileNotFoundError:pass
            else:raise FileExistsError('Existing deployment retained: '+p)
        sftp.put(str(archive),NEW+'.zip')
    program=f'''import pathlib,zipfile,hashlib,json,ast
p=pathlib.Path({NEW!r});p.mkdir(exist_ok=False)
with zipfile.ZipFile({(NEW+'.zip')!r})as z:
 for n in z.namelist():
  if p not in (p/n).resolve().parents:raise ValueError('Unsafe archive')
 z.extractall(p)
m=json.loads((p/'platform_manifest.json').read_text())
for n,h in m['files_sha256'].items():
 if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h:raise ValueError('Changed file '+n)
for n in ('ct_nodule_canvas_owner.py','ct_nodule_canvas_service.py','nodule_canvas_contract.py'):
 ast.parse((p/n).read_text())
print('STAGED_CANVAS_HASH_AND_SYNTAX_PASS')
'''
    print(transport.run(client,'/usr/local/share/pynq-venv/bin/python3 -c '+shlex.quote(program)))
    original=transport.run(client,'cat /etc/systemd/system/'+OLD_UNIT)
    old_manifest=json.loads((ROOT/'data/CT_Studio_SharedSeg_20261025_01/upgrade_provenance.json').read_text())['service_bundle_sha256']
    new_manifest=receipt['service_bundle_sha256']
    unit=original.replace(OLD,NEW).replace('ct_shared_seg_owner.py','ct_nodule_canvas_owner.py')
    unit=unit.replace(old_manifest,new_manifest).replace('Conflicts=','Conflicts='+OLD_UNIT+' ')
    (OUT/'old_service.unit').write_text(original);(OUT/'new_service.unit').write_text(unit)
    with client.open_sftp()as sftp:
        with sftp.open(NEW+'/'+UNIT,'w')as f:f.write(unit)
    transport.run(client,'install -m 644 '+shlex.quote(NEW+'/'+UNIT)+' /etc/systemd/system/'+UNIT+' && systemctl daemon-reload',sudo=True)
    print('STAGED_ONLY; old owner retained')


def activate(client):
    sys.path.insert(0,str(APP/'python'))
    from lung_live_web import BoardClient
    config=json.loads((PACKAGE/'host_config.json').read_text())
    conn=BoardClient(config['board_url'],(PACKAGE/'platform_token.txt').read_text().strip())
    health=conn.call('/v1/shared/platform',timeout=10)
    old_expected=json.loads((ROOT/'data/CT_Studio_SharedSeg_20261025_01/upgrade_provenance.json').read_text())['service_bundle_sha256']
    if (health.get('service_bundle_sha256')!=old_expected or not health.get('ready') or health.get('busy') or health.get('error')):
        raise RuntimeError('Old accepted service not confirmed idle; kept running')
    # A live web may hold a job between hardware requests. Check it too.
    from start_ct_studio_demo import read_web_info,ACTIVE
    info=read_web_info('http://127.0.0.1:8099/')
    if info:
        state=info[2]
        if state['phase']in ACTIVE or state.get('ldct',{}).get('running'):
            raise RuntimeError('Web job active; kept running')
        (OUT/'pre_activation_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
    transport.run(client,'systemctl stop '+OLD_UNIT,sudo=True)
    status=transport.run(client,'systemctl show '+OLD_UNIT+' --property=MainPID,SubState')
    if 'MainPID=0'not in status or 'SubState=dead'not in status:
        raise RuntimeError('Old owner not stopped cleanly; no new owner started')
    transport.run(client,'systemctl start '+UNIT,sudo=True)
    print('NEW_OWNER_STARTED; wait for all startup tests. No boot enable yet.')


def check():
    sys.path.insert(0,str(APP/'python'))
    from start_ct_studio_demo import configuration
    from ct_nodule_canvas_client import CanvasClient
    client,_=configuration(PACKAGE/'host_config.json');client.health()
    bundle=json.loads((PACKAGE/'shared_host_platform.json').read_text())
    value=CanvasClient(client.url,client.token,bundle).health()
    (OUT/'board_startup_health.json').write_text(json.dumps(value,indent=2))
    print('BOARD_ALL_STARTUP_AND_IDENTITY_PASS')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','activate','check','enable'))
    args=p.parse_args()
    if args.action=='check':check();return
    client=transport.connect()
    try:
        if args.action=='stage':stage(client)
        elif args.action=='activate':activate(client)
        else:
            check()
            proof=json.loads((OUT/'board_planes/result.json').read_text())
            expected=json.loads((OUT/'local_acceptance.json').read_text())['service_bundle_sha256']
            if proof.get('status')!='PHYSICAL_CANVAS_REQUIRED_CELLS_AND_DECODE_EXACT' or proof.get('service_bundle_sha256')!=expected:
                raise ValueError('Physical real-input plane comparison not accepted')
            transport.run(client,'systemctl disable '+OLD_UNIT+' && systemctl enable '+UNIT,sudo=True)
            print('ACCEPTED_NEW_SERVICE_ENABLED_FOR_BOOT')
    finally:client.close()


if __name__=='__main__':main()
