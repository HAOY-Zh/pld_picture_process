"""Idle-only deployment of raw LDCT candidate, never enabled for boot here."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from deploy_exact_speed_board_20261027 import connect,run
APP=ROOT/'data/CT_Studio_RawLDCT_20261027_05'
PACKAGE=APP/'data/ct_studio_integration_20261025'
OUT=ROOT/'data/threshold_ldct_iteration_20261027'
OLD='/home/xilinx/ct_test/ct_studio_exact_speed_20261027_04'
NEW='/home/xilinx/ct_test/ct_studio_raw_ldct_20261027_05'
OLD_UNIT='ct-studio-exact-speed-r4.service'
UNIT='ct-studio-raw-ldct-r5.service'


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','activate','inspect'));args=p.parse_args()
    proof=json.loads((OUT/'raw_candidate.json').read_text(encoding='utf8'))
    expected=hashlib.sha256((PACKAGE/'ct_studio/platform_manifest.json').read_bytes()).hexdigest()
    if expected!=proof['service_bundle_sha256']:raise ValueError('Staged candidate identity changed')
    c=connect()
    try:
        if args.action=='inspect':
            script=f"import pathlib,json;p=pathlib.Path({NEW!r})/'owner_state.json';s=json.loads(p.read_text());print(json.dumps({{k:s.get(k)for k in ['phase','error','elapsed_seconds','pid']}}))"
            print(run(c,'/usr/local/share/pynq-venv/bin/python3 -c '+shlex.quote(script),sudo=True));return
        if args.action=='stage':
            archive=Path(proof['archive'])
            if hashlib.sha256(archive.read_bytes()).hexdigest()!=proof['archive_sha256']:raise ValueError('Private archive changed')
            with c.open_sftp()as s:
                for target in (NEW,NEW+'.zip'):
                    try:s.stat(target)
                    except FileNotFoundError:continue
                    raise FileExistsError('Existing candidate preserved '+target)
                s.put(str(archive),NEW+'.zip')
            script=f'''import pathlib,zipfile,hashlib,json
p=pathlib.Path({NEW!r});p.mkdir(exist_ok=False)
with zipfile.ZipFile({(NEW+'.zip')!r})as z:
 for n in z.namelist():
  if p not in (p/n).resolve().parents:raise ValueError('Unsafe path')
 z.extractall(p)
if hashlib.sha256((p/'platform_manifest.json').read_bytes()).hexdigest()!={expected!r}:raise ValueError('Bundle mismatch')
for n,h in json.loads((p/'platform_manifest.json').read_text())['files_sha256'].items():
 if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h:raise ValueError('Changed asset '+n)
print('STAGED_PRIVATE_RAW_CANDIDATE_ALL_FILES_EXACT')
'''
            print(run(c,'/usr/local/share/pynq-venv/bin/python3 -c '+shlex.quote(script)),flush=True)
            previous=run(c,'systemctl cat '+OLD_UNIT)
            matches=re.findall(r'--expected-manifest ([0-9a-f]{64})',previous)
            if len(matches)!=1 or OLD not in previous:raise ValueError('Baseline service differs')
            unit=previous[previous.index('[Unit]'):].replace(OLD,NEW).replace('ct_ldct_lossless_owner.py','ct_ldct_raw_owner.py')
            unit=unit.replace(matches[0],expected).replace('Conflicts=','Conflicts='+OLD_UNIT+' ')
            with c.open_sftp()as s:
                with s.open(NEW+'/'+UNIT,'w')as f:f.write(unit)
            run(c,'install -m 644 '+shlex.quote(NEW+'/'+UNIT)+' /etc/systemd/system/'+UNIT+' && systemctl daemon-reload',sudo=True)
            print('STAGED_ONLY_NOT_ACTIVE_NOT_BOOT_ENABLED');return
        op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        sys.path.insert(0,str(ROOT/'data/CT_Studio_ExactSpeed_20261027_04/python'))
        from start_ct_studio_demo import read_web_info
        web=read_web_info('http://127.0.0.1:8099/')
        if web is not None and web[2]['phase']in('loading','connecting','computing','processing','reconstructing'):
            raise RuntimeError('Current web task active; preserve it')
        token=(PACKAGE/'platform_token.txt').read_text().strip()
        health=json.load(op.open(urllib.request.Request('http://192.168.2.99:8766/v1/shared/platform',headers={'Authorization':'Bearer '+token}),timeout=8))
        if (health.get('service_bundle_sha256')!='ee06e8fddd97ba4a316b68dd4781affe4c362d41cb1616c156eb01299d1d1d3e' or
                not health.get('ready')or health.get('busy')or health.get('error')):
            raise RuntimeError('Baseline must be matched, healthy and idle; no forced switch')
        run(c,'systemctl stop '+OLD_UNIT,sudo=True)
        stopped=run(c,'systemctl show '+OLD_UNIT+' -p MainPID -p SubState')
        if 'MainPID=0'not in stopped or 'SubState=dead'not in stopped:raise RuntimeError('Owner did not exit normally')
        run(c,'systemctl start '+UNIT,sudo=True)
        print('RAW_CANDIDATE_STARTING_ALL_ACTUAL_SELFTESTS; NOT_BOOT_ENABLED')
    finally:c.close()


if __name__=='__main__':main()
