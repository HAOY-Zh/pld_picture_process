"""Versioned deployment for compact transport; only idle owners may switch."""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys
import zipfile
import deploy_shared_seg_board as transport

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_DualExact_20261026_02'
PACKAGE=APP/'data/ct_studio_integration_20261025'
BOARD=PACKAGE/'ct_studio'
OUT=ROOT/'data/dual_pipeline_optimization_20261026'
OLD='/home/xilinx/ct_test/ct_studio_nodule_compact_20261026_01'
NEW='/home/xilinx/ct_test/ct_studio_dual_exact_20261026_02'
OLD_UNIT='ct-studio-nodule-compact.service'
UNIT='ct-studio-dual-exact-r2.service'


def stage(c):
    value=json.loads((OUT/'local_acceptance.json').read_text())
    if value['status']!='LOCAL_PASS_BOARD_PENDING':raise ValueError('Local verification missing')
    archive=OUT/'board_dual_exact_20261026_02.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED)as z:
            for p in BOARD.rglob('*'):
                if p.is_file() and not (p.parent.name=='exact_io_native' and p.suffix in ('.dll','.json')):z.write(p,p.relative_to(BOARD).as_posix())
    with c.open_sftp()as s:
        for name in (NEW,NEW+'.zip'):
            try:s.stat(name)
            except FileNotFoundError:pass
            else:raise FileExistsError('Existing deployment retained: '+name)
        s.put(str(archive),NEW+'.zip')
    script=f'''import pathlib,zipfile,json,hashlib
p=pathlib.Path({NEW!r});p.mkdir(exist_ok=False)
with zipfile.ZipFile({(NEW+'.zip')!r})as z:
 for n in z.namelist():
  if p not in (p/n).resolve().parents:raise ValueError('Unsafe path')
 z.extractall(p)
m=json.loads((p/'platform_manifest.json').read_text())
for n,h in m['files_sha256'].items():
 if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h:raise ValueError('Changed file '+n)
print('STAGED_COMPACT_ALL_HASHES_PASS')
'''
    print(transport.run(c,'/usr/local/share/pynq-venv/bin/python3 -c '+shlex.quote(script)))
    old=transport.run(c,'cat /etc/systemd/system/'+OLD_UNIT)
    expected=re.findall(r'--expected-manifest ([0-9a-f]{64})',old)
    if len(expected)!=1:raise ValueError('Unexpected original service unit')
    unit=old.replace(OLD,NEW).replace('ct_nodule_compact_owner.py','ct_dual_io_owner.py')
    unit=unit.replace(expected[0],value['service_bundle_sha256']).replace('Conflicts=','Conflicts='+OLD_UNIT+' ')
    (OUT/'old_service.unit').write_text(old);(OUT/'new_service.unit').write_text(unit)
    with c.open_sftp()as s:
        with s.open(NEW+'/'+UNIT,'w')as f:f.write(unit)
    transport.run(c,'install -m 644 '+shlex.quote(NEW+'/'+UNIT)+' /etc/systemd/system/'+UNIT+' && systemctl daemon-reload',sudo=True)
    print('STAGED_ONLY; production unchanged',flush=True)


def activate(c):
    sys.path.insert(0,str(APP/'python'))
    from start_ct_studio_demo import read_web_info,ACTIVE
    info=read_web_info('http://127.0.0.1:8099/')
    if info:
        s=info[2]
        if s['phase']in ACTIVE or s.get('ldct',{}).get('running')or(s.get('nodule',{}).get('job')or{}).get('status')=='RUNNING':
            raise RuntimeError('Web job active; retained')
        (OUT/'pre_activation_state.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    from lung_live_web import BoardClient
    config=json.loads((PACKAGE/'host_config.json').read_text())
    client=BoardClient(config['board_url'],(PACKAGE/'platform_token.txt').read_text().strip())
    health=client.call('/v1/shared/platform',timeout=10)
    expected=json.loads((ROOT/'data/CT_Studio_NoduleCompact_20261026_01/data/ct_studio_integration_20261025/shared_host_platform.json').read_text())['service_bundle_sha256']
    if health.get('service_bundle_sha256')!=expected or not health.get('ready')or health.get('busy')or health.get('error'):
        raise RuntimeError('Original owner not confirmed matched and idle')
    transport.run(c,'systemctl stop '+OLD_UNIT,sudo=True)
    state=transport.run(c,'systemctl show '+OLD_UNIT+' --property=MainPID,SubState')
    if 'MainPID=0'not in state or 'SubState=dead'not in state:raise RuntimeError('Old owner not cleanly stopped')
    transport.run(c,'systemctl start '+UNIT,sudo=True)
    print('NEW_COMPACT_OWNER_STARTING_SELFTESTS',flush=True)


def check():
    sys.path.insert(0,str(APP/'python'))
    from start_ct_studio_demo import configuration
    from ct_dual_io_client import CachedCanvasClient as CompactClient
    client,_=configuration(PACKAGE/'host_config.json');client.health()
    bundle=json.loads((PACKAGE/'shared_host_platform.json').read_text())
    result=CompactClient(client.url,client.token,bundle).health()
    (OUT/'board_health.json').write_text(json.dumps(result,indent=2))
    print('COMPACT_ALL_STARTUP_PASS')


def start_from_verified_stopped(c):
    """Manual repair path for a software-only startup error, never DMA recovery."""
    proof=json.loads((OUT/'startup_failure_idle_evidence.json').read_text())
    if proof['status']!='READ_ONLY_THREE_SNAPSHOTS_ALL_ENGINES_IDLE_AND_STABLE':
        raise ValueError('Physical idle evidence missing')
    sys.path.insert(0,str(APP/'python'))
    from start_ct_studio_demo import read_web_info,ACTIVE
    info=read_web_info('http://127.0.0.1:8099/')
    if info and (info[2]['phase']in ACTIVE or (info[2].get('nodule',{}).get('job')or{}).get('status')=='RUNNING'):
        raise RuntimeError('Web task active')
    for unit in (OLD_UNIT,'ct-studio-dual-exact.service',UNIT):
        value=transport.run(c,'systemctl show '+unit+' --property=MainPID,SubState')
        if 'MainPID=0'not in value or 'SubState=dead'not in value:
            raise RuntimeError('Owner still exists: '+unit)
    transport.run(c,'systemctl start '+UNIT,sudo=True)
    print('REPAIR_REVISION_STARTING_ALL_SELFTESTS',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','activate','check','inspect','enable','start_stopped'));a=p.parse_args()
    if a.action=='check':check();return
    c=transport.connect()
    try:
        if a.action=='stage':stage(c)
        elif a.action=='activate':activate(c)
        elif a.action=='start_stopped':start_from_verified_stopped(c)
        elif a.action=='inspect':
            script=f"import json,pathlib; s=json.loads(pathlib.Path({NEW!r}+'/owner_state.json').read_text()); print(json.dumps({{k:s.get(k)for k in ('phase','error','elapsed_seconds')}}))"
            print(transport.run(c,'/usr/local/share/pynq-venv/bin/python3 -c '+shlex.quote(script),sudo=True))
        else:
            check();proof=json.loads((OUT/'nodule_full_case/result.json').read_text());seg=json.loads((OUT/'seg_optimized/result.json').read_text())
            if seg['status']!='OPTIMIZED_ALL_351_HEADS_INPUTS_AND_3D_MASKS_EXACT_PASS':raise ValueError('Segmentation equality missing')
            if proof.get('status')!='DUAL_LIVE_FULL_CASE_ALL_HEADS_AND_CANDIDATES_EXACT_PASS':
                raise ValueError('Whole-case equality missing')
            transport.run(c,'systemctl disable '+OLD_UNIT+' && systemctl enable '+UNIT,sudo=True)
            print('COMPACT_ACCEPTED_BOOT_ENABLED')
    finally:c.close()


if __name__=='__main__':main()
