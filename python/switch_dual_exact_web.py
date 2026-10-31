"""Restore and switch only the idle verified 8099 frontend to compact transport."""
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'data/CT_Studio_NoduleCompact_20261026_01'
NEW=ROOT/'data/CT_Studio_DualExact_20261026_02'
OUT=ROOT/'data/dual_pipeline_optimization_20261026'
PS='C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'


def state():
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}));url='http://127.0.0.1:8099'
    html=op.open(url,timeout=4).read().decode();token=re.search(r'<meta name="ui-token" content="([^"]+)"',html)[1]
    return json.load(op.open(urllib.request.Request(url+'/api/state',headers={'X-UI-Token':token}),timeout=6))


def main():
    proof=json.loads((OUT/'physical_planes/result.json').read_text())
    if proof['status']!='PHYSICAL_DUAL_FULL_HEAD_EXACT_PASS':raise ValueError('Physical equality missing')
    s=state()
    if s['phase']in ('loading','connecting','computing','processing','reconstructing'):raise RuntimeError('Task active; retained')
    saved=OUT/'cutover_state.json';saved.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    owner=int(subprocess.check_output([PS,'-NoProfile','-Command',
        'Get-NetTCPConnection -LocalPort 8099 -State Listen | Select-Object -ExpandProperty OwningProcess'],text=True).strip())
    proc=json.loads(subprocess.check_output([PS,'-NoProfile','-Command',
        f"Get-CimInstance Win32_Process -Filter 'ProcessId={owner}' | Select-Object CommandLine | ConvertTo-Json -Compress"],text=True))
    if str(OLD/'python/ct_workspace_web.py').lower()not in proc['CommandLine'].lower():raise RuntimeError('Different frontend retained')
    subprocess.run([PS,'-NoProfile','-Command',f'Stop-Process -Id {owner} -ErrorAction Stop'],check=True)
    logs=OUT/'web_live';logs.mkdir(exist_ok=False)
    with (logs/'stdout.log').open('xb')as stdout,(logs/'stderr.log').open('xb')as stderr:
        p=subprocess.Popen([str(ROOT/'.venv-lung/Scripts/python.exe'),'-B','-u','-X','utf8',
            str(NEW/'python/ct_workspace_web.py'),'--config',str(NEW/'data/ct_studio_integration_20261025/host_config.json'),
            '--port','8099','--output',str(ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results'),
            '--resume-session',str(saved)],cwd=NEW,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW)
    (logs/'pid').write_text(str(p.pid))
    for _ in range(200):
        if p.poll()is not None:raise RuntimeError((logs/'stderr.log').read_text(errors='replace'))
        try:current=state()
        except OSError:time.sleep(.5);continue
        for key in ('case_id','known_slices','volume_ml','full_volume_ready'):
            if current[key]!=s[key]:raise ValueError('Restored state differs: '+key)
        if current['nodule']['result']!=s['nodule']['result']:raise ValueError('Previous candidates changed')
        (OUT/'web_migrated.json').write_text(json.dumps(dict(status='COMPACT_8099_RESTORED',case_id=s['case_id'],known_slices=s['known_slices'],prior_candidates_preserved=True),indent=2))
        print('COMPACT_WEB_8099_RESTORED_PASS',flush=True);return
    raise TimeoutError('New web starting; retained, inspect logs')


if __name__=='__main__':main()
