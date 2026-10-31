"""Switch only the confirmed idle 8099 workspace, retaining saved case results."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'data/CT_Studio_PC_Paired_20261025_01'
NEW = ROOT/'data/CT_Studio_SharedSeg_20261025_01'
OUT = ROOT/'data/shared_seg_web_upgrade_20261025'
PS = 'C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
ACTIVE = ('loading', 'connecting', 'computing', 'processing', 'reconstructing')


def read_state(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    root = 'http://127.0.0.1:'+str(port)
    html = opener.open(root+'/', timeout=3).read().decode()
    token = re.search(r'<meta name="ui-token" content="([^"]+)"', html)[1]
    state = json.load(opener.open(urllib.request.Request(root+'/api/state', headers={'X-UI-Token': token}), timeout=5))
    return state


def require_idle(value):
    if (value['phase'] in ACTIVE or value.get('ldct', {}).get('running')
            or (value.get('roi', {}).get('job') or {}).get('running')
            or (value.get('nodule', {}).get('job') or {}).get('status') in ('RUNNING', 'PENDING')):
        raise RuntimeError('Current web task is active; kept running')


def launch(port, state_file):
    folder = OUT/('web_'+str(port))
    folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb') as out, (folder/'stderr.log').open('xb') as err:
        process = subprocess.Popen([str(ROOT/'.venv-lung/Scripts/python.exe'), '-B', '-u', '-X', 'utf8',
            str(NEW/'python/ct_workspace_web.py'), '--config', str(NEW/'data/ct_studio_integration_20261025/host_config.json'),
            '--port', str(port), '--output', str(OLD/'data/ct_workspace_demo/results'), '--resume-session', str(state_file)],
            cwd=NEW, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
    (folder/'pid').write_text(str(process.pid))
    deadline = time.monotonic()+100
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('New web exited: '+(folder/'stderr.log').read_text(errors='replace'))
        try:
            state = read_state(port)
        except (OSError, urllib.error.URLError):
            time.sleep(.5)
            continue
        if state.get('segmentation_engine', {}).get('interface') != 'ct_shared_segmentation_v1':
            raise ValueError('New web did not select the shared segmentation engine')
        return process, state
    raise TimeoutError('New web not ready; process retained, check '+str(folder))


def main():
    result = json.loads((OUT/'full_case_parity/result.json').read_text())
    if result['status'] != 'PASS_ALL_SAVED_SLICES_EXACT' or result['different_head_codes'] != 0:
        raise ValueError('Missing full-case exact acceptance')
    state = read_state(8099)
    require_idle(state)
    state_file = OUT/'cutover_state.json'
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf8')
    preview, candidate = launch(8098, state_file)
    for key in ('case_id', 'known_slices', 'volume_ml', 'full_volume_ready'):
        if candidate.get(key) != state.get(key):
            raise ValueError('New restored web state differs: '+key)
    print('CANDIDATE_8098_RESTORED_PASS', flush=True)
    current = read_state(8099)
    require_idle(current)
    if current['revision'] != state['revision'] or current['case_id'] != state['case_id']:
        raise RuntimeError('Old UI changed during migration; original kept running')
    command = "Get-NetTCPConnection -LocalPort 8099 -State Listen | Select-Object -ExpandProperty OwningProcess"
    owner = int(subprocess.check_output([PS, '-NoProfile', '-Command', command], text=True).strip())
    command = f"Get-CimInstance Win32_Process -Filter 'ProcessId={owner}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    info = json.loads(subprocess.check_output([PS, '-NoProfile', '-Command', command], text=True))
    if str(OLD/'python/ct_workspace_web.py').lower() not in info['CommandLine'].lower():
        raise RuntimeError('8099 belongs to another application; kept running')
    subprocess.run([PS, '-NoProfile', '-Command', f'Stop-Process -Id {owner} -ErrorAction Stop'], check=True)
    print('SAVED_IDLE_OLD_WEB_STOPPED', flush=True)
    actual, live = launch(8099, state_file)
    for key in ('case_id', 'known_slices', 'volume_ml', 'full_volume_ready'):
        if live.get(key) != state.get(key):
            raise ValueError('Live migrated case differs: '+key)
    # The candidate performed reads/restoration only. No active hardware command.
    preview.terminate()
    preview.wait(timeout=10)
    wrapper = ROOT/'启动最新版CT工作台.cmd'
    (OUT/'old_launcher.cmd').write_bytes(wrapper.read_bytes())
    raw = wrapper.read_bytes().replace(b'CT_Studio_PC_Paired_20261025_01', b'CT_Studio_SharedSeg_20261025_01')
    wrapper.write_bytes(raw)
    receipt = dict(status='LIVE_8099_SHARED_SEGMENTATION', case_id=live['case_id'], known_slices=live['known_slices'],
        volume_ml=live['volume_ml'], segmentation_engine=live['segmentation_engine'], launcher_updated=True,
        previous_results_preserved=True, web_pid=actual.pid)
    (OUT/'web_migration.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
