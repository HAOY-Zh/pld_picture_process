"""Launch the accepted new web with an explicitly saved session, no inference."""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_NoduleFull_20261025_01'
OUT=ROOT/'data/nodule_full_web_upgrade_20261025'
sys.path.insert(0,str(APP/'python'))
from start_ct_studio_demo import configuration,read_web_info


def main():
    client,_=configuration(APP/'data/ct_studio_integration_20261025/host_config.json')
    client.health()
    if read_web_info('http://127.0.0.1:8099/')is not None:
        raise RuntimeError('8099 already has a web session; retained, migrate explicitly')
    folder=OUT/'web_live_20261026';folder.mkdir(exist_ok=False)
    with (folder/'stdout.log').open('xb')as stdout,(folder/'stderr.log').open('xb')as stderr:
        process=subprocess.Popen([str(ROOT/'.venv-lung/Scripts/python.exe'),'-B','-u','-X','utf8',
            str(APP/'python/ct_workspace_web.py'),'--config',str(APP/'data/ct_studio_integration_20261025/host_config.json'),
            '--port','8099','--output',str(ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results'),
            '--resume-session',str(OUT/'speed_case_resume_20261026.json')],cwd=APP,stdin=subprocess.DEVNULL,
            stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW)
    (folder/'pid').write_text(str(process.pid))
    for _ in range(180):
        if process.poll()is not None:raise RuntimeError((folder/'stderr.log').read_text(errors='replace'))
        try:info=read_web_info('http://127.0.0.1:8099/')
        except (OSError,RuntimeError):info=None
        if info is not None:
            state=info[2]
            assert state['known_slices']==351
            assert state['segmentation_engine']['interface']=='ct_shared_segmentation_v1'
            assert state['nodule']['service_available']is True
            (OUT/'new_web_restored_20261026.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
            print('LIVE_NEW_WEB_8099_RESTORED_351_PASS',flush=True)
            return
        time.sleep(.5)
    raise TimeoutError('Web still starting; process retained, inspect its logs')


if __name__=='__main__':main()
