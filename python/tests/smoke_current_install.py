"""Actual HTTP smoke of a fresh installed snapshot; no board is contacted.

Run with an installation whose board URL is loopback port 1. Tests both full
configured and desktop modes, loads a synthetic NPZ and fetches all three MPRs.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib import request, parse, error
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from create_demo_ct import create


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    config=args.config.resolve();setup=json.loads(config.read_text(encoding='utf8'))
    assert setup['board_url']=='http://127.0.0.1:1','Use a dedicated non-board smoke installation'
    out=args.report.resolve().parent;out.mkdir(parents=True,exist_ok=True)
    phantom=out/'synthetic_ct.npz';create(phantom)
    with np.load(phantom,allow_pickle=False)as z:expected=z['hu'].astype(np.float32)
    opener=request.build_opener(request.ProxyHandler({}));modes={}
    for mode in ('configured','desktop'):
        with socket.socket()as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
        url='http://127.0.0.1:'+str(port)
        logs=out/(mode+'.log');output=out/(mode+'_results')
        command=[sys.executable,'-B','-X','utf8',str(ROOT/'python/ct_workspace_web.py'),
            '--config',str(config)]if mode=='configured'else[sys.executable,'-B','-X','utf8',str(ROOT/'python/ct_workspace_web.py'),'--desktop']
        command+=['--port',str(port),'--output',str(output)]
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt'else 0
        with logs.open('xb')as stream:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=stream,stdin=subprocess.DEVNULL,creationflags=flags)
            try:
                deadline=time.monotonic()+45
                while True:
                    if proc.poll()is not None:raise RuntimeError('Web exited: '+str(logs))
                    try:
                        with opener.open(url+'/',timeout=2)as r:html=r.read().decode('utf8')
                        break
                    except (error.URLError,TimeoutError):
                        if time.monotonic()>deadline:raise
                        time.sleep(.2)
                token=re.search(r'<meta name="ui-token" content="([^"]+)"',html)[1]
                def api(path,values=None):
                    payload=None if values is None else json.dumps(values).encode()
                    headers={'X-UI-Token':token}
                    if payload is not None:headers['Content-Type']='application/json'
                    with opener.open(request.Request(url+path,data=payload,headers=headers),timeout=15)as r:return json.load(r)
                state=api('/api/state')
                assert state['host_speed_profile']=='exact_host_raw_r3'
                assert state['desktop_only']==(mode=='desktop')and not state['board_connected']
                if mode=='configured':
                    assert state['ldct']['fpga_configured']and state['nodule']['service_available']
                    assert state['segmentation_engine']and state['roi']['fpga_configured']
                api('/api/load',dict(path=str(phantom),confirm_hu=True,load_expert=False,load_slices=''))
                deadline=time.monotonic()+20
                while True:
                    state=api('/api/state')
                    if state['phase']=='error':raise RuntimeError(state.get('error'))
                    if state.get('shape')==[8,128,128]and state['phase']=='idle':break
                    if time.monotonic()>deadline:raise TimeoutError('Synthetic CT load incomplete')
                    time.sleep(.1)
                for axis,index in ((0,3),(1,64),(2,64)):
                    plane=api('/api/plane?'+parse.urlencode(dict(case=state['case_id'],axis=axis,index=index,stage='body')))
                    hu=np.frombuffer(base64.b64decode(plane['hu']),dtype='<f4').reshape(plane['shape'])
                    np.testing.assert_array_equal(hu,np.take(expected,index,axis))
                    assert plane['computed_pixels']==0 and plane['expert_comparison']is None
                modes[mode]=dict(http_200=True,profile=state['host_speed_profile'],synthetic_shape=state['shape'],
                    all_three_mpr_hu_exact=True,board_connected=False,fpga_calls=0)
            finally:
                proc.terminate();proc.wait(timeout=10)
    report=dict(status='FRESH_INSTALL_ACTUAL_WEB_PASS',modes=modes,patient_data_used=False,board_accessed=False)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
