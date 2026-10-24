"""TEST ONLY synthetic CT/mesh fixture. Does not run a model or access a board."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from ct_workspace_web import WorkspaceController, workspace_handler
from lung_live_web import reconstruction_plan
from lung_volume_viewer import mesh
from plotly.offline import get_plotlyjs

p=argparse.ArgumentParser();p.add_argument('--folder',required=True,type=Path);p.add_argument('--port',type=int,default=0)
args=p.parse_args();folder=args.folder.resolve();folder.mkdir(parents=True,exist_ok=True)
hu=np.full((12,64,80),-1000,np.float32)
z,y,x=np.indices(hu.shape)
body=((y-32)/29)**2+((x-40)/35)**2<1
lung=(((y-32)/22)**2+((x-25)/11)**2<1)|(((y-32)/22)**2+((x-55)/11)**2<1)
lung &= (z>0)&(z<11)
hu[body]=35;hu[lung]=-790
path=folder/'synthetic_ct_TEST_ONLY.npz'
np.savez(path,hu=hu,spacing_zyx_mm=[2.5,.7,.9],origin_lps_mm=[120,-45,300],orientation=[-1,0,0,0,1,0])
controller=WorkspaceController(None,{},folder/'results_TEST_ONLY')
controller.submit('load',dict(path=str(path),confirm_hu=True,load_expert=False))
while controller.snapshot()['phase']=='loading':time.sleep(.01)
if controller.status['phase']!='idle':raise RuntimeError(controller.status['error'])
# Synthetic display setup ONLY, deliberately not a production inference backend.
controller.body=lung.copy();controller.raw=lung.copy();controller.cleaned=lung.copy();controller.known[:]=True
controller.receipts={i:'SYNTHETIC_NO_MODEL_'+str(i) for i in range(len(hu))}
controller.surfaces={'body':mesh(lung,controller.geom),'cleaned':mesh(lung,controller.geom)}
plan=reconstruction_plan(controller.known,controller.details,controller.geom)
controller._update(volume_ready=True,full_volume_ready=True,volume_scope='input_volume',reconstruction=plan,
    volume_job_id='SYNTHETIC_NO_MODEL',region_volume_ml=float(lung.sum()*.7*.9*2.5/1000))
token='synthetic-browser-test-only'
handler=workspace_handler(controller,token,get_plotlyjs().encode())
class TestHandler(handler):
    def send(self,status,raw,content_type='application/json; charset=utf-8'):
        if content_type.startswith('text/html'):
            raw=raw.replace(b'<body>',('<body><div style="position:fixed;right:12px;bottom:6px;z-index:9999;background:#fff0d6;color:#8b4300;font:12px sans-serif;padding:7px">TEST ONLY · 合成 CT / 合成肺表面 · 未执行模型 / 未连接板卡</div>').encode())
        return super().send(status,raw,content_type)
server=ThreadingHTTPServer(('127.0.0.1',args.port),TestHandler)
print('TEST_URL=http://127.0.0.1:'+str(server.server_port),flush=True)
try:server.serve_forever()
finally:server.server_close()
