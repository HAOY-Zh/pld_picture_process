"""TEST ONLY: 512x512 synthetic CT and six-operation fake hardware platform."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python'))
from test_ct_platform import FakePlatformHardware, Service, handler_for, HOST, MANIFEST_SHA
from test_lung_live import TOKEN, MODEL, RunningServer
from ct_platform_client import PlatformClient
from ct_workspace_web import WorkspaceController, workspace_handler
from plotly.offline import get_plotlyjs

p=argparse.ArgumentParser();p.add_argument('--folder',required=True,type=Path);p.add_argument('--fast',action='store_true');a=p.parse_args()
if a.fast:
    from test_ct_platform_fast import FakeSharedHardware as FakePlatformHardware,Service,handler_for,HOST
folder=a.folder.resolve();folder.mkdir(parents=True,exist_ok=True)
y,x=np.mgrid[:512,:512];hu=np.full((8,512,512),-1000,np.float32)
hu[:,((y-256)**2/224**2+(x-256)**2/208**2)<1]=30
for cx in [176,336]:hu[:,((y-256)**2/152**2+(x-cx)**2/64**2)<1]=-800
g=dict(spacing_zyx_mm=[3.,1.,1.],origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
np.savez(folder/'synthetic_ct.npz',hu=hu,**g)
np.savez(folder/'synthetic_expert.npz',mask=(hu==-800).astype(np.uint8),**g)
board=RunningServer(handler_for(Service(FakePlatformHardware(),TOKEN,MANIFEST_SHA)))
controller=WorkspaceController(PlatformClient(board.url,TOKEN,HOST),MODEL,folder/'results_TEST_ONLY')
base=workspace_handler(controller,TOKEN,get_plotlyjs().encode())
class Handler(base):
    def send(self,status,raw,content_type='application/json; charset=utf-8'):
        if content_type.startswith('text/html'):
            raw=raw.replace(b'<body>',b'<body><div style="position:fixed;right:12px;bottom:6px;z-index:100;background:#fff0d6;color:#8b4300;font:11px sans-serif;padding:6px">TEST ONLY: synthetic CT / fake hardware</div>')
        return super().send(status,raw,content_type)
server=ThreadingHTTPServer(('127.0.0.1',8091),Handler)
print('TEST ONLY http://127.0.0.1:8091/ ; fake unified hardware ; binary transport='+str(a.fast),flush=True)
try:server.serve_forever()
finally:server.server_close();board.close()
