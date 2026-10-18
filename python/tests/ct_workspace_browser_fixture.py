"""TEST ONLY: synthetic HU and explicitly fake FPGA for workspace browser checks."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from test_lung_live import FakeHardware, InferenceService, TOKEN, MODEL, RunningServer, board_handler
from ct_workspace_web import WorkspaceController, BoardClient, workspace_handler
from plotly.offline import get_plotlyjs

p=argparse.ArgumentParser();p.add_argument('--folder',required=True,type=Path)
p.add_argument('--port',type=int,default=8091);args=p.parse_args()
folder=args.folder.resolve();folder.mkdir(parents=True,exist_ok=True)
hu=np.full((8,64,64),-1000,np.float32)
y,x=np.mgrid[:64,:64];body=((y-32)**2/28**2+(x-32)**2/26**2)<1
hu[:,body]=30
for cx in [22,42]:hu[:,((y-32)**2/19**2+(x-cx)**2/8**2)<1]=-800
np.savez(folder/'synthetic_ct.npz',hu=hu,spacing_zyx_mm=[3.,1.,1.],
         origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
np.savez(folder/'synthetic_expert.npz',mask=(hu==-800).astype(np.uint8),
         spacing_zyx_mm=[3.,1.,1.],origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
board=RunningServer(board_handler(InferenceService(FakeHardware(),TOKEN)))
controller=WorkspaceController(BoardClient(board.url,TOKEN),MODEL,folder/'results_TEST_ONLY')
handler=workspace_handler(controller,TOKEN,get_plotlyjs().encode())
class LabeledHandler(handler):
    def send(self,status,raw,content_type='application/json; charset=utf-8'):
        if content_type.startswith('text/html'):
            raw=raw.replace('<body>'.encode(),'<body><div style="position:fixed;right:12px;bottom:6px;z-index:100;background:#fff0d6;color:#8b4300;font:11px sans-serif;padding:6px">TEST ONLY · 合成影像 / 模拟硬件</div>'.encode())
        return super().send(status,raw,content_type)
server=ThreadingHTTPServer(('127.0.0.1',args.port),LabeledHandler)
print(f'TEST ONLY http://127.0.0.1:{args.port}/',flush=True)
try:server.serve_forever()
finally:server.server_close();board.close()
