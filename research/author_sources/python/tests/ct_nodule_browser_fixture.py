"""TEST ONLY: syntheticCT, fake FPGA and fake candidates for browser QA."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from test_lung_live import FakeHardware, InferenceService, TOKEN, MODEL, RunningServer, board_handler
from test_ct_nodule import RecordingService
from ct_workspace_web import WorkspaceController, BoardClient, workspace_handler
from lung_unet import torch_module
from plotly.offline import get_plotlyjs


class BrowserService(RecordingService):
    def predict(self,hu,mask,geometry,progress,cancel):
        progress('TEST ONLY synthetic detection',0,1)
        time.sleep(1.5)
        r=super().predict(hu,mask,geometry,progress,cancel)
        r['candidates'][0].update(center_index_zyx=[3.,32.,22.],
            center_lps_xyz_mm=(np.asarray(geometry['origin_lps_mm'])+
                np.asarray(geometry['basis_lps_zyx'])@np.array([3.,32.,22.])).tolist(),
            bbox_zyx=[2,5,28,37,18,27])
        return r


p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--port',type=int,default=8093)
args=p.parse_args();folder=args.folder.resolve();folder.mkdir(parents=True,exist_ok=True)
torch_module().set_num_threads(2)
hu=np.full((8,64,64),-1000,np.float32);y,x=np.mgrid[:64,:64]
hu[:,((y-32)**2/28**2+(x-32)**2/26**2)<1]=30
for cx in (22,42):hu[:,((y-32)**2/19**2+(x-cx)**2/8**2)<1]=-800
np.savez(folder/'synthetic_ct.npz',hu=hu,spacing_zyx_mm=[3.,1.,1.],origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
hw=FakeHardware();service=BrowserService();board=RunningServer(board_handler(InferenceService(hw,TOKEN)))
controller=WorkspaceController(BoardClient(board.url,TOKEN),MODEL,folder/'results_TEST_ONLY',nodule_service=service)
base=workspace_handler(controller,TOKEN,get_plotlyjs().encode())
class Handler(base):
    def do_GET(self):
        if self.path=='/api/test-fixture':
            if self.allowed(True):self.send(200,dict(TEST_ONLY=True,hardware_calls=hw.calls,nodule_calls=service.calls))
            return
        super().do_GET()
    def send(self,status,raw,content_type='application/json; charset=utf-8'):
        if content_type.startswith('text/html'):
            raw=raw.replace(b'<body>',('<body><div style="position:fixed;right:12px;bottom:6px;z-index:100;background:#fff0d6;'
                'color:#8b4300;font:11px sans-serif;padding:6px">TEST ONLY · 合成CT / 模拟硬件 / 模拟候选</div>').encode())
        super().send(status,raw,content_type)
server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
print('TEST ONLY browser fixture ready',flush=True)
try:server.serve_forever()
finally:server.server_close();board.close()
