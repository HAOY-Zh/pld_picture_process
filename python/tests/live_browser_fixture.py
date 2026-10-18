"""LOCAL UI TEST ONLY: explicit fake hardware, synthetic HU, no FPGA claims."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from test_lung_live import FakeHardware, InferenceService, TOKEN, MODEL, RunningServer, board_handler
from lung_live_web import BoardClient, Controller, handler_for
from plotly.offline import get_plotlyjs
from http.server import ThreadingHTTPServer

folder=Path(sys.argv[1]).resolve();folder.mkdir(parents=True,exist_ok=True)
hu=np.full((8,64,64),-1000,np.float32)
yy,xx=np.mgrid[:64,:64];body=((yy-32)**2/28**2+(xx-32)**2/26**2)<1
hu[:,body]=30
for cx in [22,42]:hu[:,((yy-32)**2/19**2+(xx-cx)**2/8**2)<1]=-800
np.savez(folder/'synthetic_ct.npz',hu=hu,spacing_zyx_mm=[3.,1.,1.],origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
np.savez(folder/'synthetic_expert.npz',mask=(hu==-800).astype(np.uint8),spacing_zyx_mm=[3.,1.,1.],origin_lps_mm=[0,0,0],orientation=[1,0,0,0,1,0])
hardware=FakeHardware();service=InferenceService(hardware,TOKEN)
board=RunningServer(board_handler(service))
controller=Controller(BoardClient(board.url,TOKEN),MODEL,folder/'results')
html=(ROOT/'python/lungseg/live_viewer.html').read_text(encoding='utf-8').replace('__UI_TOKEN__',TOKEN)
html=html.replace('肺区分割 · 在线 FPGA','TEST STUB（非 FPGA 实测） · 在线 FPGA')
server=ThreadingHTTPServer(('127.0.0.1',8089),handler_for(controller,TOKEN,html.encode(),get_plotlyjs().encode()))
print('TEST ONLY http://127.0.0.1:8089; synthetic input '+str(folder/'synthetic_ct.npz'),flush=True)
try:server.serve_forever()
finally:server.server_close();board.close()
