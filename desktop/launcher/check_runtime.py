"""Exercise original PC numerical operations with synthetic input only."""
import hashlib
import json
from pathlib import Path
import sys

APP=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(APP/'python'))
import numpy as np
import torch
from lung_live_data import input_codes,native_masks
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from plotly.offline import get_plotlyjs
from ldct_contract import validate_release
from shared_patch_contract import validate_assets

torch.set_num_threads(2)
model=json.loads((APP/'data/ct_studio_integration_20261025/model.json').read_text('utf-8'))
y,x=np.indices((512,512));hu=((x*7+y*13)%1800-1100).astype(np.float32)
codes=input_codes(hu,model)
head=(np.arange(256*256,dtype=np.int64)%256).astype(np.uint8).reshape(1,256,256)
raw,body,plan=native_masks(head,hu,{'spacing_zyx_mm':[1.,.75,.75]},model)
g=gaussian_filter(hu,1)
volume=np.zeros((9,11,13),np.float32);volume[2:7,2:9,2:11]=1
vertices,faces,_,_=marching_cubes(volume,.5)
package=APP/'data/ct_studio_integration_20261025'
release,descriptor=validate_release(package/'ct_studio/ldct')
bundle=json.loads((package/'shared_host_platform.json').read_text('utf-8'))
validate_assets(package/'ct_studio',bundle['shared_nodule'])
sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
print(json.dumps(dict(input_codes_sha256=sha(codes),restored_raw_sha256=sha(raw),
    restored_body_sha256=sha(body),gaussian_sha256=sha(g),mesh_vertices_sha256=sha(vertices),
    mesh_faces_sha256=sha(faces),plotly_js_bytes=len(get_plotlyjs().encode()),
    ldct_identity=descriptor['weights_sha256'],shared_bundle=bundle['service_bundle_sha256']),sort_keys=True))
