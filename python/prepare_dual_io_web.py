"""Prepare a separate release; preserve previously accepted apps and bitstreams."""
import ast
import json
from pathlib import Path
import shutil
from nodule_compact_contract import digest

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/CT_Studio_NoduleCompact_20261026_01'
DST=ROOT/'data/CT_Studio_DualExact_20261026_02'


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def replace_once(s,a,b):
    if s.count(a)!=1:raise ValueError('Source anchor differs: '+a[:90])
    return s.replace(a,b)


def main():
    DST.mkdir(exist_ok=False)
    for name in ('python','pynq','configs'):
        shutil.copytree(SRC/name,DST/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(SRC/'requirements-pc-verified.txt',DST/'requirements-pc-verified.txt')
    package=DST/'data/ct_studio_integration_20261025'
    shutil.copytree(SRC/'data/ct_studio_integration_20261025',package,
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','build','*_native.so','*_build.json','*_startup_*.npz','*_startup_*.json'))
    board=package/'ct_studio';added=[]
    for name in ('dual_io_contract.py','ct_dual_io_client.py'):
        shutil.copy2(ROOT/'python'/name,DST/'python'/name)
    for name in ('dual_io_contract.py',):
        shutil.copy2(ROOT/'python'/name,board/name);added.append(name)
    for name in ('ct_dual_io_service.py',):
        shutil.copy2(ROOT/'pynq'/name,board/name);added.append(name)
    for name in ('exact_io_runtime.py','exact_native_io.py'):
        shutil.copy2(ROOT/'pynq'/name,board/'shared_cnn'/name);added.append('shared_cnn/'+name)
    native=board/'shared_cnn/exact_io_native';native.mkdir()
    # Existing bounded native compiler builds a new source/binary identity.
    (native/'nodule_quantized_pack.cpp').write_bytes((ROOT/'pynq/nodule_quantized_pack.cpp').read_bytes()+b'\n'+
        (ROOT/'pynq/exact_native_io.cpp').read_bytes())
    added.append('shared_cnn/exact_io_native/nodule_quantized_pack.cpp')
    owner=(board/'ct_nodule_compact_owner.py').read_text()
    owner=replace_once(owner,'from ct_nodule_compact_service import start','from ct_dual_io_service import start')
    (board/'ct_dual_io_owner.py').write_text(owner,encoding='utf8');added.append('ct_dual_io_owner.py')
    manifest=json.loads((board/'platform_manifest.json').read_text())
    for name in added:manifest['files_sha256'][name]=digest((board/name).read_bytes())
    manifest['dual_io_interface']='ct_dual_exact_io_v3';write(board/'platform_manifest.json',manifest)
    bundle=digest((board/'platform_manifest.json').read_bytes())
    for name in ('host_platform.json','shared_host_platform.json'):
        path=package/name;m=json.loads(path.read_text());m['service_bundle_sha256']=bundle
        m['dual_io_interface']='ct_dual_exact_io_v3';write(path,m)
    path=DST/'python/start_ct_studio_demo.py';s=path.read_text(encoding='utf8')
    s=replace_once(s,'from ct_shared_seg_client import SharedSegStudioClient',
        'from ct_dual_io_client import DualSegStudioClient as SharedSegStudioClient')
    path.write_text(s,encoding='utf8')
    path=DST/'python/nodule_detector.py';s=path.read_text(encoding='utf8')
    s=replace_once(s,'from ct_nodule_compact_client import CompactClient as CanvasClient',
        'from ct_dual_io_client import CachedCanvasClient as CanvasClient')
    s=replace_once(s,'from nodule_compact_contract import INTERFACE','from dual_io_contract import INTERFACE')
    s=replace_once(s,"PIPELINE = 'int8_canvas_compact_frozen_cross_layer_v2'",
        "PIPELINE = 'int8_canvas_cached_exact_frozen_cross_layer_v3'")
    path.write_text(s,encoding='utf8')
    path=DST/'python/lung_live_web.py';s=path.read_text(encoding='utf8')
    s=replace_once(s,'import copy\n','import copy\nfrom concurrent.futures import ThreadPoolExecutor\n')
    start=s.index('        for done, z in enumerate(indices, 1):',s.index('    def _run('))
    finish=s.index('        if self.cancel.is_set():\n            self._update(phase=\'cancelled\'); return\n        plan = reconstruction_plan',start)
    block=s[start:finish]
    work=block[block.index('            raw, body, plan = native_masks'):]
    new='        def finish_slice(done,z,head,receipt,codes):\n'+work+'''
        # One host result in flight; device commands remain strictly serial.
        # The context manager drains outstanding host work on errors/cancel.
        pending=None
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='seg-result') as executor:
            for done,z in enumerate(indices,1):
                if self.cancel.is_set():break
                self._update(active_slice=int(z))
                codes=input_codes(self.hu[z],self.model)
                head,receipt=self.client.infer(codes)  # No retries or fallback.
                if pending is not None:pending.result()
                pending=executor.submit(finish_slice,done,z,head,receipt,codes)
            if pending is not None:pending.result()
'''
    s=s[:start]+new+s[finish:];path.write_text(s,encoding='utf8')
    for path in (DST/'python').glob('*.py'):ast.parse(path.read_text(encoding='utf8'))
    write(DST/'dual_exact_release.json',dict(schema='ct_dual_exact_release_v3',source=str(SRC),
        service_bundle_sha256=bundle,model_bit_and_threshold_unchanged=True,phase='PREPARED_NOT_PHYSICALLY_ACCEPTED'))
    print(json.dumps(dict(folder=str(DST),bundle_sha256=bundle),indent=2))


if __name__=='__main__':main()
