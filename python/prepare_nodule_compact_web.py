"""Separate release for exact compact transport/native CMA packing."""
import ast
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/CT_Studio_NoduleFull_20261025_01'
DST=ROOT/'data/CT_Studio_NoduleCompact_20261026_01'


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    from nodule_compact_contract import digest
    DST.mkdir(exist_ok=False)
    for name in ('python','pynq','configs'):
        shutil.copytree(SRC/name,DST/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(SRC/'requirements-pc-verified.txt',DST/'requirements-pc-verified.txt')
    package=DST/'data/ct_studio_integration_20261025'
    shutil.copytree(SRC/'data/ct_studio_integration_20261025',package,
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','build'))
    board=package/'ct_studio'
    for name in ('nodule_compact_contract.py','ct_nodule_compact_client.py'):
        shutil.copy2(ROOT/'python'/name,DST/'python'/name)
    shutil.copy2(ROOT/'python/nodule_compact_contract.py',board/'nodule_compact_contract.py')
    shutil.copy2(ROOT/'pynq/ct_nodule_compact_service.py',board/'ct_nodule_compact_service.py')
    for name in ('nodule_canvas_packed.py','nodule_quantized_pack.py','nodule_quantized_pack.cpp'):
        shutil.copy2(ROOT/'pynq'/name,board/'shared_cnn'/name)
    owner=(board/'ct_nodule_canvas_owner.py').read_text().replace('from ct_nodule_canvas_service import start',
        'from ct_nodule_compact_service import start')
    (board/'ct_nodule_compact_owner.py').write_text(owner,encoding='utf8')
    manifest=json.loads((board/'platform_manifest.json').read_text())
    for name in ('nodule_compact_contract.py','ct_nodule_compact_service.py','ct_nodule_compact_owner.py',
            'shared_cnn/nodule_canvas_packed.py','shared_cnn/nodule_quantized_pack.py','shared_cnn/nodule_quantized_pack.cpp'):
        manifest['files_sha256'][name]=digest((board/name).read_bytes())
    manifest['compact_interface']='ct_shared_nodule_compact_v2'
    write(board/'platform_manifest.json',manifest)
    bundle=digest((board/'platform_manifest.json').read_bytes())
    for name in ('host_platform.json','shared_host_platform.json'):
        p=package/name;m=json.loads(p.read_text());m['service_bundle_sha256']=bundle
        m['compact_interface']='ct_shared_nodule_compact_v2';write(p,m)
    p=DST/'python/nodule_detector.py';s=p.read_text(encoding='utf8')
    substitutions={
        'from ct_nodule_canvas_client import CanvasClient':'from ct_nodule_compact_client import CompactClient as CanvasClient',
        'from nodule_canvas_contract import quantize_canvas, digest, INTERFACE, shapes':'from nodule_canvas_contract import quantize_canvas, digest, shapes\nfrom nodule_compact_contract import INTERFACE',
        "PIPELINE = 'int8_canvas_frozen_cross_layer_v1'":"PIPELINE = 'int8_canvas_compact_frozen_cross_layer_v2'",
        'codes = lower_canvas(quantize_canvas(sampler.context))':'codes = quantize_canvas(sampler.context)',
        "input='zlib-lowered-uint8'":"input='zlib-five-plane-uint8-native-CMA-binary-persistent'",
    }
    for a,b in substitutions.items():
        if s.count(a)!=1:raise ValueError('Source anchor differs: '+a)
        s=s.replace(a,b)
    p.write_text(s,encoding='utf8')
    for p in (DST/'python').glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    write(DST/'compact_release.json',dict(schema='ct_nodule_compact_release_v2',source=str(SRC),
        service_bundle_sha256=bundle,numerical_model_unchanged=True,threshold_unchanged=True,
        phase='PREPARED_NOT_PHYSICALLY_ACCEPTED'))
    print(json.dumps(dict(folder=str(DST),bundle_sha256=bundle),indent=2))


if __name__=='__main__':main()
