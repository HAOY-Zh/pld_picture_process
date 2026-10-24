"""Create a separate ROI-enabled package; preserve the accepted deployment."""
import argparse
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET
import zipfile
from roi_contract import TRANSPORT, digest

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'hardware/roi_20261019'


def package(source,out):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve previous package: '+str(out))
    original=source/'ct_studio'
    manifest=json.loads((original/'platform_manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        p=(original/name).resolve()
        if original not in p.parents or digest(p.read_bytes())!=expected:raise ValueError('Original package changed: '+name)
    release=json.loads((VENDOR/'release_manifest.json').read_text())
    for key in ('ct_roi.bit','ct_roi.hwh'):
        if digest((VENDOR/key).read_bytes())!=release['assets']['hardware/'+key]['sha256']:
            raise ValueError('ROI vendor asset changed')
    modules={e.get('INSTANCE') for e in ET.parse(VENDOR/'ct_roi.hwh').iter('MODULE')}
    if not {'axi_dma_0','roi_geometry','roi_bounds','roi_hu_status','roi_count_identity'}<=modules:
        raise ValueError('ROI HWH interface names differ')
    shutil.copytree(source,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    board=out/'ct_studio';shutil.copytree(VENDOR,board/'roi')
    for name in ('ct_studio_roi_service.py','ct_studio_roi_owner.py'):
        shutil.copy2(ROOT/'pynq'/name,board/name)
    shutil.copy2(ROOT/'pynq/roi_vendor/ct_roi_processor.py',board/'ct_roi_processor.py')
    shutil.copy2(ROOT/'python/roi_vendor/ct_roi_reference.py',board/'ct_roi_reference.py')
    shutil.copy2(ROOT/'python/roi_contract.py',board/'roi_contract.py')
    shutil.copy2(ROOT/'python/roi_statistics_contract.py',board/'roi_statistics_contract.py')
    row=dict(module_version='roi-filter-1.0.0',abi_magic='0x524f4931',abi_version=1,clock_mhz=100.,
             transport=TRANSPORT,bit='roi/ct_roi.bit',hwh='roi/ct_roi.hwh',driver='ct_roi_processor.py',reference='ct_roi_reference.py')
    for key in ('bit','hwh','driver','reference'):row[key+'_sha256']=digest((board/row[key]).read_bytes())
    manifest['roi']=row
    manifest['files_sha256']={p.relative_to(board).as_posix():digest(p.read_bytes()) for p in sorted(board.rglob('*'))
        if p.is_file() and p.name!='platform_manifest.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    write(board/'platform_manifest.json',manifest)
    public=json.loads((out/'host_platform.json').read_text());public.update(roi=row,platform_manifest_sha256=digest((board/'platform_manifest.json').read_bytes()))
    write(out/'host_platform.json',public)
    config=json.loads((out/'host_config.json').read_text());config.update(studio_service='ct_studio_roi_service',board_url='http://192.168.2.99:8766')
    write(out/'host_config.json',config)
    archive=out/'ct_studio_roi_20261022_01.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(board.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(board).as_posix())
    write(out/'roi_integration.json',dict(status='PACKAGE_ONLY_BOARD_ACCEPTANCE_PENDING',roi=row,
        original_manifest_sha256=digest((original/'platform_manifest.json').read_bytes()),
        manifest_sha256=public['platform_manifest_sha256'],archive_sha256=digest(archive.read_bytes()),archive_bytes=archive.stat().st_size))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=ROOT/'data/ct_studio_ldct_fast_03')
    p.add_argument('--outdir',type=Path,default=ROOT/'data/ct_studio_roi_20261022_01');a=p.parse_args()
    print(package(a.base,a.outdir))
