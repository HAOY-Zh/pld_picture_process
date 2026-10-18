"""Package six existing basic bits and preserved segmentation without rebuilding."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import xml.etree.ElementTree as ET

from lung_live_web import BIT_SHA, MODEL_SHA

ROOT=Path(__file__).resolve().parents[1]
OPERATIONS={'window':'window_level','threshold':'threshold','erosion':'erosion',
            'dilation':'dilation','opening':'opening','closing':'closing'}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def prepare(outdir):
    outdir=Path(outdir).resolve()
    if outdir.exists():raise FileExistsError('Preserve existing platform release')
    source=ROOT/'data/lung_live_20261016/lung_live'
    live=json.loads((source/'live_manifest.json').read_text(encoding='utf-8'))
    for name,h in live['files_sha256'].items():
        if digest(source/name)!=h:raise ValueError('Preserved live package changed: '+name)
    if digest(source/'lung_network.bit')!=BIT_SHA or digest(source/'model.json')!=MODEL_SHA:
        raise ValueError('Wrong preserved segmentation release')
    specs={};protected={};hardware=[]
    for op,name in OPERATIONS.items():
        project=ROOT/'fpga'/('pynq_z2_'+name+'_axis')
        bit=project/(project.name+'.runs')/'impl_1/design_1_wrapper.bit'
        hwh=project/(project.name+'.gen')/'sources_1/bd/design_1/hw_handoff/design_1.hwh'
        tree=ET.parse(hwh);modules={m.get('INSTANCE'):m for m in tree.iter('MODULE')}
        dma=modules['axi_dma_0'];p={x.get('NAME'):x.get('VALUE') for x in dma.findall('./PARAMETERS/PARAMETER')}
        if (p.get('C_INCLUDE_SG')!='0' or int(p.get('C_SG_LENGTH_WIDTH','0'))<21 or
            p.get('C_M_AXIS_MM2S_TDATA_WIDTH')!='32' or p.get('C_S_AXIS_S2MM_TDATA_WIDTH')!='32'):
            raise ValueError('Unexpected DMA geometry: '+op)
        if op=='threshold' and not any('threshold_range_axis' in str(m.attrib) for m in modules.values()):
            raise ValueError('Threshold bit does not declare the range core')
        if op=='window':
            g={x.get('NAME'):x.get('VALUE') for x in modules['axi_gpio_0'].findall('./PARAMETERS/PARAMETER')}
            if g.get('C_IS_DUAL')!='1' or g.get('C_GPIO_WIDTH')!='16' or g.get('C_GPIO2_WIDTH')!='16':
                raise ValueError('Unexpected window GPIO')
        protected[str(bit)]=digest(bit);protected[str(hwh)]=digest(hwh)
        specs[op]=dict(bit='basic/ct_'+name+'.bit',hwh='basic/ct_'+name+'.hwh',
            bit_sha256=digest(bit),hwh_sha256=digest(hwh),shape=[512,512],
            pl_operation=op,mask_preparation='PS HU range threshold' if op not in ('window','threshold') else None)
        hardware.append((bit,hwh,specs[op]))
    outdir.mkdir(parents=True);board=outdir/'ct_platform';board.mkdir();(board/'basic').mkdir()
    # Keep all old package bytes, including nested manifest dependencies, intact.
    shutil.copytree(source,board/'segmentation',ignore=shutil.ignore_patterns('*.so','*.dll','__pycache__'))
    for bit,hwh,spec in hardware:
        shutil.copy2(bit,board/spec['bit']);shutil.copy2(hwh,board/spec['hwh'])
    for src,name in [(ROOT/'pynq/ct_platform_service.py','ct_platform_service.py'),
                     (ROOT/'python/ct_basic_processing.py','ct_basic_processing.py'),
                     (ROOT/'pynq/lung_live_service.py','lung_live_service.py')]:
        shutil.copy2(src,board/name)
    token=secrets.token_hex(32)
    for path in [outdir/'platform_token.txt',board/'platform_token.txt']:
        with path.open('x',encoding='utf-8') as f:f.write(token+'\n')
    files={p.relative_to(board).as_posix():digest(p) for p in board.rglob('*') if p.is_file()}
    manifest=dict(schema='ct_platform_package_v1',files_sha256=files,basic=specs,
        segmentation_bit_sha256=BIT_SHA,model_sha256=MODEL_SHA,bit_rebuilt=False,
        integration_board_tested=False,shape=[512,512],
        warning='Existing modules repackaged; unified switching/web DMA still requires real board acceptance')
    write(board/'platform_manifest.json',manifest)
    shutil.copy2(source/'model.json',outdir/'model.json')
    write(outdir/'host_config.json',dict(board_url='http://192.168.2.99:8765',token_file='platform_token.txt',
        model='model.json',platform_manifest='host_platform.json'))
    write(outdir/'host_platform.json',dict(platform_manifest_sha256=digest(board/'platform_manifest.json'),basic=specs))
    for name,h in protected.items():
        if digest(name)!=h:raise ValueError('Original basic file changed')
    for name,h in live['files_sha256'].items():
        if digest(source/name)!=h:raise ValueError('Original segmentation file changed')
    write(outdir/'verification.json',dict(status='packaged_existing_basic_and_segmentation_no_bit_rebuild',
        files=len(files),operations=list(specs),protected_sources_sha256=protected,
        manifest_sha256=digest(board/'platform_manifest.json'),
        package_files_sha256=files,originals_unchanged=True,integration_board_tested=False))
    print('Unified platform package:',board,'; six original basic bits + preserved segmentation; no rebuild',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',default='data/ct_platform_20261017_v2')
    prepare(p.parse_args().outdir)
