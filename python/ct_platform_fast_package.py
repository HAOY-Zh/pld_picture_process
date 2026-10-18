"""Create an isolated fast deployment. Never change the existing board release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)


def prepare(build,out):
    build=Path(build).resolve();out=Path(out).resolve()
    if ROOT/'data' not in out.parents or out.exists():raise ValueError('Need new workspace/data output')
    bit=build/'deploy/ct_basic.bit';hwh=build/'deploy/ct_basic.hwh'
    log=(build/'build.stdout.log').read_text(encoding='utf-8',errors='replace')
    if 'CT BASIC SHARED BIT BUILD PASS' not in log:raise ValueError('No passed routed timing/build result')
    modules={m.get('INSTANCE'):m for m in ET.parse(hwh).iter('MODULE')}
    if 'ct_basic_axis_0' not in modules:raise ValueError('Missing shared basic RTL')
    params={x.get('NAME'):x.get('VALUE') for x in modules['axi_dma_0'].findall('./PARAMETERS/PARAMETER')}
    if (params.get('C_INCLUDE_SG')!='0' or int(params.get('C_SG_LENGTH_WIDTH','0'))<21 or
            params.get('C_M_AXIS_MM2S_TDATA_WIDTH')!='32' or params.get('C_S_AXIS_S2MM_TDATA_WIDTH')!='32'):
        raise ValueError('Unexpected DMA geometry')
    for name,width,dual in [('axi_gpio_0',32,'1'),('axi_gpio_1',3,'0')]:
        p={x.get('NAME'):x.get('VALUE') for x in modules[name].findall('./PARAMETERS/PARAMETER')}
        if int(p['C_GPIO_WIDTH'])!=width or p.get('C_IS_DUAL')!=dual:raise ValueError('Unexpected GPIO')
    old=ROOT/'data/ct_platform_20261017_v2'
    source=old/'ct_platform'
    previous=json.loads((source/'platform_manifest.json').read_text(encoding='utf-8'))
    for n,h in previous['files_sha256'].items():
        if sha(source/n)!=h:raise ValueError('Preserved package changed: '+n)
    out.mkdir(parents=True);board=out/'ct_platform_fast';board.mkdir();(board/'basic').mkdir()
    shutil.copytree(source/'segmentation',board/'segmentation')
    for name in ['ct_platform_service.py','ct_basic_processing.py','lung_live_service.py','platform_token.txt']:
        shutil.copy2(source/name,board/name)
    shutil.copy2(ROOT/'pynq/ct_platform_fast_service.py',board/'ct_platform_fast_service.py')
    shutil.copy2(bit,board/'basic/ct_basic.bit');shutil.copy2(hwh,board/'basic/ct_basic.hwh')
    specs={}
    for mode,op in enumerate(('window','threshold','erosion','dilation','opening','closing')):
        row=previous['basic'][op]
        specs[op]=dict(row,bit='basic/ct_basic.bit',hwh='basic/ct_basic.hwh',
            bit_sha256=sha(bit),hwh_sha256=sha(hwh),shared_basic_overlay=True,mode=mode)
    files={p.relative_to(board).as_posix():sha(p) for p in board.rglob('*') if p.is_file()}
    manifest=dict(previous,files_sha256=files,basic=specs,basic_transport='ct_basic_zlib_cache_v1',
        bit_rebuilt=True,integration_board_tested=False,
        warning='New resident basic bit, routed timing and RTL simulation passed; real board startup still required')
    write(board/'platform_manifest.json',manifest)
    shutil.copy2(old/'model.json',out/'model.json');shutil.copy2(old/'platform_token.txt',out/'platform_token.txt')
    write(out/'host_platform.json',dict(platform_manifest_sha256=sha(board/'platform_manifest.json'),
        basic=specs,basic_transport=manifest['basic_transport']))
    config=json.loads((old/'host_config.json').read_text(encoding='utf-8'));write(out/'host_config.json',config)
    write(out/'verification.json',dict(status='packaged_routed_shared_basic_bit_real_board_pending',
        files_sha256=files,manifest_sha256=sha(board/'platform_manifest.json'),
        previous_manifest_sha256=sha(source/'platform_manifest.json'),
        segmentation_bit_unchanged=True,segmentation_model_unchanged=True,
        real_board_startup_verified=False))
    print('Fast package:',board,'; segmentation unchanged; real board startup pending')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--outdir',default='data/ct_platform_fast_20261017_v2');a=p.parse_args();prepare(a.build,a.outdir)
