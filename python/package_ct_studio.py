"""Create a NEW local integration deployment, preserving all frozen input packages."""
import argparse
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET
from ldct_contract import sha, validate_release

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def package(host,variant,ldct,out):
    host=Path(host).resolve(); out=Path(out).resolve(); ldct=Path(ldct).resolve()
    if out.exists(): raise ValueError('Choose a new output directory; old deployments are preserved')
    board_name='ct_platform_fast' if variant=='fast' else 'ct_platform'
    original=host/board_name
    manifest=json.loads((original/'platform_manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        p=(original/name).resolve()
        if original not in p.parents or sha(p)!=digest: raise ValueError('Changed base platform: '+name)
    _,descriptor=validate_release(ldct)
    out.mkdir(parents=True)
    for p in host.iterdir():
        if p.is_file(): shutil.copy2(p,out/p.name)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    board=out/'ct_studio'; shutil.copytree(original,board,ignore=ignore)
    shutil.copytree(ldct,board/'ldct',ignore=ignore)
    # The tile driver imports ctdenoise at deployment root. Do not let the
    # developer checkout's python/ directory hide a missing board dependency.
    shutil.copytree(ldct/'python/ctdenoise',board/'ctdenoise',
                    ignore=ignore)
    for name in ('ct_studio_service.py','ct_platform_fast_service.py','ldct_processor.py','ct_denoise_tile_pl_test.py'):
        shutil.copy2(ROOT/'pynq'/name,board/name)
    shutil.copy2(ROOT/'python/ldct_contract.py',board/'ldct_contract.py')
    clocks={}
    for op,row in manifest['basic'].items():
        values={float(e.get('VALUE')) for e in ET.parse(board/row['hwh']).iter('PARAMETER')
                if e.get('NAME')=='PCW_FPGA0_PERIPHERAL_FREQMHZ'}
        if len(values)!=1 or next(iter(values)) not in (50.,100.):
            raise ValueError('Missing/unsupported basic HWH frequency: '+op)
        clocks[op]=values.pop()
    manifest.update(ldct=descriptor,basic_clocks_mhz=clocks,
        integration=dict(schema='ct_studio_ldct_integration_v1',board_verified=False,
            platform_source_commit='f2ab932500da6d954fdd2a6fb693a881a8aafd44',
            ldct_source_commit='8ccf9201c8665d36c19e31d005af5e15480e6ba9'))
    manifest['files_sha256']={p.relative_to(board).as_posix():sha(p) for p in sorted(board.rglob('*'))
                            if p.is_file() and p!=board/'platform_manifest.json'
                            and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    write(board/'platform_manifest.json',manifest)
    public=json.loads((out/'host_platform.json').read_text())
    public.update(platform_manifest_sha256=sha(board/'platform_manifest.json'),ldct=descriptor,basic_clocks_mhz=clocks)
    write(out/'host_platform.json',public)
    config=json.loads((out/'host_config.json').read_text())
    config.update(studio_service='ct_studio_service',ldct_release=str(ldct.relative_to(ROOT)) if ldct.is_relative_to(ROOT) else str(ldct))
    write(out/'host_config.json',config)
    write(out/'integration.json',dict(status='PC_PACKAGE_ONLY_BOARD_ACCEPTANCE_PENDING',variant=variant,
        ldct=descriptor,source_platform_manifest_sha256=sha(original/'platform_manifest.json'),
        integrated_platform_manifest_sha256=sha(board/'platform_manifest.json')))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',type=Path,required=True); p.add_argument('--variant',choices=('classic','fast'),required=True)
    p.add_argument('--ldct-release',type=Path,required=True); p.add_argument('--outdir',type=Path,required=True)
    a=p.parse_args(); print(package(a.host,a.variant,a.ldct_release,a.outdir))
