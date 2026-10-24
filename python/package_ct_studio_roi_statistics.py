"""New pinned ROI/RST1 deployment. Preserve previous packages and all other bits/models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET
import zipfile

from roi_contract import digest,validate_assets
from roi_statistics_contract import STATISTICS_ABI,GPIO_NAMES

ROOT=Path(__file__).resolve().parents[1]


def package(base,build,out):
    base,build,out=map(lambda p:Path(p).resolve(),(base,build,out))
    if out.exists():raise FileExistsError('Preserve existing deployment: '+str(out))
    board=base/'ct_studio'
    original=json.loads((board/'platform_manifest.json').read_text(encoding='utf-8'))
    for name,expected in original['files_sha256'].items():
        p=(board/name).resolve()
        if board not in p.parents or digest(p.read_bytes())!=expected:raise ValueError('Changed baseline '+name)
    timing={line.split('=',1)[0]:line.split('=',1)[1] for line in (build/'timing_status.txt').read_text().splitlines() if '=' in line}
    if (timing.get('SCOPE')!='FULL_ROI_PS_DMA_GPIO_OVERLAY' or timing.get('CLOCK_MHZ')!='100'
            or float(timing['WNS_NS'])<0 or float(timing['WHS_NS'])<0):raise ValueError('Full overlay timing did not pass')
    bit=build/'deploy/ct_roi_statistics.bit';hwh=build/'deploy/ct_roi_statistics.hwh'
    modules={m.get('INSTANCE') for m in ET.parse(hwh).iter('MODULE')}
    required=set(GPIO_NAMES)|{'roi_geometry','roi_bounds','roi_hu_status','roi_count_identity','axi_dma_0','roi_statistics_0'}
    if not required<=modules:raise ValueError('Missing stats HWH resources: '+str(required-modules))
    if not bit.is_file() or bit.stat().st_size<1000000:raise ValueError('Missing complete ROI bit')
    shutil.copytree(base,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    target=out/'ct_studio'
    sources={'ct_studio_roi_service.py':ROOT/'pynq/ct_studio_roi_service.py',
        'ct_studio_roi_owner.py':ROOT/'pynq/ct_studio_roi_owner.py',
        'roi_statistics_processor.py':ROOT/'pynq/roi_statistics_processor.py',
        'roi_statistics_contract.py':ROOT/'python/roi_statistics_contract.py','roi_contract.py':ROOT/'python/roi_contract.py'}
    for name,path in sources.items():shutil.copy2(path,target/name)
    statsdir=target/'roi_statistics';statsdir.mkdir()
    shutil.copy2(bit,statsdir/'ct_roi_statistics.bit');shutil.copy2(hwh,statsdir/'ct_roi_statistics.hwh')
    for name in ('timing_status.txt','timing_routed.rpt','utilization_routed.rpt','drc.rpt','block_design.tcl'):
        shutil.copy2(build/name,statsdir/name)
    rtl=ROOT/'hardware/roi_statistics_20261024/fpga'
    for name in ('pynq_z2_roi.v','pynq_z2_roi_statistics.v'):shutil.copy2(rtl/name,statsdir/name)
    row=dict(original['roi'],bit='roi_statistics/ct_roi_statistics.bit',hwh='roi_statistics/ct_roi_statistics.hwh',
        driver='roi_statistics_processor.py',statistics_abi=STATISTICS_ABI,
        statistics_contract='roi_statistics_contract.py',statistics_gpio=list(GPIO_NAMES),
        statistics_identity='0x52535431')
    for key in ('bit','hwh','driver','reference','statistics_contract'):
        row[key+'_sha256']=digest((target/row[key]).read_bytes())
    manifest=dict(original,roi=row)
    manifest['files_sha256']={p.relative_to(target).as_posix():digest(p.read_bytes()) for p in sorted(target.rglob('*'))
        if p.is_file() and p.name!='platform_manifest.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    write=lambda path,v:path.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
    write(target/'platform_manifest.json',manifest);manifest_sha=digest((target/'platform_manifest.json').read_bytes())
    public=json.loads((out/'host_platform.json').read_text(encoding='utf-8'))
    public.update(roi=row,platform_manifest_sha256=manifest_sha);write(out/'host_platform.json',public)
    config=json.loads((out/'host_config.json').read_text(encoding='utf-8'));config['studio_service']='ct_studio_roi_service'
    write(out/'host_config.json',config)
    validate_assets(target,row)
    preserved={}
    for name,h in original['files_sha256'].items():
        if name.startswith(('ldct/','segmentation/','basic_shared/')) or name.endswith(('.bit','.hwh')):
            if digest((target/name).read_bytes())!=h:raise ValueError('Preserved artifact changed: '+name)
            preserved[name]=h
    archive=out/'ct_studio_roi_statistics_20261024.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(target.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(target).as_posix())
    evidence=dict(status='BUILT_AND_PACKAGED_BOARD_ACCEPTANCE_PENDING',statistics_abi=STATISTICS_ABI,
        original_package=str(base),original_manifest_sha256=digest((board/'platform_manifest.json').read_bytes()),
        platform_manifest_sha256=manifest_sha,roi=row,clock_mhz=100,timing=timing,
        preserved_artifacts_sha256=preserved,archive_sha256=digest(archive.read_bytes()),
        archive_bytes=archive.stat().st_size,physical_board_verified=False)
    write(out/'statistics_integration.json',evidence)
    return dict(package=str(out),archive=str(archive),platform_manifest_sha256=manifest_sha,preserved_artifacts=len(preserved))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=ROOT/'data/ct_studio_roi_20261022_01')
    p.add_argument('--build',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(package(a.base,a.build,a.outdir),indent=2))
