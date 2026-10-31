"""Build a separate canvas+frozen-score web release; no board operations."""
import ast
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'data/CT_Studio_SharedSeg_20261025_01'
DST = ROOT/'data/CT_Studio_NoduleFull_20261025_01'


def write(path, value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def change(path, before, after):
    text=path.read_text(encoding='utf8')
    if text.count(before)!=1: raise ValueError('Source differs: '+str(path)+' anchor '+before)
    path.write_text(text.replace(before,after),encoding='utf8')


def main():
    from nodule_canvas_contract import digest
    DST.mkdir(exist_ok=False)
    shutil.copy2(ROOT/'data/CT_Studio_PC_Paired_20261025_01/requirements-pc-verified.txt',DST/'requirements-pc-verified.txt')
    for name in ('python','pynq','configs'):
        shutil.copytree(SRC/name,DST/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    package=DST/'data/ct_studio_integration_20261025'
    shutil.copytree(SRC/'data/ct_studio_integration_20261025',package,
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','build'))
    board=package/'ct_studio'
    for name in ('nodule_canvas_contract.py','ct_nodule_canvas_client.py','nodule_frozen_scoring.py'):
        shutil.copy2(ROOT/'python'/name,DST/'python'/name)
    shutil.copy2(ROOT/'python/nodule_canvas_detector.py',DST/'python/nodule_detector.py')
    shutil.copy2(SRC/'python/nodule_detector.py',DST/'python/nodule_raw_reference.py')
    for name in ('nodule_canvas.py','nodule_fast_input.py'):
        shutil.copy2(ROOT/'pynq'/name,DST/'python/nodule_vendor'/name)
    # Only change the import namespace; preserve algorithms and numerical rules.
    change(DST/'python/nodule_vendor/nodule_canvas.py','from nodule_plan import validate','from .nodule_plan import validate')
    for name in ('lung_nodule_continuity.py','lung_nodule_continuity_boundary.py'):
        shutil.copy2(ROOT/'python'/name,DST/'python/nodule_vendor'/name)
    change(DST/'python/nodule_vendor/lung_nodule_continuity_boundary.py',
        'from lung_nodule_continuity import (','from .lung_nodule_continuity import (')
    source_ranker=ROOT/'data/nodule_board_20261023_v1/nodule_board/nodule_ranker.json'
    shutil.copy2(source_ranker,package/'nodule_ranker.json')
    plan=json.loads((ROOT/'data/lung_nodule_sealed113_20261022_v1/plan.json').read_text())
    write(package/'nodule_scoring.json',dict(schema='frozen_nodule_scoring_v1',ranker='nodule_ranker.json',
        ranker_sha256=digest(source_ranker.read_bytes()),threshold=plan['thresholds']['cross']['4.0'],
        model_checkpoint_sha256='c1fefe55a74d1cdc81ede2f2e249190aef7d0b71a8a81378c950427ddcd8dc06',
        source='Frozen75 calibration threshold; no fitting on current scan',primary_profile='development4_fp_per_ct'))
    for name in ('nodule_canvas.py','nodule_canvas_pl.py'):
        text=(ROOT/'pynq'/name).read_text()
        for module in ('nodule_plan','nodule_pl','nodule_canvas','lung_continuous'):
            text=text.replace('from '+module+' import','from .'+module+' import')
        (board/'shared_cnn'/name).write_text(text,encoding='utf8')
    shutil.copy2(ROOT/'pynq/ct_nodule_canvas_service.py',board/'ct_nodule_canvas_service.py')
    shutil.copy2(ROOT/'python/nodule_canvas_contract.py',board/'nodule_canvas_contract.py')
    owner=(board/'ct_shared_seg_owner.py').read_text().replace('from ct_shared_seg_service import start',
        'from ct_nodule_canvas_service import start')
    (board/'ct_nodule_canvas_owner.py').write_text(owner,encoding='utf8')
    manifest=json.loads((board/'platform_manifest.json').read_text())
    for name in ('shared_cnn/nodule_canvas.py','shared_cnn/nodule_canvas_pl.py','nodule_canvas_contract.py',
                 'ct_nodule_canvas_service.py','ct_nodule_canvas_owner.py'):
        manifest['files_sha256'][name]=digest((board/name).read_bytes())
    manifest['canvas_interface']='ct_shared_nodule_canvas_v1'
    write(board/'platform_manifest.json',manifest)
    bundle=digest((board/'platform_manifest.json').read_bytes())
    for name in ('host_platform.json','shared_host_platform.json'):
        p=package/name;m=json.loads(p.read_text());m['service_bundle_sha256']=bundle
        m['canvas_interface']='ct_shared_nodule_canvas_v1';write(p,m)
    # Keep the real engine route visible; never silently use raw-CNN mode.
    p=DST/'python/nodule_workspace.py'
    change(p,"if values.get('scoring_mode')!='raw_cnn':raise ValueError('当前只提供明确的原始 CNN 实验模式')",
        "if values.get('scoring_mode')!='frozen_cross_layer':raise ValueError('当前使用冻结跨层评分模式，请刷新网页')")
    change(p,"scoring_mode='raw_cnn')","scoring_mode='frozen_cross_layer')")
    change(p,'原始 CNN 分数实验版：使用当前所选 HU；实际 FPGA 推理，检测质量尚未验收。',
        'INT8整层复用＋冻结跨层评分；系数与阈值固定。新版端到端实板质量待验收。')
    p=DST/'python/lungseg/workspace/nodule.js'
    change(p,"scoring_mode:'raw_cnn'","scoring_mode:'frozen_cross_layer'")
    change(p,"'检测当前 CT · FPGA 实验版'", "'检测当前 CT · 整层加速＋冻结评分'")
    change(p,"online?'FPGA 实验候选 · 原始CNN分数'",
        "online?(packet.detector_provenance?.ranker_applied?'FPGA 候选 · 冻结跨层评分':'FPGA 历史候选 · 原始分数')")
    change(p,'实际 FPGA 原始 CNN 实验候选；分数非恶性概率，当前输入的检测质量尚未验收。',
        "${packet.detector_provenance?.ranker_applied?'实际 FPGA 整层复用＋冻结跨层评分':'历史 FPGA 原始候选'}；分数非恶性概率，当前输入的检测质量尚未验收。")
    p=DST/'python/lungseg/workspace/index.html'
    text=p.read_text(encoding='utf8')
    text=text.replace('原始 CNN 实验检测','冻结评分结节检测').replace('原始CNN实验检测','冻结评分结节检测')
    text=text.replace('原始 CNN 实验','冻结评分整层').replace('原始CNN实验','冻结评分整层')
    text=text.replace('当前仅提供原始 CNN 分数实验模式','当前采用整层复用和冻结跨层评分')
    text=text.replace('检测当前 CT · FPGA 实验版','检测当前 CT · 整层加速＋冻结评分')
    text=text.replace('原始 CNN 分数未经标定。','采用冻结跨层评分和固定阈值，不按当前病例调参。')
    text=text.replace('完成当前小批次后停止','完成当前整层后停止')
    p.write_text(text,encoding='utf8')
    # Store provenance including all copied numerical modules and actual assets.
    files={str(p.relative_to(DST)).replace('\\','/'):digest(p.read_bytes()) for p in DST.rglob('*') if p.is_file()}
    write(DST/'nodule_full_manifest.json',dict(schema='ct_nodule_full_web_v1',files_sha256=files,
        service_bundle_sha256=bundle,source=str(SRC),scoring_threshold=plan['thresholds']['cross']['4.0'],
        board_status='NOT_DEPLOYED_OR_VERIFIED_THIS_RELEASE',no_training=True,original_model_unchanged=True))
    for p in (DST/'python').glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    print(json.dumps(dict(folder=str(DST),bundle_sha256=bundle,status='PREPARED_NOT_PHYSICAL_ACCEPTED'),indent=2))


if __name__=='__main__': main()
