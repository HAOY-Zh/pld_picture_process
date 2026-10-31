"""Stage optional wire improvement and explicit cleanup preview, without deploy."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/CT_Studio_ExactSpeed_20261026_03'
DST=ROOT/'data/CT_Studio_ExactSpeed_20261027_04'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('Unexpected anchor '+a[:90])
    return s.replace(a,b)


def main():
    if DST.exists():raise FileExistsError('Existing candidate retained')
    shutil.copytree(SRC,DST,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    package=DST/'data/ct_studio_integration_20261025';board=package/'ct_studio'
    for n in ('ct_ldct_lossless_client.py','lung_structural_cleanup.py'):
        shutil.copy2(ROOT/'python'/n,DST/'python'/n)
    n='ct_ldct_lossless_service.py';shutil.copy2(ROOT/'pynq'/n,board/n)
    owner=(board/'ct_dual_io_owner.py').read_text(encoding='utf8')
    owner=once(owner,'from ct_dual_io_service import start','from ct_ldct_lossless_service import start')
    (board/'ct_ldct_lossless_owner.py').write_text(owner,encoding='utf8')
    manifest=json.loads((board/'platform_manifest.json').read_text(encoding='utf8'))
    for name in ('ct_ldct_lossless_service.py','ct_ldct_lossless_owner.py'):
        manifest['files_sha256'][name]=sha(board/name)
    manifest['ldct_lossless_interface']='ct_ldct_zlib_v1';write(board/'platform_manifest.json',manifest)
    bundle=sha(board/'platform_manifest.json')
    for n in ('host_platform.json','shared_host_platform.json'):
        p=package/n;m=json.loads(p.read_text(encoding='utf8'));m.update(
            service_bundle_sha256=bundle,ldct_lossless_interface='ct_ldct_zlib_v1');write(p,m)
    for name in ('ct_workspace_web.py','start_ct_studio_demo.py'):
        p=DST/'python'/name;s=p.read_text(encoding='utf8')
        s=s.replace("'exact_host_r1'","'exact_host_raw_r3'")
        # Physical pilot rejected LDCT wire compression. Retain the exact V3
        # segmentation/nodule transport and original raw LDCT by default.
        if name=='ct_workspace_web.py':
            s=once(s,'    args = parser.parse_args()',
                "    parser.add_argument('--cleanup-policy',choices=('legacy','structural-preview'),default='legacy',help='Structural preview changes only optional 3-D cleanup; model/ROI body mask unchanged')\n    args = parser.parse_args()\n"
                "    if args.cleanup_policy=='structural-preview':\n"
                "        if args.resume_session:raise ValueError('Preview requires new computation; old saved cleanup is preserved')\n"
                "        import lung_live_web as core\n"
                "        from lung_structural_cleanup import clean_for_display\n"
                "        core.clean_prediction=clean_for_display")
            s=once(s,'    controller = WorkspaceController(client, model, args.output)',
                "    controller = WorkspaceController(client, model, args.output)\n    controller.status['cleanup_policy']=args.cleanup_policy")
        p.write_text(s,encoding='utf8')
    p=DST/'python/lungseg/live_viewer.html';s=p.read_text(encoding='utf8')
    s=once(s,"function validStage(){", "function validStage(){$('stage').querySelector('[value=cleaned]').textContent=state?.cleanup_policy==='structural-preview'?'结构清理预览 · 近主肺15mm保护':'三维清理后（整例完成）';")
    p.write_text(s,encoding='utf8')
    original=SRC/'data/ct_studio_integration_20261025/ct_studio'
    protected=[n for n in manifest['files_sha256']if n.endswith(('.bit','.hwh','parameters.npz','weights_integer.npz','model.json'))]
    for n in protected:
        if sha(board/n)!=sha(original/n):raise ValueError('Numerical asset changed '+n)
    for p in list((DST/'python').glob('*.py'))+[board/'ct_ldct_lossless_service.py',board/'ct_ldct_lossless_owner.py']:
        ast.parse(p.read_text(encoding='utf8'))
    write(DST/'exact_speed_candidate.json',dict(status='OFFLINE_CANDIDATE_NOT_BOARD_ACCEPTED',
        service_bundle_sha256=bundle,owner='ct_ldct_lossless_owner.py',models_weights_bits_unchanged=True,
        default_cleanup='legacy',optional_cleanup='structural-preview',production_not_modified=True))
    print('WIRE_AND_PREVIEW_CANDIDATE_PREPARED',DST,bundle)


if __name__=='__main__':main()
