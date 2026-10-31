"""Stage a new independent board/host candidate; no deployment or hardware use."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/CT_Studio_ExactSpeed_20261027_04'
DST=ROOT/'data/CT_Studio_RawLDCT_20261027_05'
OUT=ROOT/'data/threshold_ldct_iteration_20261027'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    if DST.exists():raise FileExistsError('Existing candidate preserved')
    OUT.mkdir(exist_ok=True)
    # No cases, logs, former runtime ZIPs or user credentials beyond the paired
    # private package; source package is used only for the private board staging.
    DST.mkdir();shutil.copytree(SRC/'python',DST/'python',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if (SRC/'configs').exists():shutil.copytree(SRC/'configs',DST/'configs')
    package=DST/'data/ct_studio_integration_20261025';package.parent.mkdir(parents=True)
    shutil.copytree(SRC/'data/ct_studio_integration_20261025',package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    board=package/'ct_studio'
    for name in ('ct_ldct_raw_client.py','ldct_raw_contract.py'):
        shutil.copy2(ROOT/'python'/name,DST/'python'/name)
    shutil.copy2(ROOT/'python/ldct_raw_contract.py',board/'ldct_raw_contract.py')
    shutil.copy2(ROOT/'pynq/ct_ldct_raw_service.py',board/'ct_ldct_raw_service.py')
    owner=(board/'ct_ldct_lossless_owner.py').read_text(encoding='utf8')
    assert owner.count('from ct_ldct_lossless_service import start')==1
    (board/'ct_ldct_raw_owner.py').write_text(owner.replace('from ct_ldct_lossless_service import start','from ct_ldct_raw_service import start'),encoding='utf8')
    manifest=json.loads((board/'platform_manifest.json').read_text(encoding='utf8'))
    for name in ('ct_ldct_raw_service.py','ct_ldct_raw_owner.py','ldct_raw_contract.py'):
        manifest['files_sha256'][name]=sha(board/name)
    manifest['ldct_raw_interface']='ct_ldct_raw_exact_v1';write(board/'platform_manifest.json',manifest)
    bundle=sha(board/'platform_manifest.json')
    for name in ('host_platform.json','shared_host_platform.json'):
        p=package/name;m=json.loads(p.read_text(encoding='utf8'));m.update(service_bundle_sha256=bundle,ldct_raw_interface='ct_ldct_raw_exact_v1');write(p,m)
    for name in ('ct_workspace_web.py','start_ct_studio_demo.py'):
        p=DST/'python'/name;s=p.read_text(encoding='utf8')
        assert s.count('from ct_dual_io_client import DualSegStudioClient as SharedSegStudioClient')==1
        s=s.replace('from ct_dual_io_client import DualSegStudioClient as SharedSegStudioClient','from ct_ldct_raw_client import RawLDCTStudioClient as SharedSegStudioClient')
        s=s.replace('exact_host_raw_r3','exact_host_raw_candidate_r4');p.write_text(s,encoding='utf8')
    protected=[n for n in manifest['files_sha256']if n.endswith(('.bit','.hwh','parameters.npz','weights_integer.npz','model.json'))]
    prior=SRC/'data/ct_studio_integration_20261025/ct_studio'
    assert all((board/n).read_bytes()==(prior/n).read_bytes()for n in protected)
    for p in list((DST/'python').glob('*.py'))+[board/'ct_ldct_raw_service.py',board/'ct_ldct_raw_owner.py']:
        ast.parse(p.read_text(encoding='utf8'))
    archive=OUT/'raw_ldct_board_candidate_private.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1)as z:
        for n in sorted(manifest['files_sha256']):
            assert sha(board/n)==manifest['files_sha256'][n];z.write(board/n,n)
        z.write(board/'platform_manifest.json','platform_manifest.json')
    report=dict(status='STAGED_NOT_BOARD_ACCEPTED',app=str(DST),service_bundle_sha256=bundle,
        archive=str(archive),archive_sha256=sha(archive),protected_assets_exact=len(protected),
        private_archive_contains_existing_board_credentials=True,do_not_publish_private_archive=True,
        models_bits_clock_unchanged=True,board_accessed=False,new_fpga_resources=0)
    write(OUT/'raw_candidate.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
