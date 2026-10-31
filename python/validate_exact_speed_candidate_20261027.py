"""Offline candidate/archive check and real saved-output wire-size accounting."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import zlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_ExactSpeed_20261027_04'
PACKAGE=APP/'data/ct_studio_integration_20261025'
BOARD=PACKAGE/'ct_studio'
OUT=ROOT/'data/exact_speed_iteration_20261026'
sys.path[:0]=[str(APP/'python'),str(BOARD)]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    from ct_studio_shared_service import validate_package
    from start_ct_studio_demo import configuration
    from nodule_detector import NoduleCaseProvider
    m=validate_package(BOARD);client,host=configuration(PACKAGE/'host_config.json')
    provider=NoduleCaseProvider(PACKAGE,client)
    assert type(client).__name__=='DualSegStudioClient'
    assert client.manifest['ldct_lossless_interface']=='ct_ldct_zlib_v1'
    assert provider.scoring['threshold']==0.6418673229668569
    assert sha(BOARD/'platform_manifest.json')==host['service_bundle_sha256']
    old=ROOT/'data/CT_Studio_DualExact_20261026_02/data/ct_studio_integration_20261025/ct_studio'
    protected=[n for n in m['files_sha256']if n.endswith(('.bit','.hwh','parameters.npz','weights_integer.npz','model.json'))]
    assert all(sha(BOARD/n)==sha(old/n)for n in protected)
    archive=OUT/'board_exact_speed_candidate_20261027_04.zip'
    if archive.exists():
        with zipfile.ZipFile(archive)as z:
            assert set(z.namelist())==set(m['files_sha256'])|{'platform_manifest.json'}
            for n in z.namelist():assert z.read(n)==(BOARD/n).read_bytes()
    else:
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1)as z:
            for n in sorted(m['files_sha256']):
                assert sha(BOARD/n)==m['files_sha256'][n];z.write(BOARD/n,n)
            z.write(BOARD/'platform_manifest.json','platform_manifest.json')
    result=dict(status='OFFLINE_PACKAGE_VALIDATED_NOT_DEPLOYED',service_bundle_sha256=sha(BOARD/'platform_manifest.json'),
        protected_numerical_assets=len(protected),all_numerical_assets_exact=True,default_cleanup='legacy',
        optional_cleanup='structural-preview',client=type(client).__name__,default_ldct_transport='original_v1_raw',archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        archive_private_contains_existing_board_credentials=True,do_not_publish_this_private_archive=True,board_accessed=False)
    write('candidate04_package_validation.json',result);print(json.dumps(result,indent=2),flush=True)
    state=json.loads((ROOT/'data/current_effect_audit_20261026/state.json').read_text(encoding='utf8'))
    base=ROOT/'data/CT_Studio_PC_Paired_20261025_01/data/ct_workspace_demo/results'/state['series_id']/'ldct'
    rows=state['ldct']['results'];records=[]
    for i in range(35,351,21):
        row=rows[i]
        with np.load(base/(row['result_id']+'.npz'),allow_pickle=False)as saved:raw=saved['hu'].astype('<i2').tobytes()
        meta=dict(row['hardware_receipt']);meta.pop('roundtrip_ms',None)
        before=len(json.dumps(dict(meta,hu_base64=base64.b64encode(raw).decode('ascii'))).encode('utf8'))
        t=time.perf_counter();packed=zlib.compress(raw,1);codec=(time.perf_counter()-t)*1000
        after=len(json.dumps(meta,separators=(',',':')).encode('utf8'))+4+len(packed)
        assert zlib.decompress(packed)==raw
        records.append(dict(index=i,old_json_bytes=before,new_binary_bytes=after,desktop_output_compress_ms=codec))
    summary=dict(scope='Desktop-only real-output wire size/codec check, not a speed or power benchmark.',frames=len(records),
        old_output_bytes=sum(x['old_json_bytes']for x in records),new_output_bytes=sum(x['new_binary_bytes']for x in records),
        output_bytes_reduction_percent=100*(1-sum(x['new_binary_bytes']for x in records)/sum(x['old_json_bytes']for x in records)),rows=records)
    write('ldct_wire_size_check.json',summary)
    print(json.dumps({k:v for k,v in summary.items()if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
