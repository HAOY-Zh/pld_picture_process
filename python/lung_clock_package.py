"""Prepare a tiny additive clock-calibrated entrypoint; never rewrite frozen deploy files."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes,timing

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'pynq'))
from lung_fast_benchmark import validate_package
from lung_clock_benchmark import TARGET_MHZ


def generate(root):
    original=WORKSPACE/'data/lung_fast_20261015/lung_fast'
    _,_,_,fast=validate_package(original);check_hashes(fast['source_sha256'])
    board=WORKSPACE/'data/lung_network_board_20261015'
    proof=json.loads((board/'verification.json').read_text(encoding='utf-8'))
    check_hashes(proof['packaged_source_sha256'])
    if proof['hardware']['clock_mhz']!=TARGET_MHZ or timing(board/'timing_routed.rpt')!=proof['routed_wns_ns']:
        raise ValueError('Need original 100MHz routed timing proof')
    if proof['routed_whs_ns']<0: raise ValueError('Hold timing failed')
    for name in ['lung_network.bit','lung_network.hwh']:
        if file_sha256(original/name)!=proof['package_files_sha256'][name]: raise ValueError('Routed bit/HWH changed')
    root=fresh_output(root).resolve(); deploy=root/'deploy';deploy.mkdir()
    name='lung_clock_benchmark.py';shutil.copy2(WORKSPACE/'pynq'/name,deploy/name)
    sources={str(p):file_sha256(p) for p in [WORKSPACE/'pynq'/name,Path(__file__).resolve(),
        WORKSPACE/'python/tests/test_lung_clock_benchmark.py',board/'verification.json',board/'timing_routed.rpt']}
    record=dict(schema='lung_routed_clock_benchmark_v1',target_clock_mhz=TARGET_MHZ,
        original_fast_manifest_sha256=file_sha256(original/'fast_manifest.json'),
        files_sha256={name:file_sha256(deploy/name)},source_sha256=sources,
        routed_design=dict(clock_mhz=TARGET_MHZ,wns_ns=proof['routed_wns_ns'],whs_ns=proof['routed_whs_ns'],
            bit_sha256=proof['package_files_sha256']['lung_network.bit'],hwh_sha256=proof['package_files_sha256']['lung_network.hwh']),
        warning='100MHz is routed design frequency, not overclocking. Must set only after download while idle; hardware correctness/performance at corrected runtime frequency awaits new real-board test.')
    write_json(deploy/'clock_manifest.json',record)
    write_json(root/'preparation.json',dict(status='prepared_additive_clock_entrypoint_not_board_tested',
        files_sha256={p.name:file_sha256(p) for p in deploy.iterdir()},source_sha256=sources,
        original_fast_manifest_sha256=record['original_fast_manifest_sha256']))
    print('Upload ONLY these two files into existing board lung_fast folder: '+str(deploy),flush=True)


def audit(root):
    root=Path(root).resolve();output=root/'verification.json'
    if output.exists(): raise FileExistsError('Preserve prior evidence')
    preparation=json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    check_hashes(preparation['source_sha256'])
    check_hashes({str(root/'deploy'/n):h for n,h in preparation['files_sha256'].items()})
    original=WORKSPACE/'data/lung_fast_20261015/lung_fast'
    _,_,_,fast=validate_package(original);check_hashes(fast['source_sha256'])
    if file_sha256(original/'fast_manifest.json')!=preparation['original_fast_manifest_sha256']: raise ValueError('Original deployment changed')
    raw=(root/'python_tests.log').read_bytes()
    log=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    match=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',log)
    if not match or int(match[1])<251 or 'skipped=' in log: raise ValueError('Need full 251-test regression')
    protected=json.loads((WORKSPACE/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    check_hashes(protected)
    write_json(output,dict(status='audited_additive_clock_entrypoint_not_board_tested',python_tests=int(match[1]),
        protected_medical_sources_unchanged=len(protected),runtime_clock_corrected_on_board=False,
        preparation_sha256=file_sha256(root/'preparation.json'),python_tests_sha256=file_sha256(root/'python_tests.log'),
        source_sha256=preparation['source_sha256']))
    print('Audit PASS. Same model/bit/runtime preserved; corrected-clock board run still required.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_clock_20261015')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify: audit(args.root)
    else: generate(args.root)
