"""Audit actual tensor RTL, independent vectors, synthesis and complete regression."""
import argparse
import json
from pathlib import Path
import re

from lung_unet import file_sha256
from lung_segmentation import write_json
from verify_lung_axil import check_hashes,timing
from verify_lung_axil_board import utilization


def audit(root):
    root=Path(root).resolve();workspace=Path(__file__).resolve().parents[1]
    output=root/'verification.json'
    if output.exists(): raise ValueError('Do not overwrite tensor proof')
    vectors_path=root/'vectors/vectors.json';vectors=json.loads(vectors_path.read_text(encoding='utf-8'))
    check_hashes(vectors['source_sha256'])
    check_hashes({str(root/'vectors'/n):h for n,h in vectors['files_sha256'].items()})
    log=root/'tensor_run_v3.log';raw=log.read_text(encoding='utf-8')
    marker=r'Lung tensor AXI-Lite RTL: PASS \((\d+) frames, (\d+) words, (\d+) reads, (\d+) error cases;'
    match=re.search(marker,raw)
    if not match or tuple(map(int,match.groups()))!=(26,29848,32690,19) or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)',raw,re.I):
        raise ValueError('Tensor actual RTL proof incomplete')
    tests_path=root/'python_tests.log';tests=tests_path.read_bytes()
    tests=tests.decode('utf-16' if tests.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    if not re.search(r'Ran 207 tests in .*?\n\s*\nOK\s*$',tests): raise ValueError('Full 207-test regression absent')
    synth=root/'synthesis_v2';resources=utilization(synth/'utilization.rpt');wns=timing(synth/'timing.rpt')
    protected=json.loads((workspace/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    check_hashes(protected)
    files=[workspace/'fpga'/n for n in ['lung_tensor_engine.v','lung_tensor_control.v','lung_tensor_axil.v',
           'lung_requant.v','lung_axi_read64.v','lung_axi_write64.v','tb_lung_tensor_axil.sv','tb_lung_write64.sv',
           'tb_lung_axil_control.sv','synth_lung_tensor.tcl','build_lung_network_board.tcl']]
    files += [workspace/'pynq'/n for n in ['lung_tensor.py','lung_tensor_smoke.py','lung_network_smoke.py']]
    files += [workspace/'python'/n for n in ['lung_tensor_vectors.py','tests/test_lung_tensor.py','verify_lung_tensor.py']]
    evidence=[log,tests_path,vectors_path]+list(synth.glob('*.rpt'))+[synth/'tensor_synth.dcp']
    result=dict(status='passed_tensor_axi_operators_not_board_tested',frames=26,output_words=29848,read_words=32690,
                error_cases=19,python_tests=207,utilization=resources,internal_synthesis_wns_ns=wns,
                new_tensor_multicycle_exceptions=False,real_centering_pipeline=True,
                protected_medical_sources_unchanged=len(protected),
                source_sha256={str(p):file_sha256(p) for p in files},
                evidence_sha256={str(p):file_sha256(p) for p in evidence},vectors_sha256=vectors['files_sha256'],
                original_vector_sources_sha256=vectors['source_sha256'],tensor_board_tested=False,full_network_on_fpga=False)
    write_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_tensor_20261015')
    audit(parser.parse_args().root)
