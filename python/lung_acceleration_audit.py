"""Final read-only identity audit of flow acceptance, LUT and pruning experiments."""
import json
from pathlib import Path
import re

from lung_flow_prepare import ROOT as FLOW, WORKSPACE, checked
from lung_segmentation import write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes


def run():
    lut=WORKSPACE/'data/lung_lut_20261017';pruned=WORKSPACE/'data/lung_prune_board_20261017'
    recovery=WORKSPACE/'data/lung_prune_recover_20261017';first=WORKSPACE/'data/lung_prune_20261017'
    output=WORKSPACE/'data/lung_acceleration_20261017'
    if output.exists():raise FileExistsError('Preserve final experiment audit')
    evidence={};protected={}
    def read(path):
        evidence[str(path)]=file_sha256(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def check(mapping):
        check_hashes(mapping);evidence.update(mapping)
    accepted=read(FLOW/'lung_flow_result_threeway_verification.json')
    if accepted['status']!='verified_returned_threeway_actual_fp32_and_int8_heads' or not accepted['int8_head_bit_exact']:
        raise ValueError('Flow acceptance missing')
    check({str(FLOW/n):h for n,h in accepted['returned_files_sha256'].items()})
    prep=checked(lut);protected.update(prep['protected_sources_sha256'])
    routed=read(lut/'verification.json');physical=read(lut/'physopt_verification.json')
    if routed['status']!='built_routed_concat_lut_not_board_tested' or physical['status']!='verified_100mhz_physical_only_repair':
        raise ValueError('LUT implementation proof incomplete')
    for key in ['source_sha256','packaged_source_sha256','evidence_sha256']:check(routed[key])
    check(physical['evidence_sha256']);check({str(lut/'lung_lut'/n):h for n,h in routed['package_files_sha256'].items()})
    encoding=read(lut/'python_tests_encoding.json')
    raw=(lut/'python_tests_powershell_utf16.log').read_bytes();normal=(lut/'python_tests.log').read_bytes()
    if raw.decode('utf-16').encode('utf-8')!=normal:raise ValueError('Test log normalization changed content')
    if file_sha256(lut/'python_tests.log')!=encoding['utf8_sha256'] or file_sha256(lut/'python_tests_powershell_utf16.log')!=encoding['raw_sha256']:
        raise ValueError('Test log hash changed')
    if not re.search(r'Ran 345 tests in .*?\n\s*\nOK\s*$',normal.decode('utf-8').replace('\r\n','\n')):
        raise ValueError('Complete Python regression missing')
    for root in [first,recovery]:
        protocol=read(root/'protocol.json');check(protocol['input_sha256']);check(protocol['source_sha256'])
        protected.update(protocol['input_sha256'])
        frozen=read(root/'freeze.json');check(frozen['files_sha256']);check(frozen['source_sha256'])
        completed=read(root/'verification.json')
        if completed['status']!='complete_pruning_development_experiment' or completed['test_or_covid_used']:
            raise ValueError('Wrong pruning data/experiment status')
    review=read(recovery/'review.json');check(review['input_sha256'])
    small=read(recovery/'small_slice_audit.json')
    if small['candidate_freeze_sha256']!=file_sha256(recovery/'freeze.json') or small['preview_sha256']!=file_sha256(recovery/'newly_empty_review.png'):
        raise ValueError('Small-slice review changed')
    package=read(pruned/'verification.json');check(package['source_sha256'])
    check({str(pruned/'lung_pruned'/n):h for n,h in package['package_files_sha256'].items()})
    if package['commands']!=42 or len(package['nodes_checked'])!=21 or not package['performance_experiment_only'] or package['conservative_quality_review_passed']:
        raise ValueError('Wrong candidate coverage or misleading quality status')
    if file_sha256(pruned/'host_native/lung_cpu_native.dll')!=package['native_build']['binary_sha256']:
        raise ValueError('Different native validation library')
    rtl=read(pruned/'rtl_probes/verification.json');check(rtl['source_sha256']);check(rtl['vector_sha256'])
    if rtl['status']!='passed_actual_xsim_pruned_convolutions' or rtl['frames']!=72 or rtl['outputs']!=14050:
        raise ValueError('Missing actual pruned RTL check')
    if rtl['log_sha256']!=file_sha256(pruned/'rtl_probes/rtl_run.log'):
        raise ValueError('Changed pruned RTL log')
    if rtl['package_proof_sha256']!=file_sha256(pruned/'verification.json'):
        raise ValueError('Different package in RTL test')
    check(protected);output.mkdir()
    write_json(output/'verification.json',dict(status='complete_local_acceleration_experiments_board_acceptance_pending',
        accepted_flow=dict(fpga_ms=accepted['fpga_int8_ms'],int8_head_bit_exact=True),
        lut=dict(board_tested=False,routed_design=routed['routed_design'],utilization=routed['utilization'],
            package=str(lut/'lung_lut'),arithmetic_changed=False),
        pruning=dict(board_tested=False,package=str(pruned/'lung_pruned'),quality=review['reviews'][0],
            performance_experiment_only=True,commands=42,actual_rtl_frames=72,actual_rtl_output_codes=14050,
            test_or_covid_used=False,newly_empty_slices=small['newly_empty_slices'],recovered_empty_slices=small['recovered_empty_slices']),
        protected_original_files_unchanged=len(protected),python_tests=345,
        source_sha256={str(Path(__file__).resolve()):file_sha256(__file__)},evidence_sha256=evidence,
        existing_online_service_replaced=False,board_connection_made=False,
        note='No candidate board head or latency returned yet. Original model/artifacts preserved.'))
    print('Final local audit PASS:',output/'verification.json','protected files:',len(protected),flush=True)


if __name__=='__main__':run()
