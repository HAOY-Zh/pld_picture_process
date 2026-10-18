"""Audit AXI-Lite control and optional routed single-block smoke package."""
import argparse
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from lung_segmentation import write_json
from lung_unet import file_sha256
from lung_weight_reuse_vectors import read_bank


def check_hashes(hashes):
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError('Protected file changed: '+path)


def utilization(path):
    raw = path.read_text(encoding='utf-8'); result = {}
    for label in ['Slice LUTs*', 'Slice Registers', 'DSPs', 'Block RAM Tile']:
        match = re.search(r'\|\s*'+re.escape(label)+r'\s*\|\s*([\d.]+)', raw)
        if not match: raise ValueError('Missing resource: '+label)
        result[label] = float(match[1])
    for label, maximum in [('Slice LUTs*', 53200), ('Slice Registers', 106400), ('DSPs', 220), ('Block RAM Tile', 140)]:
        if result[label] > maximum: raise ValueError('Resource budget exceeded')
    return result


def timing(path):
    raw = path.read_text(encoding='utf-8')
    match = re.search(r'WNS\(ns\).*?\n\s*-+[^\n]*\n\s*([+-]?[\d.]+)', raw, re.S)
    if not match or float(match[1]) < 0: raise ValueError('Setup timing not met: '+str(path))
    return float(match[1])


def audit(root):
    workspace = Path(__file__).resolve().parents[1]; root = Path(root).resolve()
    if (root/'verification.json').exists(): raise ValueError('Audit exists, preserve evidence')
    previous_path = workspace/'data/lung_memory_axi_20261015/verification.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    if previous['status'] != 'passed_memory_axi_convolution' or not previous['independent_dense_hwc8']:
        raise ValueError('Prior independent write/math proof missing')
    tracked = {str(previous_path):file_sha256(previous_path), **previous['source_sha256'], **previous['evidence_sha256']}
    protected = json.loads((workspace/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    tracked.update(protected); check_hashes(tracked)
    logs = {name:(root/name).read_text(encoding='utf-8') for name in ['axil_control_run_v3.log', 'axil_accel_run_v2.log']}
    for name, log in logs.items():
        if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)', log, re.I): raise ValueError('Bad RTL log: '+name)
    match = re.search(r'Lung AXI-Lite control RTL: PASS \((\d+) register checks, (\d+) writes, (\d+) reads, (\d+) weights, (\d+) commands;', logs['axil_control_run_v3.log'])
    if not match or tuple(map(int, match.groups())) != (23, 173, 31, 130, 2): raise ValueError('Control protocol coverage missing')
    h = read_bank(workspace/'data/lung_cache_axi_20261015/vectors', 'header')
    match = re.search(r'Lung AXI-Lite accelerator RTL: PASS \((\d+) frames, (\d+) words, (\d+) weight packets, (\d+) memory read bursts, (\d+) memory read beats, (\d+) error cases;', logs['axil_accel_run_v2.log'])
    if not match or tuple(map(int, match.groups())) != (h[0], previous['written_words'], h[2], h[7], h[8], 5):
        raise ValueError('Register-only real engine integration incomplete')
    tests_bytes = (root/'python_tests_final.log').read_bytes()
    tests = tests_bytes.decode('utf-16' if tests_bytes.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')
    if not re.search(r'Ran 177 tests in .*?\n\s*\nOK\s*$', tests): raise ValueError('Full regression missing')
    syn = root/'synthesis_v5'
    resources = utilization(syn/'utilization.rpt'); wns = timing(syn/'timing_synth.rpt')
    exceptions = (syn/'configuration_exceptions.rpt').read_text(encoding='utf-8')
    if not all('cycles='+str(n) in exceptions for n in [2, 3, 4]) or 'input_words0' not in exceptions:
        raise ValueError('Scoped configuration and relocated count-register exceptions absent')
    if '40.000ns' not in (syn/'configuration_timing.rpt').read_text(encoding='utf-8') or '30.000ns' not in (syn/'configuration_derived_timing.rpt').read_text(encoding='utf-8'):
        raise ValueError('Wrong configuration setup periods')
    files = [workspace/'fpga'/name for name in ['lung_axil_control.v', 'lung_conv8_axil.v', 'lung_axil_config_timing.xdc',
        'tb_lung_axil_control.sv', 'tb_lung_axil_accel.sv', 'synth_lung_axil.tcl']]
    files += [workspace/'pynq'/name for name in ['lung_axil.py', 'lung_axil_smoke.py']]
    files += [workspace/'python'/name for name in ['tests/test_lung_axil.py', 'verify_lung_axil.py']]
    evidence = [root/name for name in logs] + [root/'python_tests_final.log'] + list(syn.glob('*.rpt')) + [syn/'axil_synth.dcp', syn/'checkpoint_source.txt']
    checkpoint = Path((syn/'checkpoint_source.txt').read_text(encoding='utf-8').strip())
    evidence.append(checkpoint)
    check_hashes(tracked)
    record = dict(status='passed_axil_single_block', frames=h[0], convolution_codes=previous['codes'], written_words=previous['written_words'],
        weight_packets=h[2], python_tests=177, utilization=resources, post_synth_wns_ns=wns,
        configuration_guard_stability_rtl_asserted=True, configuration_setup_cycles=4, relocated_configuration_dsp_cycles=3,
        stream_mac_requant_single_cycle=True, protected_medical_source_count=len(protected), medical_and_previous_sources_unchanged=True,
        source_sha256={str(p):file_sha256(p) for p in files}, evidence_sha256={str(p):file_sha256(p) for p in evidence},
        previous_proof_sha256=file_sha256(previous_path), axi_lite_rtl_simulated=True,
        full_network_rtl=False, routed_timing=False, bitstream_generated=False, fpga_deployed=False)
    write_json(root/'verification.json', record)
    print(json.dumps({k:v for k,v in record.items() if not k.endswith('sha256')}, indent=2), flush=True)
    return record


def audit_board(root, core_root):
    workspace = Path(__file__).resolve().parents[1]; root = Path(root).resolve(); core_root = Path(core_root).resolve()
    if (root/'verification.json').exists(): raise ValueError('Board audit exists')
    core_proof = json.loads((core_root/'verification.json').read_text(encoding='utf-8'))
    if core_proof['status'] != 'passed_axil_single_block': raise ValueError('Control proof missing')
    check_hashes(core_proof['source_sha256']); check_hashes(core_proof['evidence_sha256'])
    prep = json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    check_hashes({prep['source']:prep['source_sha256'], str(root/'ps_settings.tcl'):prep['ps_settings_sha256']})
    deploy = root/'deploy'; bit = deploy/'lung_axil.bit'; hwh = deploy/'lung_axil.hwh'
    if not bit.is_file() or bit.stat().st_size < 1000000 or not hwh.is_file(): raise ValueError('Validated deployment files missing')
    hardware = ET.parse(hwh).getroot()
    modules = hardware.findall('.//MODULE')
    accel = [m for m in modules if m.get('INSTANCE') == 'lung_conv8_axil_0']
    ps = [m for m in modules if m.get('INSTANCE') == 'processing_system7_0']
    if len(accel) != 1 or len(ps) != 1: raise ValueError('Unexpected HWH hierarchy')
    ps_params = {p.get('NAME'):p.get('VALUE') for p in ps[0].findall('.//PARAMETER')}
    if abs(float(ps_params['PCW_FPGA0_PERIPHERAL_FREQMHZ'])-100) > .01: raise ValueError('Wrong fabric clock')
    if int(ps_params['PCW_DDR_RAM_HIGHADDR'], 0) != 0x1FFFFFFF: raise ValueError('Wrong DDR map')
    ranges = hardware.findall('.//MEMRANGE')
    if not any(int(r.get('BASEVALUE', '0'), 0) == 0x43C00000 and 'lung_conv8_axil_0' in str(r.attrib) for r in ranges):
        raise ValueError('Control base not found in HWH')
    resources = utilization(root/'utilization_routed.rpt'); wns = timing(root/'timing_routed.rpt')
    routed = (root/'timing_routed.rpt').read_text(encoding='utf-8')
    match = re.search(r'WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([+-]?[\d.]+)', routed, re.S)
    # The min/max report table normally puts WNS/TNS/WHS in the same row.
    summary = re.search(r'WNS\(ns\).*?WHS\(ns\).*?\n\s*-+[^\n]*\n\s*([^\n]+)', routed, re.S)
    if not summary: raise ValueError('Missing routed hold summary')
    hold = float(summary[1].split()[4])
    if hold < 0: raise ValueError('Routed hold timing failed')
    drc = (root/'drc.rpt').read_text(encoding='utf-8')
    if re.search(r'\|[^\n]*\b(?:Error|Critical Warning)\b', drc): raise ValueError('DRC error/critical warning')
    # Check imported IP source matches the actual RTL whose simulation passed.
    imported = {}
    for current in (workspace/'fpga').glob('lung_*.v'):
        copies = list((root/'ip_repo').rglob(current.name))
        for copy in copies:
            if file_sha256(copy) != file_sha256(current): raise ValueError('Packaged RTL differs: '+str(copy))
            imported[str(copy)] = file_sha256(copy)
    if len(imported) < 13: raise ValueError('Packaged source coverage incomplete')
    for name in ['lung_axil.py', 'lung_axil_smoke.py']:
        target = deploy/name
        if target.exists(): raise ValueError('Do not overwrite deploy scripts')
        shutil.copy2(workspace/'pynq'/name, target)
    files = list(deploy.glob('*')) + [root/name for name in ['preparation.json', 'ps_settings.tcl', 'utilization_routed.rpt', 'timing_routed.rpt', 'configuration_exceptions.rpt', 'drc.rpt']]
    record = dict(status='built_routed_single_block_smoke_not_board_tested', core_proof_sha256=file_sha256(core_root/'verification.json'),
        utilization=resources, routed_wns_ns=wns, routed_whs_ns=hold, clock_mhz=100, control_base='0x43c00000', ddr_bytes=512*1024*1024,
        packaged_sources_sha256=imported, deploy_sha256={p.name:file_sha256(p) for p in deploy.glob('*')},
        evidence_sha256={str(p):file_sha256(p) for p in files},
        builder_sha256=file_sha256(workspace/'fpga/build_lung_axil_board.tcl'), preparation_source_unchanged=True,
        full_network_rtl=False, routed_timing=True, bitstream_generated=True, fpga_deployed=False,
        warning='New single-block engineering smoke overlay only. Physical board results are still required.')
    write_json(root/'verification.json', record); write_json(deploy/'package.json', record)
    print(json.dumps({k:v for k,v in record.items() if not k.endswith('sha256')}, indent=2), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/lung_axil_20261015')
    parser.add_argument('--board')
    args = parser.parse_args()
    audit_board(args.board, args.root) if args.board else audit(args.root)
