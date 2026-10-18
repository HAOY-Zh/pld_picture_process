"""Check actual HWH reset wiring before reusing the frozen routed-board audit."""
import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import verify_lung_axil_board as board_audit


def check_reset(hardware):
    modules = {m.get('INSTANCE'): m for m in hardware.findall('.//MODULE')}
    reset = modules['proc_sys_reset_0']
    params = {p.get('NAME'): p.get('VALUE') for p in reset.findall('./PARAMETERS/PARAMETER')}
    if params['C_AUX_RESET_HIGH'] != '0' or params['C_EXT_RESET_HIGH'] != '0':
        raise ValueError('Expected explicit low-active resets')

    def connections(port):
        item = reset.find('./PORTS/PORT[@NAME="' + port + '"]')
        if item is None:
            raise ValueError('Reset port missing: ' + port)
        return {(c.get('INSTANCE'), c.get('PORT')) for c in item.findall('./CONNECTIONS/CONNECTION')}

    def constant(port, value):
        sources = connections(port)
        if len(sources) != 1:
            raise ValueError('Ambiguous reset driver: ' + port)
        instance, pin = next(iter(sources))
        module = modules[instance]
        declared = {p.get('NAME'): p.get('VALUE') for p in module.findall('./PARAMETERS/PARAMETER')}
        if module.get('MODTYPE') != 'xlconstant' or pin != 'dout' or int(declared['CONST_VAL'], 0) != value or int(declared['CONST_WIDTH'], 0) != 1:
            raise ValueError('Reset input tied to wrong constant: ' + port)

    constant('aux_reset_in', 1)
    constant('dcm_locked', 1)
    constant('mb_debug_sys_rst', 0)
    if connections('ext_reset_in') != {('processing_system7_0', 'FCLK_RESET0_N')}:
        raise ValueError('External reset disconnected from PS')
    targets = connections('peripheral_aresetn')
    expected = {('lung_conv8_axil_0', 'rst_n'), ('control_interconnect', 'aresetn'), ('memory_interconnect', 'aresetn')}
    if targets != expected:
        raise ValueError('Accelerator and interconnect reset not shared')
    return dict(auxiliary_active_low=True, auxiliary_constant=1, debug_constant=0,
                clock_locked_constant=1, external_from_ps=True, shared_reset=True)


def audit(root, core_root):
    root = Path(root)
    resets = check_reset(ET.parse(root/'deploy/lung_axil.hwh').getroot())
    workspace = Path(__file__).resolve().parents[1]
    original_hwh = workspace/'data/lung_axil_board_20261015/deploy/lung_axil.hwh'
    try:
        check_reset(ET.parse(original_hwh).getroot())
    except ValueError as exc:
        if 'wrong constant: aux_reset_in' not in str(exc):
            raise
    else:
        raise ValueError('Original stuck-reset fixture unexpectedly accepted')
    simulation = workspace/'data/lung_board_reset_sim_20261015/project/reset_check.sim/sim_1/behav/xsim/simulate.log'
    transcript = simulation.read_text(encoding='utf-8')
    marker = 'Board vendor reset RTL: PASS (original tie stays reset; corrected tie releases; external reset and lock recovery; 3 AXI-Lite ID reads)'
    if marker not in transcript or re.search(r'Fatal:|ERROR:', transcript):
        raise ValueError('Actual vendor reset/ID simulation proof missing')
    test_log = root/'python_tests.log'
    raw = test_log.read_bytes()
    tests = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')
    if not re.search(r'Ran 187 tests in .*?\n\s*\nOK\s*$', tests):
        raise ValueError('Full 187-test regression missing')
    # No new proof/package is emitted until the reset wiring itself passes.
    result = board_audit.audit_board(root, core_root)
    result['reset_wiring'] = resets
    result['reset_auditor_sha256'] = board_audit.core_audit.file_sha256(__file__)
    result['vendor_reset_and_id_simulated'] = True
    result['original_stuck_reset_hwh_rejected'] = True
    result['python_tests'] = 187
    evidence = [simulation, test_log, original_hwh, workspace/'fpga/tb_lung_board_reset.sv',
                workspace/'fpga/simulate_lung_board_reset.tcl', workspace/'python/tests/test_lung_board_reset.py',
                workspace/'python/tests/test_lung_axil_board_audit.py']
    result['reset_evidence_sha256'] = {str(p): board_audit.core_audit.file_sha256(p) for p in evidence}
    for target in [root/'verification.json', root/'deploy/package.json']:
        board_audit.core_audit.write_json(target, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board', required=True)
    parser.add_argument('--root', default='data/lung_axil_20261015')
    args = parser.parse_args()
    audit(args.board, args.root)
