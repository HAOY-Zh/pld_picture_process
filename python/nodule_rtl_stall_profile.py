"""Observe deployed exact-LUT RTL without changing its functional logic.

Replay the previously accepted real 128x128 patch through all 11 nodes. Counts
are simulation diagnostics under the existing AXI stall model, not board time.
Several signals overlap: their counts must never be added as a time breakdown.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pynq'))
from nodule_plan import sha


METRICS = {
    'active': '1',
    'mac_input_beats': '`M.accepted',
    'mac_flushing': '`M.flushing',
    'mac_emitting': '`M.emitting',
    'window_to_core_blocked': '`W.m_valid && !`W.m_ready',
    'window_read_response_wait': '`W.state == 3 && !`W.rsp_valid',
    'window_read_request_blocked': '`W.state == 2 && !`W.outside && !`W.req_ready',
    'window_issue': '`W.state == 2',
    'window_response': '`W.state == 3',
    'window_held_output': '`W.state == 4',
    'cache_refill': '`C.state == 2 || `C.state == 3',
    'cache_hit_response': '`C.direct_hit',
    'output_backpressure': '`O.s_valid && !`O.s_ready',
    'writer_wait_response': '`O.state == 3',
    'writer_send': '`O.state == 2',
}


def prepare(out):
    old = ROOT / 'data/nodule_board_20261023_v1/rtl_sim'
    accepted = json.loads((old / 'verification.json').read_text())
    for name, digest in accepted['vectors_sha256'].items():
        if sha(old / 'vectors' / name) != digest:
            raise ValueError('Changed accepted patch vector: ' + name)
    for name, digest in accepted['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Changed accepted original bench/source: ' + name)
    lut = ROOT / 'data/nodule_direct_lut_20261023_v1'
    proof = json.loads((lut / 'rtl_sim/verification.json').read_text())
    if proof['status'] != 'actual_xsim_nodule_448_exact_lut_pass':
        raise ValueError('Missing deployed LUT RTL proof')
    for name, digest in proof['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Changed accepted exact-LUT RTL/bench: ' + name)
    out.mkdir()  # Preserve earlier results; require a fresh directory.
    shutil.copytree(old / 'vectors', out / 'vectors')
    shutil.copytree(lut / 'rtl', out / 'rtl')
    for name in ['nodule_sim_engine.sv', 'run.tcl']:
        shutil.copy2(old / name, out / name)
    text = (old / 'tb_nodule_network.sv').read_text()
    definitions = '''
    `define E conv.con.dut.engine
    `define M conv.con.dut.engine.conv.conv.core.tile
    `define W conv.con.dut.engine.conv.conv.window
    `define C conv.con.dut.engine.conv.cache
    `define O conv.con.dut.engine.output_writer
'''
    definitions += ''.join('    integer prof_' + n + '=0;\n' for n in METRICS)
    definitions += '    task clear_profile; begin\n'
    definitions += ''.join('        prof_' + n + '=0;\n' for n in METRICS)
    definitions += '    end endtask\n'
    definitions += '    always @(posedge clk) if(rst_n && `E.busy) begin\n'
    definitions += ''.join('        if (' + expr + ') prof_' + n + '=prof_' + n + '+1;\n'
                           for n, expr in METRICS.items())
    definitions += '    end\n'
    definitions += '    task show_profile; begin\n'
    definitions += '        $display("RTL_PROFILE command=%0d ' + ' '.join(
        n + '=%0d' for n in METRICS) + '",f,' + ','.join('prof_' + n for n in METRICS) + ');\n'
    definitions += '    end endtask\n'
    text = text.replace('    initial begin\n', definitions + '    initial begin\n', 1)
    marker = 'conv.put(4,1);conv.finish_command();conv.host.read_reg(48,0,value);'
    if text.count(marker) != 1:
        raise ValueError('Unexpected accepted bench structure')
    text = text.replace(marker, 'clear_profile();' + marker + 'show_profile();')
    (out / 'tb_nodule_network.sv').write_text(text)
    for name in ['tb_lung_axil_control.sv', 'tb_lung_burst_write_memory.sv']:
        shutil.copy2(ROOT / 'fpga' / name, out / name)
    sources = list((out / 'rtl').glob('*.v')) + list(out.glob('*.sv'))
    identity = dict(source_sha256={str(p): sha(p) for p in sources},
                    vectors_sha256={p.name: sha(p) for p in (out / 'vectors').iterdir()},
                    baseline_proof_sha256=sha(old / 'verification.json'),
                    deployed_lut_proof_sha256=sha(lut / 'rtl_sim/verification.json'))
    (out / 'inputs.json').write_text(json.dumps(identity, indent=2))
    return sources, identity


def run(out, binary):
    out = Path(out).resolve()
    sources, identity = prepare(out)
    binary = Path(binary)
    tick = time.perf_counter()
    calls = [([str(binary / 'xvlog.bat'), '--sv', *map(str, sources)], 'compile'),
             ([str(binary / 'xelab.bat'), 'tb_nodule_network', '-s', 'stall_profile',
               '--debug', 'typical', '--mt', '2'], 'elaborate'),
             ([str(binary / 'xsim.bat'), 'stall_profile', '-tclbatch', 'run.tcl',
               '-log', 'rtl_run.log'], 'simulate')]
    for cmd, name in calls:
        print('Read-only RTL profiling:', name, flush=True)
        with (out / (name + '.log')).open('x') as stream:
            p = subprocess.run(cmd, cwd=out, stdout=stream, stderr=subprocess.STDOUT)
        if p.returncode:
            raise RuntimeError('RTL profiling failed: ' + str(out / (name + '.log')))
    log = (out / 'rtl_run.log').read_text()
    if 'Nodule continuous network RTL: PASS' not in log or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)', log, re.I):
        raise ValueError('Full-graph exactness or protocol check failed')
    commands = re.findall(r'NODULE_COMMAND_PASS command=(\d+) kind=(\d+) cycles=(\d+)', log)
    if len(commands) != 15 or [int(x[0]) for x in commands] != list(range(15)):
        raise ValueError('Incomplete all-node replay')
    names = ['stem0', 'stem1', 'pool1', 'enc1_0', 'enc1_1', 'pool2',
             'enc2_0', 'enc2_1', 'enc2_2', 'cat1', 'dec1_0', 'dec1_1', 'cat0', 'dec0', 'head']
    rows = []
    for line in re.findall(r'^RTL_PROFILE (.*)$', log, re.M):
        row = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
        row['name'] = names[row['command']]
        row['command_counter_cycles'] = int(commands[row['command']][2])
        if abs(row['active'] - row['command_counter_cycles']) > 5:
            raise ValueError('Profiler disagrees with RTL command counter')
        rows.append(row)
    if [r['command'] for r in rows] != [i for i, kind, _ in map(lambda x: tuple(map(int, x)), commands) if kind == 0]:
        raise ValueError('Incomplete convolution profiling')
    for path, digest in identity['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed simulation source during profiling')
    result = dict(status='actual_xsim_exact_lut_stall_profile_pass', board_tested=False,
                  scope='One accepted real 128x128 patch, all 11 integer nodes and DDR guards; existing synthetic AXI stalls',
                  counts_overlap=True, no_functional_rtl_changes=True,
                  elapsed_sec=time.perf_counter() - tick, convolutions=rows,
                  all_commands=commands, log_sha256=sha(out / 'rtl_run.log'), **identity)
    (out / 'profile.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--vivado-bin', default='E:/vivado2/Vivado/2022.2/bin')
    args = parser.parse_args()
    run(args.out, args.vivado_bin)
