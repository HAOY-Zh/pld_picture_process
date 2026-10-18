"""Isolated exact-math window bypass and pool row-cache experiment."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from lung_segmentation import write_json
from lung_unet import file_sha256
from lung_word_prepare import replace_once
from verify_lung_axil import check_hashes
from lung_cache_fast_prepare import RTL, BENCH, TOPS, MARKERS, cycle_rows

WORKSPACE = Path(__file__).resolve().parents[1]
BASE = WORKSPACE / 'data/lung_cache_return_20261016_v2'
TENSOR = WORKSPACE / 'data/lung_tensor_burst_20261016_v3'
PACKAGE = WORKSPACE / 'data/lung_threeway_20261016/lung_threeway'
ROOT = WORKSPACE / 'data/lung_flow_20261017'
BIN = Path('E:/vivado2/Vivado/2022.2/bin')


def window_source(text):
    text = replace_once(text, 'module lung_feature_window_step', 'module lung_feature_window_flow')
    text = replace_once(text, 'output reg[63:0]m_data', 'output wire[63:0]m_data')
    text = replace_once(text, 'output reg m_error', 'output wire m_error')
    text = replace_once(text, '    reg[2:0]state;',
        '    reg[2:0]state;\n    reg[63:0]held_data;reg held_error;')
    text = replace_once(text, '    assign m_valid=rst_n&&state==OUTPUT;', '''    // Consume a synchronous response directly when the MAC is ready. The
    // response is always accepted into the holding register if it is stalled.
    wire response_output=state==RESPONSE&&rsp_valid;
    assign m_valid=rst_n&&(state==OUTPUT||response_output);
    assign m_data=response_output?(rsp_error?{8{input_zero}}:rsp_data):held_data;
    assign m_error=response_output?rsp_error:held_error;''')
    text = text.replace('m_data<=', 'held_data<=').replace('m_error<=', 'held_error<=')
    text = replace_once(text, 'error<=error||rsp_error;state<=OUTPUT;',
        'error<=error||rsp_error;\n                if(m_ready)advance_window();else state<=OUTPUT;')
    return text


def tensor_source(text):
    # These are cache-selection decisions only. Source geometry, source base,
    # clipping and branch quantization must continue using the actual branch_b.
    text = replace_once(text, '    wire line_hit=branch_b?', '''    // In pool mode both banks describe A, so either matching bank may hit.
    // Prefer the row-parity bank only on a miss. This avoids duplicate fills
    // when two short rows share one aligned line.
    wire pool_hit_a=valid_a&&tag_a==address[31:4];
    wire pool_hit_b=valid_b&&tag_b==address[31:4];
    wire cache_bank_b=op==0?(pool_hit_a?1'b0:(pool_hit_b?1'b1:slot[1])):branch_b;
    wire line_hit=cache_bank_b?''')
    for old, new in [
        ('fetched_word<=branch_b?line_b', 'fetched_word<=cache_bank_b?line_b'),
        ('if(branch_b)valid_b<=0;else valid_a<=0;', 'if(cache_bank_b)valid_b<=0;else valid_a<=0;'),
        ('if(branch_b)line_b[', 'if(cache_bank_b)line_b['),
        ('if(branch_b)begin valid_b<=1;tag_b<=address[31:4];end',
         'if(cache_bank_b)begin valid_b<=1;tag_b<=address[31:4];end'),
    ]:
        text = replace_once(text, old, new)
    return text


def traffic(op, h, w, ca, cb, abase, bbase, dst, row_cache):
    """Independent address oracle. Counts physical prefetch, not just ABI READS."""
    oh, ow = (h//2, w//2) if op == 0 else ((h*2, w*2) if op in (1, 3) else (h, w))
    blocks = (ca+cb+7)//8 if op >= 2 else (ca+7)//8
    bounds = [(abase, abase+h*w*((ca+7)//8)), (bbase, bbase+oh*ow*((cb+7)//8))]
    tags = [None, None]
    last = None
    logical = bursts = beats = 0

    def read(branch, y, x, channel, bank):
        nonlocal last, logical, bursts, beats
        width, channels = (ow, cb) if branch else (w, ca)
        address = bounds[branch][0]+(y*width+x)*((channels+7)//8)+channel//8
        if op >= 2 and last == address:
            return
        last = address
        logical += 1
        tag = address//16
        if op == 0 and row_cache:
            if tag in tags:
                return
        if tags[bank] != tag:
            tags[bank] = tag
            bursts += 1
            beats += min(tag*16+16, bounds[branch][1])-max(tag*16, bounds[branch][0])

    for y in range(oh):
        for x in range(ow):
            for block in range(blocks):
                if op == 0:
                    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        read(0, 2*y+dy, 2*x+dx, block*8, dy if row_cache else 0)
                elif op == 1:
                    read(0, y//2, x//2, block*8, 0)
                else:
                    channels = [block*8] if ca % 8 == 0 else range(block*8, min(block*8+8, ca+cb))
                    for c in channels:
                        branch = int(c >= ca)
                        read(branch, y//2 if op == 3 and not branch else y,
                             x//2 if op == 3 and not branch else x, c-ca if branch else c, branch)
    writes, remaining = 0, oh*ow*blocks
    while remaining:
        n = min(16, 512-dst % 512, remaining)
        dst += n
        remaining -= n
        writes += 1
    return logical, bursts, beats, writes


def baseline():
    p = json.loads((BASE/'verification.json').read_text(encoding='utf-8'))
    for k in ['source_sha256', 'packaged_source_sha256', 'evidence_sha256']:
        check_hashes(p[k])
    proof = WORKSPACE/'data/lung_threeway_20261016/lung_threeway_result_threeway_verification.json'
    v = json.loads(proof.read_text(encoding='utf-8'))
    if v['status'] != 'verified_returned_threeway_actual_fp32_and_int8_heads':
        raise ValueError('Need accepted board baseline')
    protected = json.loads((WORKSPACE/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text())['protocol']['input_sha256']
    check_hashes(protected)
    return p, protected


def prepare(root):
    root = Path(root).resolve()
    if root.parent != WORKSPACE/'data' or root.exists():
        raise ValueError('Use a fresh direct workspace/data child')
    p, protected = baseline()
    root.mkdir()
    for name in ['rtl', 'baseline_rtl', 'benches', 'vectors']:
        (root/name).mkdir()
    for name, digest in p['packaged_source_sha256'].items():
        src = Path(name)
        dst = root/'baseline_rtl'/src.name
        if dst.exists() and file_sha256(dst) != digest:
            raise ValueError('Conflicting IP dependency')
        shutil.copy2(src, dst)
    for src in (root/'baseline_rtl').iterdir():
        shutil.copy2(src, root/'rtl'/src.name)
    flow = (WORKSPACE/'fpga/lung_feature_window_flow.v').read_text(encoding='utf-8')
    (root/'rtl/lung_feature_window.v').write_text(replace_once(flow,
        'module lung_feature_window_flow', 'module lung_feature_window'), encoding='utf-8')
    (root/'rtl/lung_tensor_burst_engine.v').write_text(tensor_source(
        (root/'baseline_rtl/lung_tensor_burst_engine.v').read_text(encoding='utf-8')), encoding='utf-8')
    for name in BENCH+['tb_lung_window_step.sv', 'tb_lung_cache_return.sv']:
        shutil.copy2(BASE/'benches'/name, root/'benches'/name)
    (root/'benches/lung_feature_window_baseline.v').write_text(replace_once(
        (root/'baseline_rtl/lung_feature_window.v').read_text(encoding='utf-8'),
        'module lung_feature_window(', 'module lung_feature_window_baseline('), encoding='utf-8')
    shutil.copy2(TENSOR/'tb_lung_tensor_burst_axil.sv', root/'tb_lung_tensor_burst_axil.sv')
    for src in (TENSOR/'vectors').iterdir():
        shutil.copy2(src, root/'vectors'/src.name)
    values = [int(s, 16) for s in (root/'vectors/commands.mem').read_text().split()]
    shifts = [0, 1, 509, 511]
    traffic_rows = {}
    for variant in ['baseline', 'flow']:
        rows = [traffic(*values[f*18:f*18+5], 4096+shifts[f % 4],
                        1048576+shifts[(f+1) % 4], 2097152+shifts[(f+2) % 4], variant == 'flow') for f in range(35)]
        traffic_rows[variant] = rows
    # Never regenerate gold/model/reference. Only traffic expectations change.
    for i, name in enumerate(['logical_reads', 'read_bursts', 'physical_reads', 'write_bursts']):
        actual = [int(s, 16) for s in (TENSOR/'vectors'/(name+'.mem')).read_text().split()]
        if actual != [r[i] for r in traffic_rows['baseline']]:
            raise ValueError('Independent traffic oracle disagrees with accepted baseline')
    builder = (BASE/'build_return.tcl').read_text(encoding='utf-8')
    start = builder.index('foreach f $rtl_names {')
    end = builder.index('\nset_property top', start)
    builder = builder[:start]+'foreach f $rtl_names { add_files -norecurse [file join $task_out rtl $f] }'+builder[end:]
    (root/'build_flow.tcl').write_text(builder, encoding='utf-8')
    shutil.copy2(BASE/'ps_settings.tcl', root/'ps_settings.tcl')
    (root/'run.tcl').write_text('run all\nexit\n', encoding='utf-8')
    sources = [Path(__file__).resolve(), WORKSPACE/'fpga/lung_feature_window_flow.v',
               WORKSPACE/'fpga/tb_lung_window_flow.sv', WORKSPACE/'python/lung_flow_deploy.py',
               WORKSPACE/'python/tests/test_lung_flow.py', WORKSPACE/'python/lung_tensor_burst_faults.py',
               WORKSPACE/'fpga/validate_lung_window_step.tcl', WORKSPACE/'fpga/lung_axil_config_timing.xdc']
    frozen = json.loads((BASE/'preparation.json').read_text())['frozen_vectors_sha256']
    for name in ['header.mem', 'commands.mem', 'a.mem', 'b.mem', 'gold.mem']:
        frozen[str(TENSOR/'vectors'/name)] = file_sha256(TENSOR/'vectors'/name)
    shutil.copy2(WORKSPACE/'fpga/tb_lung_window_flow.sv', root/'benches/tb_lung_window_flow.sv')
    write_json(root/'preparation.json', dict(status='prepared_flow_exact_math', model_changed=False,
        arithmetic_changed=False, baseline_proof_sha256=file_sha256(BASE/'verification.json'),
        protected_sources_sha256=protected, baseline_package_sha256={str(f):file_sha256(f) for f in PACKAGE.iterdir() if f.is_file()},
        source_sha256={str(s):file_sha256(s) for s in sources}, frozen_vectors_sha256=frozen,
        independent_traffic=traffic_rows,
        generated_sha256={str(f.relative_to(root)):file_sha256(f) for f in root.rglob('*') if f.is_file()}))
    print('Prepared isolated flow optimization:', root, flush=True)


def checked(root):
    root = Path(root).resolve()
    p = json.loads((root/'preparation.json').read_text(encoding='utf-8'))
    for k in ['source_sha256', 'frozen_vectors_sha256', 'protected_sources_sha256', 'baseline_package_sha256']:
        check_hashes(p[k])
    check_hashes({str(root/n):h for n,h in p['generated_sha256'].items()})
    return p


def simulate(root, variant):
    root = Path(root).resolve()
    p = checked(root)
    folder = root/('sim_'+variant)
    folder.mkdir()
    shutil.copy2(root/'run.tcl', folder/'run.tcl')
    rtl = root/('rtl' if variant == 'flow' else 'baseline_rtl')
    conv_files = [rtl/n for n in RTL]
    conv_files += [WORKSPACE/'fpga'/n for n in ['tb_lung_write64.sv', 'tb_lung_axil_control.sv']]
    conv_files += list((root/'benches').glob('*.sv'))
    conv_files += [root/'benches/lung_feature_window_baseline.v', WORKSPACE/'fpga/lung_feature_window_flow.v']
    tensor_files = [rtl/n for n in ['lung_tensor_burst_engine.v', 'lung_tensor_axil.v', 'lung_requant.v',
                    'lung_axi_read64.v', 'lung_axi_burst_write64.v', 'lung_tensor_control.v']]
    tensor_files += [WORKSPACE/'fpga/tb_lung_burst_write_memory.sv', root/'tb_lung_tensor_burst_axil.sv']
    shutil.copytree(root/'vectors', folder/'vectors')
    for i, name in enumerate(['logical_reads', 'read_bursts', 'physical_reads', 'write_bursts']):
        (folder/'vectors'/(name+'.mem')).write_text(''.join(f'{r[i]:08x}\n' for r in p['independent_traffic'][variant]), encoding='utf-8')

    def call(command, name):
        with (folder/(name+'.log')).open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('RTL tool failed: '+name)

    start = time.perf_counter()
    call([str(BIN/'xvlog.bat'), '--sv', *map(str, conv_files+tensor_files)], 'compile')
    names = TOPS+['tb_lung_window_step', 'tb_lung_cache_return', 'tb_lung_window_flow', 'tb_lung_tensor_burst_axil']
    markers = MARKERS+['Lung stepped window RTL: PASS', 'Lung direct cache return RTL: PASS',
        'Lung response bypass RTL: PASS', 'Lung tensor AXI-Lite RTL: PASS (35 frames, 379310 words,']
    results = {}
    for top, marker in zip(names, markers):
        print(variant+': Actual XSim '+top, flush=True)
        call([str(BIN/'xelab.bat'), top, '-s', top+'_snapshot', '--debug', 'typical'], top+'_elaborate')
        call([str(BIN/'xsim.bat'), top+'_snapshot', '-tclbatch', 'run.tcl', '-log', top+'_run.log'], top+'_tool')
        log = folder/(top+'_run.log')
        text = log.read_text(encoding='utf-8')
        if marker not in text or re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)', text, re.I):
            raise ValueError('Failed RTL: '+str(log))
        row = dict(log_sha256=file_sha256(log))
        if top in ['tb_lung_cached_axi', 'tb_lung_axil_accel']:
            row['cycles'] = cycle_rows(text)
        if top == 'tb_lung_tensor_burst_axil':
            row['cycles'] = [dict(zip(['frame','op','ca','cycles','words'], map(int, m))) for m in re.findall(
                r'BURST_CYCLES frame=(\d+) op=(\d+) ca=(\d+) cycles=(\d+) words=(\d+)', text)]
            if len(row['cycles']) != 35 or '26 error cases;' not in text:
                raise ValueError('Incomplete tensor coverage')
        results[top] = row
    write_json(folder/'verification.json', dict(status='passed_actual_xsim_flow_'+variant,
        results=results, source_sha256=p['source_sha256'], wall_seconds=time.perf_counter()-start))
    print(variant+': All ten actual RTL checks PASS', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--simulate', choices=['baseline', 'flow'])
    a = ap.parse_args()
    simulate(a.root, a.simulate) if a.simulate else prepare(a.root)
