"""Independently audit frozen-code HWC8 writes, exact RTL PASS and Z7020 OOC."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from lung_segmentation import write_json
from lung_unet import file_sha256
from lung_weight_reuse_vectors import read_bank


def audit(root):
    root = Path(root).resolve()
    workspace = Path(__file__).resolve().parents[1]
    destination = root/'verification.json'
    if destination.exists():
        raise ValueError('Audit exists; preserve previous evidence')
    manifest = root/'vectors/vectors.json'
    info = json.loads(manifest.read_text(encoding='utf-8'))
    source = Path(info['source'])
    tracked = {str(manifest):file_sha256(manifest), **info['source_sha256']}
    if file_sha256(workspace/'python/lung_write_vectors.py') != info['generated_by_sha256']:
        raise ValueError('Write generator changed')
    tracked.update({str(root/'vectors'/name):digest for name, digest in info['outputs_sha256'].items()})
    previous_path = workspace/'data/lung_cache_axi_20261015/verification.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    if previous['status'] != 'passed_cached_axi_read_convolution' or not previous['previous_independent_math_banks_unchanged']:
        raise ValueError('Prior math/cache proof missing')
    tracked[str(previous_path)] = file_sha256(previous_path)
    tracked.update(previous['source_sha256'])
    tracked.update(previous['evidence_sha256'])
    protected = json.loads((workspace/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    tracked.update(protected)
    for path, digest in tracked.items():
        if file_sha256(path) != digest:
            raise ValueError('Protected source/evidence changed: ' + path)

    h = read_bank(source, 'header')
    frames = read_bank(source, 'frames')
    codes, lanes, coords, lasts = [read_bank(source, n) for n in ['codes', 'lanes', 'coords', 'lasts']]
    desc = read_bank(root/'vectors', 'descriptors')
    addresses = read_bank(root/'vectors', 'addresses')
    words = read_bank(root/'vectors', 'words')
    wh = read_bank(root/'vectors', 'header')
    if wh != [h[0], h[4], info['words']] or len(desc) != h[0]*7 or len(addresses) != wh[2] or len(words) != wh[2]:
        raise ValueError('Invalid write fixture counts')
    code_cursor = word_cursor = 0
    # This second implementation uses dense pixel byte arrays and flattened
    # coordinate indexing, not the generator's nested-loop packer.
    for f in range(h[0]):
        row = frames[f*16:f*16+16]
        base, oh, ow, blocks, block, offset, count = desc[f*7:f*7+7]
        if (oh, ow) != (row[1]+2*row[5]-row[4]+1, row[2]+2*row[5]-row[4]+1):
            raise ValueError('Output geometry differs from frozen convolution')
        if not (1 <= oh <= 512 and 1 <= ow <= 512 and 1 <= blocks <= 12 and 0 <= block < blocks
                and base >= h[1] and base+oh*ow*blocks <= 2**29 and offset == word_cursor
                and count == oh*ow and row[13] == code_cursor):
            raise ValueError('Unsafe/overlapping output allocation or descriptor offsets')
        active = [i for i in range(8) if row[8] >> i & 1]
        n = count*len(active)
        dense = np.full((count, 8), row[7], dtype=np.uint8)
        touched = set()
        for i in range(code_cursor, code_cursor+n):
            position = coords[i]
            x, y, lane = position & 1023, position >> 10 & 1023, lanes[i]
            if lane not in active or x >= ow or y >= oh or (y*ow+x, lane) in touched:
                raise ValueError('Invalid/duplicate golden output coordinate')
            if bool(lasts[i]) != (lane == active[-1]) or bool(position >> 20) != (lane == active[-1] and x == ow-1 and y == oh-1):
                raise ValueError('Frozen output markers inconsistent')
            touched.add((y*ow+x, lane));dense[y*ow+x, lane] = codes[i]
        expected_data = np.ascontiguousarray(dense).reshape(-1).tobytes()
        actual_data = b''.join(word.to_bytes(8, 'little') for word in words[offset:offset+count])
        expected_addresses = (base + np.arange(count, dtype=np.int64)*blocks + block).tolist()
        if expected_data != actual_data or addresses[offset:offset+count] != expected_addresses:
            raise ValueError('Independent dense HWC8 reconstruction differs')
        if len(touched) != n:
            raise ValueError('Incomplete channel coverage')
        code_cursor += n;word_cursor += count
    if (code_cursor, word_cursor) != (h[4], wh[2]):
        raise ValueError('Incomplete independent write coverage')

    logs = {name:(root/name).read_text(encoding='utf-8') for name in
        ['write_protocol_run.log', 'output_writer_run.log', 'memory_axi_run_v2.log']}
    for name, log in logs.items():
        if re.search(r'Fatal:|ERROR:|WARNING:.*(?:readmem|file|data)', log, re.I):
            raise ValueError('Invalid RTL log: ' + name)
    marker = re.search(r'Lung memory AXI convolution RTL: PASS \((\d+) frames, (\d+) codes, (\d+) writes, (\d+) read bursts, (\d+) read beats, (\d+) rejection/error cases;', logs['memory_axi_run_v2.log'])
    expected = (h[0], h[4], wh[2], h[7], h[8], 14)
    if not marker or tuple(map(int, marker.groups())) != expected or not all(expected):
        raise ValueError('Missing exact nonempty memory integration PASS')
    for name, marker in [('write_protocol_run.log', 'Lung AXI write64 protocol RTL: PASS (10 cases;'),
                         ('output_writer_run.log', 'Lung output writer RTL: PASS (23 cases, 1049 writes;')]:
        if marker not in logs[name]:
            raise ValueError('Missing protocol proof: ' + name)
    test_bytes = (root/'python_tests.log').read_bytes()
    tests = test_bytes.decode('utf-16' if test_bytes.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')
    if not re.search(r'Ran 167 tests in .*?\n\s*\nOK\s*$', tests) or re.search(r'^FAILED|^ERROR:', tests, re.M):
        raise ValueError('Python regression incomplete')
    util = (root/'synthesis/utilization.rpt').read_text(encoding='utf-8')
    resources = {}
    for label in ['Slice LUTs*', 'Slice Registers', 'DSPs', 'Block RAM Tile', 'RAMB36/FIFO*', 'RAMB18']:
        match = re.search(r'\|\s*'+re.escape(label)+r'\s*\|\s*([\d.]+)', util)
        if not match:
            raise ValueError('Missing resource: ' + label)
        resources[label] = float(match[1])
    if resources['Block RAM Tile'] != 31.5 or resources['DSPs'] > 80 or resources['Slice LUTs*'] > 53200 or resources['Slice Registers'] > 106400:
        raise ValueError('Unexpected resource mapping/budget')
    timing = (root/'synthesis/timing_synth.rpt').read_text(encoding='utf-8')
    match = re.search(r'WNS\(ns\).*?\n\s*-+[^\n]*\n\s*([+-]?[\d.]+)', timing, re.S)
    if not match or float(match[1]) < 0 or not re.search(r'core_clk\s+.*?10\.000', timing):
        raise ValueError('Internal OOC timing target not met')
    source_files = [workspace/'fpga'/n for n in ['lung_axi_write64.v', 'lung_output_writer.v', 'lung_conv8_memory_axi.v',
        'tb_lung_write64.sv', 'tb_lung_output_writer.sv', 'tb_lung_memory_axi.sv', 'synth_lung_memory_axi.tcl']]
    source_files += [workspace/'python'/n for n in ['lung_write_vectors.py', 'verify_lung_memory_axi.py', 'tests/test_lung_write_vectors.py']]
    evidence = [root/name for name in logs] + [root/'python_tests.log'] + [root/'synthesis'/name for name in
        ['utilization.rpt', 'utilization_hier.rpt', 'timing_synth.rpt', 'memory_axi_synth.dcp']]
    for path, digest in tracked.items():
        if file_sha256(path) != digest:
            raise ValueError('Protected source changed during audit: ' + path)
    report = dict(status='passed_memory_axi_convolution', frames=h[0], codes=h[4], written_words=wh[2],
        write_payload_bytes=wh[2]*8, read_bursts=h[7], read_beats=h[8], independent_dense_hwc8=True,
        original_convolution_math_unchanged=True, delayed_final_b_and_canaries=True, protocol_errors_and_shared_reset=True,
        python_tests=167, utilization=resources, post_synth_wns_ns=float(match[1]),
        internal_post_synth_100mhz_timing_met=True, protected_medical_source_count=len(protected),
        medical_and_previous_sources_unchanged=True, previous_proof_sha256=file_sha256(previous_path),
        source_sha256={str(p):file_sha256(p) for p in source_files},
        evidence_sha256={str(p):file_sha256(p) for p in evidence}, axi_read_write_rtl_simulated=True,
        axi_lite_control=False, full_network_rtl=False, real_ddr_measured=False, routed_timing=False,
        bitstream_generated=False, fpga_deployed=False)
    write_json(destination, report)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/lung_memory_axi_20261015')
    audit(parser.parse_args().root)
