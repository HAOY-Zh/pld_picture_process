"""Real-board single-block smoke test; synthetic integer inputs, no CT/training.

Run only after installing the matching new bit/hwh and lung_axil.py.
All reference calculations here are CPU checks, NOT FPGA inference results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from lung_axil import LungAxil


def feature_words(q, zero):
    channels, height, width = q.shape
    data = np.full((height, width, (channels+7)//8, 8), zero, np.uint8)
    for c in range(channels):
        data[:, :, c//8, c%8] = q[c]
    return data.reshape(-1, 8).copy().view('<u8').reshape(-1)


def weight_packets(weights, output_block):
    cout, cin, kernel, _ = weights.shape
    result = []
    for ky in range(kernel):
        for kx in range(kernel):
            for block in range((cin+7)//8):
                packet = 0
                for o in range(8):
                    for i in range(8):
                        oc, ic = output_block*8+o, block*8+i
                        if oc < cout and ic < cin:
                            packet |= (int(weights[oc, ic, ky, kx]) & 255) << (8*(o*8+i))
                result.append(packet)
    return result


def reference(q, weights, bias, iz, oz, multiplier=1 << 30, shift=31, padding=1, relu=False):
    centered = q.astype(np.int64)-iz
    if padding:
        centered = np.pad(centered, ((0, 0), (padding, padding), (padding, padding)))
    kernel = weights.shape[-1]
    windows = np.lib.stride_tricks.sliding_window_view(centered, (kernel, kernel), axis=(1, 2))
    accumulators = np.einsum('cyxij,ocij->oyx', windows, weights.astype(np.int64), dtype=np.int64, optimize=True)
    accumulators += np.asarray(bias, np.int64)[:, None, None]
    codes = []
    for value in accumulators.reshape(-1):
        product = int(value)*multiplier
        quotient, remainder = divmod(abs(product), 1 << shift)
        quotient += int(2*remainder > (1 << shift) or (2*remainder == (1 << shift) and quotient & 1))
        rounded = (-quotient if product < 0 else quotient)+oz
        codes.append(max(oz if relu else 0, min(255, rounded)))
    return np.asarray(codes, np.uint8).reshape(accumulators.shape)


def run(args):
    from pynq import Overlay, allocate
    bit = Path(args.bit).resolve(); hwh = bit.with_suffix('.hwh')
    if not bit.is_file() or not hwh.is_file():
        raise FileNotFoundError('Need same-basename matching new bit and hwh')
    if args.report and Path(args.report).exists():
        raise FileExistsError('Choose a new report name, preserve board evidence')
    overlay = Overlay(str(bit), download=True)
    driver = LungAxil(overlay.lung_conv8_axil_0.mmio, timeout=args.timeout)
    rng = np.random.default_rng(1510)
    reports = []
    cases = [(1, 1, 1, 1, 1, 0), (3, 4, 7, 9, 3, 1), (11, 13, 9, 11, 3, 0), (96, 9, 3, 5, 3, 1)]
    for case, (cin, cout, height, width, kernel, padding) in enumerate(cases):
        q = rng.integers(0, 256, (cin, height, width), dtype=np.uint8)
        weights = rng.integers(-3, 4, (cout, cin, kernel, kernel), dtype=np.int8)
        bias = rng.integers(-20, 21, cout, dtype=np.int32)
        if case == 0:
            q.fill(255); weights.fill(1); bias.fill(0)
        iz, oz = 128, 128
        expected = feature_words(reference(q, weights, bias, iz, oz, padding=padding), oz)
        pixels = (height+2*padding-kernel+1)*(width+2*padding-kernel+1)
        blocks = (cout+7)//8
        src = allocate(shape=(height*width*((cin+7)//8),), dtype=np.uint64)
        dst = None
        try:
            dst = allocate(shape=(pixels*blocks,), dtype=np.uint64)
            src[:] = feature_words(q, iz); dst[:] = np.uint64(0xDEADBEEFCAFEF00D)
            expected_so_far = np.full((pixels, blocks), np.uint64(0xDEADBEEFCAFEF00D), np.uint64)
            for block in range(blocks):
                driver.configure(int(src.physical_address), int(dst.physical_address), height, width, cin, cout, block, kernel, padding)
                first = block*8; count = min(8, cout-first)
                lane_bias = [int(bias[first+i]) if i < count else 0 for i in range(8)]
                driver.load_weights(weight_packets(weights, block), lane_bias, [1 << 30]*8, [31]*8, iz, oz, (1 << count)-1, False)
                begin = time.perf_counter(); stats = driver.run(src, dst); elapsed = (time.perf_counter()-begin)*1000
                expected_so_far[:, block] = expected.reshape(pixels, blocks)[:, block]
                different = int(np.count_nonzero(np.asarray(dst).reshape(pixels, blocks) != expected_so_far))
                if different:
                    raise AssertionError('case {} block {}: {} memory words differ'.format(case, block, different))
                row = dict(case=case, output_block=block, same=True, different_words=different,
                    cpu_call_ms=elapsed, **stats)
                reports.append(row); print(json.dumps(row), flush=True)
            if case == 0:
                # Same address, same resident weights, changed contents: exercise
                # CPU cache synchronization and the engine's per-command flush.
                q = 255-q; src[:] = feature_words(q, iz)
                stats = driver.run(src, dst)
                revised = feature_words(reference(q, weights, bias, iz, oz, padding=padding), oz)
                if not np.array_equal(dst, revised) or np.array_equal(revised, expected):
                    raise AssertionError('Same-address update produced stale output')
                reports.append(dict(case='same_address_update', same=True, different_words=0, **stats))
        finally:
            # Driver timeout poisons reuse; do NOT release possibly in-flight
            # buffers to an allocator while PL may still access their addresses.
            if driver._poisoned or driver._inflight:
                _TIMED_OUT_BUFFERS.extend([src, dst])
            else:
                if dst is not None: dst.freebuffer()
                src.freebuffer()
    report = dict(status='passed_real_board_single_block_smoke', bit_sha256=hashlib.sha256(bit.read_bytes()).hexdigest(),
        hwh_sha256=hashlib.sha256(hwh.read_bytes()).hexdigest(), results=reports, full_network=False,
        warning='Synthetic engineering smoke only; not full U-Net, medical accuracy or clinical validation')
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Real-board single-block smoke: PASS', flush=True)
    return report


_TIMED_OUT_BUFFERS = []
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bit', default='/home/xilinx/ct_test/lung_axil.bit')
    parser.add_argument('--report')
    parser.add_argument('--timeout', type=float, default=5.0)
    run(parser.parse_args())
