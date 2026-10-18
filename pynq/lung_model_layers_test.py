"""Real-board isolated frozen U-Net convolution layers, not full-network inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from lung_axil import LungAxil
from lung_axil_smoke import feature_words, weight_packets

SCHEMA = 'frozen_model_layer_board_v1'
NAMES = ['enc1.0','enc1.3','enc2.0','enc2.3','enc3.0','enc3.3','neck.0','neck.3',
         'dec3.0','dec3.3','dec2.0','dec2.3','dec1.0','dec1.3','head']
SENTINEL = np.uint64(0xDEADBEEFCAFEF00D)
_TIMED_OUT_BUFFERS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_layer(archive, row):
    arrays = {key:archive[row['prefix']+'_'+key] for key in
              ['input','weight','bias','multiplier','shift','expected']}
    q, w, b, m, s, expected = arrays.values()
    if q.dtype != np.uint8 or q.ndim != 3 or tuple(q.shape) != tuple(row['input_shape']):
        raise ValueError('Invalid CHW input')
    if w.dtype != np.int8 or w.ndim != 4 or w.shape[1] != q.shape[0] or w.shape[2:] not in [(1,1),(3,3)]:
        raise ValueError('Invalid OIHW weights')
    cout, cin, kernel, _ = w.shape; _, height, width = q.shape; pad = row['padding']
    if not 1 <= min(cin, cout) or max(cin, cout) > 96 or not 1 <= min(height, width) or max(height, width) > 512:
        raise ValueError('Layer outside accelerator capacity')
    if pad not in (0,1) or kernel != row['kernel'] or (kernel == 1 and pad):
        raise ValueError('Invalid convolution geometry')
    shape = (cout, height+2*pad-kernel+1, width+2*pad-kernel+1)
    if min(shape) <= 0 or expected.dtype != np.uint8 or expected.shape != shape or list(shape) != row['output_shape']:
        raise ValueError('Invalid expected output')
    if any(a.dtype != np.int32 or a.shape != (cout,) for a in [b,m,s]):
        raise ValueError('Invalid channel parameters')
    if np.any(m <= 0) or np.any(s < 0) or np.any(s > 62):
        raise ValueError('Invalid per-channel requantization')
    if any(type(row[k]) is not int or not 0 <= row[k] <= 255 for k in ['input_zero','output_zero']) or type(row['relu']) is not bool:
        raise ValueError('Invalid quantization metadata')
    return arrays


def lane_parameters(arrays, block):
    cout = len(arrays['bias']); start = block*8; count = min(8, cout-start)
    if count <= 0 or start < 0:
        raise ValueError('Output block outside layer')
    values = [[int(arrays[key][start+i]) if i < count else fill for i in range(8)]
              for key, fill in [('bias',0),('multiplier',1),('shift',0)]]
    return values, (1 << count)-1


def run_layer(driver, allocate, arrays, row):
    q, weights = arrays['input'], arrays['weight']
    cin, height, width = q.shape; cout = weights.shape[0]
    input_words = feature_words(q, row['input_zero'])
    expected = feature_words(arrays['expected'], row['output_zero'])
    blocks = (cout+7)//8; pixels = len(expected)//blocks
    src = allocate(shape=(len(input_words),), dtype=np.uint64); dst = None
    results = []
    try:
        dst = allocate(shape=(len(expected)+16,), dtype=np.uint64)
        src[:] = input_words; dst[:] = SENTINEL
        wanted = np.full(len(dst), SENTINEL, dtype=np.uint64)
        for block in range(blocks):
            print('{} block {}/{}: configure/load/start'.format(row['name'],block+1,blocks), flush=True)
            driver.configure(int(src.physical_address), int(dst.physical_address), height, width, cin, cout,
                             block, row['kernel'], row['padding'])
            (bias, multiplier, shift), keep = lane_parameters(arrays, block)
            driver.load_weights(weight_packets(weights, block), bias, multiplier, shift,
                                row['input_zero'], row['output_zero'], keep, row['relu'])
            begin = time.perf_counter(); stats = driver.run(src, dst)
            elapsed = (time.perf_counter()-begin)*1000
            wanted[:len(expected)].reshape(pixels,blocks)[:,block] = expected.reshape(pixels,blocks)[:,block]
            different = int(np.count_nonzero(np.asarray(dst) != wanted))
            if different:
                raise AssertionError('{} block {}: {} output/canary words differ'.format(row['name'],block,different))
            result = dict(layer=row['name'], output_block=block, same=True, different_words=0,
                          suffix_canary_words=16, cpu_call_ms=elapsed, **stats)
            results.append(result); print(json.dumps(result), flush=True)
    finally:
        if driver._poisoned or driver._inflight:
            _TIMED_OUT_BUFFERS.extend([src,dst])
        else:
            if dst is not None: dst.freebuffer()
            src.freebuffer()
    return results


def run(args):
    package = Path(args.package).resolve(); folder = package.parent
    manifest = json.loads(package.read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA or [r['name'] for r in manifest['layers']] != NAMES:
        raise ValueError('Need the frozen original 15-layer fixture package')
    if args.report and Path(args.report).exists():
        raise FileExistsError('Choose a new report name; do not overwrite evidence')
    for name, digest in manifest['files_sha256'].items():
        target = (folder/name).resolve()
        if target.parent != folder or sha(target) != digest:
            raise ValueError('Package file changed: '+name)
    bit = Path(args.bit).resolve(); hwh = bit.with_suffix('.hwh')
    if sha(bit) != manifest['bit_sha256'] or sha(hwh) != manifest['hwh_sha256']:
        raise ValueError('Use the exact reset-corrected verified bit/HWH')
    vectors = folder/'vectors.npz'
    with np.load(vectors, allow_pickle=False) as archive:
        for row in manifest['layers']: read_layer(archive,row)
    print('Fixture/hash checks passed. Loading verified single-block overlay...',flush=True)
    from pynq import Overlay, allocate
    overlay = Overlay(str(bit))
    print('Overlay loaded. Checking accelerator ID/ABI...',flush=True)
    driver = LungAxil(overlay.lung_conv8_axil_0.mmio, timeout=args.timeout)
    results = []
    with np.load(vectors, allow_pickle=False) as archive:
        for row in manifest['layers']:
            results.extend(run_layer(driver,allocate,read_layer(archive,row),row))
    if len(results) != manifest['expected_commands']:
        raise AssertionError('Incomplete layer command coverage')
    report = dict(status='passed_real_board_frozen_model_convolution_layers',
                  package_sha256=sha(package), bit_sha256=sha(bit), hwh_sha256=sha(hwh),
                  case=manifest['case'], slice_index=manifest['slice_index'], results=results,
                  layer_count=len(manifest['layers']), full_network=False,
                  warning='Isolated real convolution layers with CPU-generated upstream inputs; pool/resize/concat and full CT segmentation are NOT tested on PL')
    if args.report: Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Real-board frozen model convolution layers: PASS',flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default='/home/xilinx/ct_test/lung_model_layers/layers.json')
    parser.add_argument('--bit', default='/home/xilinx/ct_test/lung_axil.bit')
    parser.add_argument('--report')
    parser.add_argument('--timeout', type=float, default=30.0)
    run(parser.parse_args())
