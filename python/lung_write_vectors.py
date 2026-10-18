"""HWC8 output fixtures from frozen convolution codes; no medical inference."""
import argparse
import json
from pathlib import Path

from lung_rtl_vectors import hex_bank
from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256
from lung_weight_reuse_vectors import read_bank


def output_plan(base, height, width, blocks, block, keep, fill, results):
    if not (1 <= height <= 512 and 1 <= width <= 512 and 1 <= blocks <= 12
            and 0 <= block < blocks and 1 <= keep <= 255 and 0 <= fill <= 255
            and 0 <= base and base + height * width * blocks <= 2**29):
        raise ValueError('Invalid output configuration/physical span')
    lanes = [i for i in range(8) if keep >> i & 1]
    if len(results) != height * width * len(lanes):
        raise ValueError('Wrong output stream length')
    writes = []
    index = 0
    for y in range(height):
        for x in range(width):
            word = bytearray([fill] * 8)
            for i, lane in enumerate(lanes):
                code, got_lane, got_x, got_y, last, final = results[index]
                index += 1
                if (got_lane, got_x, got_y, last, final) != (lane, x, y, i == len(lanes)-1,
                        i == len(lanes)-1 and y == height-1 and x == width-1) or not 0 <= code <= 255:
                    raise ValueError('Invalid output stream metadata/code')
                word[lane] = code
            writes.append((base + (y * width + x) * blocks + block, int.from_bytes(word, 'little')))
    return writes


def generate(args):
    source = Path(args.source).resolve()
    manifest = source / 'vectors.json'
    info = json.loads(manifest.read_text(encoding='utf-8'))
    tracked = {str(manifest): file_sha256(manifest), **info['source_sha256']}
    tracked.update({str(source / name): digest for name, digest in info['outputs_sha256'].items()})
    for path, digest in tracked.items():
        if file_sha256(path) != digest:
            raise ValueError('Frozen fixture changed: ' + path)
    h = read_bank(source, 'header')
    frames, codes, lanes, lasts, coords = [read_bank(source, n) for n in ['frames', 'codes', 'lanes', 'lasts', 'coords']]
    if len(h) != 9 or not all(h) or len(frames) != h[0]*16:
        raise ValueError('Bad source header')
    if any(len(a) != h[4] for a in [codes, lanes, lasts, coords]):
        raise ValueError('Bad source output lengths')
    descriptors, addresses, words = [], [], []
    used_outputs = 0
    for f in range(h[0]):
        row = frames[f*16:f*16+16]
        oh = row[1] + 2*row[5] - row[4] + 1
        ow = row[2] + 2*row[5] - row[4] + 1
        count = oh * ow * row[8].bit_count()
        if row[13] != used_outputs or row[15] != oh*ow:
            raise ValueError('Source frame offsets differ')
        results = [(codes[i], lanes[i], coords[i] & 1023, coords[i] >> 10 & 1023,
                    bool(lasts[i]), bool(coords[i] >> 20)) for i in range(used_outputs, used_outputs+count)]
        base, blocks = 65536 + (f % 3)*511, f % 12 + 1
        block = f % blocks
        writes = output_plan(base, oh, ow, blocks, block, row[8], row[7], results)
        descriptors.extend([base, oh, ow, blocks, block, len(words), len(writes)])
        addresses.extend(a for a, _ in writes)
        words.extend(w for _, w in writes)
        used_outputs += count
    if used_outputs != h[4]:
        raise ValueError('Incomplete frozen output coverage')
    out = fresh_output(args.outdir)
    hex_bank(out/'header.mem', [h[0], h[4], len(words)], 32)
    hex_bank(out/'descriptors.mem', descriptors, 32)
    hex_bank(out/'addresses.mem', addresses, 32)
    hex_bank(out/'words.mem', words, 64)
    report = dict(status='complete_output_write_vectors', source=str(source), frames=h[0], codes=h[4],
        words=len(words), write_payload_bytes=8*len(words), source_sha256=tracked,
        generated_by_sha256=file_sha256(__file__), outputs_sha256={p.name:file_sha256(p) for p in out.glob('*.mem')},
        warning='One AXI beat per pixel/block; simulated memory, not board/DDR/FPS. No old vector duplication.')
    write_json(out/'vectors.json', report)
    print({k:report[k] for k in ['status', 'frames', 'codes', 'words', 'write_payload_bytes']}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/lung_cache_axi_20261015/vectors')
    parser.add_argument('--outdir', required=True)
    generate(parser.parse_args())
