"""Create a separate synthetic delivery; never modify the accepted private pack.

Build-time dependencies: NumPy, PyTorch CPU and g++. Board runtime needs no Torch.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'pynq'), str(ROOT / 'python')]
from lung_nodule_shared import checked_package, load_model
from lung_cpu import compile_native, NativeUNet
from nodule_cpu import NativeNodule
from nodule_plan import sha, lower_stem, requant
from lung_axil_smoke import feature_words


def dense_reference(model, params, image):
    import torch
    torch.set_num_threads(2)
    values = {'input': image}
    with torch.inference_mode():
        for node in model['nodes']:
            a = values[node['sources'][0]]
            if node['kind'] == 'conv':
                row = node['layer']; prefix = row['prefix']
                acc = torch.nn.functional.conv2d(
                    torch.from_numpy(a.astype(np.float64) - row['input_zero'])[None],
                    torch.from_numpy(params[prefix + '_weight'].astype(np.float64)),
                    padding=row['padding'])[0].numpy()
                if not np.array_equal(acc, np.rint(acc)):
                    raise ArithmeticError('Nonintegral FP64 accumulator')
                acc = acc.astype(np.int64) + params[prefix + '_bias'][:, None, None]
                value = requant(acc, params[prefix + '_multiplier'][:, None, None],
                                params[prefix + '_shift'][:, None, None], row['output_zero'], row['relu'])
            elif node['kind'] == 'pool':
                c, h, w = a.shape
                value = a.reshape(c, h // 2, 2, w // 2, 2).max((2, 4))
            elif node['kind'] == 'fused_concat':
                q = node['quant']; b = values[node['sources'][1]]
                aa = np.repeat(np.repeat(a, 2, axis=1), 2, axis=2)
                value = np.concatenate([
                    requant(aa.astype(np.int64) - q['za'], q['ma'], q['sa'], q['zo']),
                    requant(b.astype(np.int64) - q['zb'], q['mb'], q['sb'], q['zo'])])
            else:
                raise ValueError('Unknown node')
            values[node['name']] = np.ascontiguousarray(value)
    return values


def synthetic_image(channels, size, seed):
    # Analytic patterns plus seeded noise, without reading any medical image.
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size))
    base = (x * 3 + y * 5) % 256
    disk = (x - size * .43)**2 + (y - size * .56)**2 < (size * .21)**2
    images = []
    for c in range(channels):
        plane = (base + c * 13 + rng.integers(-12, 13, (size, size))) % 256
        plane[disk] = 180 + c * 4
        images.append(plane.clip(0, 255).astype(np.uint8))
    return np.ascontiguousarray(images)


def build(source, output, compiler=None):
    source, old = checked_package(source)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    target = output / 'lung_nodule_shared'; target.mkdir(parents=True)
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.py', '.cpp', '.h', '.bit', '.hwh'):
            shutil.copy2(p, target / p.name)
    shutil.copytree(source / 'rtl', target / 'rtl')
    # Labels must describe the new fixtures, not claim these exact fixtures ran on board.
    benchmark = target / 'lung_nodule_shared_benchmark.py'
    benchmark.write_text(benchmark.read_text(encoding='utf-8').replace('real nodule patches',
                         'synthetic nodule patches'), encoding='utf-8')
    builddir = output / 'native_build'; builddir.mkdir()
    for n in ['lung_cpu_native.cpp', 'lung_cpu_kernel.h']:
        shutil.copy2(target / n, builddir / n)
    lib, _ = compile_native(builddir, compiler)
    counts = {}; identities = {}; seeds = {'unet_original': 2026102401, 'unet_pruned': 2026102401, 'nodule': 2026102402}
    for label in ['unet_original', 'unet_pruned', 'nodule']:
        model, params = load_model(source, label)
        dst = target / 'models' / label; dst.mkdir(parents=True)
        for n in ['model.json', 'parameters.npz']:
            shutil.copy2(source / 'models' / label / n, dst / n)
        identities[label] = {n: sha(dst / n) for n in ['model.json', 'parameters.npz']}
        refs = {}; count = 0
        for case in range(4 if label == 'nodule' else 1):
            image = synthetic_image(5 if label == 'nodule' else 1,
                                    128 if label == 'nodule' else 256, seeds[label] + case)
            if label == 'nodule':
                image = lower_stem(image)
            dense = dense_reference(model, params, image)
            cls = NativeNodule if label == 'nodule' else NativeUNet
            for threads in [1, 2]:
                with cls(lib, model, params, threads) as net:
                    net.forward(image)
                    for i, node in enumerate(model['nodes']):
                        np.testing.assert_array_equal(net.node_codes(i), dense[node['name']])
                        count += 1
            if label == 'nodule':
                refs.update({'case' + str(case) + '_' + n: v for n, v in dense.items()})
            else:
                np.savez_compressed(dst / 'input.npz', codes=image)
                refs = {n['name']: feature_words(dense[n['name']], n['zero']) for n in model['nodes']}
            print(label, case, 'synthetic independent FP64 vs C++ 1/2-thread nodes EXACT', flush=True)
        np.savez_compressed(dst / 'reference.npz', **refs)
        counts[label] = count
    proof = dict(status='synthetic_independent_cpu_references_exact', board_tested=False,
                 nodes_checked=counts, fixture_seeds=seeds, model_identities=identities,
                 provenance='All inputs generated from analytic patterns and seeded noise. No patient arrays read or copied. Bit/weights are unchanged from accepted pack. New synthetic fixtures still require physical board acceptance.')
    (target / 'synthetic_validation.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    manifest = {k: copy.deepcopy(old[k]) for k in ['schema', 'clock_mhz', 'bit_sha256', 'hwh_sha256',
        'compatibility', 'all_models_resident_cma_bytes', 'route', 'utilization']}
    manifest.update(board_tested=False, fixture_kind='synthetic_only', rtl_changed=False,
        model_parameters_quantization_changed=False, overlay_downloads_per_session=1,
        source_accepted_manifest_sha256=sha(source / 'shared_manifest.json'),
        files_sha256={p.relative_to(target).as_posix(): sha(p) for p in sorted(target.rglob('*')) if p.is_file()})
    (target / 'shared_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    checked_package(target)
    archive = output / 'lung_nodule_shared_public_20261024.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(target.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failure')
    return dict(archive=archive.name, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                package_manifest_sha256=sha(target / 'shared_manifest.json'), bit_sha256=manifest['bit_sha256'],
                files_sha256={p.relative_to(output).as_posix(): sha(p) for p in sorted(target.rglob('*')) if p.is_file()},
                validation=proof)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-package', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--compiler')
    a = p.parse_args()
    result = build(a.source_package, a.output, a.compiler)
    (Path(a.output) / 'asset_manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
