"""Maintainer: build a credential-free, synthetic-only runtime delivery copy.

Requires the original local accepted packages. Recipients use ct_studio_runtime.py,
not this maintainer command. Historical packages are never rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pynq'))
from ct_studio_runtime import VARIANTS, sha, read, write


def synthetic_volume():
    z, y, x = np.mgrid[:8, :512, :512]
    body = ((x-256)/195)**2 + ((y-256)/171)**2 <= 1
    lungs = (((x-172)/64)**2 + ((y-248)/105)**2 <= 1) | (((x-340)/64)**2 + ((y-248)/105)**2 <= 1)
    hu = np.full(z.shape, -1000, np.int16)
    hu[body] = 35
    hu[lungs & body] = -780
    hu += np.random.default_rng(20261018).integers(-12, 13, size=hu.shape, dtype=np.int16)
    return hu


def independent_nodes(model, arrays, image):
    """FP64 direct integer-valued conv + independent Q31 reference, no native C++."""
    import torch
    from lung_fixed_network import requant_tensor
    torch.set_num_threads(2)
    state = {'input': torch.from_numpy(image.copy())[None]}
    trace = {}
    with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False):
        for node in model['nodes']:
            a = state[node['sources'][0]]
            if node['kind'] == 'conv':
                row = node['layer']; prefix = row['prefix']
                w = torch.from_numpy(arrays[prefix+'_weight'].astype(np.float64))
                acc = torch.nn.functional.conv2d(a.to(torch.float64)-row['input_zero'], w, padding=row['padding'])
                acc += torch.from_numpy(arrays[prefix+'_bias'].astype(np.float64)).reshape(1,-1,1,1)
                if not torch.equal(acc, acc.round()) or not torch.isfinite(acc).all():
                    raise AssertionError('Non-exact FP64 accumulator')
                result = requant_tensor(acc.to(torch.int64), arrays[prefix+'_multiplier'],
                    arrays[prefix+'_shift'], row['output_zero'], row['relu'])
            elif node['kind'] == 'pool':
                result = torch.nn.functional.max_pool2d(a.to(torch.float64), 2).to(torch.uint8)
            else:
                q = node['quant']; b = state[node['sources'][1]]
                a = a.repeat_interleave(2,2).repeat_interleave(2,3)
                result = torch.cat((requant_tensor(a.to(torch.int64)-q['za'], [q['ma']], [q['sa']], q['zo']),
                    requant_tensor(b.to(torch.int64)-q['zb'], [q['mb']], [q['sb']], q['zo'])), 1)
            state[node['name']] = result
            trace[node['name']] = result[0].numpy().copy()
    return trace


def clean_manifest(original, files, **extra):
    # Deliberate allowlist: do not carry patient identity, workstation paths,
    # old fixture acceptance, or original credential hashes into a new release.
    return dict(schema=original['schema'], files_sha256=files,
        fixture='synthetic_engineering_phantom_v1', case='SYNTHETIC_CT_STUDIO', slice_index=3,
        fixture_board_tested=False, warning='New synthetic fixture; board startup validation is required. Not clinical evidence.', **extra)


def build(out, compiler):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Use a new maintainer build directory')
    out.mkdir(parents=True)
    public = out / 'public'; public.mkdir()
    old = ROOT / 'data/ct_platform_20261017_v2'
    fast_old = ROOT / 'data/ct_platform_fast_20261017_v2'
    protected = {}
    for base in (old, fast_old):
        for p in base.rglob('*'):
            if p.is_file(): protected[p] = sha(p)
    seg_source = old / 'ct_platform/segmentation'
    model = read(seg_source / 'model.json')
    with np.load(seg_source / 'parameters.npz', allow_pickle=False) as a:
        arrays = {k:a[k] for k in a.files}
    demo = public / 'ct_studio_demo'; demo.mkdir()
    hu = synthetic_volume()
    np.savez_compressed(demo / 'synthetic_ct_512.npz', hu=hu, spacing_zyx_mm=[2.,1.,1.],
        origin_lps_mm=[0.,0.,0.], orientation=[1.,0.,0.,0.,1.,0.])
    from lung_live_data import input_codes
    from lung_cpu import compile_native, NativeUNet
    from lung_axil_smoke import feature_words
    image = input_codes(hu[3], model)
    trace = independent_nodes(model, arrays, image)
    native = out / 'native'; native.mkdir()
    for name in ('lung_cpu_native.cpp','lung_cpu_kernel.h'):
        shutil.copy2(seg_source/name, native/name)
    library, _ = compile_native(native, compiler)
    checks = []
    for threads in (1,2):
        with NativeUNet(library, model, arrays, threads) as net:
            net.forward(image)
            for i, node in enumerate(model['nodes']):
                np.testing.assert_array_equal(net.node_codes(i), trace[node['name']])
            checks.append(dict(threads=threads, matching_nodes=21))
    reference = {n['name']:feature_words(trace[n['name']], n['zero']) for n in model['nodes']}
    for variant, (name, board_name) in VARIANTS.items():
        source = (old / 'ct_platform' if variant == 'classic' else fast_old / 'ct_platform_fast')
        original = read(source / 'platform_manifest.json')
        for n,h in original['files_sha256'].items():
            if sha(source/n) != h: raise ValueError('Original source changed: '+n)
        host = public / name; board = host / board_name; seg = board / 'segmentation'
        seg.mkdir(parents=True)
        # Minimal dependency closure actually used by original network startup.
        names = set(read(seg_source/'fast_manifest.json')['files_sha256'])
        names.update(read(seg_source/'clock_manifest.json')['files_sha256'])
        names.update(('lung_selected_benchmark.py', 'lung_live_service.py'))
        names -= {'input.npz','reference.npz','continuous_manifest.json','cpu_compare_manifest.json'}
        for n in sorted(names): shutil.copy2(seg_source/n, seg/n)
        np.savez_compressed(seg/'input.npz', codes=image)
        np.savez_compressed(seg/'reference.npz', **reference)
        for n in ('continuous_manifest.json','cpu_compare_manifest.json','fast_manifest.json','clock_manifest.json','selection_manifest.json'):
            old_m = read(seg_source/n)
            hashes = {f:sha(seg/f) for f in old_m['files_sha256']}
            extra = {}
            if n == 'continuous_manifest.json':
                extra = dict(bit_sha256=sha(seg/'lung_network.bit'), hwh_sha256=sha(seg/'lung_network.hwh'))
            if n == 'cpu_compare_manifest.json': extra['original_continuous_package_sha256'] = sha(seg/'continuous_manifest.json')
            if n == 'fast_manifest.json': extra['original_manifest_sha256'] = sha(seg/'continuous_manifest.json')
            if n == 'clock_manifest.json':
                extra.update(target_clock_mhz=100, original_fast_manifest_sha256=sha(seg/'fast_manifest.json'), routed_design=old_m['routed_design'])
            write(seg/n, clean_manifest(old_m, hashes, **extra))
        write(seg/'live_manifest.json', dict(schema='lung_live_fpga_v1',
            accepted_bit_sha256=sha(seg/'lung_network.bit'), live_board_integration_tested=False,
            files_sha256={p.name:sha(p) for p in sorted(seg.iterdir())},
            warning='Same accepted model/bit; new synthetic self-test fixture, board verification pending.'))
        (board/'basic').mkdir()
        for row in original['basic'].values():
            for kind in ('bit','hwh'):
                if not (board/row[kind]).exists(): shutil.copy2(source/row[kind], board/row[kind])
        for n in ('ct_platform_service.py','ct_basic_processing.py','lung_live_service.py'):
            shutil.copy2(source/n, board/n)
        if variant == 'fast': shutil.copy2(source/'ct_platform_fast_service.py',board/'ct_platform_fast_service.py')
        manifest = dict(schema='ct_platform_package_v1', basic=original['basic'],
            shape=[512,512], segmentation_bit_sha256=sha(seg/'lung_network.bit'),
            model_sha256=sha(seg/'model.json'), integration_board_tested=False,
            fixture='synthetic_engineering_phantom_v1', credential_generated_by='python/ct_studio_runtime.py install',
            files_sha256={p.relative_to(board).as_posix():sha(p) for p in sorted(board.rglob('*')) if p.is_file()})
        if variant == 'fast': manifest['basic_transport']='ct_basic_zlib_cache_v1'
        write(board/'platform_manifest.json',manifest)
        shutil.copy2(seg/'model.json',host/'model.json')
        host_m=dict(platform_manifest_sha256=sha(board/'platform_manifest.json'), basic=manifest['basic'])
        if variant == 'fast': host_m['basic_transport']=manifest['basic_transport']
        write(host/'host_platform.json',host_m)
        write(host/'host_config.json',dict(board_url='http://192.168.2.99:8765',token_file='platform_token.txt',
            model='model.json',platform_manifest='host_platform.json'))
    write(demo/'synthetic_reference_checks.json',dict(status='PASS_SYNTHETIC_CPU_REFERENCE_ONLY',
        nodes=21, commands=51, native_cpp_vs_independent_fp64_q31=checks,
        model_sha256=sha(seg_source/'model.json'), weights_sha256=sha(seg_source/'parameters.npz'),
        phantom_sha256=sha(demo/'synthetic_ct_512.npz'), input_sha256=sha(seg/'input.npz'),
        reference_sha256=sha(seg/'reference.npz'), patient_data_included=False, board_tested=False))
    archive=out/'ct_studio_runtime_20261018.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(public.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(public).as_posix())
    files={p.relative_to(public).as_posix():sha(p) for p in sorted(public.rglob('*')) if p.is_file()}
    write(out/'asset_manifest.json',dict(schema='ct_studio_public_assets_v1',
        archive=archive.name,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        files_sha256=files, synthetic_only=True, credentials_included=False,
        default_segmentation='original_unpruned_cache_return_100mhz',
        segmentation_bit_sha256=sha(seg/'lung_network.bit'), segmentation_model_sha256=sha(seg/'model.json'),
        segmentation_weights_sha256=sha(seg/'parameters.npz'),
        fast_basic_board_tested=False, classic_unified_startup_board_tested=False))
    for path,digest in protected.items():
        if sha(path)!=digest: raise ValueError('Protected historical package changed')
    write(out/'build_verification.json',dict(status='PASS_SANITIZED_SYNTHETIC_DELIVERY_BUILD',
        historical_files_unchanged=len(protected), public_files=len(files), archive_sha256=sha(archive),
        independent_reference_checked=True, board_accessed=False))
    print('PUBLIC RUNTIME READY:', archive, archive.stat().st_size, 'bytes; no credentials/patient fixtures')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compiler')
    args=parser.parse_args()
    build(args.output,args.compiler)
