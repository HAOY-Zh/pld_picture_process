"""Seal exact native host addon against the already accepted prefetch hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python')]
from nodule_plan import load_package,sha


def package(root):
    root=Path(root).resolve();target=root/'nodule_host_fast'
    if target.exists():raise FileExistsError('Preserve sealed host addon')
    old=ROOT/'data/nodule_prefetch_20261024_v2/nodule_prefetch'
    old_manifest=json.loads((old/'prefetch_manifest.json').read_text())
    release=json.loads((old.parent/'release.json').read_text())
    if release['status']!='routed_exact_prefetch_ready_for_board' or sha(old/'prefetch_manifest.json')!=release['manifest_sha256']:
        raise ValueError('Wrong frozen prefetch release')
    for n,h in old_manifest['files_sha256'].items():
        if sha(old/n)!=h:raise ValueError('Changed frozen prefetch artifact')
    parent=ROOT/'data/nodule_board_20261023_v1/nodule_board';load_package(parent)
    if old_manifest['parent_manifest_sha256']!=sha(parent/'nodule_manifest.json'):raise ValueError('Changed canonical model')
    proof=root/'local_final/verification.json';p=json.loads(proof.read_text())
    if p['status']!='real_448_native_host_context_head_decode_candidates_exact' or p['board_tested']:
        raise ValueError('Missing real CNN host integration proof')
    for f,h in p['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Host code changed after actual CNN verification')
    if sha(root/'local_final/outputs.npz')!=p['outputs_sha256']:raise ValueError('Changed real CNN output evidence')
    if sha(old/'slice_case.npz')!=p['slice_payload_sha256']:raise ValueError('Different physical test case')
    target.mkdir()
    frozen=['nodule_canvas.py','nodule_canvas_pl.py','nodule_canvas_cpu.py','nodule_fast_input.py',
            'nodule_canvas_input.py','nodule_canvas_input.cpp','nodule_canvas_direct.py',
            'slice_case.json','slice_case.npz','lung_fast_io.cpp','lung_fast_io.h','lung_network.bit','lung_network.hwh']
    for n in frozen:shutil.copy2(old/n,target/n)
    new=['nodule_sampling_native.cpp','nodule_native_sampling.py','nodule_fast_decode.py','nodule_host_pipeline.py','nodule_host_benchmark.py']
    for n in new:shutil.copy2(ROOT/'pynq'/n,target/n)
    shutil.copy2(ROOT/'docs/肺结节端到端前后处理加速_20261024.md',target/'README.md')
    dev=target/'development';dev.mkdir()
    for f in [Path(__file__),ROOT/'python/nodule_native_host_local.py',ROOT/'python/verify_nodule_native_host.py',
              ROOT/'python/tests/test_nodule_native_host.py',ROOT/'python/tests/test_nodule_host_driver.py']:
        shutil.copy2(f,dev/f.name)
    ev=target/'evidence';ev.mkdir()
    for f in [proof,root/'local_final/outputs.npz',old/'prefetch_manifest.json',old.parent/'release.json']:
        shutil.copy2(f,ev/f.name)
    for n in ['timing_routed.rpt','utilization_routed.rpt','drc.rpt']:shutil.copy2(old.parent/n,ev/n)
    # Include the actual unchanged Verilog for teammate inspection; no build
    # intermediate projects or desktop-specific compiled DLLs are shipped.
    shutil.copytree(old/'rtl',target/'rtl')
    manifest=dict(schema='nodule_exact_native_host_addon_v1',board_tested=False,local_numerical_tests=7,driver_integration_tests=1,
        parent_manifest_sha256=sha(parent/'nodule_manifest.json'),prefetch_manifest_sha256=sha(old/'prefetch_manifest.json'),
        bit_sha256=sha(target/'lung_network.bit'),hwh_sha256=sha(target/'lung_network.hwh'),clock_mhz=100,
        model_parameters_quantization_threshold_changed=False,xy_sampling_values_changed=False,
        online_sampling_each_run=True,cross_plane_cache=False,
        files_sha256={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()})
    (target/'host_manifest.json').write_text(json.dumps(manifest,indent=2))
    code="import sys;sys.path[:0]=sys.argv[1:3];import nodule_host_benchmark,nodule_host_pipeline,nodule_native_sampling;assert 'torch' not in sys.modules and 'sklearn' not in sys.modules;print('portable host imports PASS')"
    smoke=subprocess.run([sys.executable,'-B','-c',code,str(target),str(parent)],cwd=root,text=True,capture_output=True,check=True)
    archive=root/'nodule_host_fast_20261024.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in target.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Bad host archive')
        for n,h in manifest['files_sha256'].items():
            if hashlib.sha256(z.read('nodule_host_fast/'+n)).hexdigest()!=h:raise ValueError('Archive/hash mismatch')
    receipt=dict(status='exact_native_host_ready_for_board',board_tested=False,bit_changed=False,
        archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),manifest_sha256=sha(target/'host_manifest.json'),
        real_cnn_full_plane_exact=True,local_numerical_tests=7,driver_integration_tests=1,imports=smoke.stdout.strip(),
        sources_sha256={str(ROOT/'pynq'/n):sha(ROOT/'pynq'/n) for n in new})
    (root/'release.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT/'data/nodule_native_host_20261024_v1'));package(p.parse_args().root)
