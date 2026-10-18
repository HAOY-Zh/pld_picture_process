"""Build an independently audited prepared-runtime package, preserving all old proofs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'pynq'))
from lung_continuous_test import load_package
from lung_fast import prepare_block
from lung_axil_smoke import weight_packets
from lung_fast_benchmark import validate_package


def generate(root):
    original=WORKSPACE/'data/lung_cpu_compare_20261015/lung_cpu_compare'
    manifest,model,params=load_package(original)
    prior=json.loads((original/'cpu_compare_manifest.json').read_text(encoding='utf-8'))
    check_hashes({str(original/n):h for n,h in prior['files_sha256'].items()})
    check_hashes(prior['source_sha256']);check_hashes(manifest['source_sha256'])
    root=fresh_output(root).resolve();folder=root/'lung_fast';folder.mkdir()
    for p in original.iterdir():
        if p.name in prior['files_sha256'] or p.name=='cpu_compare_manifest.json': shutil.copy2(p,folder/p.name)
    additions=['lung_fast.py','lung_fast_io.cpp','lung_fast_io.h','lung_fast_benchmark.py']
    source={str(Path(__file__).resolve()):file_sha256(__file__)}
    for name in additions:
        p=WORKSPACE/'pynq'/name;shutil.copy2(p,folder/name);source[str(p)]=file_sha256(p)
    for name in ['test_lung_fast.py','lung_fast_io_test.cpp']:
        p=WORKSPACE/'python/tests'/name;source[str(p)]=file_sha256(p)
    packets=0;blocks=0
    for node in model['nodes']:
        if node['kind']!='conv': continue
        row=node['layer']; arrays={k:params[row['prefix']+'_'+k] for k in ['weight','bias','multiplier','shift']}
        for block in range((node['shape'][0]+7)//8):
            prepared=prepare_block(arrays,block,row)
            if list(prepared['packets'])!=weight_packets(arrays['weight'],block): raise AssertionError('Changed frozen weight packet order')
            packets+=len(prepared['words']);blocks+=1
    record=dict(schema='lung_prepared_runtime_v1',source_sha256=source,
        original_manifest_sha256=file_sha256(folder/'continuous_manifest.json'),
        files_sha256={p.name:file_sha256(p) for p in folder.iterdir()},
        frozen_weight_blocks=blocks,weight_packets_per_frame=packets,weight_mmio_words_per_frame=packets*16,
        model_changed=False,bit_changed=False,board_tested=False,
        warning='Host-runtime optimization only. Speed, ARM compilation and real cache/MMIO behavior await board measurement.')
    write_json(folder/'fast_manifest.json',record);validate_package(folder)
    # Real input through independent arithmetic emulators of the NEW schedule.
    # Never ship test doubles or substitute this for physical board results.
    sys.path.insert(0,str(WORKSPACE/'python/tests'))
    from test_lung_fast import CachedMemory,DeviceConv,DeviceTensor
    from lung_fast import FastUNet
    memory=CachedMemory();network=FastUNet(DeviceConv(memory),DeviceTensor(memory),memory.allocate,model,params)
    checked=[]
    with np.load(folder/'input.npz',allow_pickle=False) as a: image=a['codes']
    with np.load(folder/'reference.npz',allow_pickle=False) as refs:
        def observe(name,words):
            if not np.array_equal(words,refs[name]): raise AssertionError('PC actual-input emulated node differs: '+name)
            checked.append(name);print('PC new schedule '+name+': exact frozen node match',flush=True)
        try:
            head,stats=network.predict_codes(image,observe)
            second,second_stats=network.predict_codes(image)
            if not np.array_equal(head,second) or len(checked)!=21: raise AssertionError('Emulated repeat differs')
            with np.load(WORKSPACE/'data/lung_continuous_20261015/lung_continuous_result_head.npz',allow_pickle=False) as a:
                if not np.array_equal(head,a['head_codes']): raise AssertionError('New emulated schedule differs from returned original board head')
        finally: network.close()
    write_json(root/'pc_verification.json',dict(status='passed_pc_prepared_runtime_real_input_not_board_tested',
        nodes_checked=checked,head_matches_accepted_original_board_head=True,repeated_frames_checked=2,
        separate_cpu_cache_and_device_memory=True,cma_resident_bytes=stats['cma_resident_bytes'],
        package_sha256=file_sha256(folder/'fast_manifest.json'),source_sha256=source,
        native_mmio_runtime_on_arm_verified=False,performance_known=False))
    check_hashes(source);check_hashes(manifest['source_sha256'])
    print('Prepared upload folder: '+str(folder),flush=True)


def audit(root):
    import re
    root=Path(root).resolve();out=root/'verification.json'
    if out.exists(): raise FileExistsError('Preserve existing verification')
    folder=root/'lung_fast';manifest,_,_,fast=validate_package(folder)
    check_hashes(fast['source_sha256']);check_hashes(manifest['source_sha256'])
    pc_path=root/'pc_verification.json';pc=json.loads(pc_path.read_text(encoding='utf-8'))
    if pc['package_sha256']!=file_sha256(folder/'fast_manifest.json') or len(pc['nodes_checked'])!=21:
        raise ValueError('PC proof changed/incomplete')
    raw=(root/'python_tests.log').read_bytes()
    log=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    match=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',log)
    if not match or int(match[1])<245 or 'skipped=' in log: raise ValueError('Need full regression including compiled native protocol test')
    protected=json.loads((WORKSPACE/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    check_hashes(protected)
    write_json(out,dict(status='prepared_audited_host_optimization_not_board_tested',
        board_performance_known=False,arm_native_execution_verified=False,
        pc_nodes_checked=21,python_tests=int(match[1]),protected_medical_sources_unchanged=len(protected),
        package_sha256=file_sha256(folder/'fast_manifest.json'),pc_proof_sha256=file_sha256(pc_path),
        python_tests_sha256=file_sha256(root/'python_tests.log'),source_sha256=fast['source_sha256']))
    print('Audit PASS; board timings and real-MMIO validation still required.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_fast_20261015')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify: audit(args.root)
    else: generate(args.root)
