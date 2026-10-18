"""Prepare and audit the same-board native CPU / guarded PL comparison package."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_continuous import unpack_words
from lung_continuous_test import load_package
from lung_cpu import compile_native,NativeUNet


def generate(args):
    workspace=Path(__file__).resolve().parents[1]
    original=workspace/'data/lung_continuous_20261015'
    manifest,model,parameters=load_package(original/'lung_continuous')
    accepted_path=original/'board_report_verification.json'
    accepted=json.loads(accepted_path.read_text(encoding='utf-8'))
    if accepted['status']!='verified_returned_real_board_continuous_output': raise ValueError('Actual original continuous PL head must be accepted first')
    if accepted['head_codes_sha256']!=file_sha256(original/'lung_continuous_result_head.npz'):
        raise ValueError('Actual board output changed')
    check_hashes(manifest['source_sha256']);check_hashes(accepted['source_sha256'])
    root=fresh_output(args.root).resolve(); target=root/'lung_cpu_compare';target.mkdir()
    for name in list(manifest['files_sha256'])+['continuous_manifest.json']:
        shutil.copy2(original/'lung_continuous'/name,target/name)
    additions=['lung_cpu_kernel.h','lung_cpu_native.cpp','lung_cpu.py','lung_cpu_compare.py']
    tracked={str(accepted_path):file_sha256(accepted_path),
             str(Path(__file__).resolve()):file_sha256(__file__),
             str(workspace/'python/tests/test_lung_cpu.py'):file_sha256(workspace/'python/tests/test_lung_cpu.py')}
    for name in additions:
        source=workspace/'pynq'/name;shutil.copy2(source,target/name);tracked[str(source)]=file_sha256(source)
    compare=dict(schema='lung_same_board_cpu_pl_v1',original_continuous_package_sha256=file_sha256(target/'continuous_manifest.json'),
                 original_real_board_acceptance_sha256=file_sha256(accepted_path),
                 files_sha256={p.name:file_sha256(p) for p in target.iterdir()},source_sha256=tracked,
                 warning='No ARM timing measured yet; C++ native baseline is self-written, not a best-possible-library claim. Benchmark current guarded PL without per-node reference comparisons.')
    write_json(target/'cpu_compare_manifest.json',compare)
    load_package(target)
    native=root/'host_native';native.mkdir()
    for name in ['lung_cpu_kernel.h','lung_cpu_native.cpp']: shutil.copy2(target/name,native/name)
    library,build=compile_native(native,args.compiler)
    with np.load(target/'input.npz',allow_pickle=False) as archive: image=archive['codes']
    with np.load(target/'reference.npz',allow_pickle=False) as archive:
        expected={n['name']:unpack_words(archive[n['name']],n['shape']) for n in model['nodes']}
    with np.load(original/'lung_continuous_result_head.npz',allow_pickle=False) as archive:
        actual_pl_head=archive['head_codes']
    for threads in [1,2]:
        with NativeUNet(library,model,parameters,threads) as network:
            head=network.forward(image)
            if not np.array_equal(head,actual_pl_head): raise ValueError('Native complete head differs from actual board output')
            for index,node in enumerate(model['nodes']):
                if not np.array_equal(network.node_codes(index),expected[node['name']]):
                    raise ValueError('Native actual-input node mismatch: '+node['name'])
                print('PC native {} worker(s), {}: exact frozen node match'.format(threads,node['name']),flush=True)
    # Compile the SAME intrinsic microkernel to ARMv7 assembly. This does not
    # execute ARM instructions on the PC and is NOT a board performance result.
    assembly=native/'armv7_neon_check.s'
    command=[args.arm_clang,'--target=armv7-none-eabi','-mcpu=cortex-a9','-mfpu=neon','-O3','-std=c++11',
             '-ffreestanding','-DLUNG_NEON_ONLY','-S',str(native/'lung_cpu_native.cpp'),'-o',str(assembly)]
    process=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if process.returncode: raise RuntimeError('ARM NEON code generation failed:\n'+process.stdout)
    text=assembly.read_text(encoding='utf-8')
    if not re.search(r'vmlal\.s16',text): raise ValueError('ARM assembly lacks widened signed NEON MAC')
    check_hashes(tracked);check_hashes(manifest['source_sha256'])
    write_json(root/'native_verification.json',dict(status='passed_pc_native_numeric_and_arm_compile_checks_not_board_tested',
        cpu_threads_checked=[1,2],nodes_checked_per_run=21,actual_board_head_match=True,
        input_codes_unchanged=True,retraining=False,recalibration=False,arm_runtime_tested=False,
        pc_native_build=build,arm_compile_command=command,arm_compile_stdout=process.stdout,
        arm_assembly_sha256=file_sha256(assembly),package_sha256=file_sha256(target/'cpu_compare_manifest.json'),
        source_sha256=tracked,files_sha256=compare['files_sha256'],
        warning='PC native execution and ARM cross-compilation only. No ARM CPU timings or speedups exist until the user runs the new board program.'))
    print('Prepared same-board comparison:',target,flush=True)


def audit(root):
    root=Path(root).resolve(); output=root/'verification.json'
    if output.exists(): raise FileExistsError('Preserve previous comparison proof')
    native_path=root/'native_verification.json'; native=json.loads(native_path.read_text(encoding='utf-8'))
    if native['status']!='passed_pc_native_numeric_and_arm_compile_checks_not_board_tested': raise ValueError('Missing numeric/compile proof')
    folder=root/'lung_cpu_compare';check_hashes(native['source_sha256'])
    check_hashes({str(folder/n):h for n,h in native['files_sha256'].items()});load_package(folder)
    if native['package_sha256']!=file_sha256(folder/'cpu_compare_manifest.json'): raise ValueError('Changed comparison manifest')
    if native['arm_assembly_sha256']!=file_sha256(root/'host_native/armv7_neon_check.s'): raise ValueError('ARM assembly changed')
    log_path=root/'python_tests.log';raw=log_path.read_bytes()
    log=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    match=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',log)
    if not match or int(match[1])<236 or 'skipped=' in log: raise ValueError('Full 236-test regression required, including actual native compilation')
    workspace=Path(__file__).resolve().parents[1]
    protected=json.loads((workspace/'data/lung_fixed_network_20261014/screen/evaluation.json').read_text(encoding='utf-8'))['protocol']['input_sha256']
    check_hashes(protected)
    result=dict(status='prepared_audited_same_board_cpu_pl_comparison_not_board_tested',
                cpu_board_time_known=False,speedup_known=False,arm_runtime_tested=False,
                protected_medical_sources_unchanged=len(protected),python_tests=int(match[1]),
                python_tests_sha256=file_sha256(log_path),native_proof_sha256=file_sha256(native_path),
                package_sha256=file_sha256(folder/'cpu_compare_manifest.json'),
                source_sha256=native['source_sha256'],package_files_sha256=native['files_sha256'])
    write_json(output,result);print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_cpu_compare_20261015')
    parser.add_argument('--compiler',default='E:/vivado2/Vivado/2022.2/tps/mingw/6.2.0/win64.o/nt/bin/g++.exe')
    parser.add_argument('--arm-clang',default='E:/vivado2/Vivado/2022.2/tps/llvm/7.0/win64/bin/clang++.exe')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify: audit(args.root)
    else: generate(args)
