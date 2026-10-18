"""Independently audit package identities and emulate continuous memory flow.

This is CPU emulation of host orchestration, NOT real-board acceptance.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

from lung_fixed_network import checked_contract
from lung_segmentation import write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes
from lung_continuous_emulator import Memory,Conv,Tensor
from lung_continuous_package import make_plan

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_axil_smoke import feature_words
from lung_continuous import ContinuousUNet,unpack_words
from lung_continuous_test import load_package


def audit(root):
    root = Path(root).resolve(); folder = root/'lung_continuous'; proof_path = root/'verification.json'
    if proof_path.exists(): raise FileExistsError('Preserve previous verification evidence')
    manifest,model,parameters = load_package(folder); check_hashes(manifest['source_sha256'])
    workspace = Path(__file__).resolve().parents[1]
    path,contract,protected = checked_contract(workspace/'data/lung_fpga_readiness_20261014/integer_contract_v3/contract.json')
    old_folder = workspace/'data/lung_network_board_20261015/lung_network'
    old = json.loads((old_folder/'layers.json').read_text(encoding='utf-8'))
    if model != make_plan(old['layers'],contract): raise ValueError('Plan differs from independently reconstructed frozen topology')
    with np.load(path.parent/'parameters.npz',allow_pickle=False) as frozen:
        for row in old['layers']:
            prefix = row['prefix']
            for key,suffix in [('weight','weight_int8'),('bias','bias_int32'),('multiplier','multiplier_int32'),('shift','right_shift_int32')]:
                if not np.array_equal(parameters[prefix+'_'+key],frozen[prefix+'_'+suffix]):
                    raise ValueError('Frozen model parameter differs: '+prefix+'_'+key)
    with np.load(folder/'input.npz',allow_pickle=False) as source: image = source['codes']
    with np.load(old_folder/'vectors.npz',allow_pickle=False) as original:
        if not np.array_equal(image,original['enc1_0_input']): raise ValueError('Original CT input codes changed')
        wanted = {}
        mapping = {'pool0':'enc2_0','pool1':'enc3_0','pool2':'neck_0',
                   'cats.0':'dec3_0','cats.1':'dec2_0','cats.2':'dec1_0'}
        for node in model['nodes']:
            key = node['layer']['prefix']+'_expected' if node['kind']=='conv' else mapping[node['name']]+'_input'
            wanted[node['name']] = feature_words(original[key],node['zero'])
    with np.load(folder/'reference.npz',allow_pickle=False) as reference:
        if set(reference.files) != set(wanted): raise ValueError('Reference node coverage mismatch')
        for name,words in wanted.items():
            if not np.array_equal(words,reference[name]): raise ValueError('Reference changed: '+name)
    memory = Memory(); conv = Conv(memory); tensor = Tensor(memory)
    network = ContinuousUNet(conv,tensor,memory.allocate,model,parameters)
    observed = []
    def observe(name,words):
        if not np.array_equal(words,wanted[name]): raise AssertionError('Actual upstream CPU emulation differs: '+name)
        observed.append(name)
        print('  PC-only actual-input emulation matched frozen reference:',name,flush=True)
    actual,stats = network.predict_codes(image,observe)
    if not np.array_equal(actual,unpack_words(wanted['head'],model['nodes'][-1]['shape'])):
        raise AssertionError('Emulated head extraction mismatch')
    if len(observed)!=21 or len(memory.events)!=51 or not all(b.freed for b in memory.buffers):
        raise AssertionError('Incomplete scheduling/release proof')
    # Confirm address identity for every real dependency, not just output equality.
    addresses = {'input':memory.buffers[0].physical_address}; offset = 0
    for node in model['nodes']:
        count = (node['shape'][0]+7)//8 if node['kind']=='conv' else 1
        events = memory.events[offset:offset+count]; offset += count
        for event in events:
            if event['sources'] != [addresses[n] for n in node['sources']]:
                raise AssertionError('CPU fixture replaced a dependency')
        addresses[node['name']] = events[0]['dst']
    log_path=root/'python_tests.log'; raw=log_path.read_bytes()
    log=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    match=re.search(r'Ran (\d+) tests in .*?\n\s*\nOK\s*$',log)
    if not match or int(match[1])<219: raise ValueError('Full regression proof missing')
    check_hashes(protected); check_hashes(manifest['source_sha256']); load_package(folder)
    result=dict(status='prepared_audited_continuous_network_not_board_tested',
                cpu_emulation_only=True,continuous_network_on_real_board=False,
                frozen_parameters_and_ct_input_unchanged=True,reference_nodes_matched=len(observed),
                hardware_commands_planned=stats['command_count'],cma_peak_bytes=stats['cma_peak_bytes'],
                protected_source_count=len(protected),python_tests=int(match[1]),
                package_sha256=file_sha256(folder/'continuous_manifest.json'),
                files_sha256=manifest['files_sha256'],source_sha256=manifest['source_sha256'],
                audit_sources_sha256={str(p):file_sha256(p) for p in [Path(__file__).resolve(),workspace/'python/lung_continuous_emulator.py']},
                tests_log_sha256=file_sha256(log_path),
                warning='CPU emulation proves scheduling/arithmetic agreement only; new continuous FPGA run still required. No CPU timing is reported as FPGA timing.')
    write_json(proof_path,result)
    print('PC-only continuous scheduler audit: PASS; real-board continuous test still required',flush=True)
    print('Planned peak CMA bytes:',stats['cma_peak_bytes'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_continuous_20261015')
    audit(parser.parse_args().root)
