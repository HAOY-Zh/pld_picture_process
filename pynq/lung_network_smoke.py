"""Dual-engine regression: old conv smoke + 15 real conv layers + PL tensor ops.

This does NOT schedule a complete network continuously. No PyTorch on the board.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import lung_axil_smoke
import lung_model_layers_test
from lung_model_layers_test import sha
from lung_tensor import TensorAxil
from lung_tensor_smoke import run_cases


def run(args):
    folder=Path(__file__).resolve().parent
    package=folder/'network_manifest.json';manifest=json.loads(package.read_text(encoding='utf-8'))
    if manifest.get('schema')!='lung_dual_engine_smoke_v1': raise ValueError('Wrong dual-engine package')
    for name,digest in manifest['files_sha256'].items():
        path=(folder/name).resolve()
        if path.parent!=folder or sha(path)!=digest: raise ValueError('Changed deployment file: '+name)
    report_path=Path(args.report).resolve()
    conv_path=report_path.with_name(report_path.stem+'_conv.json')
    layers_path=report_path.with_name(report_path.stem+'_layers.json')
    if any(p.exists() for p in [report_path,conv_path,layers_path]):
        raise FileExistsError('Use a fresh report name; preserve previous board evidence')
    bit=folder/'lung_network.bit'
    print('1/3: regression of already-verified small convolution cases',flush=True)
    conv=lung_axil_smoke.run(SimpleNamespace(bit=str(bit),report=str(conv_path),timeout=args.timeout))
    print('2/3: regression of all 15 real frozen convolution layers',flush=True)
    layers=lung_model_layers_test.run(SimpleNamespace(bit=str(bit),package=str(folder/'layers.json'),
                                                     report=str(layers_path),timeout=args.timeout))
    print('3/3: loading overlay for independent PL tensor operators',flush=True)
    from pynq import Overlay,allocate
    overlay=Overlay(str(bit))
    print('Overlay loaded; reading tensor ID/ABI',flush=True)
    driver=TensorAxil(overlay.lung_tensor_axil_0.mmio,timeout=args.timeout)
    with np.load(folder/'tensor_vectors.npz',allow_pickle=False) as archive:
        tensors=run_cases(driver,allocate,archive,manifest['tensor_cases'])
    result=dict(status='passed_real_board_dual_engine_regression',package_sha256=sha(package),
                bit_sha256=sha(bit),hwh_sha256=sha(bit.with_suffix('.hwh')),
                convolution_smoke_sha256=sha(conv_path),real_convolution_layers_sha256=sha(layers_path),
                convolution_commands=len(conv['results']),real_convolution_commands=len(layers['results']),
                tensor_results=tensors,continuous_network=False,
                warning='Both PL engines individually checked; whole-network scheduling/continuous inference still not tested')
    report_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Real-board dual-engine regression: PASS',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',default='/home/xilinx/ct_test/lung_network_smoke_result.json')
    parser.add_argument('--timeout',type=float,default=30.0)
    run(parser.parse_args())
