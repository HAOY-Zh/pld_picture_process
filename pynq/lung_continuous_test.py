"""One actual continuous PL U-Net slice, with read-only node comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np

from lung_axil import LungAxil
from lung_continuous import ContinuousUNet, validate_model
from lung_model_layers_test import sha
from lung_tensor import TensorAxil


def load_package(folder):
    folder = Path(folder).resolve(); path = folder/'continuous_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'lung_continuous_board_v1': raise ValueError('Wrong continuous package')
    for name,digest in manifest['files_sha256'].items():
        target = (folder/name).resolve()
        if target.parent != folder or sha(target) != digest: raise ValueError('Changed deployment file: '+name)
    model = json.loads((folder/'model.json').read_text(encoding='utf-8'))
    with np.load(folder/'parameters.npz',allow_pickle=False) as archive:
        parameters = {k:archive[k] for k in archive.files}
    # Inference parameters intentionally contain NO reference intermediate inputs/outputs.
    wanted = {n['layer']['prefix']+'_'+k for n in model['nodes'] if n['kind']=='conv'
              for k in ['weight','bias','multiplier','shift']}
    if set(parameters) != wanted: raise ValueError('Unexpected inference parameter arrays')
    validate_model(model,parameters)
    return manifest,model,parameters


def run(args):
    folder = Path(__file__).resolve().parent
    manifest,model,parameters = load_package(folder)
    report_path = Path(args.report).resolve()
    output_path = report_path.with_name(report_path.stem+'_head.npz')
    if report_path.exists() or output_path.exists(): raise FileExistsError('Choose a fresh report name; preserve evidence')
    with np.load(folder/'input.npz',allow_pickle=False) as archive: image = archive['codes']
    checked = []
    with np.load(folder/'reference.npz',allow_pickle=False) as reference:
        def observe(name,words):
            expected = reference[name]
            different = int(np.count_nonzero(words != expected)) if words.shape == expected.shape else -1
            if different: raise AssertionError('{}: {} FPGA output/reference words differ'.format(name,different))
            checked.append(dict(node=name,output_words=len(words),same=True,different_words=0))
            print('  continuous output/reference: SAME ({} words)'.format(len(words)),flush=True)
        print('Hashes and plan checked. Loading SAME dual-engine bit once...',flush=True)
        from pynq import Overlay,allocate
        bit = folder/'lung_network.bit'; overlay = Overlay(str(bit))
        print('Overlay loaded; checking both accelerator IDs/ABIs...',flush=True)
        conv = LungAxil(overlay.lung_conv8_axil_0.mmio,timeout=args.timeout)
        tensor = TensorAxil(overlay.lung_tensor_axil_0.mmio,timeout=args.timeout)
        network = ContinuousUNet(conv,tensor,allocate,model,parameters)
        codes,stats = network.predict_codes(image,observer=observe)
    if (len(checked) != 21 or stats['convolution_commands'] != 45 or stats['tensor_commands'] != 6
            or stats['command_count'] != 51): raise AssertionError('Incomplete frozen network coverage')
    # Store actual hardware head codes, not reference predictions or a CPU forward.
    with output_path.open('xb') as stream: np.savez_compressed(stream,head_codes=codes)
    result = dict(status='passed_real_board_continuous_unet_codes',continuous_network=True,
                  core_forward_on_pl=True,package_sha256=sha(folder/'continuous_manifest.json'),
                  bit_sha256=sha(bit),hwh_sha256=sha(bit.with_suffix('.hwh')),
                  case=manifest['case'],slice_index=manifest['slice_index'],node_checks=checked,
                  head_codes_sha256=sha(output_path),head_shape=list(codes.shape),**stats,
                  warning='Single validation slice bit-exact integer core forward only. HU preprocessing, sigmoid/native resize, body/3D cleanup, clinical accuracy and sustained FPS are NOT validated by this test')
    with report_path.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print('Real-board continuous U-Net: PASS (21 nodes, 51 commands, all outputs bit-exact)',flush=True)
    print('Actual FPGA head codes saved to: '+str(output_path),flush=True)
    print('Diagnostic slice host time: {:.1f} ms; peak CMA: {:.3f} MiB'.format(stats['host_total_ms'],stats['cma_peak_bytes']/2**20),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',default='/home/xilinx/ct_test/lung_continuous_result.json')
    parser.add_argument('--timeout',type=float,default=30.0)
    run(parser.parse_args())
