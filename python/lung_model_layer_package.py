"""Export one fixed CT slice's 15 isolated real conv layers for the working bit."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from lung_fixed_network import FixedUNet, checked_contract
from lung_fpga_contract import conv_accumulator, requantize
from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256, mapped_npy, read_manifest, torch_module
from verify_lung_axil import check_hashes


def generate(args):
    workspace = Path(__file__).resolve().parents[1]
    proof_path = workspace/'data/lung_axil_board_20261015_resetfix/board_smoke_verification.json'
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    if proof['status'] != 'verified_returned_real_board_single_block_smoke':
        raise ValueError('Verify original returned board smoke first')
    contract_path, contract, tracked = checked_contract(args.contract)
    dataset_path = Path(args.dataset).resolve(); root, dataset = read_manifest(dataset_path)
    record = next(r for r in dataset['cases'] if r['split'] == 'validation')
    cache = root/record['cache_folder']/'images.npy'; index = record['slices']//2
    tracked.update({str(dataset_path):file_sha256(dataset_path), str(cache):file_sha256(cache),
                    str(proof_path):file_sha256(proof_path),
                    str(workspace/'python/lung_fixed_network.py'):file_sha256(workspace/'python/lung_fixed_network.py')})
    torch = torch_module(); torch.set_num_threads(4)
    with mapped_npy(cache) as images:
        if images.shape != (record['slices'],256,256) or images.dtype not in (np.float16,np.float32):
            raise ValueError('Expected original 256x256 normalized CT cache')
        image = torch.from_numpy(np.array(images[index],dtype=np.float32,copy=True))[None,None]
    if not torch.isfinite(image).all(): raise ValueError('Nonfinite cached CT input')
    grid = contract['input_quantization']
    q = torch.quantize_per_tensor(image,grid['scale'],grid['zero_point'],torch.quint8).int_repr()
    with np.load(contract_path.parent/'parameters.npz',allow_pickle=False) as source:
        model = FixedUNet(contract,source,'cpu'); inputs = {}; original_conv = model.conv
        def capture(a,name):
            inputs[name] = a.detach().cpu().numpy()[0].copy()
            return original_conv(a,name)
        model.conv = capture
        with torch.inference_mode(): _, trace = model.forward_codes(q,capture=True)
        arrays = {}; layers = []
        for row in contract['layers']:
            name = row['name']; prefix = name.replace('.','_')
            values = dict(input=inputs[name], weight=source[prefix+'_weight_int8'],
                          bias=source[prefix+'_bias_int32'], multiplier=source[prefix+'_multiplier_int32'],
                          shift=source[prefix+'_right_shift_int32'], expected=trace[name][0])
            # Independent dense INT64 NumPy accumulation, not reuse of Torch output.
            acc = conv_accumulator(values['input'],values['weight'],values['bias'],row['input_zero_point'],row['padding'][0])
            independent = requantize(acc,values['multiplier'],values['shift'],row['output_zero_point'],row['fused_relu'])
            if not np.array_equal(independent,values['expected']):
                raise ValueError('Dense independent reference differs: '+name)
            layers.append(dict(name=name,prefix=prefix,input_shape=list(values['input'].shape),
                               output_shape=list(independent.shape),kernel=values['weight'].shape[-1],
                               padding=row['padding'][0],input_zero=row['input_zero_point'],
                               output_zero=row['output_zero_point'],relu=row['fused_relu']))
            arrays.update({prefix+'_'+key:np.array(value,copy=True) for key,value in values.items()})
            print('Real CT layer reference matched:',name,list(independent.shape),flush=True)
    output = fresh_output(args.outdir); deploy = output/'lung_model_layers'; deploy.mkdir()
    np.savez_compressed(deploy/'vectors.npz',**arrays)
    for name in ['lung_axil.py','lung_axil_smoke.py','lung_model_layers_test.py']:
        shutil.copy2(workspace/'pynq'/name,deploy/name)
        tracked[str(workspace/'pynq'/name)] = file_sha256(workspace/'pynq'/name)
    tracked[str(Path(__file__).resolve())] = file_sha256(__file__)
    check_hashes(tracked)
    manifest = dict(schema='frozen_model_layer_board_v1',case=record['case'],slice_index=index,
                    selection='First existing validation case, middle slice; no expert masks read or used for selection',
                    layers=layers,expected_commands=sum((r['output_shape'][0]+7)//8 for r in layers),
                    output_codes=sum(int(np.prod(r['output_shape'])) for r in layers),
                    bit_sha256=proof['bit_sha256'],hwh_sha256=proof['hwh_sha256'],
                    reference='Frozen complete integer U-Net; every conv independently matched dense NumPy INT64 + Q31 RNE',
                    full_network=False,retraining=False,recalibration=False,test_set_used=False,
                    warning='15 isolated real convolution layers, NOT an end-to-end FPGA U-Net or accuracy evaluation',
                    files_sha256={p.name:file_sha256(p) for p in deploy.iterdir()},source_sha256=tracked)
    write_json(deploy/'layers.json',manifest)
    print('Layer package ready:',deploy,'commands:',manifest['expected_commands'],flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',default='data/lung_fpga_readiness_20261014/integer_contract_v3/contract.json')
    parser.add_argument('--dataset',default='data/lung_learning_20261009/dataset.json')
    parser.add_argument('--outdir',required=True)
    generate(parser.parse_args())
