"""Prepare a NEW continuous-network package without changing any frozen artifact."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from lung_fixed_network import checked_contract
from lung_segmentation import fresh_output, write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_axil_smoke import feature_words
from lung_continuous import validate_model
from lung_continuous_test import load_package


def make_plan(layers,contract):
    by_name = {r['name']:r for r in layers}; nodes = []
    def conv(name,source):
        row = by_name[name]
        nodes.append(dict(name=name,kind='conv',sources=[source],shape=row['output_shape'],zero=row['output_zero'],layer=row))
    def pool(number,source):
        row = by_name[source]; c,h,w = row['output_shape']; name = 'pool'+str(number)
        nodes.append(dict(name=name,kind='pool',sources=[source],shape=[c,h//2,w//2],zero=row['output_zero'],
                          quant=dict(za=row['output_zero'],zo=row['output_zero'])))
        return name
    source = 'input'
    for number,prefix in enumerate(['enc1','enc2','enc3','neck']):
        conv(prefix+'.0',source); conv(prefix+'.3',prefix+'.0'); source = prefix+'.3'
        if number < 3: source = pool(number,source)
    for number,(prefix,skip) in enumerate([('dec3','enc3.3'),('dec2','enc2.3'),('dec1','enc1.3')]):
        cat = contract['concat'][number]; a,b = cat['branches']; name = 'cats.'+str(number)
        if [a['source'],b['source']] != [source,skip]: raise ValueError('Frozen concat branch order changed')
        quant = dict(za=a['input_zero_point'],zb=b['input_zero_point'],zo=cat['output_zero_point'],
                     ma=a['multiplier'],sa=a['right_shift'],mb=b['multiplier'],sb=b['right_shift'])
        nodes.append(dict(name=name,kind='fused_concat',sources=[source,skip],shape=by_name[prefix+'.0']['input_shape'],
                          zero=cat['output_zero_point'],quant=quant))
        conv(prefix+'.0',name); conv(prefix+'.3',prefix+'.0'); source = prefix+'.3'
    conv('head',source)
    return dict(schema='lung_continuous_plan_v1',input_shape=by_name['enc1.0']['input_shape'],
                input_zero=contract['input_quantization']['zero_point'],input_quantization=contract['input_quantization'],
                output_scale=contract['layers'][-1]['output_scale'],output_zero=by_name['head']['output_zero'],
                nodes=nodes,commands=sum((r['output_shape'][0]+7)//8 for r in layers)+6,
                storage='HWC8 little-endian; device DDR buffers with 16 suffix guard words',
                core='15 frozen conv + 3 maxpool + 3 fused nearest/requant-concat, no CPU operator fallback')


def generate(args):
    workspace = Path(__file__).resolve().parents[1]; board = Path(args.board).resolve()
    proof_path = board/'verification.json'; proof = json.loads(proof_path.read_text(encoding='utf-8'))
    if proof['status'] != 'built_routed_dual_engine_regression_not_board_tested': raise ValueError('Wrong routed board proof')
    source = board/'lung_network'; original = source/'network_manifest.json'
    if file_sha256(original) != proof['package_sha256']: raise ValueError('Board package proof mismatch')
    dual = json.loads(original.read_text(encoding='utf-8'))
    check_hashes({str(source/n):h for n,h in dual['files_sha256'].items()})
    path,contract,tracked = checked_contract(args.contract)
    layers = json.loads((source/'layers.json').read_text(encoding='utf-8'))
    check_hashes(layers['source_sha256'])
    model = make_plan(layers['layers'],contract); parameters = {}; reference = {}
    next_input = {'pool0':'enc2_0','pool1':'enc3_0','pool2':'neck_0',
                  'cats.0':'dec3_0','cats.1':'dec2_0','cats.2':'dec1_0'}
    with np.load(source/'vectors.npz',allow_pickle=False) as old, np.load(path.parent/'parameters.npz',allow_pickle=False) as frozen:
        image = old['enc1_0_input']
        for node in model['nodes']:
            if node['kind'] == 'conv':
                row = node['layer']; prefix = row['prefix']
                for key,suffix in [('weight','weight_int8'),('bias','bias_int32'),('multiplier','multiplier_int32'),('shift','right_shift_int32')]:
                    value = old[prefix+'_'+key]
                    if not np.array_equal(value,frozen[prefix+'_'+suffix]): raise ValueError('Frozen parameter mismatch')
                    parameters[prefix+'_'+key] = value
                value = old[prefix+'_expected']
            else: value = old[next_input[node['name']]+'_input']
            reference[node['name']] = feature_words(value,node['zero'])
    validate_model(model,parameters)
    output = fresh_output(args.outdir); target = output/'lung_continuous'; target.mkdir()
    for name in ['lung_axil.py','lung_axil_smoke.py','lung_model_layers_test.py','lung_tensor.py',
                 'lung_continuous.py','lung_continuous_test.py']:
        shutil.copy2(workspace/'pynq'/name,target/name)
        tracked[str(workspace/'pynq'/name)] = file_sha256(workspace/'pynq'/name)
    for name in ['lung_network.bit','lung_network.hwh']: shutil.copy2(source/name,target/name)
    np.savez_compressed(target/'parameters.npz',**parameters)
    np.savez_compressed(target/'input.npz',codes=image)
    np.savez_compressed(target/'reference.npz',**reference)
    write_json(target/'model.json',model)
    tracked.update(layers['source_sha256'])
    for file in [original,proof_path,source/'vectors.npz',source/'layers.json',Path(__file__).resolve()]:
        tracked[str(file)] = file_sha256(file)
    check_hashes(tracked)
    manifest = dict(schema='lung_continuous_board_v1',case=layers['case'],slice_index=layers['slice_index'],
                    selection=layers['selection'],retraining=False,recalibration=False,test_set_used=False,
                    status='prepared_continuous_network_not_board_tested',
                    board_regression_evidence='User screenshot: Real-board dual-engine regression PASS; raw JSON not yet returned',
                    files_sha256={p.name:file_sha256(p) for p in target.iterdir()},source_sha256=tracked,
                    board_proof_sha256=file_sha256(proof_path),
                    warning='Host scheduling prepared; only an actual new continuous real-board run establishes end-to-end PL core correctness')
    write_json(target/'continuous_manifest.json',manifest)
    load_package(target)
    print('Prepared continuous network:',target,'nodes',len(model['nodes']),'commands',model['commands'],flush=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board',default='data/lung_network_board_20261015')
    parser.add_argument('--contract',default='data/lung_fpga_readiness_20261014/integer_contract_v3/contract.json')
    parser.add_argument('--outdir',required=True)
    generate(parser.parse_args())
