"""Independently check real layer fixtures, frozen parameters and graph inputs."""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

from lung_fixed_network import checked_contract
from lung_fpga_contract import conv_accumulator, requantize, maxpool2, nearest2
from lung_segmentation import write_json
from lung_unet import file_sha256, mapped_npy, read_manifest, torch_module
from verify_lung_axil import check_hashes

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_model_layers_test import read_layer, sha


def audit(root):
    root=Path(root).resolve(); output=root/'verification.json'
    if output.exists(): raise ValueError('Do not overwrite fixture proof')
    folder=root/'lung_model_layers'; package=folder/'layers.json'
    manifest=json.loads(package.read_text(encoding='utf-8'))
    check_hashes(manifest['source_sha256'])
    check_hashes({str(folder/n):h for n,h in manifest['files_sha256'].items()})
    workspace=Path(__file__).resolve().parents[1]
    path,contract,_=checked_contract(workspace/'data/lung_fpga_readiness_20261014/integer_contract_v3/contract.json')
    dataset_path=workspace/'data/lung_learning_20261009/dataset.json'
    cache_root,dataset=read_manifest(dataset_path)
    record=next(r for r in dataset['cases'] if r['split']=='validation')
    if manifest['case']!=record['case'] or manifest['slice_index']!=record['slices']//2:
        raise ValueError('Fixture case/slice selection changed')
    torch=torch_module(); torch.set_num_threads(4)
    with mapped_npy(cache_root/record['cache_folder']/'images.npy') as images:
        image=torch.from_numpy(np.array(images[manifest['slice_index']],dtype=np.float32,copy=True))[None]
    grid=contract['input_quantization']
    first=torch.quantize_per_tensor(image,grid['scale'],grid['zero_point'],torch.quint8).int_repr().numpy()
    expected_inputs={'enc1.0':first}; outputs={}; codes=commands=0
    if [r['name'] for r in manifest['layers']]!=[r['name'] for r in contract['layers']]:
        raise ValueError('Frozen layer order changed')
    with np.load(folder/'vectors.npz',allow_pickle=False) as archive, np.load(path.parent/'parameters.npz',allow_pickle=False) as parameters:
        for row,frozen in zip(manifest['layers'],contract['layers']):
            name=row['name']; prefix=row['prefix']; values=read_layer(archive,row)
            if name.endswith('.3'): expected_inputs[name]=outputs[name[:-1]+'0']
            elif name in ['enc2.0','enc3.0','neck.0']:
                previous={'enc2.0':'enc1.3','enc3.0':'enc2.3','neck.0':'enc3.3'}[name]
                expected_inputs[name]=maxpool2(outputs[previous])
            elif name in ['dec3.0','dec2.0','dec1.0']:
                number=['dec3.0','dec2.0','dec1.0'].index(name)
                previous=['neck.3','dec3.3','dec2.3'][number]; skip=['enc3.3','enc2.3','enc1.3'][number]
                cat=contract['concat'][number]; branches=[]
                for image,branch in zip([nearest2(outputs[previous]),outputs[skip]],cat['branches']):
                    count=len(image)
                    branches.append(requantize(image.astype(np.int64)-branch['input_zero_point'],
                        np.full(count,branch['multiplier'],np.int64),np.full(count,branch['right_shift'],np.int64),cat['output_zero_point']))
                expected_inputs[name]=np.concatenate(branches,axis=0)
            elif name=='head': expected_inputs[name]=outputs['dec1.3']
            if not np.array_equal(values['input'],expected_inputs.pop(name)):
                raise ValueError('Upstream CT/graph input mismatch: '+name)
            for key,suffix in [('weight','weight_int8'),('bias','bias_int32'),
                               ('multiplier','multiplier_int32'),('shift','right_shift_int32')]:
                if not np.array_equal(values[key],parameters[prefix+'_'+suffix]):
                    raise ValueError('Frozen channel parameter mismatch: '+name)
            if (row['input_shape']!=frozen['input_shape'][1:] or row['output_shape']!=frozen['output_shape'][1:] or
                row['input_zero']!=frozen['input_zero_point'] or row['output_zero']!=frozen['output_zero_point'] or
                row['padding']!=frozen['padding'][0] or row['relu']!=frozen['fused_relu']):
                raise ValueError('Frozen layer metadata mismatch')
            acc=conv_accumulator(values['input'],values['weight'],values['bias'],row['input_zero'],row['padding'])
            computed=requantize(acc,values['multiplier'],values['shift'],row['output_zero'],row['relu'])
            if not np.array_equal(computed,values['expected']): raise ValueError('Independent output mismatch: '+name)
            outputs[name]=computed; codes+=computed.size; commands+=(len(computed)+7)//8
            print('Independent CT/graph/parameter/output check:',name,'PASS',flush=True)
    if codes!=manifest['output_codes'] or commands!=manifest['expected_commands']: raise ValueError('Coverage totals differ')
    raw=(root/'python_tests.log').read_bytes()
    tests=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    if not re.search(r'Ran 198 tests in .*?\n\s*\nOK\s*$',tests): raise ValueError('Full regression proof absent')
    check_hashes(manifest['source_sha256'])
    result=dict(status='prepared_verified_real_model_layer_fixtures_not_board_tested',layers=len(outputs),commands=commands,
                output_codes=int(codes),independent_ct_graph_parameters_outputs_match=True,
                package_sha256=sha(package),python_tests=198,python_tests_sha256=sha(root/'python_tests.log'),
                source_sha256=manifest['source_sha256'],files_sha256=manifest['files_sha256'],
                verifier_sha256=file_sha256(__file__),full_network_on_fpga=False,model_layers_board_tested=False)
    write_json(output,result); print('Real layer package verified:',len(outputs),'layers,',commands,'commands,',codes,'codes',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_model_layers_20261015')
    audit(parser.parse_args().root)
