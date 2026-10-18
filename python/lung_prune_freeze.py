"""Freeze complete candidate files and cross-check PTQ/contract/checkpoint identity."""
import argparse
import json
from pathlib import Path

import numpy as np

from lung_fpga_contract import multiplier
from lung_prune import restore_candidate,restore_pruned_quantized
from lung_segmentation import write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes


def run(root):
    root=Path(root).resolve()
    if (root/'freeze.json').exists():raise FileExistsError('Preserve candidate freeze')
    protocol=json.loads((root/'protocol.json').read_text());proof=json.loads((root/'verification.json').read_text())
    if proof['status']!='complete_pruning_development_experiment' or proof['protocol_sha256']!=file_sha256(root/'protocol.json'):
        raise ValueError('Incomplete or changed experiment')
    check_hashes(protocol['input_sha256']);check_hashes(protocol['source_sha256'])
    for result in proof['results']:
        folder=root/result['candidate'];original,saved=restore_candidate(folder/'best.pt')
        model,quantized=restore_pruned_quantized(folder/'int8_model.pt');modules=dict(model.named_modules())
        if quantized['source_checkpoint_sha256']!=file_sha256(folder/'best.pt') or quantized['kept']!=saved['kept']:
            raise ValueError('Quantized checkpoint identity differs')
        contract=json.loads((folder/'contract.json').read_text());core=json.loads((folder/'core_manifest.json').read_text())
        if contract['checkpoint_sha256']!=file_sha256(folder/'best.pt'):raise ValueError('Contract checkpoint changed')
        if [r['name'] for r in contract['layers']]!=[r['name'] for r in core['layers']]:raise ValueError('Different graph order')
        with np.load(folder/'parameters.npz') as parameters,np.load(folder/'int8_core_parameters.npz') as exported:
            for row in contract['layers']:
                name=row['name'];prefix=name.replace('.','_');module=modules[name];weight=module.weight()
                np.testing.assert_array_equal(weight.int_repr().numpy(),parameters[prefix+'_weight_int8'])
                np.testing.assert_array_equal(weight.int_repr().numpy(),exported[prefix+'_weight_int8'])
                np.testing.assert_array_equal(weight.q_per_channel_scales().numpy(),exported[prefix+'_weight_scale'])
                if list(original.get_submodule(name).weight.shape)!=row['weight_shape'] or module.scale!=row['output_scale'] or module.zero_point!=row['output_zero_point']:
                    raise ValueError('Graph/grid shape differs')
                scales=weight.q_per_channel_scales().numpy();bias=np.rint(module.bias().detach().numpy()/(row['input_scale']*scales)).astype(np.int32)
                m,s=multiplier(row['input_scale']*scales/row['output_scale'])
                for key,value in [('bias_int32',bias),('multiplier_int32',m),('right_shift_int32',s)]:
                    np.testing.assert_array_equal(parameters[prefix+'_'+key],value)
        for row in json.loads((folder/'fixed_evaluation.json').read_text())['cases']:
            if file_sha256(folder/row['case']/'fixed_native.npz')!=row['mask_sha256']:raise ValueError('Fixed validation mask changed')
    files={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
    write_json(root/'freeze.json',dict(status='frozen_pruning_candidate_identity_verified',files_sha256=files,
        fixed_graph_and_quantized_weights_verified=True,original_inputs_unchanged=True,
        test_or_covid_used=False,board_tested=False,source_sha256={str(Path(__file__).resolve()):file_sha256(__file__)}))
    print('Frozen candidate identity and',len(files),'files:',root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    run(parser.parse_args().root)
