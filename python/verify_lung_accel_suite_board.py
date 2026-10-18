"""Independently verify both returned threeway heads, clocks and suite arithmetic."""
import argparse
import json
from pathlib import Path
import sys
from lung_unet import file_sha256
from lung_segmentation import write_json
from verify_lung_axil import check_hashes
from verify_lung_threeway_combo import verify as verify_model
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_accel_suite import validate


def run(root,package):
    root=Path(root).resolve();package=Path(package).resolve();manifest=validate(package)
    dest=root/'suite_verification.json'
    if dest.exists():raise FileExistsError('Preserve previous suite verification')
    record=json.loads((root/'suite_result.json').read_text(encoding='utf-8'))
    if record['status']!='completed_six_way_board_measurement_pending_independent_return_audit':
        raise ValueError('Not a completed suite')
    if (record['suite_manifest_sha256']!=file_sha256(package/'suite_manifest.json') or
            record['bit_sha256']!=manifest['bit_sha256'] or set(record['models'])!={'original','pruned'}):
        raise ValueError('Different suite identity')
    check_hashes({str(root/n):h for n,h in record['files_sha256'].items()})
    proofs={}
    for label in manifest['models']:
        raw=json.loads((root/(label+'_threeway.json')).read_text(encoding='utf-8'))
        if raw!=record['models'][label]:raise ValueError('Different per-model summary')
        verify_model(root,package/label,label)
        proofs[label]=file_sha256(root/(label+'_threeway_verification.json'))
    a,b=record['models']['original'],record['models']['pruned']
    for key in ['fp32_ms','cpu_int8_ms','fpga_int8_ms']:
        if abs(record['pruning_latency_reduction'][key]-(1-b[key]/a[key]))>1e-12:
            raise ValueError('Different pruning arithmetic')
    proof=dict(status='verified_returned_all_acceleration_six_way_board_results',models=record['models'],
        pruning_latency_reduction=record['pruning_latency_reduction'],bit_sha256=record['bit_sha256'],
        suite_result_sha256=file_sha256(root/'suite_result.json'),model_proofs_sha256=proofs,
        verifier_sha256=file_sha256(__file__),
        scope='One shared CT slice repeated; not all-case board Dice or web end-to-end speed')
    write_json(dest,proof);print(json.dumps(proof,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--package',default='data/lung_all_accel_20261017_v2/lung_all_accel')
    a=p.parse_args();run(a.root,a.package)
