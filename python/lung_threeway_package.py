"""Combine audited new hardware and independently verified original FP32 addon."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from lung_segmentation import fresh_output,write_json
from lung_unet import file_sha256
from verify_lung_axil import check_hashes
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pynq'))
from lung_selected_benchmark import validate_selection
from lung_threeway_benchmark import validate


def generate(root,hardware,fp32):
    root=Path(root).resolve();hardware=Path(hardware).resolve();fp32=Path(fp32).resolve()
    workspace=Path(__file__).resolve().parents[1]
    if root.parent!=workspace/'data':raise ValueError('Use fresh workspace/data child')
    p=json.loads((hardware/'verification.json').read_text(encoding='utf-8'))
    if p['status']!='built_routed_cache_return_not_board_tested':raise ValueError('Wrong new hardware proof')
    for k in ['source_sha256','packaged_source_sha256','evidence_sha256']:check_hashes(p[k])
    source=hardware/'lung_cache_return';check_hashes({str(source/n):h for n,h in p['package_files_sha256'].items()})
    f=json.loads((fp32/'fp32_manifest.json').read_text(encoding='utf-8'));check_hashes(f['source_sha256'])
    check_hashes({str(fp32/n):h for n,h in f['files_sha256'].items()})
    root=fresh_output(root).resolve();target=root/'lung_threeway';target.mkdir()
    for item in source.iterdir():
        if not item.is_file():raise ValueError('No host cache in hardware package')
        shutil.copy2(item,target/item.name)
    names=['lung_fp32.py','lung_fp32_native.cpp','lung_fp32_kernel.h','lung_threeway_benchmark.py',
           'fp32_model.npz','fp32_input.npz','fp32_reference.npz']
    for n in names:shutil.copy2(fp32/n,target/n)
    # Original fast/clock/selection content stays byte-identical to hardware
    # package. Actual path is guarded by explicit __file__ at runtime.
    f.update(combined_hardware_proof_sha256=file_sha256(hardware/'verification.json'),
        reference_export_sha256=file_sha256(fp32/'fp32_manifest.json'),
        files_sha256={p.name:file_sha256(p) for p in target.iterdir()})
    write_json(target/'fp32_manifest.json',f);validate_selection(target);validate(target)
    sources=[Path(__file__).resolve(),workspace/'python/verify_lung_threeway_board.py',workspace/'python/tests/test_lung_threeway.py',
             workspace/'python/tests/test_lung_fp32.py']
    write_json(root/'verification.json',dict(status='prepared_threeway_original_fp32_and_routed_fpga_not_board_tested',
        board_speedup_known=False,model_changed=False,hardware_proof_sha256=file_sha256(hardware/'verification.json'),
        fp32_proof_sha256=file_sha256(fp32/'fp32_manifest.json'),source_sha256={str(p):file_sha256(p) for p in sources},
        package_files_sha256={p.name:file_sha256(p) for p in target.iterdir()}))
    print('Threeway upload package:',target,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',default='data/lung_threeway_20261016')
    ap.add_argument('--hardware',default='data/lung_cache_return_20261016_v2')
    ap.add_argument('--fp32',default='data/lung_fp32_reference_20261016_v3/lung_fp32_compare')
    a=ap.parse_args();generate(a.root,a.hardware,a.fp32)
