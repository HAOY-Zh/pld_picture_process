"""Create a two-file selection addon without modifying the routed word package."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'pynq'))
from lung_selected_benchmark import sha,write_new,validate_selection,selected_entrypoints


def prepare(root):
    root=Path(root).resolve()
    if root.exists(): raise FileExistsError('Preserve prior selection package')
    source=WORKSPACE/'data/lung_word_20261015/lung_word'
    proof=json.loads((source.parent/'verification.json').read_text(encoding='utf-8'))
    for name,digest in proof['package_files_sha256'].items():
        if sha(source/name)!=digest: raise ValueError('Changed routed package: '+name)
    if proof['routed_design']['bit_sha256']!=sha(source/'lung_network.bit'):
        raise ValueError('Wrong word bit')
    root.mkdir(parents=True);deploy=root/'deploy';deploy.mkdir()
    shutil.copy2(WORKSPACE/'pynq/lung_selected_benchmark.py',deploy/'lung_selected_benchmark.py')
    expected={n:sha(source/n) for n in ['fast_manifest.json','clock_manifest.json','lung_network.bit','lung_network.hwh']}
    expected['lung_selected_benchmark.py']=sha(deploy/'lung_selected_benchmark.py')
    write_new(deploy/'selection_manifest.json',dict(schema='lung_pinned_entrypoint_v1',files_sha256=expected,
        target_folder='/home/xilinx/ct_test/lung_word',warning='Fresh file-rooted entries; preserve driver modules and all existing board reports. No new bit or model. Board timing still pending.'))
    # Temporary local staging is NOT a board run. The routed source package
    # itself is never changed; only temporary copies are automatically removed.
    from lung_selected_benchmark import load_file
    old=load_file('_old_benchmark',WORKSPACE/'data/lung_fast_20261015/lung_fast/lung_fast_benchmark.py')
    with tempfile.TemporaryDirectory(prefix='local_check_',dir=root) as tmp:
        staging=Path(tmp)
        for p in source.iterdir(): shutil.copy2(p,staging/p.name)
        for p in deploy.iterdir(): shutil.copy2(p,staging/p.name)
        _,fast=validate_selection(staging)
        # Bind only the public entry alias; preserve imported driver owners.
        previous=sys.modules.get('lung_fast_benchmark');sys.modules['lung_fast_benchmark']=old
        try:
            with selected_entrypoints(staging,fast) as (bench,clocks,origins):
                clocks.validate_addon(staging)
                if Path(bench.__file__).parent!=staging or clocks.benchmark is not bench:
                    raise AssertionError('Pinned selected package failed')
                binding=dict(old_cached_benchmark=str(Path(old.__file__).resolve()),selected_benchmark=str(Path(bench.__file__).resolve()),
                    temporary_staging_removed_after_check=True)
        finally:
            if previous is None: sys.modules.pop('lung_fast_benchmark',None)
            else: sys.modules['lung_fast_benchmark']=previous
    write_new(root/'verification.json',dict(status='verified_local_pinned_entrypoint_not_board_tested',
        board_tested=False,bit_changed=False,model_changed=False,local_cached_import_reproduction=binding,
        routed_package_sha256=sha(source.parent/'verification.json'),selected_bit_sha256=expected['lung_network.bit'],
        deploy_sha256={p.name:sha(p) for p in deploy.iterdir()},
        source_sha256={str(p):sha(p) for p in [Path(__file__),WORKSPACE/'pynq/lung_selected_benchmark.py',
            WORKSPACE/'python/tests/test_lung_selected_benchmark.py']}))
    print('Selection addon verified:',deploy,flush=True)
    print('Upload ONLY lung_selected_benchmark.py and selection_manifest.json into existing lung_word.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lung_selected_20261015')
    prepare(parser.parse_args().root)
