"""Build a separate, hash-bound PC/board package; never edit the old delivery."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def replace_once(path, old, new):
    text = path.read_text(encoding='utf8')
    if text.count(old) != 1:
        raise ValueError('Source anchor differs: '+str(path))
    path.write_text(text.replace(old, new), encoding='utf8')


def main():
    source = ROOT/'data/CT_Studio_PC_Paired_20261025_01'
    target = ROOT/'data/CT_Studio_SharedSeg_20261025_01'
    target.mkdir(exist_ok=False)
    for name in ('python', 'pynq', 'configs'):
        shutil.copytree(source/name, target/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    runtime_name = 'data/ct_studio_integration_20261025'
    package = target/runtime_name
    shutil.copytree(source/runtime_name, package, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'build'))
    for name in ('shared_segmentation_contract.py', 'ct_shared_seg_client.py'):
        shutil.copy2(ROOT/'python'/name, target/'python'/name)
    board = package/'ct_studio'
    shutil.copy2(ROOT/'python/shared_segmentation_contract.py', board/'shared_segmentation_contract.py')
    shutil.copy2(ROOT/'pynq/ct_shared_seg_service.py', board/'ct_shared_seg_service.py')
    owner = board/'ct_shared_seg_owner.py'
    owner.write_text((board/'ct_studio_shared_owner.py').read_text().replace(
        'from ct_studio_shared_service import start', 'from ct_shared_seg_service import start'), encoding='utf8')
    from shared_segmentation_contract import descriptor, digest
    manifest = read(board/'platform_manifest.json')
    manifest['shared_segmentation'] = descriptor(board)
    for name in ('shared_segmentation_contract.py', 'ct_shared_seg_service.py', 'ct_shared_seg_owner.py'):
        manifest['files_sha256'][name] = digest((board/name).read_bytes())
    write(board/'platform_manifest.json', manifest)
    bundle = digest((board/'platform_manifest.json').read_bytes())
    host = read(package/'host_platform.json')
    host.update(shared_segmentation=manifest['shared_segmentation'], service_bundle_sha256=bundle)
    write(package/'host_platform.json', host)
    shared = read(package/'shared_host_platform.json')
    shared.update(service_bundle_sha256=bundle, shared_segmentation=manifest['shared_segmentation'])
    write(package/'shared_host_platform.json', shared)
    config = read(package/'host_config.json')
    config['workspace_output'] = str((source/'data/ct_workspace_demo/results').resolve())
    write(package/'host_config.json', config)
    for filename in ('start_ct_studio_demo.py', 'ct_workspace_web.py'):
        path = target/'python'/filename
        if filename == 'start_ct_studio_demo.py':
            replace_once(path,
                "client=ROIStudioClient(config['board_url'],token,manifest)",
                "from ct_shared_seg_client import SharedSegStudioClient\n        client=(SharedSegStudioClient if manifest.get('shared_segmentation') else ROIStudioClient)(config['board_url'],token,manifest)")
        else:
            replace_once(path,
                "client=ROIStudioClient(config['board_url'],token,manifest)",
                "from ct_shared_seg_client import SharedSegStudioClient\n                client=(SharedSegStudioClient if manifest.get('shared_segmentation') else ROIStudioClient)(config['board_url'],token,manifest)")
    replace_once(target/'python/ct_workspace_web.py', "            value = super().snapshot()",
        "            value = super().snapshot()\n            value['segmentation_engine'] = self.client.manifest.get('shared_segmentation') if self.client is not None and hasattr(self.client, 'manifest') else None")
    path = target/'python/start_ct_studio_demo.py'
    replace_once(path, "    with (folder/'web.stdout.log').open('xb') as out,(folder/'web.stderr.log').open('xb') as err:",
        "    workspace_output = ROOT/('data/ct_workspace_pc/results' if desktop else 'data/ct_workspace_demo/results')\n"
        "    if not desktop:\n        setup=json.loads(Path(config).read_text(encoding='utf8'))\n"
        "        workspace_output=Path(setup.get('workspace_output',workspace_output))\n"
        "    with (folder/'web.stdout.log').open('xb') as out,(folder/'web.stderr.log').open('xb') as err:")
    replace_once(path, "str(ROOT/('data/ct_workspace_pc/results' if desktop else 'data/ct_workspace_demo/results'))", "str(workspace_output)")
    # Record real per-slice engine identity, including mixed old/new saved slices.
    path = target/'python/lung_live_web.py'
    replace_once(path, "            write_new(self.case_folder / (job_id + '_volume.json'), dict(schema='lung_live_volume_v1',",
        "            bit_ids = sorted({json.loads((self.case_folder / rid / 'receipt.json').read_text(encoding='utf8'))['bit_sha256'] for rid in self.receipts.values()})\n"
        "            write_new(self.case_folder / (job_id + '_volume.json'), dict(schema='lung_live_volume_v1',")
    replace_once(path, "                backend='FPGA_INT8', bit_sha256=BIT_SHA, model_sha256=MODEL_SHA,",
        "                backend='FPGA_INT8', bit_sha256=bit_ids[0] if len(bit_ids)==1 else None, bit_sha256s=bit_ids, model_sha256=MODEL_SHA,")
    # An explicitly restored legacy result stays legacy; new results keep new IDs.
    path = target/'python/ct_workspace_resume.py'
    replace_once(path, "meta['bit_sha256']!=BIT_SHA", "not set(meta.get('bit_sha256s') or [meta['bit_sha256']]).issubset({BIT_SHA, getattr(c.client,'segmentation_bit_sha256',BIT_SHA)})")
    # Cancelled/stale nodule jobs are not republished by the existing restorer.
    write(target/'upgrade_provenance.json', dict(schema='ct_shared_seg_web_upgrade_v1',
        source=str(source), destination=str(target), service_bundle_sha256=bundle,
        original_models_unchanged=True, shared_segmentation=manifest['shared_segmentation'],
        model_selection='unet_original', legacy_routes_preserved=True))
    print(json.dumps(dict(target=str(target), board_folder=str(board), bundle_sha256=bundle), indent=2))


if __name__ == '__main__':
    main()
