"""Install the reviewed shared-CNN board runtime using standard-library only."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'runtime' / 'lung_nodule_shared_20261024'
DEFAULT_OUTPUT = ROOT / 'data' / 'lung_nodule_shared_public_20261024'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_relative(name):
    path = PurePosixPath(name)
    if '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe package member')
    if path.parts[0] != 'lung_nodule_shared':
        raise ValueError('Unexpected package root')
    return path


def read_assets(assets=ASSETS):
    assets = Path(assets)
    manifest = json.loads((assets / 'asset_manifest.json').read_text(encoding='utf-8'))
    name = manifest['archive']
    if Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Unsafe archive name')
    archive = assets / name
    if digest(archive.read_bytes()) != manifest['archive_sha256']:
        raise ValueError('Archive hash mismatch')
    payloads = {}
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if len(infos) != len(manifest['files_sha256']):
            raise ValueError('Wrong ZIP member count')
        for item in infos:
            checked_relative(item.filename)
            if item.is_dir() or item.filename in payloads or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Invalid ZIP member')
            data = z.read(item)
            if digest(data) != manifest['files_sha256'].get(item.filename):
                raise ValueError('Package member hash mismatch: ' + item.filename)
            payloads[item.filename] = data
    if set(payloads) != set(manifest['files_sha256']):
        raise ValueError('Incomplete runtime')
    return manifest, payloads


def verify(output=DEFAULT_OUTPUT, assets=ASSETS):
    manifest, _ = read_assets(assets)
    output = Path(output).resolve()
    for name, expected in manifest['files_sha256'].items():
        relative = checked_relative(name)
        p = (output / Path(*relative.parts)).resolve()
        if output not in p.parents or not p.is_file() or digest(p.read_bytes()) != expected:
            raise ValueError('Installed file changed or missing: ' + name)
    return dict(status='verified_synthetic_shared_runtime_files', package=str(output / 'lung_nodule_shared'),
                files_checked=len(manifest['files_sha256']), bit_sha256=manifest['bit_sha256'],
                synthetic_fixtures_board_tested=False)


def install(output=DEFAULT_OUTPUT, assets=ASSETS):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing deployment; use a fresh --output-root')
    _, payloads = read_assets(assets)
    output.mkdir(parents=True)
    for name, data in payloads.items():
        relative = checked_relative(name)
        p = output / Path(*relative.parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as stream:
            stream.write(data)
    return verify(output, assets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'verify'])
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    print(json.dumps((install if args.action == 'install' else verify)(args.output_root), indent=2))
