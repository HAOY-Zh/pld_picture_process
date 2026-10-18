"""Install and verify the public CT Studio assets. Never touches FPGA hardware."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path, PurePosixPath
import secrets
import sys
import zipfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'runtime/ct_studio_20261018'
VARIANTS = {'classic': ('ct_platform_20261017_v2', 'ct_platform'),
            'fast': ('ct_platform_fast_20261017_v2', 'ct_platform_fast')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def check_files(folder, hashes):
    folder = Path(folder).resolve()
    for name, digest in hashes.items():
        path = (folder / name).resolve()
        if folder not in path.parents or not path.is_file() or sha(path) != digest:
            raise ValueError('Missing/changed asset: ' + name)


def validate_board_url(url):
    parts = urlsplit(url)
    address = ipaddress.ip_address(parts.hostname or '')
    if (parts.scheme != 'http' or not (address.is_private or address.is_loopback)
            or parts.path not in ('', '/') or parts.query or parts.fragment or parts.username or parts.password):
        raise ValueError('Use http://PRIVATE_BOARD_IP:8765 with no credentials/path')
    _ = parts.port
    return url.rstrip('/')


def install(output, board_url):
    output = Path(output).resolve()
    board_url = validate_board_url(board_url)
    release = read(RELEASE / 'asset_manifest.json')
    archive = RELEASE / release['archive']
    if sha(archive) != release['archive_sha256']:
        raise ValueError('Public runtime archive SHA256 mismatch')
    targets = [output / name for name, _ in VARIANTS.values()] + [output / 'ct_studio_demo']
    if any(p.exists() for p in targets):
        raise FileExistsError('Preserve existing deployments; use a new --output-root or fresh clone')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(release['files_sha256']):
            raise ValueError('Unexpected runtime archive members')
        # Validate the entire archive before creating any output; no generic extractall.
        for item in z.infolist():
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in item.filename
                    or ':' in item.filename or item.is_dir()
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe archive path')
            if hashlib.sha256(z.read(item)).hexdigest() != release['files_sha256'][item.filename]:
                raise ValueError('Runtime member SHA256 mismatch: ' + item.filename)
            destination = (output / item.filename).resolve()
            if output not in destination.parents or destination.exists():
                raise ValueError('Output must be new and contained under output root')
        for name in names:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(z.read(name))
    # One installation owns one private credential; never print it or commit it.
    token = secrets.token_hex(32) + '\n'
    for name, board_name in VARIANTS.values():
        host = output / name
        board = host / board_name
        seg = board / 'segmentation'
        for path in (host / 'platform_token.txt', board / 'platform_token.txt', seg / 'live_token.txt'):
            with path.open('x', encoding='utf-8') as stream:
                stream.write(token)
        live = read(seg / 'live_manifest.json')
        live['files_sha256']['live_token.txt'] = sha(seg / 'live_token.txt')
        write(seg / 'live_manifest.json', live)
        platform = read(board / 'platform_manifest.json')
        platform['files_sha256'] = {p.relative_to(board).as_posix(): sha(p)
            for p in sorted(board.rglob('*')) if p.is_file() and p != board / 'platform_manifest.json'}
        write(board / 'platform_manifest.json', platform)
        host_manifest = read(host / 'host_platform.json')
        host_manifest['platform_manifest_sha256'] = sha(board / 'platform_manifest.json')
        write(host / 'host_platform.json', host_manifest)
        config = read(host / 'host_config.json')
        config['board_url'] = board_url
        write(host / 'host_config.json', config)
    write(output / 'ct_studio_demo/installation.json', dict(schema='ct_studio_local_install_v1',
        public_archive_sha256=release['archive_sha256'], board_url=board_url,
        credential_generated_locally=True, board_tested=False))
    print('CT STUDIO ASSETS INSTALLED (no hardware accessed; credentials not printed)')
    print('Demo:', output / 'ct_studio_demo/synthetic_ct_512.npz')
    for variant, (name, _) in VARIANTS.items():
        print(variant + ' config:', output / name / 'host_config.json')


def verify(output, variant):
    """Strictly validate deployment AND all nested original driver manifests."""
    name, board_name = VARIANTS[variant]
    host = Path(output).resolve() / name
    board = host / board_name
    platform = read(board / 'platform_manifest.json')
    check_files(board, platform['files_sha256'])
    seg = board / 'segmentation'
    for filename in ('continuous_manifest.json', 'cpu_compare_manifest.json', 'fast_manifest.json',
                     'clock_manifest.json', 'selection_manifest.json', 'live_manifest.json'):
        check_files(seg, read(seg / filename)['files_sha256'])
    fast = read(seg / 'fast_manifest.json')
    cpu = read(seg / 'cpu_compare_manifest.json')
    clocks = read(seg / 'clock_manifest.json')
    if (fast['original_manifest_sha256'] != sha(seg / 'continuous_manifest.json')
            or cpu['original_continuous_package_sha256'] != sha(seg / 'continuous_manifest.json')
            or clocks['original_fast_manifest_sha256'] != sha(seg / 'fast_manifest.json')):
        raise ValueError('Nested manifest relationship mismatch')
    config = read(host / 'host_config.json')
    validate_board_url(config['board_url'])
    manifest = read(host / config['platform_manifest'])
    if manifest['platform_manifest_sha256'] != sha(board / 'platform_manifest.json'):
        raise ValueError('Host/board deployment mismatch')
    if sha(host / config['model']) != sha(seg / 'model.json') or manifest['basic'] != platform['basic']:
        raise ValueError('Host/board model or basic bit mismatch')
    token = (host / config['token_file']).read_text(encoding='utf-8').strip()
    if len(token) < 32 or token != (board / 'platform_token.txt').read_text(encoding='utf-8').strip():
        raise ValueError('Private host/board credentials mismatch')
    # Execute the SAME pre-download validators used on PYNQ; NumPy only.
    sys.path[:0] = [str(board), str(seg)]
    from ct_platform_service import validate_package
    from lung_live_service import validate_live
    from lung_selected_benchmark import validate_selection
    from lung_fast_benchmark import validate_package as validate_network
    from lung_clock_benchmark import validate_addon
    validate_package(board)
    validate_live(seg)
    validate_selection(seg)
    validate_addon(seg)
    _, model, _, _ = validate_network(seg)
    print('CT STUDIO ' + variant.upper() + ' VERIFY PASS: all nested hashes, 21-node/51-command model; no FPGA accessed')
    return dict(variant=variant, files=len(platform['files_sha256']), nodes=len(model['nodes']),
                commands=model['commands'], hardware_accessed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'verify'))
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--board-url', default='http://192.168.2.99:8765')
    parser.add_argument('--variant', choices=tuple(VARIANTS), default='classic')
    args = parser.parse_args()
    if args.action == 'install':
        install(args.output_root, args.board_url)
    else:
        verify(args.output_root, args.variant)


if __name__ == '__main__':
    main()
