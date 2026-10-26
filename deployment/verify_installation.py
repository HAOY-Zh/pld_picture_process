"""Read-only acceptance of a freshly generated PC/board pair. No FPGA requests."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT/'data/ct_studio_integration_20261025')
    parser.add_argument('--board', action='store_true', help='Also read actual legacy/shared service identities')
    args = parser.parse_args()
    try:
        if sys.version_info[:2] != (3, 11):
            raise ValueError('Use the documented PC Python 3.11 environment')
        versions = {}
        for name in ['numpy', 'scipy', 'pydicom', 'matplotlib', 'nibabel', 'skimage', 'plotly', 'torch']:
            module = importlib.import_module(name)
            versions[name] = module.__version__
        from ct_studio_integration_release import verify
        from start_ct_studio_demo import configuration
        from ldct_contract import validate_release
        from nodule_detector import NoduleCaseProvider
        package = args.package.resolve()
        verify(package)
        config = json.loads((package/'host_config.json').read_text('utf-8'))
        client, manifest = configuration(package/'host_config.json')
        release = Path(config['ldct_release'])
        if not release.is_absolute():
            release = ROOT/release
        if validate_release(release)[1] != manifest['ldct']:
            raise ValueError('LDCT release/host identity mismatch')
        nodule = Path(config['nodule_package'])
        if not nodule.is_absolute():
            nodule = ROOT/nodule
        if nodule.resolve() != package:
            raise ValueError('Nodule asset path does not refer to this paired installation')
        provider = NoduleCaseProvider(package, client)
        if args.board:
            client.health()
            provider.client.health()
        print(json.dumps(dict(status='PASS_PAIRED_BOARD_READINESS' if args.board else 'PASS_LOCAL_DEPENDENCIES_AND_ASSETS',
            dependencies=versions, board_identity_checked=args.board,
            inference_submitted=False, credentials_printed=False), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print('CHECK FAILED: '+str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
