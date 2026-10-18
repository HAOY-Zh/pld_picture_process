"""Create an independent live demo; NEVER overwrite the accepted release."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pynq'))
from lung_selected_benchmark import validate_selection
from lung_live_service import BIT_SHA, MODEL_SHA


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2)


def prepare(source, outdir):
    source, outdir = Path(source).resolve(), Path(outdir).resolve()
    if outdir.exists(): raise FileExistsError('Use a fresh output folder; accepted releases stay untouched')
    selected, _ = validate_selection(source)
    if sha(source / 'lung_network.bit') != BIT_SHA or sha(source / 'model.json') != MODEL_SHA:
        raise ValueError('Expected the accepted cache-direct-return release')
    proof = source.parent / 'lung_threeway_result_threeway_verification.json'
    acceptance = json.loads(proof.read_text(encoding='utf-8'))
    if acceptance['status'] != 'verified_returned_threeway_actual_fp32_and_int8_heads':
        raise ValueError('Need preserved returned-board acceptance')
    for name, digest in acceptance['returned_files_sha256'].items():
        if (source.parent / name).parent != source.parent or sha(source.parent / name) != digest:
            raise ValueError('Changed returned evidence: ' + name)
    # Byte-identical copy of the accepted package, including nested manifest
    # dependencies. Adds no new bitstream and modifies no frozen helper.
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix not in ('.so', '.dll'))
    before = {p.name: sha(p) for p in files}
    protected = {str(proof): sha(proof)}
    for path in [ROOT/'python/lungseg/volume_viewer.html', ROOT/'python/lungseg/case_browser.html']:
        protected[str(path)] = sha(path)
    outdir.mkdir(parents=True)
    board = outdir / 'lung_live'; board.mkdir()
    for path in files: shutil.copy2(path, board / path.name)
    service = ROOT / 'pynq/lung_live_service.py'
    shutil.copy2(service, board / service.name)
    token = secrets.token_hex(32)
    with (board / 'live_token.txt').open('x', encoding='utf-8') as stream: stream.write(token + '\n')
    with (outdir / 'live_token.txt').open('x', encoding='utf-8') as stream: stream.write(token + '\n')
    shutil.copy2(source / 'model.json', outdir / 'model.json')
    write(board / 'live_manifest.json', dict(schema='lung_live_fpga_v1', accepted_bit_sha256=BIT_SHA,
        files_sha256={p.name: sha(p) for p in board.iterdir() if p.is_file()},
        accepted_evidence_sha256=sha(proof), live_board_integration_tested=False,
        warning='Existing FPGA release accepted; new HTTP/web path still requires board validation.'))
    write(outdir / 'host_config.json', dict(board_url='http://192.168.2.99:8765',
        token_file='live_token.txt', model='model.json'))
    if any(sha(source / name) != digest for name, digest in before.items()):
        raise AssertionError('Accepted package changed')
    if any(sha(path) != digest for path, digest in protected.items()):
        raise AssertionError('Protected acceptance/viewer changed')
    write(outdir / 'preserved_release.json', dict(schema='lung_live_preservation_v1',
        source=str(source), original_files_sha256=before, protected_files_sha256=protected,
        live_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [service,
            ROOT/'python/lung_live_data.py', ROOT/'python/lung_live_web.py',
            ROOT/'python/lung_live_reference.py',
            ROOT/'python/lungseg/live_viewer.html', Path(__file__)]},
        bit_rebuilt=False, old_offline_viewer_modified=False,
        accepted_fpga_mean_ms=acceptance['fpga_int8_ms'], live_board_integration_tested=False))
    print('New board folder: ' + str(board))
    print('Host config: ' + str(outdir / 'host_config.json'))
    print('Accepted release preserved; no synthesis/implementation/bitstream generation.')
    return outdir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'data/lung_threeway_20261016/lung_threeway')
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.outdir)
