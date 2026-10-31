"""Export the current accepted app into the separate GitHub publishing checkout.

Does not touch the active app, the board, local datasets or its private tokens.
The exported runtime has analytic self-test images and per-install credentials.
"""
from pathlib import Path
import ast
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT/'data/CT_Studio_DualExact_20261026_02'
REPO = ROOT/'data/github_integrated_review_20261025'


def replace_once(s, before, after):
    if s.count(before) != 1:
        raise ValueError('Unexpected source anchor: '+before[:90])
    return s.replace(before, after)


def main():
    for name in ('python', 'pynq', 'configs'):
        shutil.copytree(APP/name, REPO/name, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    extras = ('dual_io_contract.py', 'ct_dual_io_client.py', 'prepare_dual_io_web.py',
        'benchmark_dual_seg_web.py', 'benchmark_dual_nodule_web.py',
        'benchmark_dual_wire_pairs.py', 'verify_dual_exact_transport.py',
        'deploy_dual_exact_board.py', 'integrate_desktop_dual_exact.py',
        'finalize_desktop_dual.py', 'verify_desktop_current_case.py')
    for name in extras:
        shutil.copy2(ROOT/'python'/name, REPO/'python'/name)
    shutil.copy2(ROOT/'python/tests/test_dual_exact_io.py', REPO/'python/tests/test_dual_exact_io.py')
    for name in ('ct_dual_io_service.py', 'exact_io_runtime.py', 'exact_native_io.py', 'exact_native_io.cpp'):
        shutil.copy2(ROOT/'pynq'/name, REPO/'pynq'/name)
    for name in ('肺分割与结节检测无损调度优化_20261026.md', '桌面exe接入最新版_20261026.md',
        'LDCT耗时核对_20261026.md', '结节搬运与通信无损优化_20261026.md'):
        shutil.copy2(ROOT/'docs'/name, REPO/'docs'/name)
    board = APP/'data/ct_studio_integration_20261025/ct_studio'
    shutil.copytree(board/'shared_cnn/rtl', REPO/'hardware/ct_studio_current_20261026/shared_cnn',
        dirs_exist_ok=True)

    # Preserve the old public installer. This new entry exports exactly the
    # latest service files and adds the previously external frozen score files.
    source = (REPO/'python/ct_studio_integration_release.py').read_text(encoding='utf8')
    source = source.replace("runtime/ct_studio_integration_20261025", "runtime/ct_studio_current_20261026")
    source = source.replace("data/ct_studio_integration_20261025", "data/ct_studio_current_20261026")
    source = replace_once(source, "'ldct/fixtures/','ldct/source/','ldct/verification/'",
        "'ldct/fixtures/','ldct/source/','ldct/verification/','ldct/samples/'")
    source = replace_once(source, "    manifest=bind_manifests(board)\n", """    for name in ('nodule_scoring.json','nodule_ranker.json'):
        shutil.copy2(source.parent/name,staging/name)
    manifest=bind_manifests(board)
""")
    source = replace_once(source, "for p in sorted(board.rglob('*')):\n            if p.is_file():z.write(p,p.relative_to(staging).as_posix())",
        "for p in sorted(staging.rglob('*')):\n            if p.is_file():z.write(p,p.relative_to(staging).as_posix())")
    source = replace_once(source,
        "files={p.relative_to(staging).as_posix():sha(p)for p in sorted(board.rglob('*'))if p.is_file()}",
        "files={p.relative_to(staging).as_posix():sha(p)for p in sorted(staging.rglob('*'))if p.is_file()}")
    source = replace_once(source, "    write(out/'host_platform.json',dict(legacy,platform_manifest_sha256=legacy_sha))\n",
        """    current_fields={k:manifest[k]for k in ('shared_nodule','shared_batch','shared_segmentation',
        'canvas_interface','compact_interface','dual_io_interface')}
    write(out/'host_platform.json',dict(legacy,platform_manifest_sha256=legacy_sha,
        service_bundle_sha256=sha(board/'platform_manifest.json'),**current_fields))
""")
    source = replace_once(source,
        "legacy_contract_sha256=legacy_sha,shared_nodule=manifest['shared_nodule'],shared_batch=manifest['shared_batch']))",
        "legacy_contract_sha256=legacy_sha,**current_fields))")
    ast.parse(source)
    (REPO/'python/ct_studio_current_release.py').write_text(source, encoding='utf8')
    # Record the actual source, not the older workstation version-list file.
    (REPO/'python/requirements-current.txt').write_text(
        '# Exact numerical environment: NumPy scalar promotion affects sampling.\n'
        '-r requirements-web-fpga.txt\nnumpy==1.26.4\n', encoding='utf8')
    (REPO/'start_ct_studio_current.cmd').write_text(
        '@echo off\r\nsetlocal\r\npushd "%~dp0"\r\n'
        '".venv-web\\Scripts\\python.exe" -B -u -X utf8 python\\start_ct_studio_demo.py '
        '--config data\\ct_studio_current_20261026\\host_config.json --port 8099 %*\r\n'
        'set "RC=%ERRORLEVEL%"\r\npopd\r\nexit /b %RC%\r\n', encoding='utf8')
    ignore=REPO/'.gitignore'
    ignore.write_text(ignore.read_text(encoding='utf8')+
        '\n# Credential-free current application snapshot.\n'
        '!/runtime/ct_studio_current_20261026/ct_studio_integration_runtime.zip\n'
        '/runtime/ct_studio_current_20261026/staging/\n',encoding='utf8')
    (REPO/'evidence/ct_studio_current_20261026').mkdir(parents=True,exist_ok=True)
    evidence=dict(status='CURRENT_VERSION_SNAPSHOT_BEFORE_NEXT_OPTIMIZATION',
        source_bundle_sha256=hashlib.sha256((board/'platform_manifest.json').read_bytes()).hexdigest(),
        source_python_files_sha256={p.relative_to(APP).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((APP/'python').rglob('*'))if p.is_file()and'__pycache__'not in p.parts},
        algorithms_unchanged=True,three_dimensional_cleanup_unchanged=True,
        next_10_percent_speed_goal_not_implemented=True,
        source_selftests_physically_accepted=True,sanitized_recipient_selftests_physically_accepted=False,
        credentials_excluded=True,patient_arrays_excluded=True,
        transport_entry_note='Current actual web main retains compatible V1 segmentation client; standalone V3 client is included. This snapshot does not silently fix the entry.')
    (REPO/'evidence/ct_studio_current_20261026/source_snapshot.json').write_text(
        json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
    print('CURRENT_APPLICATION_SOURCE_EXPORTED_WITHOUT_TOUCHING_ACTIVE_APP')


if __name__ == '__main__':
    main()
