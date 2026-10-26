"""Run the downloaded editable source with an existing private portable runtime."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
# Python ._pth may point at another application's copy of the same modules.
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'desktop')]
DATA = ROOT/'data/ct_studio_dev'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def prepare(package):
    from build_private_release import validate_paired_installation
    package = Path(package).resolve()
    pair = package/'app/data/ct_studio_integration_20261025'
    if not (package/'runtime/python.exe').is_file():
        raise ValueError('需要完整解压的私有软件包，所选目录应同时包含 app 和 runtime。')
    if sys.version_info[:2] != (3, 11):
        raise ValueError('请使用配套包的 Python 3.11 启动。')
    validate_paired_installation(pair)
    versions = {}
    for name in ['numpy', 'scipy', 'pydicom', 'matplotlib', 'nibabel', 'skimage', 'plotly', 'torch']:
        module = importlib.import_module(name)
        versions[name] = module.__version__
    origins = {}
    for name in ['nodule_detector', 'nodule_workspace', 'ct_workspace_web', 'start_ct_studio_demo']:
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if ROOT/'python' not in origin.parents:
            raise ValueError('模块没有来自当前源码目录：'+name)
        origins[name] = str(origin.relative_to(ROOT))
    config = read(pair/'host_config.json')
    for key in ['token_file', 'model', 'platform_manifest']:
        config[key] = str((pair/config[key]).resolve())
    config.update(ldct_release=str(pair/'ct_studio/ldct'), nodule_package=str(pair))
    config_path = DATA/'host_config.json'
    write(config_path, config)
    from start_ct_studio_demo import configuration
    from ct_studio_shared_client import SharedPatchClient
    from ldct_contract import validate_release
    from nodule_detector import NoduleCaseProvider
    client, manifest = configuration(config_path)
    if validate_release(pair/'ct_studio/ldct')[1] != manifest['ldct']:
        raise ValueError('LDCT 资产与配套服务身份不一致。')
    NoduleCaseProvider(pair, client)
    shared = SharedPatchClient(client.url, client.token, read(pair/'shared_host_platform.json'))
    write(DATA/'settings.json', dict(desktop_package=str(package)))
    print('当前源码：'+str(ROOT), flush=True)
    print('运行环境：'+str(Path(sys.executable).resolve()), flush=True)
    print('环境、配套资产和当前源码模块检查通过。', flush=True)
    write(DATA/'diagnose.json', dict(status='PASS_OFFLINE', versions=versions, source_modules=origins,
        board_accessed=False, credentials_generated=False, package_modified=False))
    return client, shared, manifest, config_path


def fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT/'python').rglob('*')):
        if path.suffix in ('.py', '.js', '.css', '.html'):
            digest.update(path.relative_to(ROOT).as_posix().encode('utf-8'))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def choose_port(manifest, stamp):
    from start_ct_studio_demo import read_web_info, web_info
    marker_path = DATA/'session.json'
    try:
        marker = read(marker_path)
    except (OSError, ValueError):
        marker = {}
    if marker.get('source_fingerprint') == stamp and 8110 <= marker.get('port', 0) <= 8120:
        port = marker['port']
        try:
            info = web_info('http://127.0.0.1:'+str(port)+'/', manifest)
            if info is not None and info[1].get('X-UI-Token') == marker.get('ui_token'):
                return port, info
        except (OSError, ValueError, RuntimeError):
            pass
    for port in range(8110, 8121):
        try:
            if read_web_info('http://127.0.0.1:'+str(port)+'/') is None:
                return port, None
        except (OSError, ValueError, RuntimeError):
            continue
    raise RuntimeError('开发端口 8110–8120 已占用，请关闭不再使用的开发实例后重试；原会话已保留。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--desktop-package', required=True)
    parser.add_argument('--diagnose', action='store_true', help='只做离线检查，不连接板子或启动网页')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--wait', type=int, default=75)
    args = parser.parse_args()
    try:
        client, shared, manifest, config = prepare(args.desktop_package)
        if args.diagnose:
            return 0
        from desktop_bootstrap import wait_ready
        from start_ct_studio_demo import launch, web_info
        stamp = fingerprint()
        port, existing = choose_port(manifest, stamp)
        wait_ready(client, shared, max(0, min(300, args.wait)), existing)
        code = launch(SimpleNamespace(config=config, port=port, check_only=False,
            no_browser=True, wait_seconds=0))
        if code:
            return code
        url = 'http://127.0.0.1:'+str(port)+'/'
        info = web_info(url, manifest)
        if info is None:
            raise RuntimeError('开发网页尚未启动。')
        write(DATA/'session.json', dict(port=port, ui_token=info[1]['X-UI-Token'], source_fingerprint=stamp))
        print('开发版已就绪：'+url+'；运行的是当前源码，私有软件包保持原样。', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    except Exception as exc:
        print('开发版启动未完成：'+str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
