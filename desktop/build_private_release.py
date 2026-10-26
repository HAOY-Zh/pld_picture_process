"""Build the private Windows delivery from reviewed source and an existing pair.

No downloads, credential generation, board access or service installation.
The public repository contains this builder; its output contains private tokens.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAIR_NAME = 'ct_studio_integration_20261025'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args]).decode('utf-8').strip()


def validate_inputs(pair, runtime, output):
    required = ['host_config.json', 'host_platform.json', 'shared_host_platform.json',
                'model.json', 'installation.json', 'platform_token.txt',
                'ct_studio/platform_manifest.json', 'ct_studio/platform_token.txt']
    if any(not (pair/name).is_file() for name in required):
        raise ValueError('Use the complete existing paired installation directory.')
    config = read(pair/'host_config.json')
    if config.get('token_file') != 'platform_token.txt':
        raise ValueError('Expected the paired platform_token.txt; no new token is generated.')
    token = (pair/'platform_token.txt').read_text('utf-8').strip()
    if len(token) < 32 or token != (pair/'ct_studio/platform_token.txt').read_text('utf-8').strip():
        raise ValueError('The two existing credential files do not match.')
    installation = read(pair/'installation.json')
    if sha(pair/'ct_studio/platform_manifest.json') != installation['service_bundle_sha256']:
        raise ValueError('Paired service bundle identity mismatch.')
    board = (pair/'ct_studio').resolve()
    for name in ['platform_manifest.json', 'legacy_platform_manifest.json', 'ldct/manifest.json']:
        manifest = board/name
        for rel, expected in read(manifest)['files_sha256'].items():
            target = (manifest.parent/rel).resolve()
            if board not in target.parents or sha(target) != expected:
                raise ValueError('Paired asset integrity check failed: '+name)
    if not (runtime/'python.exe').is_file():
        raise ValueError('Use the complete existing portable Windows Python runtime directory.')
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError('Choose a new output name; existing deliveries are preserved.')
    for source in [ROOT, pair, runtime]:
        if output == source or output in source.parents or source in output.parents:
            # A repository-local ignored data/ destination is the supported exception.
            if source == ROOT and ROOT/'data' in output.parents:
                continue
            raise ValueError('Output must be separate from source, paired assets and runtime.')


def copy_sources(app):
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z', '--',
        'python', 'pynq', 'configs/ldct_primary_model.json',
        'start_ct_studio_integrated.cmd', 'start_ct_studio_pc.cmd']).decode('utf-8').split('\0')
    for name in filter(None, names):
        source = ROOT/name
        if source.is_symlink() or not source.is_file():
            raise ValueError('Missing or unsupported source file: '+name)
        target = app/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(ROOT/'desktop/desktop_bootstrap.py', app/'desktop_bootstrap.py')
    return [name for name in names if name]+['desktop_bootstrap.py']


def build(args):
    pair, runtime, output = (Path(p).resolve() for p in [args.paired_dir, args.runtime_dir, args.output])
    validate_inputs(pair, runtime, output)
    csc = Path(args.csc) if args.csc else Path(os.environ.get('WINDIR', 'C:/Windows'))/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    if not csc.is_file():
        raise FileNotFoundError('Windows .NET Framework compiler not found; use --csc.')
    source_commit = git('rev-parse', 'HEAD')
    source_dirty = bool(git('status', '--porcelain', '--untracked-files=normal'))
    output.mkdir(parents=True)
    (output/'.gitignore').write_text('# Private paired delivery, never commit.\n*\n', encoding='utf-8')
    app = output/'app'
    app.mkdir()
    names = copy_sources(app)
    private_pair = app/'data'/PAIR_NAME
    # Only the installation directory, never source app/data/ with patient results.
    shutil.copytree(pair, private_pair, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    config = read(private_pair/'host_config.json')
    config.update(ldct_release='data/'+PAIR_NAME+'/ct_studio/ldct', nodule_package='data/'+PAIR_NAME)
    write(private_pair/'host_config.json', config)
    model = read(app/'configs/ldct_primary_model.json')
    model.update(model_dir=config['ldct_release']+'/model', preferred_release=config['ldct_release'],
                 upgrade_100mhz=config['ldct_release'])
    model.pop('stable_50mhz', None)
    write(app/'configs/ldct_primary_model.json', model)
    write(app/'source_provenance.json', dict(
        commit=source_commit, source_worktree_dirty=source_dirty,
        export_method='Tracked reviewed working-tree files; hashes describe the delivered bytes',
        relocated_config_files=[],
        committed_export_files_sha256={name:sha(app/name) for name in names}))
    print('Source and existing paired assets copied; credentials were not regenerated.', flush=True)
    shutil.copytree(runtime, output/'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    launcher = output/'launcher'
    launcher.mkdir()
    for name in ['Launcher.cs', 'app.manifest', 'app.ico']:
        shutil.copy2(ROOT/'desktop'/name, launcher/name)
    executable = output/'CT Studio 上位机.exe'
    subprocess.run([str(csc), '/nologo', '/target:winexe', '/platform:x64', '/optimize+',
        '/reference:System.Windows.Forms.dll', '/reference:System.Drawing.dll',
        '/reference:System.Web.Extensions.dll', '/win32manifest:'+str(launcher/'app.manifest'),
        '/win32icon:'+str(launcher/'app.ico'), '/out:'+str(executable), str(launcher/'Launcher.cs')], check=True)
    (output/(executable.name+'.config')).write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<configuration><startup>'
        '<supportedRuntime version="v4.0" sku=".NETFramework,Version=v4.5"/>'
        '</startup></configuration>\n', encoding='utf-8')
    (output/'使用说明.txt').write_text(
        'CT Studio 上位机 · Windows 10/11 x64\n\n'
        '1. 完整解压 ZIP，保留 exe、app、runtime 和 launcher 的目录结构。\n'
        '2. 给板子接好电源和网线，电脑须能访问 '+config['board_url']+'。\n'
        '3. 双击 CT Studio 上位机.exe，自动检查配对并打开网页。\n\n'
        '不需要安装 Python 或下载模型。连接失败时，按窗口提示检查后点重新连接。\n'
        '关闭启动器或浏览器不会中断后台任务；再次打开可继续原会话。\n'
        '本机结果位于 app/data/，日志可在启动窗口打开。\n'
        '本软件不安装、替换或重启板端服务。私有令牌随包交接，整个软件包不要上传 GitHub。\n'
        '保留原电脑端和 FPGA 功能；基础 FPGA 任务不再重复执行电脑参考和像素对比。\n'
        '本次仅自动完成离线环境检查；实板验收范围见源码最终版说明。\n\n'
        '源码提交：'+source_commit+'\n', encoding='utf-8-sig')
    # A new source revision should not inherit an older handoff's processing hashes.
    installation = read(private_pair/'installation.json')
    installation.update(desktop_source_commit=source_commit,
        processing_patch_files_sha256={name:sha(app/name) for name in
            ['python/ct_workspace_web.py', 'python/lungseg/workspace/app.js']})
    write(private_pair/'installation.json', installation)
    subprocess.run([str(output/'runtime/python.exe'), '-B', '-X', 'utf8',
        str(app/'desktop_bootstrap.py'), '--diagnose'], cwd=app, check=True)
    files = sorted(p for p in output.rglob('*') if p.is_file())
    write(output/'desktop_manifest.json', dict(schema='ct_studio_private_desktop_v1',
        source_commit=source_commit, source_worktree_dirty=source_dirty,
        contains_existing_credentials=True, new_credentials_generated=False,
        board_modified=False, real_board_tested_by_builder=False,
        offline_diagnose_passed=True,
        files_sha256={str(p.relative_to(output)).replace('\\', '/'):sha(p) for p in files}))
    if args.zip:
        archive = output.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file():
                    z.write(p, str(p.relative_to(output.parent)))
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError('Archive integrity check failed.')
        print(json.dumps(dict(archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size)), flush=True)
    print(json.dumps(dict(output=str(output), source_commit=source_commit,
        source_worktree_dirty=source_dirty, offline_diagnose_passed=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-dir', required=True, help='Existing installed data/ct_studio_integration_20261025')
    parser.add_argument('--runtime-dir', required=True, help='Existing portable runtime/ with Python and all dependencies')
    parser.add_argument('--output', required=True, help='New private destination; never a Git-tracked folder')
    parser.add_argument('--csc', help='Optional Windows .NET Framework C# compiler')
    parser.add_argument('--zip', action='store_true', help='Also create and CRC-check a private ZIP')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
