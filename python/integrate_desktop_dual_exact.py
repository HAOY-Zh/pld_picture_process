"""Update the portable desktop shell with the accepted DualExact PC release.

The input ZIP and the current 8099 installation remain untouched. No board
installation, network reconfiguration, downloads or model conversion occur.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CURRENT=ROOT/'data/CT_Studio_DualExact_20261026_02'
OUT=ROOT/'data/CT_Studio_Desktop_DualExact_20261026_02'
EVIDENCE=ROOT/'data/desktop_dual_integration_20261026'
OLD_ROOT='CT_Studio_Desktop_20261026_01/'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unexpected source anchor: '+old[:100])
    return text.replace(old,new)


def bootstrap(source):
    s=source.replace("source['committed_export_files_sha256']","source['validated_export_files_sha256']")
    s=replace_once(s,'from ct_studio_shared_client import SharedPatchClient',
        'from ct_dual_io_client import CachedCanvasClient as SharedPatchClient')
    s=replace_once(s,"    provider=NoduleCaseProvider(CONFIG.parent,client)",
        "    if manifest.get('dual_io_interface')!='ct_dual_exact_io_v3':\n"
        "        raise LaunchError('IDENTITY','缺少已验收的 DualExact V3 配套身份。')\n"
        "    provider=NoduleCaseProvider(CONFIG.parent,client)")
    s=replace_once(s,"    from start_ct_studio_demo import configuration",'''    compat=ROOT.parent/'runtime/compat_site'
    descriptor=compat/'compat_manifest.json'
    if sha(descriptor)!=source['runtime_compat_manifest_sha256']:
        raise LaunchError('RUNTIME','数值兼容环境不完整，请重新解压配套包。')
    for rel,expected in read(descriptor)['files_sha256'].items():
        target=(compat/rel).resolve()
        if compat not in target.parents or sha(target)!=expected:
            raise LaunchError('RUNTIME','数值兼容文件缺失或改变。')
    import numpy as np
    if np.__version__!='1.26.4' or compat not in Path(np.__file__).resolve().parents:
        raise LaunchError('RUNTIME','结节采样需要已验收的 NumPy 1.26.4 兼容环境。')
    from start_ct_studio_demo import configuration''')
    s=replace_once(s,"    if legacy.get('error') or identity.get('error'):",
        "    if shared.bundle.get('dual_io_interface') and (\n"
        "            identity.get('dual_io_interface')!=shared.bundle['dual_io_interface'] or\n"
        "            identity.get('shared_segmentation')!=shared.bundle['shared_segmentation'] or\n"
        "            identity.get('dual_io_startup')!='all-nodes-and-native-layout-exact'):\n"
        "        raise LaunchError('IDENTITY','板端尚未通过新版分割与结节接口自检。')\n"
        "    if legacy.get('error') or identity.get('error'):")
    a=s.index('                state=info[2]\n');b=s.index('            return port,info',a)
    s=s[:a]+'''                state=info[2]
                release_id=read(CONFIG)['desktop_release_id']
                # Preserve other source/desktop copies even when their hardware
                # identities match. Reuse only our own accepted local backend.
                if state.get('desktop_release_id')!=release_id:continue
                if not settings.is_file():continue
                marker=read(settings)
                if (marker.get('port')!=port or marker.get('release_id')!=release_id or
                        marker.get('launch_root')!=str(ROOT.resolve())):continue
'''+s[b:]
    s=replace_once(s,"json.dumps(dict(port=port))",
        "json.dumps(dict(port=port,release_id=read(CONFIG)['desktop_release_id'],launch_root=str(ROOT.resolve())))")
    return s


def ignore(folder,names):
    return [n for n in names if n in ('__pycache__','.pytest_cache','build') or n.endswith(('.pyc','.pyo'))
        or (Path(folder).name=='exact_io_native' and n.endswith(('.dll','.json','.so')))]


def install_compat():
    """Pin scalar-promotion behavior to the actual accepted numerical environment."""
    import numpy as np
    if np.__version__!='1.26.4':raise ValueError('Build from the accepted NumPy 1.26.4 environment')
    site=Path(np.__file__).resolve().parents[1];compat=OUT/'runtime/compat_site'
    if compat.exists()and any(compat.iterdir()):raise FileExistsError('Existing compatibility layer retained')
    compat.mkdir(exist_ok=True)
    for name in ('numpy','numpy.libs','numpy-1.26.4.dist-info'):
        shutil.copytree(site/name,compat/name,ignore=ignore)
    write(compat/'compat_manifest.json',dict(schema='ct_desktop_numpy_compat_v1',numpy_version='1.26.4',
        reason='Match the accepted scalar promotion, sampling and decode behavior; no algorithm change',
        files_sha256={p.relative_to(compat).as_posix():sha(p)for p in sorted(compat.rglob('*'))if p.is_file()}))
    (OUT/'runtime/python311._pth').write_text('.\nLib\nDLLs\ncompat_site\nLib/site-packages\n../app/python\n../app\nimport site\n',encoding='ascii')


def finish_program(compile_launcher=True):
    """Compile the reviewed launcher source and freeze the final source identity."""
    for path in (OUT/'app/desktop_bootstrap.py',OUT/'launcher/desktop_bootstrap.py'):
        s=path.read_text(encoding='utf8').replace("default=75)","default=300)")
        s=s.replace('Verify committed program files too.','Verify current accepted program files too.')
        path.write_text(s,encoding='utf8')
    source=OUT/'launcher/Launcher.cs';s=source.read_text(encoding='utf-8-sig')
    if '[assembly: System.Reflection.AssemblyFileVersion('not in s:
        s=s.replace('internal sealed class Launcher : Form {',
            '[assembly: System.Reflection.AssemblyFileVersion("2026.10.26.2")]\ninternal sealed class Launcher : Form {')
    s=s.replace('Text="CT Studio 上位机";','Text="CT Studio 上位机 · 新版";')
    source.write_text(s,encoding='utf-8-sig')
    compiler=Path('C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe')
    cmd=[str(compiler),'/nologo','/target:winexe','/platform:x64','/optimize+',
        '/reference:System.Windows.Forms.dll','/reference:System.Drawing.dll','/reference:System.Web.Extensions.dll',
        '/win32manifest:'+str(OUT/'launcher/app.manifest'),'/win32icon:'+str(OUT/'launcher/app.ico'),
        '/out:'+str(OUT/'CT Studio 上位机.exe'),str(source)]
    if compile_launcher:
        result=subprocess.run(cmd,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Launcher build failed: '+result.stdout+result.stderr)
    app=OUT/'app';package=app/'data/ct_studio_integration_20261025'
    source_files={p.relative_to(app).as_posix():sha(p)for base in ('python','pynq','configs')
        for p in sorted((app/base).rglob('*'))if p.is_file()}
    source_files['desktop_bootstrap.py']=sha(app/'desktop_bootstrap.py')
    provenance=json.loads((app/'source_provenance.json').read_text(encoding='utf8'))
    bundle=sha(package/'ct_studio/platform_manifest.json')
    compat_sha=sha(OUT/'runtime/compat_site/compat_manifest.json')
    rid=hashlib.sha256(json.dumps(dict(files=source_files,bundle=bundle,runtime_compat_manifest=compat_sha),sort_keys=True).encode()).hexdigest()
    provenance.update(validated_export_files_sha256=source_files,desktop_release_id=rid,runtime_compat_manifest_sha256=compat_sha)
    write(app/'source_provenance.json',provenance)
    config=json.loads((package/'host_config.json').read_text(encoding='utf8'));config['desktop_release_id']=rid
    write(package/'host_config.json',config)
    version=json.loads((OUT/'desktop_version.json').read_text(encoding='utf8'))
    version.update(desktop_release_id=rid,original_exe_and_runtime_byte_exact=False,
        portable_runtime_byte_exact_to_input=False,launcher_compiled_from_reviewed_source=True,
        numpy_compatibility_overlay='1.26.4',runtime_compat_manifest_sha256=compat_sha,
        launcher_source_sha256=sha(source),launcher_exe_sha256=sha(OUT/'CT Studio 上位机.exe'))
    write(OUT/'desktop_version.json',version)
    write(EVIDENCE/'launcher_build.json',dict(status='REVIEWED_SOURCE_COMPILED',command=cmd,
        source_sha256=sha(source),executable_sha256=sha(OUT/'CT Studio 上位机.exe'),desktop_release_id=rid))
    print('Reviewed Windows launcher compiled; final source identity frozen.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);args=parser.parse_args()
    accepted=json.loads((ROOT/'data/dual_pipeline_optimization_20261026/production_acceptance.json').read_text())
    if accepted['status']!='DUAL_EXACT_WEB_8099_ACCEPTED':raise ValueError('Current release not accepted')
    OUT.mkdir(exist_ok=False);EVIDENCE.mkdir(exist_ok=True)
    copied={}
    with zipfile.ZipFile(args.archive)as z:
        manifest=json.loads(z.read(OLD_ROOT+'desktop_manifest.json'))['files_sha256']
        for info in z.infolist():
            if info.is_dir():continue
            if not info.filename.startswith(OLD_ROOT):raise ValueError('Unexpected ZIP root')
            rel=info.filename[len(OLD_ROOT):];parts=PurePosixPath(rel).parts
            if PurePosixPath(rel).is_absolute()or '..'in parts or '\\'in rel:raise ValueError('Unsafe ZIP path')
            if not (rel.startswith(('runtime/','launcher/'))or rel in ('CT Studio 上位机.exe','CT Studio 上位机.exe.config')):
                continue
            target=(OUT/rel).resolve()
            if OUT not in target.parents:raise ValueError('Escaping destination')
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info)as source,target.open('xb')as dest:shutil.copyfileobj(source,dest)
            digest=sha(target)
            if manifest.get(rel)!=digest:raise ValueError('Original desktop manifest mismatch: '+rel)
            copied[rel]=digest
        old_boot=z.read(OLD_ROOT+'launcher/desktop_bootstrap.py').decode('utf-8-sig')
        original_provenance=json.loads(z.read(OLD_ROOT+'app/source_provenance.json'))
        old_version=json.loads(z.read(OLD_ROOT+'desktop_version.json'))
    print('Original EXE and portable runtime verified and extracted.',flush=True)
    shutil.copytree(CURRENT/'python',OUT/'app/python',ignore=ignore)
    shutil.copytree(CURRENT/'pynq',OUT/'app/pynq',ignore=ignore)
    shutil.copytree(CURRENT/'configs',OUT/'app/configs',ignore=ignore)
    package=OUT/'app/data/ct_studio_integration_20261025'
    shutil.copytree(CURRENT/'data/ct_studio_integration_20261025',package,ignore=ignore)
    shutil.copy2(CURRENT/'requirements-pc-verified.txt',OUT/'app/requirements-pc-verified.txt')
    # Remove the source machine's workspace path; the existing launcher defaults
    # to its own app/data/ct_workspace_demo/results, independently per copy.
    config_path=package/'host_config.json';config=json.loads(config_path.read_text())
    config.pop('workspace_output',None)
    config['ldct_release']='data/ct_studio_integration_20261025/ct_studio/ldct'
    config['nodule_package']='data/ct_studio_integration_20261025'
    code=bootstrap(old_boot)
    for path in (OUT/'launcher/desktop_bootstrap.py',OUT/'app/desktop_bootstrap.py'):
        path.write_text(code,encoding='utf8')
    # Read-only release marker in the backend's existing state response. The
    # launcher's marker also binds the directory, preserving the live 8099 app.
    web=OUT/'app/python/ct_workspace_web.py';s=web.read_text(encoding='utf8')
    anchor="    server = ThreadingHTTPServer("
    if anchor not in s:
        anchor="    server=ThreadingHTTPServer("
    index=s.index(anchor,s.index('def main('))
    s=s[:index]+"    if args.config:\n        controller.status['desktop_release_id']=json.loads(Path(args.config).read_text(encoding='utf8')).get('desktop_release_id')\n"+s[index:]
    web.write_text(s,encoding='utf8')
    source_files={p.relative_to(OUT/'app').as_posix():sha(p)
        for base in ('python','pynq','configs')for p in sorted((OUT/'app'/base).rglob('*'))if p.is_file()}
    source_files['desktop_bootstrap.py']=sha(OUT/'app/desktop_bootstrap.py')
    bundle=sha(package/'ct_studio/platform_manifest.json')
    release_id=hashlib.sha256(json.dumps(dict(files=source_files,bundle=bundle),sort_keys=True).encode()).hexdigest()
    config['desktop_release_id']=release_id;write(config_path,config)
    write(package/'installation.json',dict(status='CURRENT_ACCEPTED_BOARD_PAIRED_FOR_DESKTOP',
        service_bundle_sha256=bundle,board_folder='/home/xilinx/ct_test/ct_studio_dual_exact_20261026_02',
        board_service='ct-studio-dual-exact-r2.service',board_url=config['board_url'],
        credentials_copied_unchanged=True,board_program_modified=False))
    write(OUT/'app/original_desktop_source_provenance.json',original_provenance)
    write(OUT/'app/source_provenance.json',dict(schema='ct_desktop_current_release_provenance_v2',
        original_desktop_commit=original_provenance['commit'],current_accepted_source=str(CURRENT.relative_to(ROOT)),
        validated_export_files_sha256=source_files,relocated_config_files=[],
        all_python_and_pynq_source_byte_exact_to_original_commit=False,
        includes_dual_exact_v3=True,service_bundle_sha256=bundle,desktop_release_id=release_id))
    write(OUT/'desktop_version.json',dict(product='CT Studio DualExact portable desktop',date='2026-10-26',
        original_desktop=old_version,current_pc_source=CURRENT.name,desktop_release_id=release_id,
        service_bundle_sha256=bundle,requires_system_python=False,requires_pip_install=False,
        requires_admin=False,original_exe_and_runtime_byte_exact=True,contains_existing_credentials=True))
    (OUT/'.gitignore').write_text('# Private paired desktop distribution.\n*\n',encoding='utf8')
    write(EVIDENCE/'integration_sources.json',dict(original_archive_sha256=sha(args.archive),
        original_exe_and_runtime_files_verified=len(copied),desktop_release_id=release_id,
        original_desktop_files_sha256=copied,modified_source_files=source_files))
    install_compat();finish_program()
    print(json.dumps(dict(output=str(OUT),service_bundle_sha256=bundle,desktop_release_id=release_id),indent=2),flush=True)


if __name__=='__main__':main()
