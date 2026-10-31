"""Update a private portable desktop package from an accepted local PC snapshot.

Preserves runtime and results, backs up every replaced file, regenerates source
identities, and never contacts or changes the FPGA. Run explicitly with paths.
"""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def usable(p):return not any(x in ('__pycache__','.pytest_cache') for x in p.parts) and p.suffix not in ('.pyc','.pyo')


def update(source,target,evidence):
    source=source.resolve();target=target.resolve();evidence=evidence.resolve()
    assert (source/'python/ct_workspace_web.py').is_file()
    assert (target/'runtime/compat_site/compat_manifest.json').is_file()
    evidence.mkdir(parents=True,exist_ok=False);backup=evidence/'backup';backup.mkdir()
    replaced={}
    def preserve(p):
        assert target in p.resolve().parents
        if p.is_file() and p.relative_to(target).as_posix() not in replaced:
            dest=backup/p.relative_to(target);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest);replaced[p.relative_to(target).as_posix()]=sha(p)
    def copy(src,dst):
        preserve(dst);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    app=target/'app';package=app/'data/ct_studio_integration_20261025'
    source_package=source/'data/ct_studio_integration_20261025'
    # Touch program and matched assets only; retain cases and runtime libraries.
    for folder in ('python','pynq','configs'):
        for src in sorted((source/folder).rglob('*')):
            if src.is_file() and usable(src):copy(src,app/folder/src.relative_to(source/folder))
    for src in sorted(source_package.rglob('*')):
        if src.is_file() and usable(src):copy(src,package/src.relative_to(source_package))
    for name in ('source_provenance.json','desktop_bootstrap.py'):preserve(app/name)
    for name in ('desktop_manifest.json','desktop_version.json','desktop_validation.json',
        'README_新版接入说明.md','使用说明.txt','CT Studio 上位机.exe','launcher/Launcher.cs',
        'launcher/desktop_bootstrap.py'):preserve(target/name)
    cfg=read(package/'host_config.json');cfg.pop('workspace_output',None)
    cfg['ldct_release']='data/ct_studio_integration_20261025/ct_studio/ldct'
    cfg['nodule_package']='data/ct_studio_integration_20261025'
    web=app/'python/ct_workspace_web.py';text=web.read_text(encoding='utf8')
    assert 'desktop_release_id' not in text
    pos=text.index('    server = ThreadingHTTPServer(',text.index('def main('))
    text=text[:pos]+"    if args.config:\n        controller.status['desktop_release_id']=json.loads(Path(args.config).read_text(encoding='utf8')).get('desktop_release_id')\n"+text[pos:]
    web.write_text(text,encoding='utf8')
    cs=target/'launcher/Launcher.cs';text=cs.read_text(encoding='utf-8-sig')
    text=text.replace('AssemblyFileVersion("2026.10.26.2")','AssemblyFileVersion("2026.10.27.4")')
    text=text.replace('CT Studio 上位机 · 新版','CT Studio 上位机 · 10月27日新版')
    cs.write_text(text,encoding='utf-8-sig')
    compiler=Path('C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe')
    cmd=[str(compiler),'/nologo','/target:winexe','/platform:x64','/optimize+',
        '/reference:System.Windows.Forms.dll','/reference:System.Drawing.dll','/reference:System.Web.Extensions.dll',
        '/win32manifest:'+str(target/'launcher/app.manifest'),'/win32icon:'+str(target/'launcher/app.ico'),
        '/out:'+str(target/'CT Studio 上位机.exe'),str(cs)]
    run=subprocess.run(cmd,capture_output=True,text=True)
    if run.returncode:raise RuntimeError(run.stdout+run.stderr)
    bundle=sha(package/'ct_studio/platform_manifest.json')
    write(package/'installation.json',dict(status='CURRENT_ACCEPTED_EXACT_R4_PAIRED_FOR_DESKTOP',
        service_bundle_sha256=bundle,board_service='ct-studio-exact-speed-r4.service',
        board_program_modified=False,board_url=cfg['board_url']))
    source_files={p.relative_to(app).as_posix():sha(p) for folder in ('python','pynq','configs')
        for p in sorted((app/folder).rglob('*')) if p.is_file() and usable(p)}
    source_files['desktop_bootstrap.py']=sha(app/'desktop_bootstrap.py')
    compat_sha=sha(target/'runtime/compat_site/compat_manifest.json')
    rid=hashlib.sha256(json.dumps(dict(files=source_files,bundle=bundle,runtime_compat_manifest=compat_sha),sort_keys=True).encode()).hexdigest()
    provenance=read(app/'source_provenance.json')
    provenance.update(current_accepted_source=source.name,validated_export_files_sha256=source_files,
        service_bundle_sha256=bundle,desktop_release_id=rid,runtime_compat_manifest_sha256=compat_sha)
    write(app/'source_provenance.json',provenance);cfg['desktop_release_id']=rid;write(package/'host_config.json',cfg)
    version=read(target/'desktop_version.json')
    version.update(date='2026-10-27',current_pc_source=source.name,desktop_release_id=rid,
        service_bundle_sha256=bundle,launcher_source_sha256=sha(cs),launcher_exe_sha256=sha(target/'CT Studio 上位机.exe'),
        host_speed_profile='exact_host_raw_r3',board_service='ct-studio-exact-speed-r4.service',
        default_selection_threshold=0.7655561641768475,board_program_modified_this_integration=False)
    write(target/'desktop_version.json',version)
    write(target/'desktop_validation.json',dict(status='UPDATED_PENDING_CURRENT_CONNECTION_CHECK',
        historical_validation_retained_in_backup=True,source=source.name,board_program_modified=False))
    note="""CT Studio 上位机 · 2026-10-27 新版

双击本目录“CT Studio 上位机.exe”。保留 app、runtime、launcher，不能只拷贝 exe。
配套板端：192.168.2.99:8766，ct-studio-exact-speed-r4.service。
实际网页与已验收 exact_host_raw_r3 加速版一致；NumPy 1.26.4 便携环境保持不变。
包括新版分割/结节客户端、严格档候选筛选、整例自动三维清理显示。
基础与 LDCT 仅提供 FPGA 执行；直方图均衡、伪彩已移除。ROI 电脑参考仍保留。
自动选择 8099～8109 的空闲端口，以启动窗口显示的地址为准。
已有其它网页保持运行；本软件结果保存在 app/data/ct_workspace_demo/results。
日志在 app/data/desktop_launcher。更新未改变板端程序、模型权重、定点算法或清理算法。
包含本机配套凭据，供私下使用，不要上传整个桌面目录到公共仓库。
仅用于工程教学与算法演示，不作为临床诊断依据。
"""
    (target/'使用说明.txt').write_text(note,encoding='utf-8-sig')
    (target/'README_新版接入说明.md').write_text('# 当前桌面版接入说明\n\n'+note,encoding='utf8')
    manifest=read(target/'desktop_manifest.json')
    hashes=manifest['files_sha256']
    for folder in ('app/python','app/pynq','app/configs','app/data/ct_studio_integration_20261025'):
        for p in sorted((target/folder).rglob('*')):
            if p.is_file() and usable(p):hashes[p.relative_to(target).as_posix()]=sha(p)
    for name in ('app/source_provenance.json','app/desktop_bootstrap.py','launcher/Launcher.cs',
        'launcher/desktop_bootstrap.py','CT Studio 上位机.exe','desktop_version.json','desktop_validation.json',
        '使用说明.txt','README_新版接入说明.md'):hashes[name]=sha(target/name)
    manifest.update(schema='ct_studio_portable_exact_r4_windows_v4',contains_existing_board_credentials=True)
    write(target/'desktop_manifest.json',manifest)
    write(evidence/'update.json',dict(status='UPDATED_AND_IDENTITIES_REBOUND',target=str(target),
        source=str(source),replaced_files_sha256=replaced,desktop_release_id=rid,service_bundle_sha256=bundle,
        portable_runtime_changed=False,patient_results_changed=False,board_program_modified=False))
    print(json.dumps(dict(status='UPDATED',files_backed_up=len(replaced),desktop_release_id=rid,bundle=bundle),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--desktop',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();update(a.source,a.desktop,a.evidence)
