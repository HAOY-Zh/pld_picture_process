"""Freeze the privately paired desktop ZIP and verify every archived file."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'data/CT_Studio_Desktop_DualExact_20261026_02'
EVIDENCE=ROOT/'data/desktop_dual_integration_20261026'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,m):Path(p).write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def files():
    obsolete={'launcher/build_desktop.py','launcher/finalize_desktop.py','launcher/test_local_integration.py'}
    for p in sorted(APP.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(APP).as_posix()
        if rel in obsolete or p.name=='desktop_manifest.json':continue
        if '__pycache__'in p.parts or p.suffix.lower()in('.pyc','.pyo'):continue
        if any(rel.startswith('app/data/'+name+'/')for name in
                ('desktop_launcher','ct_studio_demo_launcher','ct_workspace_demo','ct_workspace_pc')):continue
        yield p


def main():
    post=read(EVIDENCE/'portable_postprocess_compat/result.json')
    nodule=read(EVIDENCE/'portable_nodule_compat/result.json')
    lung=read(EVIDENCE/'portable_lung_board.json')
    relocation=read(EVIDENCE/'relocation_compat.json')
    gui=read(EVIDENCE/'gui_and_backend.json')
    assert post['status']=='PORTABLE_351_INPUTS_AND_ALL_NATIVE_3D_MASKS_EXACT_PASS'
    assert nodule['status']=='PORTABLE_ACTUAL_BOARD_ALL_CANVAS_HEADS_AND_CANDIDATES_EXACT_PASS'
    assert lung['status']=='PORTABLE_THREE_ACTUAL_LUNG_HEADS_EXACT_PASS'
    assert relocation['status']=='RELOCATED_PINNED_RUNTIME_AND_NUMERICS_PASS'
    assert gui['status']=='FINAL_PINNED_GUI_READY_AND_CURRENT_BACKEND_PASS'
    run=subprocess.run([str(APP/'runtime/python.exe'),'-B','-X','utf8',str(APP/'launcher/check_runtime.py'),str(APP/'app')],
        capture_output=True,text=True,encoding='utf8',check=True,timeout=90)
    numerical=json.loads(run.stdout)
    assert numerical==read(EVIDENCE/'synthetic_runtime_parity.json')['results']['current']
    validation=dict(status='PASS_PORTABLE_PINNED_RUNTIME_GUI_REAL_BOARD_AND_NUMERICAL_PARITY',
        numpy_version=post['numpy_version'],torch_version=post['torch_version'],
        portable_runtime_imports_passed=True,synthetic_numerical_parity=True,
        native_windows_gui_opened_and_inspected=True,launcher_tests_passed=14,
        relocated_chinese_and_space_path_tested=True,system_python_required=False,pip_install_required=False,
        full_351_preprocessing_and_native_masks_exact=True,full_3d_cleanup_and_volume_exact=True,
        lung_actual_board_spot_checks=len(lung['rows']),new_351_lung_board_inferences_this_integration=False,
        nodule_actual_board_canvases=nodule['canvases'],all_nodule_inputs_and_heads_exact=True,
        all_ranked_candidates_exact=nodule['raw_candidates'],selected_candidates_exact=nodule['selected_candidates'],
        real_board_connected=True,real_board_inference_performed=True,board_program_modified=False,
        original_8099_case_preserved=gui['original_8099_case_preserved'],
        tests_used_one_existing_case=True,scope='Packaging compatibility and exact regression; not new clinical accuracy evaluation')
    write(APP/'desktop_validation.json',validation)
    version=read(APP/'desktop_version.json')
    version.update(packaged_board_assets_updated=True,board_program_modified_this_integration=False,
        obsolete_author_build_tools_omitted=['build_desktop.py','finalize_desktop.py','test_local_integration.py'])
    write(APP/'desktop_version.json',version)
    (APP/'使用说明.txt').write_text('''CT Studio 上位机 · 新版免安装包（2026-10-26）

1. 完整解压到普通文件夹，例如 D:\\CT_Studio。不要在 ZIP 预览里直接运行。
2. 给已部署配套服务的 PYNQ 板子接电、连接网线。
3. 双击“CT Studio 上位机.exe”。不需要安装 Python、运行 pip 或手动启动网页。

请保留 exe、app、runtime、launcher 在同一个目录，不能只复制 exe。
启动窗口仍使用队友设计的界面。最新版肺分割调度、结节整层检测、上下文传输缓存及原有基础处理、LDCT、ROI/三维功能已打入 app。
程序自动选取 8099～8109 的空闲端口；若旧 8099 程序已打开，会保留原会话并使用另一端口。
日志位于 app/data/desktop_launcher，病例结果位于 app/data/ct_workspace_demo/results。
本机试运行中使用 8100；在另一台电脑上端口可能不同，以启动窗口显示为准。

当前配套板子：192.168.2.99:8766
当前板端服务：ct-studio-dual-exact-r2.service
本次没有重新烧写、重启或更新板端程序。当前这块板子的新版服务已经部署好，可直接连接。
若另一块板子运行旧服务，需先由负责集成的同学部署配套版本；本软件不会自动替换板端程序。

内置 NumPy 已固定为验收环境的 1.26.4。不要单独更换 runtime，否则可能改变结节采样与解码结果。
已验证：351 张预处理/分割后处理、三维清理与体积一致；三张实际肺分割板端输出一致；结节 202 个实际检测头、5,324 个评分候选和最终 10 个候选严格一致。
这些是同一既有病例的工程回归核对，不是新的临床准确率评价。

本包包含配套板端凭据，仅供指定队友私下交付，不作为公开 GitHub 发布包。
详细版本与验证范围见 desktop_version.json、desktop_validation.json 和 README_新版接入说明.md。
''',encoding='utf-8-sig')
    (APP/'README_新版接入说明.md').write_text('''# 最新网页接入桌面启动器

EXE 保留原启动窗口，由审查后的 C# 源码重新编译；实际处理程序放在 `app/python`，便携环境放在 `runtime`。桌面外壳不包含独立的推理算法。

本版 PC 来源为已实板验收的 `CT_Studio_DualExact_20261026_02`。旧 `8a2a855...` 提交仅作为原始桌面包来源，不能再声称所有处理代码与该提交一致。

已更新：新版分割和结节客户端、完整 Python/网页资源、模型及评分资产、平台配对身份、启动校验与源文件哈希。运行资产的硬件清单 SHA256 为 `503b99ab19e5be533d978f36b85c2edb24abd81af6d185a369cd5be159ba585f`。

板端仍使用已部署的 `ct-studio-dual-exact-r2.service`；打包过程没有改变 bit、模型、阈值或板端程序。

原桌面包 NumPy 2.4 与现用 1.26 的标量精度规则不同，导致结节采样输入与坐标存在差异。本版在 `runtime/compat_site` 固定 NumPy 1.26.4，并校验兼容层文件哈希。完整实板回归已通过，不能自行升级这层依赖。

启动器只复用同一桌面目录、同一代码/数值环境身份的后台。其它 8099 会话会保留；新包首次运行时不会携带本机日志、端口记录或病例。

后续更新应重新构建完整配套包并生成来源哈希，不能只换几个 Python 文件后跳过校验。源码仓库中的 `python/integrate_desktop_dual_exact.py` 和 `python/finalize_desktop_dual.py` 是本次构建与冻结入口；原作者写死 D:\\PLD 的旧构建/归档工具未作为可执行入口交付。

所有精度与验证口径见 `desktop_validation.json`。此包为含既有凭据的私下交付包，未上传 GitHub。
''',encoding='utf8')
    selected=list(files());hashes={p.relative_to(APP).as_posix():sha(p)for p in selected}
    write(APP/'desktop_manifest.json',dict(schema='ct_studio_portable_dual_exact_windows_v3',
        contains_existing_board_credentials=True,files_sha256=hashes))
    archive=APP.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True)as z:
        for index,p in enumerate(selected+[APP/'desktop_manifest.json']):
            z.write(p,APP.name+'/'+p.relative_to(APP).as_posix())
            if index and index%4000==0:print('ARCHIVED',index,flush=True)
    print('Archive written; verifying every file CRC and SHA256.',flush=True)
    with zipfile.ZipFile(archive)as z:
        names={n[len(APP.name)+1:]for n in z.namelist()}
        assert names==set(hashes)|{'desktop_manifest.json'}
        for name,expected in hashes.items():
            digest=hashlib.sha256()
            with z.open(APP.name+'/'+name)as stream:
                while True:
                    chunk=stream.read(1024*1024)
                    if not chunk:break
                    digest.update(chunk)
            if digest.hexdigest()!=expected:raise ValueError('Archive file mismatch: '+name)
    receipt=dict(status='FINAL_PRIVATE_DESKTOP_ZIP_ALL_CRC_AND_SHA256_PASS',archive=str(archive),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files=len(hashes)+1,
        no_cases_logs_or_port_markers_shipped=True,validation=validation)
    archive.with_suffix('.zip.sha256').write_text(receipt['archive_sha256']+'  '+archive.name+'\n',encoding='ascii')
    write(EVIDENCE/'final_zip_verification.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items()if k!='validation'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
