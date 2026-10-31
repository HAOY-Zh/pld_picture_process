"""Finish the reviewed source-only delivery and remove legacy image outputs from HEAD."""
import hashlib,json,shutil,subprocess,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PUBLIC=ROOT/'data/github_integrated_review_20261025'
tracked=[n for n in subprocess.check_output(['git','-C',str(PUBLIC),'ls-files','-z']).decode('utf-8').split('\0')if n]
omitted=[]
for name in tracked:
    p=PUBLIC/name
    if not name.startswith('handoff/ldct-slice-v1-20261017/'):continue
    image_output=p.suffix.lower() in ('.npy','.png','.jpg','.jpeg')
    old_archive='/archives/' in name and p.suffix.lower() in ('.zip','.sha256')
    if not (image_output or old_archive):continue
    assert PUBLIC.resolve()in p.resolve().parents
    dest=OUT/'omitted_legacy_data'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest);p.unlink();omitted.append(name)

note=PUBLIC/'handoff/ldct-slice-v1-20261017/源码保留说明_20261031.md'
note.write_text('''# 历史LDCT交接源码保留说明

本目录保留历史Python、Verilog、Tcl、模型和接口说明，便于追溯。
2026-10-31整理时移除了历史病例切片、板测图像数组和携带这些数组的旧发布压缩包。
历史manifest、测速JSON可能仍记录这些文件的哈希，仅作为当时结果记录；本目录不是当前可直接部署的运行包。
完整当前运行资产请使用仓库根目录setup_ct_studio_current.cmd，它读取runtime/ct_studio_current_20261027。
当前运行包的模型和bit未因本次整理而修改；原作者工作区中的病例和历史数组未删除。
''',encoding='utf-8')

(PUBLIC/'docs/最终完整交付说明_20261031.md').write_text('''# CT Studio 最终完整交付说明

## 当前运行版本

当前网页采用已验收的exact_host_raw_r3主机路径和共用CNN引擎，肺结节最终阈值固定为0.8738163050576156，已移除档位选择框与重筛按钮。
新检测、真实结果导入及旧会话恢复均应用固定低误报档，不改动CNN权重、几何坐标或冻结跨层评分器。提高阈值会减少候选，也可能减少检出；报告中的历史FP32质量数字与新板端准确率应区分。
基础处理与LDCT只提供实际FPGA执行；肺分割整例完成后自动显示三维清理结果。ROI独立参考入口保留。

## 解压或克隆后启动

Windows x64上安装Python 3.11，然后在项目根目录执行setup_ct_studio_current.cmd。
首次安装需要联网下载依赖；按python/requirements-current.txt固定NumPy 1.26.4，不能任意升级数值库。
运行start_ct_studio_preview.cmd打开电脑浏览界面，默认地址http://127.0.0.1:8099/。
完整FPGA功能需要把本次安装生成的data/ct_studio_current_20261027/ct_studio部署到PYNQ-Z2，并使用同次生成的凭据配对，再运行start_ct_studio_current.cmd。
完整接板、自检和开机服务步骤见CT_Studio_latest_20261027.md。不要复用作者或队友另一套安装的令牌，不关闭身份校验。

## 代码和资产位置

| 目录 | 内容 |
|---|---|
| python | 当前上位机网页、输入、后处理、调度、离线训练与评价、测试和维护脚本 |
| pynq | ARM端服务、DMA和共享计算引擎驱动、C++原生加速、板端验证脚本 |
| fpga | 逐步开发的Verilog、测试台、Vivado构建Tcl与工程源配置 |
| hardware/ct_studio_current_20261027 | 实际当前LDCT、共享CNN、ROI等硬件源快照，优先于旧实验RTL |
| hardware/ct_studio_20261018/basic_shared | 当前六种基础处理共用模块的冻结硬件来源 |
| board_files | PYNQ-Z2板卡定义和构建配套文件 |
| runtime/ct_studio_current_20261027 | 当前可安装的整数模型、bit/HWH、参数、自检向量和服务运行包 |
| models/fp32 | 三个最终浮点模型的纯张量导出，未含优化器和训练病例清单 |
| desktop | EXE启动器C#源码、manifest、图标和Python引导代码 |
| research/author_sources | 与当前运行代码不同的作者旧源码，保留研究过程，不自动覆盖当前程序 |
| matlab | 早期基础处理参考与对拍源码，未含HU图像文本数据 |
| docs | 部署、模块原理、数据集说明、历史实验及引用修订后的项目报告 |
| evidence/final_20261031 | 本次源文件收集、固定阈值与发布验证摘要 |

历史分支或日期目录保留研发过程，当前运行入口始终是根目录的current启动脚本。
同名功能的早期Python参考、SVM、YOLO、剪枝和配准原型不等于当前网页默认算法。
旧实验脚本可能依赖其阶段性数据清单和环境，不能把所有历史测试都当成无需数据的当前发布测试。

## 本次打包范围

包含项目源码、必要运行模型、bit/HWH、依赖清单和启动/部署说明。
不包含原始DICOM/NIfTI/MetaImage病例库、训练缓存、个人连接密钥、虚拟环境、Vivado生成的综合布线缓存及个人网页结果。
需要医学测试数据时，按项目报告附录中的LDCT、LCTSC、LUNA16和COVID外部肺标注来源自行下载。
运行所需的固定自检向量和模型参数仍保留；历史LDCT交接中的原始病例切片及板测数组已从本次文件树移除。
原作者本机已有完整EXE便携包保持可用；本源码交付不把其数GB运行环境和私有配对配置上传GitHub。desktop保留外壳源码，跨电脑使用优先采用上述Python安装入口。

## 版本核验

GitHub默认发布分支继续为codex/ct-studio-current-20261027，本次不合入其他人的分支修改。
桌面RAR从本次Git提交导出，另外附有GIT_COMMIT.txt和PACKAGE_SHA256.json。RAR使用实际RAR5格式，不是改后缀的ZIP。
压缩完成后检查RAR完整性，并解压逐文件核对SHA256；GitHub推送后再次核对远端提交号。
真实硬件时延和准确率仍使用已有明确范围的验收记录，本次源码整理不会生成新的板测性能或医学精度结论。

仅用于工程教学与算法演示，不作为临床诊断依据。
''',encoding='utf-8')

desktop=PUBLIC/'desktop';desktop.mkdir(exist_ok=True)
(desktop/'README.md').write_text('''# 上位机EXE外壳源码

launcher/Launcher.cs是Windows启动窗口源码，app/desktop_bootstrap.py是实际环境、配对、启动和会话复用引导代码。
EXE外壳本身不执行CNN。运行时需要完整的app源码/模型/板端配套包、Python便携环境和对应的来源哈希；只编译一个EXE不能替代这些依赖。
作者现有便携目录已同步固定0.8738策略，并通过离线诊断。其私有凭据、病例和完整便携Python环境没有复制到本源码仓库。
新电脑请先使用仓库根目录setup_ct_studio_current.cmd和start_ct_studio_current.cmd，得到同一处理网页。
维护脚本python/integrate_desktop_dual_exact.py、python/update_desktop_exact_current.py保留了各阶段打包机制；这些维护脚本使用指定阶段的目录契约，不作为普通用户首次安装入口。
''',encoding='utf-8')

# The collection helper is a maintainer utility, independent of live services.
for name in ('prepare_sources.py','finish_sources.py'):
    dst=PUBLIC/'tools/release_20261031'/name;dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OUT/name,dst)
report=dict(date='2026-10-31',omitted_from_final_tree=omitted,
    reason='Legacy medical slice/board-output arrays and old archives containing them; algorithm sources retained.',
    original_author_workspace_data_deleted=False,history_rewritten=False)
(PUBLIC/'evidence/final_20261031/legacy_data_omissions.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(omitted_legacy_data_files=len(omitted),source_files_retained=True),ensure_ascii=False))
