# 项目交接：PYNQ-Z2 医学图像处理

交接日期：2026-10-10。请将本文件交给接手的 Codex，并让它先检查实际源码和运行环境。
本文区分「板端已验证」「电脑端已运行」「仅有代码」「尚未开始」，不要把计划当作成果。
第 1–8 节为当日历史快照；2026-10-11 的模型与测试更新见第 9 节。

## 1. 项目目标和当前结论

目标是在 PYNQ-Z2 上实现可交互的医学图像处理系统：公开 CT 输入、DDR 缓存、
PL 核心运算、输出显示，并进一步实现有人工标注评价的肺分割功能。

- 三类基础功能已有独立板端验证：窗宽/窗位、阈值分割、二值形态学。
- 腐蚀、膨胀、开运算、闭运算的核心运算均已有 PL 实现。
- 窗宽/窗位及范围阈值界面已有滑块和手动输入框，呈现在浏览器里的板端 Jupyter。
- 目前是多个独立 Overlay 和演示脚本，**不是所有功能已合并为一个连续处理系统**。
- 肺分割数据准备和轻量模型代码已经具备，但**还没有训练好的模型、量化模型或肺分割 RTL**。
- 所有处理仅用于工程教学与算法演示，不作为临床诊断依据。

竞赛基础要求中，CT 输入/缓存/处理/显示链路与三类基础处理已有演示基础；
智能分析、完整系统集成、稳定连续交互、统一性能和 FPGA 资源报告仍需继续完成。
神经网络量化与 FPGA 加速、2.5D/3D、完整同屏展示等拓展要求不能声称已经完成。

## 2. 仓库与环境

- 私有仓库：`https://github.com/HAOY-Zh/pld_picture_process`，队友需要仓库访问权限。
- 已推送 `main` 的首次提交：`98fee11`，包含 147 个源码、配置和文档文件。
- GitHub 返回首次推送成功；随后独立远端复核遇到网络超时。
- 本机路径：`D:\pld_picture_process`；队友电脑应使用自己的克隆路径。
- Vivado：2022.2；器件：`xc7z020clg400-1`；开发板：PYNQ-Z2。
- 板端目录：`/home/xilinx/ct_test`，账号曾使用 `xilinx`。
- 此前板端地址为 `192.168.2.99`、Jupyter 端口为 `9090`；网络变化后需重新确认。
- 电脑：Windows / PowerShell，Python 3.11，RTX 4060 Laptop GPU（8 GiB）。
- 本机 `.venv-lung` 已安装 PyTorch `2.8.0+cu126`，CUDA 可用，GPU 前向/反向测试通过。
- 本机环境依赖已有 NumPy/SciPy/pydicom/matplotlib；新机器需要自行创建环境、安装依赖。
- 本机 torch 安装包：`D:\pytorch\torch-2.8.0+cu126-cp311-cp311-win_amd64.whl`。

`.gitignore` 排除 `data/`、`.venv-lung/`、医学图像、模型权重、安装包、Vivado
编译产物、CT 导出的 MATLAB 测试数据及旧课堂工程 `fpga/Lab_9/`。
它们仍在原电脑上，但**不会随 Git 克隆得到**。没有删除原始 CT 数据。
本交接文件是在首次上传之后新增的本地文档，创建时尚未提交或推送到 GitHub。

## 3. 已完成的基础处理

| 功能 | PL 主要源码（`fpga/`） | 板端入口（`pynq/`） |
| --- | --- | --- |
| 窗宽/窗位 | `window_level.v`、`window_level_axis.v` | `ct_window_level_interactive.py`、`ct_window_level_full_test.py` |
| 单阈值 | `threshold.v`、`threshold_axis.v` | `ct_threshold_interactive.py`、`ct_threshold_pl_test.py` |
| HU 范围阈值 | `threshold_range.v`、`threshold_range_axis.v` | `ct_threshold_range_interactive.py` |
| 腐蚀 | `erosion_3x3.v`、`erosion_axis.v` | `ct_erosion_pl_test.py` |
| 膨胀 | `dilation_3x3.v`、`dilation_axis.v` | `ct_dilation_pl_test.py` |
| 开运算 | `opening_axis.v`，内部先腐蚀再膨胀 | `ct_opening_pl_test.py` |
| 闭运算 | `closing_axis.v`，内部先膨胀再腐蚀 | `ct_closing_pl_test.py` |

各功能另有 `tb_*.v` testbench。Vivado 工程位于 `fpga/pynq_z2_*_axis/`。
范围阈值当前在 `pynq_z2_threshold_axis` 工程中，不要仅凭工程名推断还是旧单阈值版本。

### 已记录的板端结果

用户已给出上述基础功能的通过截图。窗宽/窗位由 32×32 测试扩展到整张 512×512。
形态学整图测试使用同一张 CT 的 HU 范围 `[-500,300]` 生成输入二值掩膜：

| 运算 | 输入白像素 | 输出白像素 | 软件参考与 PL 输出 |
| --- | ---: | ---: | --- |
| 腐蚀 | 86886 | 63469 | `same=True`，差异像素 0，最大误差 0 |
| 膨胀 | 86886 | 103324 | 同上 |
| 开运算 | 86886 | 79704 | 同上 |
| 闭运算 | 86886 | 92071 | 同上 |

这些数字来自已记录的演示，不是在交接时重新运行板卡得到的。
它们说明测试条件下的软件/硬件一致，不证明全部输入正确、医学效果优秀或达到实时帧率。

### 必须保留的语义

- 范围阈值输出是二值掩膜：`lower <= HU <= upper` 输出 255，否则输出 0。
  范围覆盖所有 HU 时输出全白是正确结果，不应要求它保留原图灰度。
- 阈值比较使用有符号 16 位 HU；负阈值打包不能直接当作无符号数比较。
- 窗宽/窗位软件参考要与当前 RTL 的整数除法/截断一致，不能随意改成四舍五入。
- 二值形态学用居中的全 1 的 3×3 核，非零输入为前景，输出 0/255。
- 每一级使用图像外补零；腐蚀及当前闭运算最外边界为黑。
  当前闭运算可能删除边界前景，所以截图出现少量 removed pixels 不必然是错误。
- 在 `ct_*_pl_test.py --ct` 独立形态学演示中，**PS/Python 先准备阈值掩膜，PL 做形态学**。
  不能把这个测试写成「阈值 + 形态学全部在 PL 的整条链路已验证」。
- `docs/erosion_integration.md` 包含阈值串接腐蚀的方案；方案文档不等于该组合已经完成板端验证。
- 腐蚀和膨胀可独立使用，也可组成开闭运算；以后可串接清理分割掩膜，不应强制永久绑定阈值模块。

## 4. BD 连接和接口约定

控制通路：PS `M_AXI_GP0` → `smartconnect_0` → DMA `S_AXI_LITE`，以及有参数需求时的 AXI GPIO。
DMA 访存：`M_AXI_MM2S`、`M_AXI_S2MM` → 两输入一输出的 `smartconnect_1` → PS `S_AXI_HP0`。
数据流：DMA `M_AXIS_MM2S` → 对应 PL 模块 `s_axis` → `m_axis` → DMA `S_AXIS_S2MM`。

- 当前采用单个 PS `FCLK_CLK0`，100 MHz，连接相关 AXI/DMA/算法时钟及 PS AXI 接口时钟。
- `FCLK_RESET0_N` 接复位模块；外设通常用 `peripheral_aresetn`，核对复位极性。
- DMA 使用 Simple 模式（关闭 Scatter-Gather），长度寄存器位宽已改为 23。
- 已完成版本的 AXI-Stream 为 32 位：每个 beat 对应一个像素，不是一个 beat 打包四个像素。
- HU 输入取 `TDATA[15:0]` 的有符号值；掩膜/灰度输出在低 8 位，高位为零。
- 形态学输入取低字节，DMA 缓冲通常用 `uint32`，整帧 512×512 共 1 MiB。
- 完整像素 word 的 `TKEEP=0xF`，整帧最后一个 word 标 `TLAST`。
- 形态学几何默认为 512×512；涉及行缓存/补边，不是可直接任意尺寸运行的模块。
- 范围阈值 GPIO 为一个 32 位通道：低 16 位是下限，高 16 位是上限，均编码为 int16。
- 窗宽/窗位当前用双 GPIO 通道：channel1 为有符号 WL，channel2 为无符号 WW，各 16 位。
- 当前多个 wrapper 采用保守握手，**不能声称一像素/时钟**；行缓存也不能未看综合就声称用了 BRAM。

修改 BD 后需 Validate Design、生成 Output Products、更新 HDL Wrapper，
以 `design_1_wrapper` 为综合顶层，再综合/实现/生成比特流。
误把裸算法模块设为顶层，会把 AXIS 端口当板外 IO，可能在写 bit 时出现 NSTD-1 / UCIO-1。
不要通过降低 DRC 严重级别来绕过这个顶层错误。

PYNQ 加载需要**同一次构建、同名配对**的 `.bit` 和 `.hwh`，放在板端目录；
`Overlay(path, download=True)` 会配置 PL，不需要再额外用 Vivado 烧录一次。
元数据缺失报错应检查 HWH；找不到 `axi_dma_0` 时先检查 `overlay.ip_dict`、IP 实例名及文件配对。
已有 `.xpr`/Tcl 可能含本机路径或增量 DCP 引用，换电脑不能保证一键编译成功。

## 5. 肺分割数据与人工标注

数据集：公开 LCTSC，CT + RTSTRUCT，任务参考为专家左肺/右肺轮廓的并集。
它不是「低 HU 都算肺」，也不是病灶分割或左右肺分类任务。

本机原始路径为 `D:\CT2\lctsc`。清点记录共 60 个病例目录、6827 张 CT、60 个 RTSTRUCT。
19 个目录只有标注没有 CT，暂不用于开发：

- `LCTSC-Train-S1-001` 至 `LCTSC-Train-S1-012`。
- `LCTSC-Test-S1-102` 至 `LCTSC-Test-S1-104`。
- `LCTSC-Test-S1-201` 至 `LCTSC-Test-S1-204`。

24 个 Train 和 17 个 Test 通过文件头及引用缺失清点。17 个 Test 是候选，
不能声称其所有像素和轮廓已完成检查；它们的图像未用于目前模型训练/验证。
**不能因为完整病例分割效果差而删除它，也不能使用测试集调参。**

已完成的训练缓存：`data/lung_learning_20261009/dataset.json`，状态为 `complete`，
24 个病例成功、失败 0：18 个训练病例/3002 张切片，6 个验证病例/1120 张切片。
固定 seed 为 `20261009`，患者级、机构分层划分；验证病例为：

```text
LCTSC-Train-S2-006
LCTSC-Train-S2-007
LCTSC-Train-S2-008
LCTSC-Train-S3-002
LCTSC-Train-S3-003
LCTSC-Train-S3-008
```

此前看过并用于算法诊断的 S2-001、S3-001 固定留在训练组。
不要随机拆分同一患者的相邻切片到训练/验证两侧，也不要把验证集 Dice 称为独立测试精度。
S1 训练 CT 缺失，不能声称当前开发验证覆盖三家机构。

### 数据转换的保护检查

`python/lungseg/dicom_io.py` 根据物理坐标排序、用 RescaleSlope/Intercept 得到 HU，
检查切片引用 SOP UID、Frame of Reference、图像方向/间距及轮廓所在平面，不按文件名猜位置。
支持平面闭合轮廓及 XOR 轮廓；当前只支持满足检查条件的经典单帧 CT 网格。
左右肺偶有重叠：模型缓存保留参考信息、记录重叠，二肺并集只计一次，不擅自修改标签。
像素化极小有效轮廓可能覆盖不到任何像素中心，已修复为允许空掩膜并记录计数；
物理坐标/平面错误检查仍保留，不能把该修复扩大成忽略全部对齐错误。

每病例缓存含 `images.npy`、`labels.npy`、`reference_native.npy`、`case.json` 和
`reference_alignment.png`。需继续抽查人工轮廓对齐；转换成功不等于全量人工审核完成。

本机清点和早期分析：`data/lung_validation_20261009/download_audit/`、
`data/lung_validation_20261009/initial_findings.md`。这些产物不在 GitHub，需要另外提供或重建。

## 6. 算法基线和轻量模型

### 已运行的 CPU 规则基线

`HU < -400` → 2D 8 邻域连通域 → 排除图像边缘连通区域 → 最小面积 25 mm² →
保留最大两个候选 → 一次 3×3 闭运算 → 填孔。

- 训练病例 S2-001：160 层，体素合并 Dice 0.919944、IoU 0.851756。
- 训练病例 S3-001：204 层，Dice 0.359736、IoU 0.219316，明显误选扫描床附近包围的空气。
- 这是两个训练病例上的 CPU 基线，不是神经网络结果、独立测试精度或 FPGA 分割结果。
- 因此暂不把简单规则冻结成最终肺分割硬件，先验证学习型方案。

### 已写代码、尚未正式训练的模型

入口 `python/lung_unet.py`，子命令 `prepare`、`smoke`、`train`、`evaluate`。

- 小型 2D U-Net 类网络，通道 8/16/32/64，三次下采样。
- 3×3/1×1 Conv、BatchNorm、ReLU、MaxPool、最近邻上采样、skip 拼接。
- 输入由原切片缩放为 256×256，HU 裁剪至 `[-1000,400]`，归一化到 `[0,1]`。
- 训练图像缓存 float16，训练标签最近邻缩放；包含无肺切片。
- Loss：BCEWithLogits + soft Dice；成对水平翻转；Adam；CUDA 混合精度。
- 评价把概率图恢复到原始尺寸后以 0.5 阈值二值化，与未缩放人工标签比较。
- 每轮选择模型依据六个验证病例各自整体 Dice 的平均值。
- 已有 `last.pt`/`best.pt` 保存逻辑、可信 checkpoint 恢复、患者划分指纹检查和训练日志代码。
- 评价代码包含规则基线对照、代表层及较差层图像；**这不等于已有模型评价结果**。

目前本机没有 `.pt` 权重，正式训练及两轮短训练均未开始。
GPU smoke PASS 只说明前向、反向和参数更新可运行，不说明分割准确。
尚未完成 INT8/INT4、BN 折叠、PL 卷积/缓存设计或 FPGA 推理。
网络 MAC 数不能替代 LUT/FF/BRAM/DSP 或端到端 FPS 报告。
现有模型与 CPU 基线的计时范围不同，不能直接用其时间相除宣称加速比。

## 7. 当前明确的问题与下一步

最近一次完整测试：20 项，19 项通过、1 项 ERROR。
失败项：`python/tests/test_lung_unet.py` 的 `ModelTests.test_summary_and_slice_indexing`。
失败发生在退出 `TemporaryDirectory` 清理时：`SliceDataset` 持有 `np.load(..., mmap_mode='r')`
数组，Windows 无法删除仍打开的 `images.npy`（WinError 32）。不是网络下载失败。
当前 `SliceDataset` 还没有显式关闭映射的方法。不要简单忽略清理异常就宣称全部通过。

建议接手顺序：

1. 为数据集及相关读取流程补上可靠的资源释放，在测试/训练结束或异常路径正确调用，重新跑全套测试。
2. 检查若干训练/验证病例人工轮廓与 CT 对齐，保留固定患者划分和难例。
3. 在 GPU 上做两轮短训练，验证整个训练、保存、恢复和评价流程。
4. 在验证病例上看 Dice/IoU、空肺层假阳性及失败切片，不只展示最好的一层。
5. 完整流程稳定后再决定正式训练轮数，固定模型后才做独立测试和量化精度对比。
6. 精度与算力约束明确后，再划分 PS/PL 推理模块、设计量化卷积和缓存；暂不直接开始新的肺分割 RTL。
7. 完整系统整合时再做处理模式选择、连续处理、分割掩膜/轮廓叠加、FPS/延迟/资源显示等。

### 原电脑可继续使用的命令

从项目根目录执行；原电脑已有环境和缓存，不要重新下载/覆盖它们：

```powershell
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -v
.\.venv-lung\Scripts\python.exe python/lung_unet.py smoke --device cuda
.\.venv-lung\Scripts\python.exe python/lung_unet.py train --dataset data/lung_learning_20261009/dataset.json --outdir data/lung_training_20261010/short_run --epochs 2 --device cuda
.\.venv-lung\Scripts\python.exe python/lung_unet.py evaluate --dataset data/lung_learning_20261009/dataset.json --checkpoint data/lung_training_20261010/short_run/best.pt --outdir data/lung_training_20261010/short_run_evaluation --device cuda
```

输出目录应为新目录；同名已有结果时先检查，不删除已有结果。
明确指定 CUDA 而不可用时应报告原因，不默默改 CPU 后称 GPU 训练成功。

新机器没有本机缓存时，可根据自己的数据路径重建（先安装基线依赖，目录要新建）：

```powershell
python python/lung_segmentation.py audit --input "D:\CT2\lctsc" --outdir data/lung_validation_20261009/download_audit
python python/lung_unet.py prepare --input-root "D:\CT2\lctsc" --audit data/lung_validation_20261009/download_audit/audit.json --outdir data/lung_learning_20261009
```

`prepare --resume-cache` 仅用于本流程生成的匹配缓存，重试失败病例而不更改划分。
如果队友下载的病例集合不同，应先核对患者集合，不能把新的划分结果当成原实验直接比较。

## 8. 接手时优先阅读

- `README.md`：仓库范围、状态和环境。
- `docs/lung_unet_training.md`：模型训练/恢复/评价流程。
- `docs/lung_segmentation.md`：CT、RTSTRUCT、规则算法与参考评价。
- `python/lung_unet.py`、`python/lungseg/`、`python/tests/`：当前电脑端工作重点。
- `fpga/*.v`、`fpga/tb_*.v`：已实现硬件及接口，注意签名、边界和背压。
- `pynq/ct_*_interactive.py`、`pynq/ct_*_pl_test.py`：板端交互与演示的真实 PS/PL 分工。
- `docs/erosion_integration.md`：接口/边界/握手说明，组合链路要另行验证。

用户希望分步骤推进，关心性能和准确性，不能以「改硬件麻烦」为由回避必要优化。
同时要清楚说明每一步在电脑、PS、PL、浏览器的哪个位置执行。
保留已有源码与原始数据；除明确要求外，不删除文件、不上传患者数据、不把未验证功能写成完成。

## 9. 2026-10-11 接续更新

- 已修复 Windows 内存映射文件占用，22 项工程测试全部通过，GPU smoke 通过。
- 固定原患者划分，实际完成两轮 GPU 训练，参数 122321 个，保存了 `best.pt` 和 `last.pt`。
- 六病例原始分辨率平均验证 Dice 0.9591，固定 CPU 规则基线 0.5959。
- 已保存逐病例指标、原尺寸预测、正常层及失败层对比；小面积肺切片仍有明显漏分，空肺层仍有假阳性。
- 产物位于本机 `data/lung_training_20261011/{short_run,short_run_evaluation}`，未上传 GitHub。
- 以上为短训练开发验证，不是独立测试精度。正式训练、量化、肺分割 PL 推理与系统集成尚未完成。
- 已有基础 FPGA 模块、原始 CT、人工标注和患者划分均未修改；没有读取官方测试病例图像。
- 下一步先做较完整训练和失败层分析，再固定模型进行独立评价，然后考虑量化/硬件。

详细逐病例结果和命令见 [模型训练说明](lung_unet_training.md)。

## 10. 2026-10-11 早停续训更新（接替上一节两轮状态）

- `python/lung_unet.py` 增加早停（默认 patience 8、Dice min_delta 0.001）、
  验证平台期学习率减半（patience 3、下限 1e-5）、训练曲线和摘要。
- 新断点保存调度器与早停状态，续训不重置。兼容旧两轮断点，并把历史最佳权重带到新目录；
  如果 last 不是 best，要求匹配的同目录 best.pt。已早停断点不会静默重新启动。
- 原患者划分、人工标签、模型结构、256×256 输入和阈值 0.5 不变。
- 实际从第 2 轮续训至第 13 轮，触发早停；最佳模型是第 12 轮，平均病例验证 Dice 0.966576。
- 27 项工程测试全部通过。
- 空参考切片出现任意前景：205/516 降至 40/516；空层 FP 像素 22917 降至 11351。
- 49 张小肺区切片平均逐切片 Dice：0.499085 降至 0.462885，尚未解决小区域漏分。
  S2-006 第 25 层现在全部漏分，Dice 0。不要只展示整体改善或删除困难层。
- 产物在 `data/lung_training_20261011/{early_stop_run,early_stop_evaluation}`；
  原 short_run 和 short_run_evaluation 保留，当前改动和权重未提交/推送 GitHub。
- 这是电脑 GPU 开发验证，不是独立测试、INT8 或 PL 推理。
  下一步优先针对小区域做受控改进，固定模型后再独立测试和量化/硬件实现。

## 11. 2026-10-11 小肺区受控优化更新

- 实际完成两组 GPU 训练及六病例完整评价，各最多 13 轮；每组从同一随机种子初始化，
  保持患者划分、人工参考、模型结构、阈值 0.5 与最佳模型选择标准。
- 512 输入实验：重读原 DICOM（不是放大旧缓存），逐病例校验原尺寸标注 SHA256；
  24 个病例的划分、身份、顺序和标注保持不变。选中第 10 轮，整体 Dice 0.963092，
  49 张小肺区平均 Dice 0.427707，空层误分 41/516、11301 像素。
  算量变为 4 倍但未改善目标，因此不采用，也不声称所有高分辨率方案无效。
- 原 256 早停最佳模型：整体 Dice 0.966576，小区 Dice 0.462885，
  空层误分 40/516、11351 像素；仍保留为当前基线。
- 256 加权采样实验：仅依据训练患者原尺寸人工标签，为 149 张小肺区切片赋权 4；
  有放回抽样，每轮仍抽 3002 张，期望小区占比从 4.96% 变为 17.28%。
  不改标签、不使用验证或官方测试病例计算权重，默认训练仍为原均匀采样。
  选中第 13 轮，整体 Dice 0.965430，小区 Dice 0.521037，空层误分 52/516、47698 像素。
  网络参数和推理算量不变，但误分增加，因此是局部改善，不替换原模型。
- S2-006 索引 25：原模型完全漏分（Dice 0）；加权模型 Dice 0.670659，
  TP=280、FP=249、FN=26。配对图保留旧失败切片，不只挑好的层。
- 代码新增 `rebuild-resolution`、可配置 GPU smoke、训练 `--small-lung-weight`，
  以及评价 `--compare-evaluation`。断点恢复禁止静默改变采样设置。
  新比较图 `model_comparison_slice_*.png` 的黄色是旧模型；
  原 `comparison_slice_*.png` 的黄色仍是规则基线，不要混淆。
- 34 项工程测试通过；完整命令、结果表和评价口径见 `docs/lung_unet_training.md`。
- 本机新产物：`data/lung_learning_20261011_512/`；
  `data/lung_training_20261011/{resolution_512_run,resolution_512_evaluation,small_sampling_run,small_sampling_evaluation}`。
  原缓存、模型及基础 PL 模块未覆盖。本轮没有生成新比特流，也没有提交/推送 GitHub。
- 仍未完成独立测试、INT8/INT4、智能分割 PL 推理；不要把 GPU 结果称为板端结果。
  下一步目标是降低小肺区漏分且控制空层假阳性，之后固定模型再做独立测试与量化。

## 12. 2026-10-11 相邻三切片 2.5D 实验更新

- 用户同意继续肺部提取的相邻多切片实验，未来系统可提供 2D/2.5D 模式；
  本轮没有实现独立 3D 浏览/重建、3D 卷积或肺叶分割，也没有修改 PL 模块。
- `train` / `smoke` 新增 `--context-slices 1|3`，默认 1 保持旧 2D 模式。
  三通道仅来自同一患者同一序列的前/中/后物理层，首尾重复最近边缘层；
  只使用中间层人工标签，不跨患者取邻居，不把相邻掩膜或人工答案送入模型。
  三通道与中心目标同步翻转；训练/推理共享邻层提取逻辑。
- 训练与评价前读取源 CT 文件头，核对患者、序列、Frame、SOP、物理顺序、
  形状、方向、间距及无剪切，失败则停止。24 个病例通过，层间距 1.25..3 mm。
  本版不统一跨病例的毫米范围，也不重新缩放或改写原标签/缓存。
- 新断点保存输入通道模式；旧断点没有该字段时按 1 通道恢复。
  禁止在续训时混用 2D/2.5D 权重；评价自动从权重恢复模式，报告/图片明确标注。
- 39 项工程测试全部通过，包括三通道训练、续训、评价，GPU smoke 通过。
  参数 122465；256 输入每切片卷积 MAC 675282944，比原 2D 增加约 1.42%。
  这不是实际 GPU/PL 加速比；输入样本有三层，还需单独核算带宽与缓存。
- 用原 18 个训练/6 个验证患者，256 输入、阈值 0.5、普通均匀采样和原损失；
  同一种子重新初始化，最多 13 轮，第 11 轮早停，选中第 8 轮。
  与旧基线的学习率轨迹/初始化参数不完全相同，不声称多种子或充分收敛结论。
- 实际完整评价：整体 Dice 0.961381、小区 Dice 0.227280；
  空层误分 13/516、23665 像素；原 2D 对应 0.966576、0.462885、40/516、11351。
  误分张数下降但像素总数增加，且小区漏分更重，因此不替换原 2D 基线。
  S2-006 索引 25 仍全漏分（FN=306）。不得挑第 3 轮的小区较好指标冒充最终选择结果。
- 新产物在 `data/lung_training_20261011/{context_25d_run,context_25d_evaluation}`，
  配对图 `model_comparison_slice_*.png`：青=专家，黄=旧2D，绿=本轮2.5D，
  最后列红=多分、蓝=漏分、灰=正确。旧结果、模型和数据全部保留。
- 未读官方测试像素，未完成独立测试、量化或智能分割 PL 推理；本轮没有 Git 提交/推送。
  可以保留 2.5D 作为研究模式，当前不应把它称为准确性优化成功或完整拓展功能已验收。

## 13. 2026-10-11 冻结原 2D 模型后的独立测试

- 用户要求将全部可用预留病例用于独立测试，已实际完成，而不只是划分清单。
  新入口 `python/lung_heldout.py`；说明 `docs/lung_heldout_evaluation.md`。
  原 `lung_unet.py` 对官方 Test 进入训练的禁止保持不变。
- 固定 `data/lung_training_20261011/early_stop_run/best.pt`（epoch 12），
  2D、256 输入、原 HU 归一化、概率阈值 0.5。没有训练、换权重或调参。
  读取测试数据前保存权重/代码/开发清单 SHA256 和测试协议。
- 下载清单全部 24 个官方测试目录均尝试。17 个预留候选全通过并评价，2705 层；
  7 个 S1 目录仍无 CT，均只有 RTSTRUCT。未因低 Dice 剔除病例。
  原开发 18 train / 6 validation 不变；41 个可用病例均进入各自用途。
- 患者哈希、Frame、Series、CT SOP 与开发集及其他测试病例无重复；
  物理层序、间距、方向、形状及 RTSTRUCT/CT 关联通过检查。
  保留空参考层；人工标签原始尺寸并集，不用软件阈值充当人工标签。
- 平均病例 Dice **0.9654666003861169**，IoU **0.9334746463660962**，
  精确率 **0.9623008563500502**，召回率 **0.9691597236954715**；
  最低 S2-203 **0.9448277795021224**，最高 S3-104 **0.9846515763510415**。
  固定规则基线同批平均病例 Dice **0.6087231653430938**。
- 小肺区 100 层逐切片 Dice **0.6148025433651741**；
  中区 478 层 **0.9216768577287392**，大区 953 层 **0.968925720032602**；
  空参考 1174 层，有前景 67 层，FP 18828 像素。小区漏分、空层误分仍需关注。
- 产物 `data/lung_training_20261011/heldout_2d_evaluation/`，含冻结协议、
  缓存清单、覆盖/失败统计、`evaluation.json`、`case_metrics.csv`、每病例
  `prediction_native.npy` 和最大肺区/最差非空层/空层错误的对比图。
  青=人工、黄=规则、绿=模型；误差图红=多分、蓝=漏分、灰=正确选中。
- 43 项工程测试通过；全部 17 个保存的预测/参考重新算指标一致，输入/参考缓存、
  冻结权重/代码/开发清单哈希核对不变。原数据、历史实验和权重未覆盖。
- 这是可用 17 例子集，不是官方全 24 例或临床评价；仍是电脑 GPU 推理，
  不是 PL 推理。没有新增比特流、INT8/INT4，没有 Git 提交或推送。
  如果看到这些分数后再针对它们调算法，这批就不再是新的盲测，需要另留独立集合。

## 14. 2026-10-11 INT8 静态量化的电脑验证

- 用户同意在 COVID 数据下载期间做量化验证。新增 `python/lung_quantize.py`、
  `python/tests/test_lung_quantize.py`、`docs/lung_int8_quantization.md`；未修改 PL。
- 冻结原 2D epoch 12；所有 18 train / 3002 层无标签校准，所有 6 validation /
  1120 层评价。不使用已看过的 17 测试病例或新 COVID 数据，不训练、不调阈值。
- 复制原模型，折叠 14 个 BN；所有训练输入 logits 对拍最大误差 0.000185013，
  rtol/atol=0.0002 通过。GPU 校准观察器，CPU x86 转成 15 个真实量化卷积。
  权重 INT8 逐输出通道，激活 UINT8 affine，skip concat 重缩放；卷积没有浮点残留。
  Sigmoid/原始尺寸概率插值和输入预处理仍浮点在核心外。
- 首次 `int8_ptq_validation/` 在 GPU→CPU 转换时失败，未跑真实验证病例；
  已保留 failure.json。修复 observer 工厂闭包仍指向原 GPU 模块的问题，
  不重置校准激活，不改变量化策略；新目录 `int8_ptq_validation_v2/` 已完整完成。
- 本轮 FP32 平均病例 Dice **0.966575711271031**，INT8 **0.9664116430184805**，
  下降 **0.00016406825255055324**；IoU 0.935520891→0.935219276。
  小区 49 层 Dice **0.46290112771513486→0.4625200571207048**；
  空参考层有前景均 40/516，FP 像素 **11353→10773**。通过预声明的 Dice 工程筛查，
  不代表小区问题解决或临床验收。FP32 关闭 TF32 重算，与历史微小差异不代表改权重。
- 实际 INT8 权重 121616 字节，bias FP32 1412 字节，合计 123028，不含 scale 等；
  同一折叠浮点核心权重+bias 487876 字节。卷积 MAC 665845760 / 切片不变。
  naive 三条 full-frame skip 总 917504 字节；后续需 DDR/BRAM 调度，非资源报告。
- 导出实际权重/scale/zero point/bias，以及明确标为 proposed 的 INT32 bias；
  保守累加最大上界 7053984。UINT8 zero point 需要补偿，不能直接强转有符号。
  CPU 量化后端不等于 RTL 位精确 oracle；固定 multiplier/shift、bias、round/saturate
  仍需规范/对拍。不能把 int8_model.pt 当成比特流烧录。
- 48 项全工程测试通过，含 CUDA→CPU 回归、15 量化卷积/3 concat、权重导出、
  重载输出逐值一致、只读训练输入不读标签、拒绝 validation/Test 校准。
- 新产物包含 protocol/calibration/core_manifest/evaluation、int8_model.pt、
  int8_core_parameters.npz、逐病例 CSV、两种原始尺寸预测和困难层比较图。
  data 被 Git 忽略；原权重/开发清单哈希不变，没有 Git 提交或推送。
- 当前实现依赖现有 PyTorch 2.8 CPU x86 量化后端，该 eager API 有弃用提示；
  未升级或新增依赖。迁移 Torch/CPU 后端后必须重新验证，不保证可直接在板卡跑。
  下一步需先定义硬件位精确算术、缓存/DDR计划；外部数据下载好后另做冻结方案测试。

## 15. 2026-10-12 COVID CT 外部测试完成

- 用户下载 `D:\CT3\Covid-19.zip`（国内镜像）和 `Lung_Mask.zip`（官方标注）。
  原压缩包保持只读，不全量解压覆盖。官方标注 MD5
  `972b3d9b4c7b64d2c518b936bd98cb47` 匹配；镜像 ZIP 不是官方原 ZIP。
- 新增 `python/lung_external.py`、`python/tests/test_lung_external.py`、
  `docs/lung_external_evaluation.md`；仅将 NiBabel 5.x 加入离线依赖与当前电脑虚拟环境。
  没改旧模型、训练划分、量化策略、板卡程序或 RTL；没升级 PyTorch。
- 读外部像素前冻结 FP32 epoch 12 与对应 INT8 权重、代码、旧开发/测试清单与源 ZIP 哈希。
  主测试预声明为 `coronacases_001..010` 原 HU；其他十份 Radiopaedia 是窗口化
  UINT8 0–255，完整审计后编码排除，不输入 HU 模型，也不按低 Dice 剔除。
- 全部 20 CT / 20 官方人工肺部标注完成 CRC、解码和物理网格检查，无数据失败。
  标签 0/1/2 的左右肺并集，不用感染标注、不修改/插值专家标签。
  通过 affine 做无损轴置换/翻转转到 LPS 再 ZYX；旧 24 开发/17 测试 DICOM
  均为 `[1,0,0,0,1,0]`，输入方向匹配。十份最大肺区标注叠加图均查看，无明显错位。
- 镜像 CT 单位头为 unknown，官方标签为 mm；按同一数值 affine 的官方 mm 网格解释。
  NIfTI 不含原始 PatientID/SOP；无完全相同归一化图像缓存，但不能据此宣称已证明
  患者身份完全独立或检测所有重采样重复数据。
- 主测试全 10 份 / **2581 层**（含空层）完成：平均体数据 Dice
  **FP32 0.9585901534399891，INT8 0.959075389033645**；
  IoU 0.9212548419246211 / 0.922127966041878。
  最低 003：0.8991588651570478 / 0.9005771904903374；最高 002：约 0.97333 / 0.97349。
- 小区 80 层 Dice **0.5768270293718732 / 0.5719930609246198**；
  空参考 425 层有前景均 **74** 层，误分像素 **72625 / 71429**。
  003 肺内较亮区域明显漏分。量化整体影响小，不等于泛化失败或小区/空层问题已解决。
  与此前 LCTSC 测试人群不同，不能直接把均值当同条件模型排名/诊断正确率。
- 53 项全工程测试通过，新增轴置换/翻转、真实坐标错位、斜切/冲突 q/s、非法标签、
  灰度冒充 HU、冻结/权重关联、真实 FP32/INT8 输出保存和失败中止测试。
  十份两种保存预测重新计算指标一致；源 ZIP、旧模型/代码/清单哈希检查通过。
- 产物 `data/lung_external_20261012/{prepared,evaluation}`；含 frozen protocol、
  全包审计、排除原因、HU/参考/输入缓存、十份原尺寸预测、CSV、逐层指标与对比图。
  青=人工，黄=FP32，绿=INT8；红=多分，蓝=漏分，灰=正确。
  困难示例 `evaluation/coronacases_003/comparison_slice_080.png`。
- 仍是电脑 GPU FP32 / CPU x86 INT8；GPU/CPU时间不能当板端 FPS 或同硬件加速比。
  PYNQ-Z2 的 DSP/LUT/BRAM 使用、DDR 调度、定点位精确实现与智能 PL 推理仍待验证。
  本轮没有 Git 提交或推送。若根据此次失败调算法，这批外部数据已被观察，
  不能再宣称后续优化是在全新盲测集验证；应另留患者级独立数据。
