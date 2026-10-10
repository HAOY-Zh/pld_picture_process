# PYNQ-Z2 医学图像处理项目

基于 PYNQ-Z2 / Zynq-7000 的医学图像处理工程，包含 Verilog 图像处理模块、
AXI DMA 数据通路、板端交互程序，以及肺部 CT 分割的离线开发代码。

**仅用于工程教学与算法演示，不作为临床诊断依据。**

## 当前进度

- 窗宽/窗位调节：已有 PL 模块、DMA 测试和滑块/手动输入界面。
- 阈值分割：支持单阈值及 HU 范围阈值，已有板端交互程序。
- 二值形态学：腐蚀、膨胀、开运算、闭运算均有独立 PL 模块和测试程序。
- 上述基础模块已进行板端软件参考结果对比；验证仅代表已运行的测试条件，
  不代表完整系统已集成，也不代表所有输入与性能指标均已验证。
- 肺分割：已有 CT/RTSTRUCT 读取、人工轮廓转换、规则基线、Dice/IoU 评价，
  以及轻量 2D U-Net 类模型的训练/评价代码。目前完成本地数据准备与 GPU
  前向/反向计算检查，**尚未正式训练、量化或部署到 PL**。

## 目录

| 目录 | 内容 |
| --- | --- |
| `fpga/` | Verilog、testbench、Tcl、Vivado `.xpr` 工程和 `.bd` 连接配置 |
| `pynq/` | 板端 Overlay 加载、DMA 测试和交互显示程序 |
| `matlab/` | CT 读取、基础处理与 RTL 对比的 MATLAB 代码 |
| `python/` | 肺分割数据准备、离线算法、模型训练/评价及工程测试 |
| `docs/` | 模块集成、数据转换和训练流程说明 |

## 开发环境与数据通路

- 开发板：PYNQ-Z2，器件 `xc7z020clg400-1`。
- 当前 Vivado 工程版本：2022.2。
- 板端：PYNQ / Python / Jupyter。
- 离线开发：Python 3.11；本机模型环境使用 PyTorch 2.8.0 + CUDA 12.6。

基础处理的数据流为 DDR → DMA MM2S → PL 图像处理模块 → DMA S2MM → DDR。
PS 负责数据组织、参数配置、传输控制和显示；对应基础处理的核心运算由 PL 执行。
离线肺分割模型代码当前在电脑 CPU/GPU 上运行，不是 FPGA 推理实现。

## 使用入口

1. 板端基础处理：查看 `pynq/ct_window_level_interactive.py`、
   `pynq/ct_threshold_range_interactive.py` 及各形态学测试脚本。
2. 腐蚀模块集成：见 [集成说明](docs/erosion_integration.md)。
3. 人工标注、规则基线和精度评价：见 [肺分割开发说明](docs/lung_segmentation.md)。
4. 模型数据准备、短训练和评价：见 [轻量模型训练说明](docs/lung_unet_training.md)。

程序内的数据路径、板端文件路径需按自己的环境配置。
现有 Vivado 工程及 Tcl 可能含本机路径或生成文件引用；换电脑后需检查源文件位置、
顶层与增量检查点设置，重新生成 Output Products、综合、实现和比特流。
本仓库提供开发源码和配置，尚不是已验证的一键重建系统。

### 离线 Python 基线环境

在项目根目录运行：

```powershell
python -m venv .venv-lung
.\.venv-lung\Scripts\python.exe -m pip install -r python/requirements-lung.txt
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -v
```

PyTorch 需按训练说明另行安装；没有 PyTorch 时模型测试会跳过。
当前完整环境测试还有一项 Windows 临时文件释放问题待修复，不能声称全部测试通过。
工程测试不等于医学精度验证。

## 数据与上传范围

肺分割使用公开 LCTSC CT/RTSTRUCT 数据，需自行获取并遵守原始数据使用条件。
数据集不随本仓库分发；患者级训练/验证划分与人工标注转换说明见 `docs/`。

`.gitignore` 排除本地 `data/`、虚拟环境、医学图像、模型权重、安装包、
Vivado 缓存/综合实现产物，以及由 CT 导出的 MATLAB 测试数据。
旧的 `fpga/Lab_9/` 课堂工程也不纳入本项目上传。被忽略的文件仍保留在本地。

PL 部署需要匹配的 `.bit` 与 `.hwh`，它们不进入源码提交；请在本地重新生成。
本仓库不包含 AMD/Xilinx 工具安装包或独立的第三方 IP 授权。
