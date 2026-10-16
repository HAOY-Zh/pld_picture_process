# 二维 CT 刚性配准原型

本模块面向题目一拓展要求④，提供二维同模态图像的平移和旋转配准。
与已有去噪、肺分割模块独立开发；可接收统一窗宽窗位后的 CT 灰度图。
本交付是算法与硬件接口原型，不表示整套竞赛系统或真实医学配准已完成。
仅用于工程教学与算法演示，不作为临床诊断依据。

## 当前状态

- Python 软件参考、搜索、定点插值和 SSD 数据格式已有自动测试。
- 合成图像演示采用已知变换，结果位于 `demo/synthetic_cpu/`；不是患者 CT。
- `fpga/registration_axis.v` 提供帧缓存、几何变换、双线性插值和 SSD 累加 RTL。
- 默认 128×128 RTL 已通过 Yosys 前端和逻辑连接检查；功能仿真仍待完成。
- `fpga/tb_registration_axis.sv` 提供 6 帧自检、输入间隙、输出背压及协议错误检查。
- `pynq/registration_board.py` 提供预期 Overlay 的 DMA 接入代码，尚未上板验证。
- 当前包没有 `.bit/.hwh`，没有 Vivado 工程，没有真实配对 CT/CT-MRI 验证数据。
- 软件测试记录及硬件检查的实际执行状态见 `VERIFICATION.md`。

## 覆盖题目要求的方式

| 要求 | 实现位置 |
| --- | --- |
| 几何变换 | PS 产生旋转、平移参数并转换为逆向采样矩阵；PL 逐行、逐像素累加源坐标 |
| 插值 | PL 读取源图 4 个邻域像素并进行定点双线性插值 |
| 代价函数计算 | PL 累加有效重叠区域的 SSD，并输出有效像素数；PS 归一化为 MSE |
| 参数优化 | PS 有界粗到细坐标搜索，逐候选调用 PL 评价器；不宣称优化器也在 PL |
| 结果显示 | 软件输出配准前后双色叠加与误差图；后续接统一界面 |

二维 CT 刚性配准是图像配准的一种工程路线，最终适用性需以实际数据证明。
当前 SSD/MSE 方案面向同一灰度映射下的同模态输入；不能直接称为 CT-MRI 配准。
不同模态需要适合其强度关系的度量（如互信息）和实际成对影像，并重新验证。
二维算法不解决真实呼吸变形、离面运动或三维解剖变化。

## 软件运行

Python 3.11/3.12，依赖 `numpy` 和 `Pillow`。DICOM 输入另外需要 `pydicom`。
从本包根目录运行，命令中的 `python` 应替换为自己的项目虚拟环境解释器。

```powershell
python -m unittest discover -s tests -v
python python/ct_registration.py demo --outdir demo/my_synthetic_run
python python/ct_registration.py demo --fixed path/to/fixed.png --moving path/to/moving.png --outdir demo/my_pair
```

输出目录必须是新目录。支持 PNG、二维 uint8 NPY 及单张 DICOM/IMA。
DICOM 在软件端按同一窗宽窗位映射到 8 位灰度，保留原始文件不变；这不是三维
DICOM 配准器。第一版两张图尺寸必须相同，软件不默默缩放或重采样。
图像方向、物理层面、间距和切片对应关系应由数据准备流程确认。
真实配准需要有意义的初始位置及重叠；默认搜索范围 ±8°、±10 像素。
这是有界局部搜索，不保证任意输入都找到全局最优，也不检测所有错误配准。

矩阵含义固定为：

```text
moving_x = a * fixed_x + b * fixed_y + c
moving_y = d * fixed_x + e * fixed_y + f
```

这称为输出坐标到输入坐标的逆向采样，不要把矩阵方向反过来。
源图像域外输出 0，并以有效位排除出 SSD。边界恰好落在最后一行/列时有效，
插值邻点钳位在最后一行/列。MSE = SSD / 有效像素数；无重叠或重叠比例不足的
候选在 PS 拒绝。实际评价应增加固定 ROI/同一覆盖区域、对应点误差和失败案例，
避免通过缩小重叠范围获得表面更低的误差。

## 定点及 AXI 协议

- 32 位 AXI Stream，一个 word 对应一对固定图/浮动图像素。
- 6 个有符号 32 位系数采用 Q16.16；PS 四舍五入量化。
- 坐标累加器为有符号 32 位，PS 应检查整个输出域内不溢出。
- 插值使用源坐标的小数高 8 位构造权重；最终加 32768 后右移 16 位。
- 输入连续提供 `N = WIDTH*HEIGHT` 个 word，`TKEEP=0xF`，最后一个 `TLAST=1`。
- 输入 `[7:0]` 为 fixed，`[15:8]` 为 moving，高位保留。
- 输出前 N 个 word：`[7:0]` 为配准图，`[15:8]` 为 fixed，`[16]` 为有效位。
- 输出随后 4 个 word：SSD 低 32 位、SSD 高 32 位、有效像素数、协议状态。
- 输出 `TLAST` 在第 N+4 个 word，而不是最后一个图像像素。
- 状态 bit0 表示输入 TKEEP 不全，bit1 表示输入 TLAST 位置错误。
- 未满 N 个输入像素会等待后续数据，不提供包丢失恢复；联调时异常要复位整个通路。
- 支持输出背压，输出 word 在 valid=1、ready=0 时保持。
- 系数在每帧首个输入握手时锁存；新一帧必须等前一帧输出结束。

不能把这个协议直接替换为 AXI 视频的逐行 TLAST。

## PYNQ 与 Vivado 接入

先保持独立工程。复制既有 DMA 工程到自己新建的位置，然后：

1. 添加 `fpga/registration_axis.v`，将默认几何设为 128×128。
2. `DMA M_AXIS_MM2S -> registration_axis -> DMA S_AXIS_S2MM`。
3. 维持 32 位流、Simple DMA 和匹配的 DDR HP 接口；单个 100 MHz 时钟可先沿用。
4. 增加 3 个双通道输出、每通道 32 位的 AXI GPIO，命名 `coeff_ab`、`coeff_cd`、`coeff_ef`。
   channel1/channel2 分别连接 a/b、c/d、e/f。GPIO 由 PS AXI 控制。
5. 保持已有复位结构和 `design_1_wrapper` 顶层。Validate、仿真、综合、实现、生成 bit。
6. `.bit` 和 `.hwh` 使用同一次构建的匹配文件。DMA 接收缓冲长度是 N+4。
7. 用 `pynq/registration_board.py` 的 `RegistrationBoard` 加载实际路径，先测试恒等变换，
   再测试整数/小数平移、旋转、边界、无重叠和连续多帧，与 Python 定点参考比较。

板端逐像素检查入口：

```powershell
python pynq/verify_board.py --bit /home/xilinx/ct_test/ct_registration.bit --size 128
```

```python
from registration_board import RegistrationBoard
with RegistrationBoard('/home/xilinx/ct_test/ct_registration.bit', (128,128)) as board:
    result = board.register(fixed_uint8, moving_uint8)
```

本包初版每评价一个候选都重新传两张图，以减少状态控制复杂度；可能被 DMA/Python
调度开销限制，不能承诺加速比。后续可改为一次载入、多次参数评价、只返回代价。

128×128 的两张 8 位图逻辑容量共 32 KiB，512×512 共 512 KiB；这不是综合后的
BRAM 占用，实际还有映射粒度和其他资源。默认实现使用全帧随机读，按序读取
4 个邻点，不能直接用原有局部滤波行缓存替代。扩展到 512×512 前应评估资源、
定点精度、时序和 DDR 分块方案。当前架构不宣称一像素/时钟。

## 硬件仿真入口

矢量已提供，重建时使用新的目录或先保留旧结果：

```powershell
python python/ct_registration.py vectors --outdir tests/new_vectors
```

testbench 默认读取 `tests/vectors/`。在 Vivado 命令环境从本包根目录执行：

```powershell
xvlog fpga/registration_axis.v
xvlog -sv fpga/tb_registration_axis.sv
xelab tb_registration_axis -s registration_sim
xsim registration_sim -runall
```

检查完整 PASS、像素/有效位/SSD/计数/协议状态和 TLAST，不能只看编译成功。
随后上板记录单候选 DMA 延迟、纯 PL 计算周期、完整配准耗时、资源和时序。

## 参考资料

- 题目一附录 A 拓展要求④：图像配准或多模态融合，体现几何变换、插值及代价计算硬件化。
- SimpleITK registration overview: https://simpleitk.readthedocs.io/en/v2.3.0/registrationOverview.html
- AMD Warp Transform: https://docs.amd.com/r/2024.2-English/Vitis_Libraries/vision/api-reference.html_1_107

代码为独立新增模块，没有修改队友已有工程。
