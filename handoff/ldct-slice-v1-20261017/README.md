# LDCT slice v1：最新优化主模型，50/100 MHz 已验收

本包对应《LDCT 模块接入与 100 MHz 优化交接要求》。主模型为 **300 轮续训中的第 118 轮最佳权重，经离线等价通道缩放及 range-safe 量化**；没有重新训练。默认选择已验证的 100 MHz 版，50 MHz 版保留回退。

入口配置：[configs/ldct_primary_model.json](../../configs/ldct_primary_model.json)。两个频率版使用完全相同的模型和整数权重。

| 标识 | SHA256 |
|---|---|
| 原训练最佳 checkpoint | `0fd09c0901d395384d7d28d272935d121bb005dc2ef318f31add8a06b1ce24f5` |
| 当前等价参数化模型 | `2cc0ba07080f950c87427c574472d067e411c7379a7394332290fba12f6f8d5b` |
| 当前整数权重 | `00d7d04f33219c381ad748ae4c869a046538ef7c176e96e2031202b15b0ae022` |

## 接入入口

- [100 MHz 完整目录](releases/100mhz/) / [可直接上传解压的 ZIP](archives/ldct_100mhz.zip)
- [50 MHz 完整目录](releases/50mhz/) / [回退 ZIP](archives/ldct_50mhz.zip)
- [LDCTProcessor 源码](../../pynq/ldct_processor.py) / [借用资源契约测试](../../python/tests/test_ldct_processor.py)
- 每个版本的 `manifest.json` 是该版本频率、GPIO/AXIS 协议、方法、尺寸和文件 SHA256 的依据。

统一服务获取硬件锁，核验文件后加载对应 bit/HWH，在引擎空闲时设置 FCLK0 并读回校验，再创建和传入两个 4096 元素的 uint32 CMA 缓冲。处理类不创建 Overlay、不改时钟、不分配/释放 CMA、不复位、不自动重试或回退到 CPU。

```python
from pathlib import Path
import json, sys
from pynq import Clocks

bundle = Path("/absolute/path/to/selected/release")
manifest = json.loads((bundle / "manifest.json").read_text())
sys.path.insert(0, str(bundle / "pynq"))
from ldct_processor import LDCTProcessor

# dma/gpio/tx_buffer/rx_buffer 已由统一服务准备并持有。
processor = LDCTProcessor(
    dma=dma, gpio=gpio, tx_buffer=tx_buffer, rx_buffer=rx_buffer,
    model_dir=bundle / "model",
    read_clock_mhz=lambda: float(Clocks.fclk0_mhz),
    expected_clock_mhz=manifest["clock_mhz"], timeout_s=10.0,
)
result = processor.process_slice(hu, method="cnn")
processor.assert_idle()
enhanced_hu = result["hu"]
state = processor.get_state()
# 同一个 processor 可继续处理下一张，无需重新加载 bit 或分配缓冲。
```

输入：非空二维原始 HU，int16 或 float32，每个维度 1..512。浮点使用 `np.rint`，检查有限值和舍入后的 INT16 范围，越界拒绝。输出为独立 int16 数组，同尺寸、同坐标，无显示窗或缩放。spacing/origin/direction 由平台保留。

支持 `cnn/bypass/gaussian5/bilateral5/joint7_guide5/joint7_guide3/guided3/bilateral7`；CNN 为 64×64 内部块、halo=3、58×58 核心，512×512 共81块。其他滤波方法核心48×48。

`total_ms` 包含 HU 打包、缓存、DMA、PL、轮询、拼图和独立结果形成，不含文件、加载、初始化、绘图或 CPU 参考。返回实际前后频率、块数、模型ID。

故障后 `faulted/dma_may_access` 保守保留，统一服务必须处理恢复，不能直接释放缓冲/切换 bit。`assert_idle()` 只检查真实状态，不执行复位。PYNQ3.1.1 首次提交前的 DMA.Idle=0 特例，只在 RS=1、LENGTH=0、驱动首次传输标记为真、无活动缓冲且本对象从未提交过传输时接受；部分启动后不适用此特例。已用模拟对象覆盖部分启动、超时、错误、频率变化等路径，没有在板上故意制造总线挂死。

## 硬件与实际速度

| 版本 | 目标/读回 PL | WNS / WHS | LUT / BRAM / DSP |
|---|---:|---:|---:|
| 稳定版 | 50 / 50 MHz | +2.911 / +0.027 ns | 8895 / 46.5 / 78 |
| 提频版 | 100 / 100 MHz | +0.276 / +0.007 ns | 9386 / 46.5 / 74 |

两个版本均提供匹配 bit/HWH、冻结 RTL/ROM、构建脚本、布局布线后时序/DRC/资源报告。各自78帧/51607输出像素压力仿真通过，含极值、背压、裁边、退化尺寸和连续tile。

100 MHz 版缩短了配置判帧、地址计算、CNN 加法树和传统滤波的长组合路径。CNN 的卷积次数、运行状态周期和数值规则保持一致；双边/引导滤波增加了流水阶段，故本次频率性能表固定比较 CNN，不把其他滤波器的变化混入提频倍数。没有放宽时序约束或给50MHz bit直接超频。

PYNQ3.1.1、Python3.10.4、ARM650MHz；相同输入、整数权重、PS驱动、compact58和计时边界，顺序50→100→100→50。每阶段各预热3次，每版本每张图共正式20次，两张图共40次。

| 版本 | 平均 ms/张 | 中位数 ms/张 | P95 ms/张 |
|---|---:|---:|---:|
| 50 MHz | 515.619 | 514.812 | 523.835 |
| 100 MHz | 373.936 | 374.044 | 377.308 |

整帧减时约27.5%，加速约1.38倍。PL计算外仍有PS/缓存/DMA/轮询/拼图，因此不按频率直接推算2倍整机收益。两张样例各项统计及所有逐次记录见 [原始 JSON](evidence/clock_comparison/result.json)、[CSV](evidence/clock_comparison/timings.csv)、[独立审计](evidence/clock_comparison_audit.json)。

两版均通过15个接口实板用例，覆盖全部声明方法、完整512×512、67×69、1×69、67×1、1×1及float32舍入。同一对象连续处理；实际输出 NPY 随记录提供，均与独立 NumPy 整数参考逐像素一致。频率每帧前后检查，阶段结束资源状态干净。较早的 legacy48/fast48/fast58/compact58 原始对照保存在 `evidence/earlier_model_comparison/`，不与本次接口时延混算。

## 效果、数据与浮点参考

数据源为 IDC LDCT-and-Projection-data 的 Siemens 胸部重建图像。训练12人：C002/C004/C012/C016/C021/C027/C081/C095/C099/C107/C166/C179；验证C050/C052；测试C067/C077。演示为C067/180、C077/190，未进入训练；测试患者此前已经查看，不称盲测。

既有76张固定软件测试图、患者等权：原始LDCT RMSE192.726/SSIM0.395723，当前优化整数版69.873/0.674202；旧整数版70.365/0.669995，新原量化版70.653/0.670651。HU RMSE不裁剪；PSNR/SSIM按[-1000,1000]窗、data_range=2000、SSIM窗口7计算；显示窗[-1000,400]不改变HU。完整结果见 [quality_test76.json](evidence/quality_test76.json)。这与两张实板演示图是不同范围。

对应原始/等价参数化 checkpoint 位于各 release/model/，FP32模型实现见 [model_fp32_reference.py](model_fp32_reference.py)。三层replicate padding卷积1→8→8→1，前两层ReLU，HU/1024输入，输出为输入减预测噪声；737参数。Torch只在电脑浮点参考时需要，板端接口不依赖Torch或GPU。

无专家肺标签，不宣称已提高分割Dice或三维精度。网页、HTTP、CT读取/空间信息、3D及全局硬件调度由平台接入；本包验证的是可供平台调用的 LDCT 模块。

## 复现与文件完整性

每个 release 含冻结向量和 `source/replay_simulation.ps1`，无需下载完整患者缓存即可重放RTL仿真；其 README.txt 给出完整构建与调用示例。根目录入口 `fpga/run_ct_tile_checks.ps1` 默认使用主模型配置，缺少本地缓存时使用本包C067样例生成完整78例。

```powershell
# NumPy即可运行，不需要训练或连接板子。
python -m unittest discover -s python/tests -p test_ldct_processor.py -v
python -m unittest discover -s python/tests -p test_ct_tile_driver.py -v

# 修改本机工具路径；输出目录必须是新的。
powershell -ExecutionPolicy Bypass -File fpga/run_ct_tile_checks.ps1 -Python C:\Path\python.exe -VivadoBin C:\Path\Vivado\2018.2\bin -OutDir D:\NewSim -Stress
```

构建与仿真使用 Vivado2018.2。失败的前三次100MHz实验只保留时序/源码证据，未作为可用bit交付。旧模型、旧包、旧结果没有覆盖；本次提交只包含模块所需源码及本交接目录，没有上传整套患者数据或板端登录辅助脚本。

`SHA256.json` 为本交接目录文件校验清单；每个 release 另有 `manifest.json`。发布内容保持原始字节，避免换行转换导致哈希失配。
