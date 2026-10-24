# PYNQ-Z2 可选 ROI HU 定量 Verilog 模块

## 做了什么

`fpga/pynq_z2_roi_statistics.v` 是新加的可综合 Verilog 模块，实例化同目录
`pynq_z2_roi.v`。后者与 10 月 19 日已经验证的导入版逐字一致。
目标 PYNQ-Z2 `xc7z020clg400-1`，Vivado 2018.2，时钟 100 MHz，低有效同步复位。
两份 `.v` 按 Verilog 编译，只有 testbench 用 SystemVerilog。

`fpga/reference` 附原 ROI 包使用的 PYNQ 板卡 PS 资源配置来源记录、
2018.2 适配预设与开源许可。该预设来自 Xilinx/PYNQ v2.5 的 Pynq-Z2 硬件 Tcl，
是资源/PS 配置参考；不要在共享整机中另建 PS7 或覆盖 owner 已验证的 DDR/时钟配置。

原来的筛选、Mask、计数、32-bit AXIS、配置锁存与错误状态保持原 ABI。
新逻辑将每个选中像素对应的**原始有符号 int16 HU** 同输出 Mask 对齐，在输出握手时
累计总和、平方和、最小值和最大值。背压时保持 HU 对齐，最后一个输出握手时发布整层统计。
不是对 0/1 Mask 本身计算 HU，也不是在网页假装 FPGA 计算。
HU 对齐依赖本包基础 core 的一级弹性输出寄存器；如果队友改变其流水级数，
必须同时修改对齐逻辑并重新对拍，不能只替换同名模块。

核心仍然无背压时一拍一个像素。累计只需寄存器和一个 16×16 乘法；
不保存 CT 帧，不额外占用 BRAM。CPU 读出少量统计字再求均值/标准差。
512×512 最坏情况下 sum 范围在有符号 34 位以内，sumsq 最大为 2^48，
64 位输出有充足余量；测试包含全 -32768、全 32767 和正负交替的完整 512×512 帧。

## 接入现有工程

1. 打开整机的 PYNQ-Z2 工程，用 Add Sources 添加两份 `.v`，类型选 Verilog。
   或在 Tcl Console `source .../fpga/add_statistics_to_open_project.tcl`。
   脚本核对器件，检查同名基础源码是否一致，不重复添加，不更改整机 top。
2. 在 BD 中 Add Module `pynq_z2_roi_statistics`；用它替换原 ROI 模块实例，
   将现有配置、DMA 输入输出、clk/rst_n 和状态/计数信号接回。
   不要同时串联两个 ROI 模块；新模块内部已包含原 core。
3. 额外输出接至整机 owner 管理的只读 AXI GPIO 通道，或并入其已有寄存器映射：

| 端口/读取名 | 位宽和含义 |
| --- | --- |
| hu_sum_out / sum_lo,sum_hi | signed 64，低/高各 32 位，选中 HU 总和 |
| hu_sumsq_out / sumsq_lo,sumsq_hi | unsigned 64，低/高各 32 位，选中 HU 平方和 |
| hu_minmax_out / minmax | 低 16 位 min，高 16 位 max，分别按 signed16 解码 |
| statistics_valid_out / valid | 1 位；可以零扩展至 GPIO32；无协议错误且输出帧完成才为 1 |
| statistics_identity_out / identity | 常数 0x52535431，ASCII RST1，确认扩展统计 ABI |

4. 保持只有整机 owner 配置 Overlay、时钟、CMA、GPIO 地址和任务队列。
   新模块无板外 PL 引脚，不给它乱分配板级 XDC 引脚；DDR/FIXED_IO 沿用 PS7 和原板资源配置。
   owner 必须在最后输出完成后、下一输入帧开始前，在同一任务锁内读完统计。
5. 重新生成整机 bit/HWH，检查整体资源、时钟跨域与全系统时序，再真机对拍。
   **10 月 19 日的旧 bit/HWH 不包含新增统计端口，不能用旧文件读取这些寄存器。**

无选中体素时 count=0、sum=sumsq=minmax=0；软件将 min/max/mean/std 显示为 null。
新帧开始 valid 清零，帧结束前不能拿上一帧数据当当前结果。协议错误时 valid 保持 0。
count 继续从原 count_out 读取；status 必须为 0x01000002。

`python/roi_linkage_adapter.py::read_roi_statistics` 接收 owner 注入的 read_u32 回调。
不分配新寄存器地址，不创建 GPIO/Overlay，不改变服务。方差用整数分子
`count*sumsq-sum*sum` 再转换，避免两个较大浮点数相减引入消减误差。
跨多层 ROI 可由 owner 顺序累计各层 count/sum/sumsq 和非空 min/max，
物理体积仍然是筛选区域体积，不能标成真实结节体积。

## 实际验证与局限

- `scripts/run_statistics.ps1 -Synthesize` 使用用户提供的
  `D:\vivado\Vivado\2018.2\bin` 工具。
- NumPy 独立生成 9 帧、803,180 个像素的期望 Mask/count/sum/sumsq/min/max。
  XSim 对每个输出 Mask、TLAST、TKEEP 和每帧统计逐项比较，另检查背压保持。
- `results/fpga_statistics` 保存真实编译、仿真、OOC 综合/布局布线日志、资源、时序报告和 CSV。
- OOC 只约束内部 clk=10 ns，不知道整机 AXI 外部延迟、PARTPIN 位置或真实时钟树。
  日志保留了 no_input/output_delay、HD.CLK_SRC/PARTPIN 等 OOC 警告。
  因此内部时序的正裕量不能等价于整机时序闭合，更不能当成实板已测吞吐/延迟。
- 新统计模块还未合入共享 CNN 整机，未生成或部署新版整机 bit/HWH，没有实板或功耗结果。
  资源占用小于器件总量也不代表与共享 CNN 同时放置一定足够，整机必须再评估。

验收记录与 SHA-256 见 `results/verification.json`，数值从实际报告提取，不手填预期值。
