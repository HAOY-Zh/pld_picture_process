本项目的核心硬件开发目录。

主要内容包括：

Verilog / SystemVerilog / HLS 源码
AXI-Stream 数据通路
行缓存与滑动窗口模块
基础图像处理模块
LDCT 增强核心模块
AI 核心算子加速模块
分割后处理模块
轮廓提取与图像叠加模块
Testbench
FPGA 资源与性能测试相关文件

后续重点记录：

模块划分
数据流路径
缓存机制
时钟域设计
定点化策略
DDR / BRAM 访问方式
LUT / FF / BRAM / DSP 占用
延迟、吞吐率与 FPS