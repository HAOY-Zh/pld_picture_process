# PYNQ-Z2 同板 ARM CPU 与当前 PL 整网比较

## 口径

比较同一份冻结 INT8 U-Net、同一张 `LCTSC-Train-S2-006` 第 72 层、256×256、batch=1，
输入为内存中 UINT8 CHW 编码，输出为完整网络的 UINT8 CHW head 编码。
不包含 CT 文件读取、HU/输入归一化、sigmoid/原尺寸插值、人体约束、三维清理和显示。
软件安装、编译、模型/常量准备和下载 bit 是一次性准备，不进入每帧计时。
这不是整幅原始 CT 的端到端显示延迟，也不是全病例医学精度证明。

板端实测环境：`armv7l`，`/usr/bin/g++` 和 NumPy 可用，Torch/ONNX Runtime 未安装。
新测试不需要联网下载包，不需要新 BD 或新 bit。

## 软件实现

`pynq/lung_cpu_native.cpp` / `lung_cpu_kernel.h` 是原生 C++11 数值实现，
`lung_cpu.py` 只用 ctypes 调用，不在 Python 循环里做卷积。
实际乘加为整数，保持非零激活零点、ACC32、Q31 signed RNE、ReLU/饱和裁剪的冻结语义。
模型载入时检查保守 ACC32 上界，运行中不使用浮点近似替代整数舍入。

- 卷积展开一个空间位置的中心化输入 patch，一次复用给四个输出通道。
- ARMv7 使用 NEON signed int16 乘法扩展到 int32 的向量累加，不使用不存在于 Cortex-A9 的 SDOT。
- OpenMP 静态空间划分，分别实测 1/2 工作线程。
- 权重转置/扩展及工作区在模型载入时准备，不在每像素分配内存。
- 拼接的标量重定标以预计算的 256 项精确查表实现；无近似舍入或更改模型。

NEON 依据 [Arm 官方 Intrinsics 参考](https://arm-software.github.io/acle/neon_intrinsics/)。
这是**自编优化 C++/NEON 基线**，不是 ARM 上所有成熟推理库的最佳性能承诺。
不能因为采用 C++ 就把它写成“最快 CPU”，也不能用本机 x86 的时间冒充 ARM 时间。

## 已做与未做的验证

本机原生 C++ 单/双线程已分别逐节点对齐同一真实 CT 的 21 个冻结输出，
head 也与用户已回传的实际 FPGA head 完全相等。
同一 NEON 微内核已经交叉编译为 Cortex-A9 ARMv7 汇编，检查有 `vmlal.s16`。
这**不是 ARM 指令已经在板上运行**；完整 ARM 编译与实际时间仍待用户运行。
新增 7 项原生实现测试和完整 236 项回归通过。准备与编译证明位于
`data/lung_cpu_compare_20261015/native_verification.json`，完整验收记录为同目录 `verification.json`；
没有任何本机时间被写成板端 CPU 速度。

本机初次测试数值全部通过，但进程退出异常；定位并去掉 ctypes DLL 中非平凡 C++ TLS 析构，
改为平凡 TLS 错误缓冲后重新编译、数值和正常退出测试通过。旧 DLL 不随部署包上传。

## 实测流程与限制

`lung_cpu_compare.py` 会先检查所有包文件哈希，然后在板上用
`-O3 -std=c++11 -shared -fPIC -fopenmp -mcpu=cortex-a9 -mfpu=neon` 编译。
要求实际 NEON 和 OpenMP 启用；编译失败时停止，不悄悄回退为很慢的 Python 实现。
已编译二进制仅在源文件、命令及二进制自身哈希全部一致时复用。

CPU 单线程、CPU 双线程、当前 PL 各预热一次，测量三次，记录各次、均值、中位数、最小值和最大值。
每次输出验证在计时结束之后；CPU 全部 21 节点另在预热之后、计时之外验证。
CPU 结果先单独保存，之后才加载一次匹配 overlay 执行 PL 比较。
CPU 单次 C++ 调用期间没有逐层打印；看到 `running...` 后应等待该次返回。

PL 没有逐节点参考比较及进度打印，但**仍保留既有调度器的安全哨兵和未调度通道块检查**。
PL 当前每帧的分配/释放、权重打包/装载、MMIO 调度、缓存同步、轮询都在整次调用计时内。
CPU 工作区常驻，而当前 PL 缓冲是逐帧申请；报告明确披露这种当前实现差异，不能称为两侧最佳性能对比。
不拿 PL 仅硬件计数器的时间与 CPU 整次调用时间计算主加速比。

比值 `CPU 平均毫秒 / PL 平均毫秒`：大于 1 才是当前 PL 更快，小于 1 就如实报告 PL 更慢。
线程数 1/2 的结果全部展示，不能只挑较慢的 CPU 线程配置。
测试期间不要同时运行训练、其他模型、PL 应用或重载 bit。
FPGA 超时/异常沿用既有缓冲保留机制；不要盲目重跑或用局部复位取消在途 AXI。

## 上传运行

新目录不修改旧 bit、旧调度器和旧证明。电脑 PowerShell：

```powershell
Set-Location "D:\pld_picture_process\data\lung_cpu_compare_20261015"
& "C:\Program Files\Git\usr\bin\scp.exe" -r lung_cpu_compare xilinx@192.168.2.99:/home/xilinx/ct_test/
```

板端 Jupyter：

```python
%cd /home/xilinx/ct_test/lung_cpu_compare
%run lung_cpu_compare.py --runs 3 --warmup 1 --report /home/xilinx/ct_test/lung_cpu_compare_result.json
```

预期最后为 `Same-board CPU/PL benchmark: PASS (all outputs match)`，
并分别显示单线程、双线程 CPU 耗时与 CPU/PL 比值，以及当前带安全检查的 PL 耗时。
这句话表示测速完成且输出一致，不意味着比值一定大于 1。
原始报告为 `lung_cpu_compare_result.json`；CPU 阶段另保存 `lung_cpu_compare_result_cpu.json`。
如需再次运行，指定新的报告名；两个报告都不能覆盖已有证据。

## 用户终端返回的实测结果

用户粘贴的同板运行日志显示全部输出相同：单线程均值 1789.498 ms，双线程 921.987 ms，
旧 PL 5405.506 ms，引擎活动约 1814.3 ms。相对双线程 ARM，旧 PL 约慢 5.86 倍。
完整原始 JSON 尚未复制回本机，此处是终端报告记录，不是新增的文件级验收证明。
后续实现见[同 bit 主机运行时优化](lung_fast_runtime.md)，不改变这份原始基线和证明。
