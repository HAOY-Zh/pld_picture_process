# 同 bit 的连续网络运行时优化

这是第一阶段的主机/调度优化，**不是新计算核**。用户已回传终端耗时，原始 JSON 尚待取回核验。
原模型、量化参数、输入、21 节点/51 命令、bit/HWH 和全部旧证明不变。

## 为什么先做这一步

用户贴出的同板测速：单核 ARM 平均 1789.498 ms，双核 921.987 ms，
旧 guarded PL 平均 5405.506 ms；其中约 1814.3 ms 是将计数器按假定的 100 MHz 折算，
3591.2 ms 是相应的差值。**当时未核实实际 FCLK，不能将差值全部认定为主机开销。**
这是用户终端提供的数据，原始测速 JSON 尚未回传核验；不能冒充本机已接收的完整证据。
即使按 100 MHz 的折算，现有计算/访存核仍比双核 ARM 慢。因此后续还需要 RTL 优化。

## 优化版用户终端结果与已确认的频率差异

用户回传同次测速：双核 ARM 874.777 / 891.830 / 881.634 ms，均值 882.747 ms；
旧 PL 单次 5374.097 ms；新 PL 2986.808 / 2994.459 / 2994.439 ms，均值 2991.902 ms。
相比同次旧 PL 单次值，耗时下降约 44.3%、约 1.80 倍；仍比双核 ARM 慢约 3.39 倍。
原版只有一个测量样本，不能据此宣称具有统计稳定性的精确提速倍数。
全部 21 节点和所有 head 均 SAME。权重装载仅约 34–39 ms，输入同步约 7–9 ms，帧末 guards 约 8 ms。
这是终端结果记录，不改写之前的本地准备证明为原始文件验收证明。

检查发现，部署 HWH 中 `PCW_IO_IO_PLL_FREQMHZ=1600.000`，FCLK0 的两级分频为 4 和 4。
设计标称由 1600 / 16 得到 100 MHz；实际板卡启动 PLL 不一定与这份 PS 配置相同。
假如实际源为 1000 MHz，则为 62.5 MHz，181.43 百万周期实际约 2902.9 ms，
而不是 1814.3 ms，几乎可解释新 PL 的约 2.99 秒。
后续用户在同一 Jupyter 内核截图确认 **FCLK0=62.5 MHz，CPU=650 MHz**。
三次 `execute_wait_ms` 约 2913.867 / 2914.268 / 2914.565 ms，
与按 62.5 MHz 换算的引擎活动约 2902.94 ms 一致，整帧其他开销约 89 ms。
**先前将约 1.18 秒差值全部解释为主机开销的说法不成立，现已更正。**
这是运行后寄存器读回结合用户同内核截图的诊断；旧测速没有前后时钟快照，不能倒填为当时独立记录的频率。

用户完成过的只读诊断（保留供复现）：

```python
from pynq import Clocks
import json
print("PL FCLK0 MHz:", Clocks.fclk0_mhz)
print("CPU MHz:", Clocks.cpu_mhz)
with open("/home/xilinx/ct_test/lung_fast_result.json") as f:
    result = json.load(f)
for i, row in enumerate(result["optimized_pl"]["details"], 1):
    print(i, row["timing_ms"], "unclassified_ms:", row["unclassified_host_ms"])
```

不下载 bit，不重跑网络，不设置频率。`Clocks.fclk0_mhz` 根据当前时钟寄存器计算频率，
见 [PYNQ 官方实现](https://pynq.readthedocs.io/en/v3.1/_modules/pynq/ps.html)。
读到的是检查时的频率，不是旧测速时频率的独立历史证明；如中途重载过设计，需要后续重测并记录前后频率。
不要把余下约 1.18 秒差值称为 Python 轮询/主机开销，也不要直接更改 PLL 或超频。

## 新增：加载后校准到路由频率复测

增量入口 `pynq/lung_clock_benchmark.py` 不修改原版测速/驱动/运行时/bit/HWH。
它临时包装本次同步调用的 Overlay 构造和网络入口：bit 加载后，检查两个 IP 的 ID/ABI 和 idle；
只通过 PYNQ 的 FCLK0 setter 设为 100 MHz，读回检查 CPU 仍为原频率；随后才允许网络分配和启动。
异常时恢复 Python 入口绑定，但**不会自动回退时钟、局部复位或取消在途 AXI**。
每个整网 forward 前后都读回 FCLK0 和 CPU，新增 `_clock.json` 记录；读时钟的开销包含在外部整帧墙钟内。

同 bit 的路由证明为 100 MHz、WNS +0.425 ns、WHS +0.013 ns，校准不是提高到超过设计的频率。
原 HWH 按 Vivado 1600 MHz PLL 假设给出 4×4 分频；新入口按当前源频率选分频，
不更改 PLL、CPU 或 DDR 设置，也不改 HWH 来伪造元数据。原始证明及包文件哈希保持不变。

只需上传两个小文件到**已有** `lung_fast` 目录，电脑 PowerShell：

```powershell
Set-Location "D:\pld_picture_process\data\lung_clock_20261015\deploy"
& "C:\Program Files\Git\usr\bin\scp.exe" lung_clock_benchmark.py clock_manifest.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_fast/
```

Jupyter（板上不要同时运行其他 PL 工作）：

```python
%cd /home/xilinx/ct_test/lung_fast
%run lung_clock_benchmark.py --runs 3 --warmup 1 --report /home/xilinx/ct_test/lung_fast_100mhz_result.json
```

仍会重新实测双核 CPU，再完整检查优化版 21 节点、预热和测速，默认不再跑旧 PL 对照。
启动时应显示 `Clock correction after bit download: 62.500 -> 100.000 MHz; CPU unchanged 650.000 MHz`，
如当前已经为 100 MHz，允许 100→100 且不重复设置。若频率/CPU/身份/忙状态不符则停止。
不建议在旧测速命令之前单独写 `Clocks.fclk0_mhz=100`：后续 Overlay 加载可能再次应用 4×4 分频。

新证据前缀为 `lung_fast_100mhz_result`，包括主 JSON、`_head.npz`、`_cpu.json`、`_diagnostic.json`、`_clock.json`。
旧 `lung_fast_result*` 不覆盖。当前只有本地空闲保护/校准顺序/频率监测/异常恢复测试，
**尚未取得校准后 100 MHz 的物理板端结果，不能预先宣称时延或已超过 ARM。**
改频后 DDR 等待的时钟周期数可能变化，不能仅按 62.5/100 线性缩放保证新整帧耗时。

后续用户截图已确认：100 MHz 前后监测通过、21 节点全部相同，PL 平均 1924.060 ms，
CPU 双线程 890.895 ms；引擎活动约 1842.582 ms，其他约 81.478 ms。
仍慢于 CPU，但主机开销已经不是主要瓶颈；原始报告/实际 head 待回传，不改写旧本地准备证明。
下一阶段是[字级并行拼接 RTL](lung_word_tensor.md)，不是继续归咎于 Python 等待时间。

## 已实现

- 固定权重在构造时打包一次，而不是每帧重新构造 Python 512-bit 整数。
  已核对真实模型全部 45 个权重块、1909 个包与原版逐包一致。
- 本地 C++ 批量执行权重窗口 MMIO 写入、push 和完成检查；ARM 使用显式 `dmb sy`。
  仍然是 AXI-Lite 单字访问，**不是 AXI burst**，每帧 30544 个权重字及参数仍需实际传输。
  一次准备不等于全模型权重已经常驻 PL。
- 22 个 DDR 分配跨帧保留，不再每帧分配、填充和释放。
  当前常驻 CMA 7834368 字节，约 7.47 MiB，旧按生命期释放版本峰值约 2.25 MiB。
  这是用少量板外 DDR 换取调度简化，不增加 BRAM/DSP。
- 中间结果在 PL→DDR→PL 间流转，不做逐命令的整块 CPU 缓存同步。
  CPU 只更新并 flush 输入；所有缓冲初次初始化后 flush 一次，随后 CPU 不写中间张量。
  CPU 读结果/诊断快照前 invalidate，避免把过期 CPU 数据 flush 回 DDR 覆盖 PL 输出。
- 保留驱动几何/地址检查、写出字数、完成序列、超时及错误状态。
  改为每帧结束检查全部分配的 canary；**不再每条命令扫描所有缓冲及未调度通道块**。
  上板测速前另有一次全部 21 节点的完整参考对照，计时区间不做参考比较。
- 超时/中断后仍保留 CMA 和驱动所有者、拒绝重用；绝不靠释放缓冲或局部复位取消在途 AXI。

新增实现为 `pynq/lung_fast.py`、`lung_fast_io.cpp/.h`、`lung_fast_benchmark.py`。
冻结原版驱动与调度器没有修改，可继续独立复现旧结果。

## 本地验证的边界

9 项新增测试覆盖新调度的完整网络、变化输入连续帧、CPU 缓存/设备 DDR 分离模拟、
权重与尾通道排列、诊断快照隔离、分配失败、越界、完成计数和超时保留。
其中 C++ 模板装载器实际编译执行了脚本化总线测试，包括 1/2/128 包、计数回绕、
超时、拒绝及不完整装载。真实 MMIO 入口也已编译，非法参数的无写入拒绝路径经过执行。
这**不是 ARM 板端 MMIO 成功路径测试**，不能证明实际板上时延或缓存行为。

真实验证切片通过新调度的独立算术模拟，全部 21 个节点匹配冻结输出，
末端编码与先前回传的真实 FPGA 输出相同；同一运行时重复执行也一致。
模拟器不放入部署目录，不提供 CPU 计算替代路径。
证明位于 `data/lung_fast_20261015/pc_verification.json`；完整测试/来源审计另存于该目录。

## 用户现在需要做的步骤

无需 Vivado/BD，无需重新生成 bit。保持板上没有其他 PL 程序同时运行。
电脑 PowerShell：

```powershell
Set-Location "D:\pld_picture_process\data\lung_fast_20261015"
& "C:\Program Files\Git\usr\bin\scp.exe" -r lung_fast xilinx@192.168.2.99:/home/xilinx/ct_test/
```

板端 Jupyter 新内核：

```python
%cd /home/xilinx/ct_test/lung_fast
%run lung_fast_benchmark.py --runs 3 --warmup 1 --old-runs 1 --report /home/xilinx/ct_test/lung_fast_result.json
```

先编译本地 C++（不联网），运行双核 ARM，再加载同一个 overlay 一次，
运行原版 PL 对照，然后检查优化版全部节点、预热和测速。
setup 的权重准备/DDR 分配单独显示并排除在稳态计时之外，与 CPU 常驻模型设置一致；
每帧物理权重装载、输入同步、配置、轮询、帧末 guards、head 拷贝均在计时内。
原版默认只有 1 次测量用于确认量级；若需要同等采样，设 `--old-runs 3`。

结果保存 `lung_fast_result.json`、`lung_fast_result_head.npz`，
另有 `_cpu.json` 和 `_diagnostic.json` 阶段文件；复测必须换新报告名。
`Outputs: PASS` 只表示输出相同。`CPU/PL < 1` 时仍未实现相对双核 ARM 加速，必须如实报告。

取回原始证据：

```powershell
Set-Location "D:\pld_picture_process\data\lung_fast_20261015"
& "C:\Program Files\Git\usr\bin\scp.exe" xilinx@192.168.2.99:/home/xilinx/ct_test/lung_fast_result.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_fast_result_head.npz xilinx@192.168.2.99:/home/xilinx/ct_test/lung_fast_result_diagnostic.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_fast_result_cpu.json .
```

## 后续硬件优化方向

仍需减少 tensor concat 按字节串行处理，以及卷积计算与输出写回互相等待；
逐步引入字级 concat、读算写重叠、写回 FIFO/burst，并进行逐项 RTL 仿真和路由时序检查。
不能仅通过删除检查或提高时钟宣称满足加速。当前阶段暂停在需要用户运行真实板卡的步骤，
待完整计时分解返回后再锁定下一阶段瓶颈和优先级。
