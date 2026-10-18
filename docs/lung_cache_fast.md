# 缓存命中与通道连续寻址优化

## 已有实板结果与本轮边界

用户 2026-10-16 回传终端：确切路径入口确认字级拼接 bit
`0c7dc65f4c76825b5abfe5333488d54b59dd5123c06c749e42ff091c525337fa`，
实际 FCLK0 100 MHz、CPU 650 MHz，21 节点与每次 head 均逐码一致。
双核 ARM 891.049 ms；PL 1686.432 ms，活动引擎均值 1602.856 ms，外围约 83.576 ms。
相较之前旧 bit 100 MHz 的用户实测 1924.060 ms，PL 时延下降约 12.35%，但仍比双核 ARM 慢约 1.89 倍。
跨次基线不是同次 A/B；以上来自用户终端，原始 JSON/head 尚待取回文件级核验。

本轮不改变模型、训练/验证/测试划分、缩放、权重、偏置、舍入或医学后处理。
保留已经通过实板的字级拼接张量引擎，不再把主要优化目标放在约 84 ms 的外围开销上。

### 本轮用户回传终端（原始文件待取回）

新缓存/窗口 bit `c30c9ad5fa86f6fd08ca9ec48649137144af9116e5fa9b80f25b8c89b2eb5905` 已由
绑定入口确认下载，100 MHz/CPU 650 MHz，21 节点及最终 head 全部 SAME。
双核 ARM 三次 878.463 / 890.054 / 898.118 ms，均值 888.878 ms；
PL 三次 1146.599 / 1143.327 / 1152.504 ms，均值 1147.477 ms。
活动引擎均值 1060.827 ms，外围约 86.650 ms，常驻 CMA 7.471 MiB。
相对上一轮用户终端的 1686.432 ms，本轮耗时下降 31.96%，速度约为上一轮的 1.470 倍；
跨次结果不是同次 A/B，CPU/PL 同次比例为 0.775，仍未实现相对双核 ARM 的加速。
还需减少约 258.599 ms（本轮总耗时的 22.54%）才能追平该次 ARM 均值。
即使完全消除外围开销，1060.827 ms 的硬件活动仍大于 ARM 888.878 ms，后续必须继续优化硬件。

本机尚无此次 `lung_cache_fast_result*.json` 和实际 head，不能据终端总量确定逐层瓶颈或冒充文件级验收。
下一步取回现有六份结果（无需重跑），核验身份/实际输出并统计 21 节点、51 命令的平均周期，
再据实际占比选择数据供给流水、输出写回或计算端的优化。不修改既有构建证明的历史状态。

后续上述六份原始文件已取回，通过 `python/verify_lung_prepared_board.py` 独立验收；
实际 head 65,536 个编码一致，验收另存为 `lung_cache_fast_result_verification.json`。
逐层均值为卷积 660.884 ms、张量 399.943 ms；因此下一阶段选择[张量突发搬运](lung_tensor_burst.md)，
不再仅凭终端总耗时猜测瓶颈。

## 两条硬件改动

1. `fpga/lung_feature_cache_fast.v`：在请求被接收的时钟沿，同步读取三路 BRAM 的 tag 与 data；
   下一拍比较 tag 并选择数据，移除原命中路径的三个串行状态。
   容量仍为 3 路、256 组、每行 16 个 64-bit 字；FIFO 替换、帧清空、裁剪突发和错误填充策略不变。
   仍然是同步 BRAM，不用组合 RAM，也没有增加 DDR 并发事务或绕过 AXI 故障检查。
2. `fpga/lung_feature_window_fast.v`：同一卷积采样点的连续 HWC8 通道块，采用寄存器地址 `+1`，
   不重复坐标乘法/边界准备；移除地址已经寄存后额外空等的一拍。
   padding 仍输出输入零点，不读 DDR；输出 backpressure 时所有数据/标签稳定，错误按原规则粘滞记录。

旧源文件和旧工程不覆盖。准备器只在新实验目录生成同接口影子源与机械派生构建脚本；
卷积 MAC、Q31/RNE、输出写回、AXI 读写主机、AXI-Lite、复位连接和原时序例外均保持原实现。

## 完成的本地验证

新旧实现分别实际运行六组 XSim：读缓存卷积、DDR 完整写回、AXI-Lite 寄存器启动、
窗口维度/地址边界、缓存故障协议、默认容量。
每种实现的前三组分别覆盖 147 帧；读缓存组与完整写回组核对 68,849 个原金标准编码，
寄存器组核对 8,866 个输出字。包括截断通道、非零零点、读写阻塞、延迟最终 B、canary、
14 项内存拒绝/错误、5 项寄存器集成错误、28 次缓存协议读取及 18,437 次默认容量读取。
窗口边界另覆盖 H/W=512、C=96、最大地址、越界与共享复位。

相同总线阻塞策略下，147 帧总周期：

| 范围 | 原周期 | 新周期 | 原/新 |
| --- | ---: | ---: | ---: |
| 缓存读到卷积输出 | 3,513,595 | 1,766,502 | 1.989 |
| 寄存器 START 到完成写回 | 3,486,904 | 1,731,127 | 2.014 |

这是相同独立金标准与相同 BFM 策略的局部 RTL 结果，不是整网实板加速倍数。
独立卷积缓存读链综合：6,526 LUT、8,142 FF、70 DSP、31.5 BRAM Tile；
100 MHz 内部 setup WNS +1.273 ns。不能把独立综合资源或时序当作整机路由结果。
本次完整工程回归 266 项通过（96.229 秒）；原参考向量未修改。

第一轮验收器曾把两个旧 PASS 计数误写为 5,232/29，已按冻结旧日志改为 6,055/28；
实际 RTL 日志均保留，经修正的严格验收器复核通过。没有调整金标准或 HDL 来迎合输出。

## 构建和交付

实验目录 `data/lung_cache_fast_20261016`。
`python/lung_cache_fast_prepare.py` 准备、执行/复核仿真；
`python/lung_cache_fast_deploy.py --rtl-only` 独立检查实际日志、冻结向量与综合报告。
整机使用该目录的 `build_cache.tcl`，沿用字级张量引擎以及旧 BD、复位、DDR、100 MHz 和时序约束。

只有整机路由、DRC、时序、完整回归及原医学输入保护检查全部通过，
部署审计器才会创建 `lung_cache_fast/`。
新包重新绑定 bit/HWH 与分层 manifest，内置确切路径测速入口，避免同一 Jupyter 内核缓存旧入口。
原始模型、输入、参考编码、驱动和 C++ CPU 基线不变。
旧 `lung_word` 目录和报告保留，不覆盖。

最终整机路由和交付审计已通过：100 MHz 下 WNS +0.328 ns、WHS +0.015 ns；
DRC 无 Error/Critical Warning（仍有原 DSP/互连等 Warning，未声称零警告）。
整机 18,227 LUT（34.26%）、19,699 FF（18.51%）、113 DSP（51.36%）、31.5 BRAM Tile（22.50%）。
1,182 个受保护医学输入与原模型未变，所有打包依赖逐项核对。
证明 `data/lung_cache_fast_20261016/verification.json`，状态 `built_routed_cache_window_fast_not_board_tested`；
板端新性能和精度仍待实际运行，不能把本地审计当成实板验收。

新 bit SHA256：`c30c9ad5fa86f6fd08ca9ec48649137144af9116e5fa9b80f25b8c89b2eb5905`。
上传目录 `data/lung_cache_fast_20261016/lung_cache_fast/` 仅含声明的部署文件，不带桌面 Python 缓存。
首次打包时发现目录内的桌面 `__pycache__` 被当作文件求哈希；已修正打包器，禁止本地检查写入缓存，
重新通过完整打包审计。硬件与仿真未变，也没有为此改动旧包。

上传/执行命令（仅在本轮最终审计通过、包实际存在后使用）：

```powershell
Set-Location "D:\pld_picture_process\data\lung_cache_fast_20261016"
& "C:\Program Files\Git\usr\bin\scp.exe" -r lung_cache_fast xilinx@192.168.2.99:/home/xilinx/ct_test/
```

```python
%cd /home/xilinx/ct_test/lung_cache_fast
%run lung_selected_benchmark.py --runs 3 --warmup 1 --report /home/xilinx/ct_test/lung_cache_fast_result.json
```

完成后取回本轮六份证据：

```powershell
Set-Location "D:\pld_picture_process\data\lung_cache_fast_20261016"
& "C:\Program Files\Git\usr\bin\scp.exe" xilinx@192.168.2.99:/home/xilinx/ct_test/lung_cache_fast_result.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_cache_fast_result_clock.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_cache_fast_result_selection.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_cache_fast_result_diagnostic.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_cache_fast_result_cpu.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_cache_fast_result_head.npz .
```

上次必须正常结束、没有其他 PL 程序并行运行，才可重新下载。
超时/AXI 状态不确定时不清除驱动缓存、不释放在途 CMA、不盲目重载，需安全的共享板级恢复。
只有新 bit 的 21 节点和最终 head 验证后，才接受实板时延。

取回旧字级拼接结果用于文件级核验（不覆盖新报告）：

```powershell
Set-Location "D:\pld_picture_process\data\lung_selected_20261015"
& "C:\Program Files\Git\usr\bin\scp.exe" xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result_clock.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result_selection.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result_diagnostic.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result_cpu.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result_head.npz .
```
