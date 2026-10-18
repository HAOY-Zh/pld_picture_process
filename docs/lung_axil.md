# AXI-Lite 控制与独立单块上板冒烟工程

2026-10-15：控制 RTL、板端 MMIO 驱动、寄存器启动的实际卷积仿真与独立综合已完成。
完整 U-Net、层间算子/调度和肺分割板端精度仍未完成。
仅用于工程教学与算法演示，不作为临床诊断依据。

## 当前边界与下一次上板

`lung_conv8_axil.v` 将新 `lung_axil_control.v` 接到原 `lung_conv8_memory_axi.v`。
CPU 只配置、装载权重、启动和显示；卷积、定点量化、特征缓存及 AXI 读写由 PL 实现。
目前每个命令仍只有一个输出通道块；CPU MMIO 装载不是高吞吐权重 DMA，也不是整网调度。

本轮额外创建独立板级冒烟工程，先验证真实 PS GP0 控制与 HP0 DDR 通路，再扩展层间算子和整网。
不打开或改动六个旧基础处理工程，不覆盖旧 bit/hwh，不改模型或人工标注。
新工程继承原 closing 工程 PS XCI 的 350 个有效用户参数，保持 512 MiB DDR，覆盖 FCLK0 为 100 MHz。
只制作 PL overlay，**不能用它生成/替换现有 Linux SD 的 BOOT/FSBL**。

## 寄存器，32 位 AXI-Lite，逻辑范围 0x000～0x3ff

板级工程预留控制物理基址 `0x43c00000`；实际以新 HWH 为准，不使用旧 DMA overlay 地址。
所有寄存器按 4 字节对齐，支持 WSTRB；零 WSTRB 对齐写入无作用。
未知/只读地址写入、未对齐读写、保留位或非法操作返回 SLVERR。
AW/W 独立缓存，一笔 B、R 响应；返回值在背压期间稳定。PROT 不实现访问权限保护。

| 偏移 | 名称 | 内容 |
| --- | --- | --- |
| 0x000 | ID | 只读 `0x4c554e47` |
| 0x004 | CTRL | 写命令，读为 0 |
| 0x008 | STATUS | 状态及粘滞标记，读取不清除 |
| 0x00c | VERSION | 只读 `0x00010000` |
| 0x010 / 0x014 | INPUT / OUTPUT | 输入、完整输出张量的 **物理字节地址**，须 8 字节对齐 |
| 0x018 | SHAPE | `[9:0]=H, [25:16]=W` |
| 0x01c | GEOMETRY | `[6:0]=Cin, bit8=3×3, bit9=pad1, [19:16]=输出块数, [27:24]=块索引` |
| 0x020 | QUANT | `[7:0]=keep, [15:8]=输入零点, [23:16]=输出零点, bit24=ReLU` |
| 0x024 | WRITTEN | 最近命令成功 B 响应的输出 word 数 |
| 0x028 / 0x02c | HITS / MISSES | 最近命令缓存命中/缺失数 |
| 0x030 | CYCLES | 命令被引擎接收至含最后 B 响应完成的周期，32 位饱和 |
| 0x034 | COMPLETED | 完成命令数，含核心拒绝的非法配置/错误命令，32 位回绕 |
| 0x038 | WEIGHT_BEATS | 实际权重握手总拍数，32 位回绕 |
| 0x03c | REJECTED | 被拒绝的控制写入数，32 位回绕 |
| 0x040 | LOAD_COUNT | 未完结权重块已接收的拍数 |
| 0x100～0x13c | WEIGHTS[16] | 512 位权重暂存，低地址对应低位 |
| 0x140～0x15c | BIASES[8] | 各输出 lane 的有符号 INT32 偏置位模式 |
| 0x160～0x17c | MULTIPLIERS[8] | Q31 非负整数乘子 |
| 0x180～0x19c | SHIFTS[8] | 各 lane 右移位数 0～62 |

CTRL 只接受以下操作：`1=START`、`2=CLEAR`、`4=PUSH` 加可选 `8=FIRST`、`16=LAST`。
例如单拍权重块写 `28`；多拍首拍 `12`、中间 `4`、末拍 `20`。
FIRST 必须与新块一致，最多 128 拍，要求第 128 拍必须 LAST。
CLEAR 只清 DONE/ERROR/CONTROL_ERROR，**不复位、不取消在途事务、不清权重**。

STATUS 位：`0=BUSY, 1=DONE, 2=计算错误, 3=权重就绪, 4=可排队START,
5=权重块未结束, 6=权重待握手, 7=启动等待/待握手, 8=控制写入错误`。
DONE/计算错误在新 START 自动清除，控制错误也可用 CLEAR 清除。
完成事件与 CLEAR 同拍时，完成事件优先，不丢失 DONE。
STATUS 位4不保证引擎此拍就握手；START 可以排队，字段保持稳定。

计算或命令/权重待握手期间，配置与暂存数据均锁定；权重块装载中仅允许继续更新 WEIGHTS 暂存。
量化参数在首拍权重握手时进入原核心，不能把更新暂存器当作自动更新驻留权重。
几何/地址错误沿用核心检查，以 DONE+计算错误结束；START 前的地址不对齐直接 SLVERR。
错误结果必须整次丢弃，已写 DDR 数据不会回滚。真实缓冲区的大小和访问权限仍由驱动保证。

## 配置等待与时序，不是全局放宽

首轮寄存器接入暴露原先 OOC 外部参数未计入的配置乘法/范围检查路径，100 MHz WNS 为 -8.052 ns。
控制器增加显式四边沿配置稳定窗口，窗口及计算期间禁止更新配置，之后才发出命令 VALID。
真实 RTL 测试逐次断言 VALID 前配置至少稳定四周期；握手背压继续保持不变。

`lung_axil_config_timing.xdc` 只将控制器 H/W、几何与基址到引擎的捕获路径设为 setup 4 / hold 3。
Vivado 可将配置 FF 吸收/复制到专用计数乘法 DSP，因此另外对明确命名的
`input_words0/output_words0/command_words0` 配置计数 DSP 使用更保守的 setup 3 / hold 2。
不包含运行时 `row_offset`、卷积 MAC、重缩放、AXI、流数据或 VALID 控制状态路径。
不能把该约束移植给一个没有配置锁定/等待窗口的模块。

正式综合证据在 `data/lung_axil_20261015/synthesis_v5/`。
RTL 未变，v5 复用 v3 检查点，仅重新报告上述约束；原负时序及约束诊断报告保留。
内部 WNS **+1.273 ns**；LUT **8,363 (15.72%)**、FF **9,686 (9.10%)**、DSP **73 (33.18%)**、BRAM Tile **31.5 (22.50%)**。
这些仍是单块 OOC 数据，不代表整网或板级路由已通过。完整板级证据需另看 board verification。

## 测试与审计

- 控制协议真实 XSim：23 个寄存器核对，173 写、31 读、130 权重拍、2 个命令。
- 整链路真实 XSim：只从 AXI-Lite 配置/装载/启动，147 个冻结卷积任务，6055 权重拍、8866 次 64 位写回全一致。
- 每任务检查命中/缺失、周期与完成计数、未写区域哨兵值；最后 B 响应前强制保持 BUSY。
- 实测忙时配置/重复 START 被拒绝，三种非法配置不访问内存；读/B 错误正确上报；共享复位清除控制和在途事务。
- 177 项 Python 回归通过，包含 MMIO 调用、参数预检查、缓冲区 flush/invalidate、超时与中断安全。
- 独立复核上一阶段 HWC8/卷积证明、正式日志、资源/配置例外与 1182 个模型/医学来源哈希。

正式日志为 `axil_control_run_v3.log`、`axil_accel_run_v2.log`；审计 `data/lung_axil_20261015/verification.json`。
测试没有重训、修改标签或对独立测试集调参。

## 板级构建与物理测试

### 2026-10-15 已生成的独立工程

**后续发现板级复位错误：以下原始 deploy 禁止继续加载。**
首次上板未通过，原检查漏掉了低有效辅助复位被接到常量 0 的问题。
修正工程与新增验收见 [首次上板复位问题](lung_axil_board_reset_issue.md)。
旧资源/时序数据仅说明旧版实现结果，不表示板端可正常工作。

工程：`data/lung_axil_board_20261015/project/lung_axil.xpr`。
BD：该工程 Sources 下的 `design_1`，由 Tcl 自动创建，不需要再手工接线。
控制通路为 PS GP0 → control_interconnect → 加速器 AXI-Lite；
数据通路为加速器 AXI4 master → memory_interconnect → PS HP0 → DDR，不使用 AXI DMA。
全链路共享 FCLK0 100 MHz 和复位，未修改旧基础处理工程。

完整布线结果：LUT 10,306 (19.37%)，FF 11,933 (11.22%)，DSP 73 (33.18%)，
BRAM Tile 31.5 (22.50%)；WNS +0.571 ns，WHS +0.016 ns。
DRC 无 Error/Critical Warning，但有 278 条 DSP 流水线/OPMODE 类 Warning，不能称为零告警。
`verification.json` 与 `deploy/package.json` 保存来源、资源、时序和部署文件哈希。
报告解析兼容另有 5 项单测通过；未改写原有 177 项回归的冻结证据。

上板文件位于 `data/lung_axil_board_20261015/deploy/`：
`lung_axil.bit`、`lung_axil.hwh`、`lung_axil.py`、`lung_axil_smoke.py` 和 `package.json`。
这些仅支持单卷积模块实板验证，尚未实板运行，也不代表完整 U-Net 已部署。

`python/lung_axil_board.py` 只读提取已有 PS XCI，输出新的 PS 配置构建产物。
`fpga/build_lung_axil_board.tcl` 在新目录封装核心，接 PS GP0/HP0、两个 SmartConnect、共享复位。
完成 BD 验证、综合、实现和时序/DRC检查后才复制同名 `lung_axil.bit/.hwh` 到 deploy。
`verify_lung_axil_board.py --board DIR` 校验控制证明、封装源码/HWH、实际时钟/DDR/基址、路由 setup/hold 和 DRC，再复制板端脚本并记录哈希。
该入口兼容布线资源表中 `Slice LUTs` 可选的星号，复用冻结核验器的其余检查，不改动既有证明的源码哈希。

```powershell
.\.venv-lung\Scripts\python.exe python/lung_axil_board.py --outdir data/lung_axil_board_new
Push-Location data/lung_axil_board_new
& 'E:\vivado2\Vivado\2022.2\bin\vivado.bat' -mode batch -source ../../fpga/build_lung_axil_board.tcl -tclargs D:/pld_picture_process/data/lung_axil_board_new build
Pop-Location
.\.venv-lung\Scripts\python.exe python/verify_lung_axil_board.py --board data/lung_axil_board_new
```

板端测试使用 `pynq/lung_axil.py` 和 `pynq/lung_axil_smoke.py`，不需要 CT 或模型权重下载。
在板端 Jupyter 内，将 bit/hwh 与两个 Python 文件放在 `/home/xilinx/ct_test/`，新建内核后运行：

```python
%cd /home/xilinx/ct_test
%run lung_axil_smoke.py --bit /home/xilinx/ct_test/lung_axil.bit --report /home/xilinx/ct_test/lung_axil_board_result.json
```

报告名必须未存在，不覆盖板端证据。先停止旧 DMA/其他 PL 程序，不能同时运行不同 overlay。
它测试小型合成整数输入的 1×1/3×3、补边/valid、尾通道、多块与同地址更新，打印逐项一致结果和计数。
CPU 参考只用于对拍，不冒充 FPGA 输出；CPU 调用耗时也不冒充完整 U-Net FPS。
成功须实际看到 `Real-board single-block smoke: PASS`，电脑仿真/生成 bit 都不能替代它。

驱动会校验实际分配地址和 nbytes，并执行输入/输出 flush 与完成后 invalidate。
若超时或中断且可能有事务在途，不释放缓冲区，也不再发新命令；保留 Jupyter 内核和缓冲区，
应先进行整个开发板/AXI 系统的可靠复位或断电重启，再关闭内核/复用内存，不能只清状态或局部复位。
真实共享复位与缓存一致性仍需板端确认。

单块物理测试通过后，继续层间 pool/nearest/concat、缓冲复用与整网调度，再验证完整肺分割。
