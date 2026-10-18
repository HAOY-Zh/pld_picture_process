# PL 层间张量操作与双引擎工程

## 范围

此前用户截图已显示冻结 U-Net 的 15 个卷积层、45 次输出块全部实板 PASS。
该次截图对应的原始层报告尚未单独回传；已正式核验的旧板端 JSON 是 7 次小型卷积 smoke。
这两类证据不混写。新双引擎包会重新执行这两组卷积回归，生成各自原始 JSON。

新增 `lung_tensor_engine/control/axil.v` 为独立 PL 模块，复用已验证的 AXI 读/写主机和 Q31 舍入模块。
数据保持 **HWC8，64-bit word，小端字节 lane0 为低 8 位**。
PS 只配置地址、尺寸、量化参数和启动命令，池化/放大/拼接的核心操作在 PL。

支持操作：

| op | 操作 | A 输入尺寸 | B 输入尺寸 | 输出尺寸 |
| --- | --- | --- | --- | --- |
| 0 | 2×2 stride2 最大池化 | H×W×Ca，H/W 偶数 | 无 | H/2×W/2×Ca |
| 1 | 最近邻 2× 放大 | H×W×Ca，H/W≤256 | 无 | 2H×2W×Ca |
| 2 | 两分支重定标后拼接 | H×W×Ca | H×W×Cb | H×W×(Ca+Cb) |
| 3 | A 放大后与 B 重定标拼接 | H×W×Ca，H/W≤256 | 2H×2W×Cb | 2H×2W×(Ca+Cb) |

op3 融合解码器放大与拼接，避免单独放大结果的中间 DDR 写回。
拼接顺序固定 A（解码器）先、B（编码器 skip）后；Ca/Cb 不要求是 8 的倍数。
输出通道末尾不足一 word 的槽位填输出零点；池化/放大保持原量化网格。
concat 每个分支各用自己的输入零点、Q31 乘子和移位，一次 signed RNE 后加输出零点并裁剪 UINT8。
没有 ReLU、双重舍入或偷偷使用浮点的分支重定标。

## DDR 与安全

张量模块只保留小型控制寄存器、单 word 缓存和逐字节重定标流水线，不展开整帧到 BRAM。
最大 H/W 为 512，通道最多 96，拼接总通道最多 96；地址 8-byte 对齐，完整张量范围不得溢出 32-bit 物理空间。
输出不得与任一完整输入张量重叠；两个只读输入可以重叠。
读取和写回按状态机串行推进，AXI AW/W 独立握手，完成必须等最后 B 响应。
读/写错误使整个命令失效；之前写入的数据不回滚。缺失 RLAST 时先排空实际剩余读响应，才允许完成。
没有本地 soft reset、自动超时重启或 CLEAR 取消事务。恢复必须共享复位/整板恢复。
主机超时或中断时保留可能在途的缓冲，不能直接释放给分配器。

## 控制寄存器

双引擎工程中卷积地址为 `0x43c00000`，新增张量地址为 `0x43c10000`，各 1 KiB。
均共享 FCLK0 100 MHz、DDR HP0 和复位。低有效辅助复位明确接常量 1。

| 偏移 | 内容 |
| --- | --- |
| 0x00 / 0x0c | ID=`0x54454e53` / VERSION=`0x10000` |
| 0x04 | 1 START；2 CLEAR sticky 状态，不取消事务 |
| 0x08 | bit0 BUSY，bit1 DONE，bit2 ERROR，bit8 CONTROL_ERROR |
| 0x10 / 0x14 / 0x18 | A、B、输出的物理**字节**地址 |
| 0x1c | H[9:0]、W[25:16]，是 A 输入尺寸 |
| 0x20 | op[1:0]、Ca[14:8]、Cb[22:16] |
| 0x24 | 输入 A 零点[7:0]、B 零点[15:8]、输出零点[23:16] |
| 0x28 / 0x2c | A Q31 乘子 / 右移 0～62 |
| 0x30 / 0x34 | B Q31 乘子 / 右移 0～62 |
| 0x38 / 0x3c / 0x40 | 成功写回 word 数 / 读回 word 数 / 周期 |
| 0x44 / 0x48 | 完成命令数 / 拒绝控制写入数 |

AW/W 独立保存，支持 WSTRB、B/R 背压；运行时配置锁定。
未知/不对齐/只读/保留位写入 SLVERR 并记 CONTROL_ERROR；非法几何由引擎完成为 ERROR，不能到达 DDR。

## 已完成的本机验证

真实 XSim 通过 **26 组算子、29,848 个输出 word、32,690 次读取、19 组错误场景**。
包含 512 边界尺寸、96 通道、非 C8 对齐拼接、signed ties/裁剪、shift0/62、脏输入 padding，
以及冻结 CT 的激活与真实拼接参数。
AXI-Lite 启动、配置锁定、独立 AW/W、B/R 保持、最后 B 延迟、RRESP/RID/BRESP/BID 错误、
错误 RLAST 排空、输出区哨兵和共享在途复位后再次启动均已验证。
正式日志为 `data/lung_tensor_20261015/tensor_run_v3.log`。
独立主机地址模型也复算得出相同的 32,690 次读取。

初次综合发现零点相减→乘法路径 WNS -1.987 ns；新增真实的 signed9 中心化与参数寄存级后，
26 组 RTL 回归仍通过，独立综合 100 MHz WNS **+1.273 ns**。
新模块没有多周期豁免；原卷积模块精确的配置等待约束保持原样。
独立 OOC 占用：**2,205 LUT、1,667 FF、8 DSP、0 BRAM Tile**，不是完整板级占用。
207 项 Python 回归通过，1182 个受保护模型/医学来源哈希未改。
核验记录：`data/lung_tensor_20261015/verification.json`。

## 新工程与上板

`fpga/build_lung_network_board.tcl` 在 `data/lung_network_board_20261015/project/lung_network.xpr`
创建独立 BD：PS GP0 → 1进2出控制 SmartConnect → 两个引擎的 AXI-Lite；
两个引擎的 AXI master → 2进1出内存 SmartConnect → PS HP0 → DDR。
脚本会检查最终布线 setup/hold，再输出同名 `lung_network.bit/.hwh`。
`python/verify_lung_network_board.py` 核验 HWH 的两个地址空间、四路共享复位、时钟、DDR、
实际封装 RTL 来源、最终时序/资源/DRC，再打包到 `data/lung_network_board_20261015/lung_network/`。

2026-10-15 新 bit 已生成并核验：整板 100 MHz，WNS **+0.425 ns**、WHS **+0.013 ns**；
LUT **15,415（28.98%）**、FF **17,766（16.70%）**、DSP **81（36.82%）**、BRAM Tile **31.5（22.50%）**。
DRC 无 Error/Critical Warning，仍有 294 条 Warning，包含 DSP 流水线/OPMODE 以及 1 条指向 SmartConnect
内部复位管线无可布线负载的 RTSTAT-10；不能称为零告警。打包检查了卷积 IP 的 13 个可达 RTL 依赖和
张量 IP 的 6 个可达 RTL 依赖，均与仿真源码一致。
最终构建记录 `data/lung_network_board_20261015/verification.json` 的状态为
`built_routed_dual_engine_regression_not_board_tested`；仍待用户实际运行新 bit 的三阶段回归。

新包包含原真实层数据的一份副本，**只为新副本更新 bit/HWH 哈希绑定**，不改原始层数据包。
原来的 `lung_axil.bit/.hwh` 和全部旧证明保持不动。

在电脑 PowerShell 上传新目录：

```powershell
Set-Location "D:\pld_picture_process\data\lung_network_board_20261015"
& "C:\Program Files\Git\usr\bin\scp.exe" -r lung_network xilinx@192.168.2.99:/home/xilinx/ct_test/
```

停止其他正在使用 PL 的程序，在板端 Jupyter 运行：

```python
%cd /home/xilinx/ct_test/lung_network
%run lung_network_smoke.py --report /home/xilinx/ct_test/lung_network_smoke_result.json
```

依次执行：7 次旧小型卷积、45 次真实卷积层、26 次张量算子。
每阶段前检查文件哈希、打印加载/读 ID 的进度；前阶段异常会直接停止，不会继续加载下一阶段。
原始阶段报告分别为 `_conv.json`、`_layers.json`；最终汇总包含全部张量结果。
新报告及任一子报告必须不存在，不覆盖证据。预期最后输出：

```text
Real-board dual-engine regression: PASS
```

**这仍是双引擎独立回归，不是完整网络连续处理。**
它通过后，再实现 PS 层间调度、skip 缓冲生命周期和端到端 FPGA 整数 U-Net；
随后才核对最终 CT 掩膜、人工标注精度和完整链路耗时。
