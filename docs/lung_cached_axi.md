# 特征缓存与 AXI4 读通路

2026-10-15：完成可综合 RTL 和真实 XSim 验证，未修改旧窗口/卷积核心、模型、医学数据、BD 或 bit/hwh。
仅用于工程教学与算法演示，不作为临床诊断依据。

## 本轮硬件

- `lung_feature_cache.v`：256 组、三路组相联、每行 16 个 64 位字；有效特征容量 **96 KiB**。
  标签与数据为同步 BRAM，valid/替换指针复位，RAM 内容不清零；命中不改变 FIFO 替换顺序。
- `lung_axi_read64.v`：真正的 AXI4 AR/R 读主机 RTL，64 位数据、固定 ID 0、INCR 突发、
  最长 16 拍、一次只有一笔事务。包括长度、地址/4 KiB 检查，AR/R 背压和错误处理。
- `lung_conv8_cached_axi.v`：将新缓存/读主机接入未改动的窗口/权重复用/64 路卷积链路。
  每个命令仍计算一个输出通道块的整张特征图，输出保留坐标、通道、LAST 和错误标记。

现在仿真中的外部存储通过 **AXI 内存模型**服务，不是将 Python 算好的窗口传给硬件。
但 AXI RTL 仿真通过不等于已经连接 PYNQ DDR，更不等于完整网络已经上板。

## 地址、范围与一致性

上游沿用 HWC8 64 位 **word 地址**：

```text
word = base + ((y * width + x) * ceil(channels / 8) + channel_block)
AXI byte address = word * 8
```

AXI 物理地址为 32 位，因此读桥拒绝 word 地址 >=2^29，不能将旧逻辑窗口允许的全部
32 位 word 地址误当作 32 位物理字节地址。实际 PYNQ 分配的缓冲区、DDR 容量和访问权限还要由后续驱动保证。
base 必须来自按 8 字节对齐的真实缓冲区地址除以 8，不能直接填 Python 数组索引或虚拟地址。

每次命令捕获 `base + H*W*ceil(Cin/8)` 的合法范围，并清空缓存有效位。
缺失时按 16 字对齐缓存行读取，但首尾突发裁剪到合法特征范围；不读缓冲区前后，不跨 4 KiB 边界。
padding 仍由旧窗口控制器补输入零点，不访问缓存/AXI。

每帧清空有效位是正确性策略：即使 base 相同，PS 可能已经更新该地址的数据。
当前没有跨帧/跨输出块的“同地址自动保留”或 CPU 缓存一致性协议；未来驱动还需处理 flush/invalidate。
调用期间输入特征必须保持不变，不能 CPU/PL 同时覆写。帧内命中只表示已成功填充的同一行。

## 错误与恢复

RRESP 错误、错误 RID、过早 RLAST 或预期长度处缺失 RLAST，都使本次填充失败：
**整行不提交有效标签**，当前读取报错；重试必须重新缺失读取，不能命中半行/错误数据。
窗口将失败读取替换为真实零点，结果/帧错误会向外传递，不能将其当有效分割。

RLAST 迟到时，桥在报告错误后继续排空到实际 LAST，不接受新的 AR；不会把残留数据交给下一事务。
没有响应或永远没有 LAST 时没有自动超时，需要后续驱动/系统恢复机制。
复位假定读主机、缓存与存储响应服务共同取消在途事务；实机连接 AXI 互联后必须明确排空/隔离策略，
不能只复位本模块再允许旧 ID=0 的响应混入新帧。

接口只实现 **读**：没有 AW/W/B 写回、AXI-Lite 配置寄存器、完整软件驱动、中断、CDC 或网络调度。
暂时不要把这些内部参数端口全部 Make External，或直接替换旧板端工程的 DMA 模块。

## 验证与读取量

原 147 个通道块任务的黄金数据、窗口、坐标和卷积结果保持字节一致。

| 同一批冻结向量 | 数量 |
| --- | ---: |
| 卷积输出，全数一致 | 68849 |
| 窗口拍，全数一致 | 371974 |
| 窗口发出的单字读取请求 | 312018 |
| 缓存命中 | 309406 |
| 实际 AXI 突发 | 2612 |
| 实际 AXI 64 位读取拍 | 41606 |

相对无缓存的 312018 字读取，仿真读取载荷减少 **86.6655%**，每帧统计和每个 AR 地址/长度均匹配独立模型。
这只是该批冻结向量的流量，**不是整网/真实 DDR 带宽、延迟或 FPS 提升 86.7%**。
单请求、标签查询、状态切换和串行量化仍有开销，缓存容量不能直接换算加速倍数。

额外三套 RTL：

- 缓存协议 28 次读取：三路 FIFO 冲突、未对齐首尾、越界拒绝、同地址新帧、坏填充后重试、
  RRESP/RID、早/晚 RLAST、4 KiB、物理地址限制及在途共享复位。
- AXI 协议 10 组：字→字节地址、AR/R 停顿、非法长度/地址、4 KiB、早晚 LAST、隔离排空及复位。
- 默认几何 18437 次读取：连续装入 **9216 字**（三行×256 像素×96 通道），第二遍全部命中，
  再测同组第四标签替换。这验证默认容量，而不只依赖小图或缩小参数。

96 KiB 是分组缓存，不是固定“三整行 512 像素×96 通道”缓存；后者需要 144 KiB，超出本缓存容量。
超容量仍可通过重新缺失保持数值正确，但命中率会下降。本轮没有完整网络的全尺寸 AXI 性能实测。

163 项 Python 工程测试通过，四套真实 XSim 额外执行。测试台的保留字变量和复位释放后同调度区采样问题
已修正；早期日志保留，不作为通过证据。正式 AXI 协议记录为 `axi_protocol_run_v2.log`，
缓存协议为 `cache_protocol_run_v2.log`；另有 `cached_axi_run.log` 和 `cache_capacity_run.log`。
验收同时检查 Fatal/文件警告及精确非空 PASS，不只看 XSim 退出码。

`verify_lung_cached_axi.py` 用第二种 FIFO 队列实现独立复算所有读取、替换、突发和每帧统计，
确认数学黄金值与前阶段独立卷积证明一致，并核验四套日志、资源/时序和 1182 个冻结医学/模型来源哈希。
正式记录：`data/lung_cache_axi_20261015/verification.json`。本轮未训练、调分割参数或选择独立测试病例。

## Z7020 实际独立综合

| 缓存＋AXI 读＋窗口/卷积 | 使用量 | Z2 占比 |
| --- | ---: | ---: |
| LUT | 6961 / 53200 | 13.08% |
| FF | 8212 / 106400 | 7.72% |
| DSP | 70 / 220 | 31.82% |
| BRAM Tile 等效 | 31.5 / 140（28 RAMB36＋7 RAMB18） | 22.50% |
| 10 ns 内部综合后 WNS / TNS | +1.273 ns / 0 ns | 非布局布线 |

特征缓存独占 24 BRAM Tile 等效（含标签），原权重块占 7.5；读桥不新增 DSP/BRAM。
命中查询/替换表还消耗 LUT/FF，不能只按有效字节数估算资源。
正式报告在 `data/lung_cache_axi_20261015/synthesis/`。
这里只是 OOC 综合，接口延迟/时钟来源未完整约束，没有连接真实互联/DDR或布局布线，不能宣称整机 100 MHz。
也不能用本读通路的 22.5% BRAM 占比代表完整 U-Net 所有缓存与控制已经放得下。

## 下一步与复现

接下来补输出打包/AXI 写回与 AXI-Lite 控制寄存器，然后做层间缓冲、pool/nearest/concat、整网调度。
完成控制/读写闭环后再接 BD，做完整实现/路由、bit/hwh、板端数值/精度与内存/延迟/带宽实测。

依赖前阶段本地冻结向量与审计文件；这些数据不会随源码 Git 提交分发。选择新目录保留证据，
按自己的 Vivado 路径调整，在根目录执行：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_cache_vectors.py --outdir data/lung_cache_axi_new/vectors
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -v
Push-Location data/lung_cache_axi_new
& 'E:\vivado2\Vivado\2022.2\bin\xvlog.bat' --sv ../../fpga/lung_requant.v ../../fpga/lung_mac_tile8.v ../../fpga/lung_mac_tile8_reuse.v ../../fpga/lung_feature_window.v ../../fpga/lung_conv8_window.v ../../fpga/lung_feature_cache.v ../../fpga/lung_axi_read64.v ../../fpga/lung_conv8_cached_axi.v ../../fpga/tb_lung_cached_axi.sv ../../fpga/tb_lung_cache_protocol.sv ../../fpga/tb_lung_axi_read64.sv
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_cached_axi -s cached_axi
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' cached_axi -runall -log cached_axi_run.log
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_cache_protocol -s cache_protocol
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' cache_protocol -runall -log cache_protocol_run_v2.log
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_cache_capacity -s cache_capacity
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' cache_capacity -runall -log cache_capacity_run.log
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_axi_read64 -s axi_protocol
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' axi_protocol -runall -log axi_protocol_run_v2.log
& 'E:\vivado2\Vivado\2022.2\bin\vivado.bat' -mode batch -source ../../fpga/synth_lung_cached_axi.tcl -tclargs D:/pld_picture_process/data/lung_cache_axi_new/synthesis
Pop-Location
.\.venv-lung\Scripts\python.exe python/verify_lung_cached_axi.py --root data/lung_cache_axi_new
```
