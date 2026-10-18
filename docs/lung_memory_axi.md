# 单卷积块的 AXI 读—计算—写回闭环

2026-10-15：新增可综合输出写回 RTL，完成真实 XSim、独立复核与 Z7020 独立综合。
旧窗口、缓存、卷积核心、模型权重、医学数据、BD、bit/hwh 均未改动。
仅用于工程教学与算法演示，不作为临床诊断依据。

## 做到了什么

`lung_conv8_memory_axi.v` 将原 `lung_conv8_cached_axi.v` 接到新 `lung_output_writer.v`、
`lung_axi_write64.v`：从 AXI 内存取输入、硬件卷积/量化、打包后通过 AXI 写回。
外部仿真服务只模拟内存与握手，不替硬件计算卷积。

一个命令仍只计算一张特征图的一个输出通道块，不是完整 U-Net。
完成条件改为最后一个输出字收到 B 响应且内部确认完成，不再以卷积吐出最后一个字节为准。
写回等待期间持续 busy，不接新命令，也不接替换权重。

## 命令与输出布局

权重加载端口与原模块相同。合法权重块首拍 `w_first` 锁存输出 keep 与输出零点；
调用方必须遵循既有 FIRST/LAST 协议，并在命令开始前完成权重加载。

- `c_base`：输入缓冲区的 64 位 word 基址，非字节地址。
- `c_height/c_width/c_channels/c_kernel3/c_padding`：原输入几何；步长仍为 1。
- `c_output_base`：完整输出张量的 word 基址。
- `c_output_blocks`：完整输出张量的 `ceil(Cout/8)`，1～12。
- `c_output_block`：本次计算的输出块索引，必须小于 blocks。

输入 H/W 为 1～512、Cin 为 1～96；1×1 不补边，3×3 支持 pad=1 或 valid。
valid 卷积输出 H/W 分别减 2，输入两轴必须均不小于 3。

```text
output_word = output_base + ((y * output_width + x) * output_blocks + output_block)
AXI byte address = output_word * 8
```

输出按像素收集 keep 中启用的 lane，顺序递增；检查坐标、lane、像素 LAST 和帧 LAST。
一个完整像素块组成 64 位字，lane 0 位于最低字节。
未启用的 lane 填输出零点，WSTRB=0xff；不是保留旧内存里的尾通道。
其余输出块的整个 word 不写入。后续调度必须算完需要的所有输出块，再启动消费该张量的下一层。

输入和输出**完整张量分配范围**均须落在 32 位 AXI 字节地址可表示范围内；
允许最后一个 word 为 `0x1fffffff`，但不能溢出 `2^29` word 的右开端点。
完整输入/输出范围不允许重叠，不能原地覆盖。当前没有动态内存分配、权限保护或缓存一致性硬件。
真实 PYNQ 驱动还需验证缓冲区大小/物理地址，并正确 flush/invalidate；B 响应不能替代 CPU 缓存同步。

## AXI 与错误边界

写桥是 AXI4 单拍 INCR、64 位、固定 ID 0，一次一笔在途事务；AWLEN=0、AWSIZE=3、WLAST=1。
AW 与 W 独立握手，任何一个先到都不会丢失；在背压下保持地址/数据不变。
等两通道都接收后等待 B，检查 BRESP/BID，再向打包器报告确认。
它不是多拍突发写入或高吞吐 DMA；不同输出块在 HWC8 中跨 word，因此目前统一使用单拍。
单拍且 8 字节对齐，不跨 4 KiB 边界。写桥单独支持非零 WSTRB；打包器使用完整字写。

非法配置不发 AR/AW/W，并以 error 完成。
读错误、输出字节错误或元数据错误使整个命令失败，当前坏像素及后续像素不再发起写入，
继续接收剩余正常长度流。B 错误也标记命令失败并禁止后续写入。
`written_words` 只统计收到成功 B 响应的字数，不是错误帧的可用像素数。
先前已写入的字不会回滚，错误响应对应的内存也不能假定未修改；软件必须丢弃整个失败命令的输出。

对畸形流按预期计数检查终止，过早 frame_last 可结束错误命令；如果输入流永久停止则没有超时。
若生产者与接收者因畸形流失步，必须执行系统级恢复，不保证继续复用残留流。
复位假定本模块、读/写主机、缓存及响应服务共同取消在途事务；不能只复位此核心，再把旧响应当新事务。
未实现 AXI-Lite、软件超时/中断、CDC、权重 DMA、层间算子或网络调度。

## 实际验证

沿用上一阶段 147 个冻结任务，不复制原输入/权重向量，不重新推理或训练。
新向量只记录输出分配、地址及 64 位黄金字。

| 验证项 | 实际结果 |
| --- | ---: |
| 卷积任务 | 147 |
| 原累加值/量化值/坐标逐项对拍 | 68,849 个结果全部一致 |
| AXI 输出写回 | 8,866 个 64 位字，70,928 字节载荷 |
| 输入 AXI 读取 | 2,612 个突发，41,606 拍，与前阶段相同 |
| 写桥协议 | 10 组 |
| 打包器 | 23 组，1,049 次写入，另有在途共享复位 |
| 完整链路非法配置/读写错误 | 14 组，另有在途共享复位 |
| Python 工程回归 | 167 项通过 |

整链路测试逐个任务检查所有写地址/数据、输出块间隔、2 万 word 模拟分配内的未写区域哨兵值，
并在每帧最后一次 B 响应前强制停顿，检查不会提前 done、接受新命令或覆盖权重。
写桥测试包含 AW 先/W 先、响应确认背压、部分 WSTRB、4 KiB 两侧/物理末端、BRESP/BID 与共享复位。
打包器额外检查稀疏 keep、尾通道、512×1/1×512、坐标/lane/LAST 错误、读错误传递和非法范围。

`verify_lung_memory_axi.py` 用密集像素字节数组独立重建 HWC8，不调用生成器的打包函数；
核验三套日志的精确非空 PASS、原数学/缓存证明、当前综合与 **1182 个冻结医学/模型来源**哈希。
正式证据为 `data/lung_memory_axi_20261015/verification.json`。
正式整链路日志是 `memory_axi_run_v2.log`；早期测试台动态文件名读取失败的日志保留，不作为通过证据。
验收不只看 XSim 退出码；Fatal、文件读取警告或结果数量不足均拒绝。

## Z7020 实际独立综合

| 整个读缓存＋卷积＋打包写回链路 | 使用量 | Z2 占比 |
| --- | ---: | ---: |
| LUT | 7,305 / 53,200 | 13.73% |
| FF | 8,232 / 106,400 | 7.74% |
| DSP | 73 / 220 | 33.18% |
| BRAM Tile 等效 | 31.5 / 140 | 22.50% |
| 10 ns 综合后内部 WNS | +1.273 ns | 非布局布线 |

相对前阶段增加 344 LUT、20 FF、3 DSP，没有新增 BRAM。
这些不是整个 U-Net 或完整系统的资源；仍缺 AXI-Lite、网络控制、层间操作与板级互联。
OOC 没有完整接口延迟/时钟来源约束，也未布局布线，不能宣称整机已达到 100 MHz 或给出真实 DDR/FPS。

## 复现与下一步

保留并校验通过的旧 `data/lung_cache_axi_20261015/vectors` 是只读依赖。
整链路测试台用固定相对路径读取该目录，避免 Windows XSim 动态字符串文件名兼容问题。
在项目根目录生成到新目录，随后从该目录运行 Vivado（本机安装路径按需修改）：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_write_vectors.py --outdir data/lung_memory_axi_new/vectors
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -v *> data/lung_memory_axi_new/python_tests.log
Push-Location data/lung_memory_axi_new
& 'E:\vivado2\Vivado\2022.2\bin\xvlog.bat' --sv ../../fpga/lung_requant.v ../../fpga/lung_mac_tile8.v ../../fpga/lung_mac_tile8_reuse.v ../../fpga/lung_feature_window.v ../../fpga/lung_conv8_window.v ../../fpga/lung_feature_cache.v ../../fpga/lung_axi_read64.v ../../fpga/lung_conv8_cached_axi.v ../../fpga/lung_axi_write64.v ../../fpga/lung_output_writer.v ../../fpga/lung_conv8_memory_axi.v ../../fpga/tb_lung_write64.sv ../../fpga/tb_lung_output_writer.sv ../../fpga/tb_lung_memory_axi.sv
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_axi_write64 -s write_protocol
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' write_protocol -runall -log write_protocol_run.log
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_output_writer -s output_writer
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' output_writer -runall -log output_writer_run.log
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_memory_axi -s memory_axi
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' memory_axi -runall -log memory_axi_run_v2.log
& 'E:\vivado2\Vivado\2022.2\bin\vivado.bat' -mode batch -source ../../fpga/synth_lung_memory_axi.tcl -tclargs D:/pld_picture_process/data/lung_memory_axi_new/synthesis
Pop-Location
.\.venv-lung\Scripts\python.exe python/verify_lung_memory_axi.py --root data/lung_memory_axi_new
```

每一步都应检查工具退出码、日志与精确 PASS；已有证据目录不覆盖。
接下来补 **AXI-Lite 配置/状态控制**，然后推进层间缓冲、pool/nearest/concat、整网调度。
BD/真实 DDR 接入、完整实现时序、新 bit/hwh、板端数值/精度/延迟验证仍在后面。
