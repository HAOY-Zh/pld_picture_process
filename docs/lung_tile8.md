# 64 路并行 MAC、共享重缩放与 BRAM 输入包缓存

2026-10-14：实现并验证 8×8 MAC tile，含真实 BRAM 的组输入缓存。
原模型、定点规则、CT、人工标注、既有 UI、BD 与 bit/hwh 均未更换。
**本阶段截至 2026-10-14 尚无完整 U-Net RTL、AXI/DMA 接入、持续权重复用、整图窗口控制或新比特流。**
2026-10-15 后续已单独实现并验证持续权重复用，见 [权重复用说明](lung_weight_reuse.md)；本页保留原阶段记录。
仅用于工程教学与算法演示，不作为临床诊断依据。

## 本轮实际硬件

- `fpga/lung_mac_tile8.v`：每个有效输入拍提供 8 个输入通道、8×8 权重，
  并行执行 64 个有符号 MAC。输入 UINT8 减零点后用有符号 9 位，权重有符号 8 位。
- 每个输入 lane 保留部分累加，八路用 33/34/35 位流水加法树归约，最后加入一次
  INT32 bias；部分累加及最终 INT32 范围错误都会标记，不能让回绕成为有效结果。
- 八个输出依次共享原来的四级 `lung_requant`。通道选择先经过弹性寄存器，
  8 项元数据 FIFO 保持通道标签、累加值、错误与 LAST 同量化结果对应。
- `fpga/lung_mac_tile8_cached.v`：128 拍缓存，72 位激活/keep 和 512 位权重
  放同步 BRAM，加载完整一组后播放到 tile，输出组全部消费后才接受下一组。
  RAM 内容本身不清零复位，通过有效位/长度/读指针保证不读未写区域。

这是实际 Verilog，不是 Python 分割或“用 CPU 代替 PL”。Python 只生成数学测试向量。
但本轮仍是在电脑上仿真/综合 RTL，尚未在板卡执行肺分割。

## 内部接口约定

所有端口都在同一个 `clk` 域，低有效同步 `rst_n`。输入只在 valid 且 ready 时接受；
first/last 标识一组输出的全部输入拍，第一拍捕获 bias、乘数/移位、零点、输出 keep 和 ReLU。
之后改变这些配置不能影响当前组；input keep 可以随每拍变化。

| 信号 | 位宽及含义 |
| --- | --- |
| activations | 64 位；lane 0 是最低字节，共 8 个 UINT8 |
| weights | 512 位；字节 `output_lane*8+input_lane`，INT8 二补码 |
| input_keep | 8 位；尾输入通道屏蔽，屏蔽槽任意垃圾值都不参与计算 |
| output_keep | 8 位；屏蔽输出不产生结果，支持稀疏掩码 |
| biases / multipliers / shifts | 每输出 lane 分别 32/32/6 位，低位对应 lane 0 |
| m_lane / m_last | 当前有效通道 0..7；LAST 在最后一个保留通道，而不是固定 lane 7 |
| m_acc / m_code / m_error | INT32 累加诊断值、UINT8 输出、无效计算/配置标志 |

外部控制器仍需保存 `(layer, y, x, output_channel_base)`，将 `m_lane` 转成实际通道。
若输出 valid 而下游未 ready，数据、通道、LAST 和错误必须保持不变。
零输出 mask、缺失 FIRST、累加溢出有明确错误；缓存超过 DEPTH 会截住写地址并标错，
不能覆盖 RAM/读未初始化地址，错误结果不得用于有效分割。
生产模型最大输入组是 `Cin=96, K=3`：`ceil(96/8)*3*3=108` 拍，低于 128。

**这不是 AXI 端口。** 不要把 512 位权重/64 位激活直接接现有 32 位 DMA，
不要将这些内部端口全部 Make External，也不要用本模块替换现有板端 bit 文件。
跨时钟域及 AXI 寄存器/流桥还需后续设计。

## 数值与协议实测

`python/lung_tile8_vectors.py` 将冻结的 15 层×3 组算子向量重组，另加
Cin/Cout=3/4、11/13、1/1 的尾通道/非零零点用例。无医学标签或精度调参。
权重及激活的屏蔽槽故意填垃圾，不能仅靠填 0 蒙混过关。

正式 XSim：直连和缓存各自通过 **8700 组、367338 拍、68046 个输出**对拍，
与原标量整数卷积/重缩放一致，掩码有效的标量 MAC 总数为 23371332。
还单独验证稀疏输出 mask、输入气泡、输出背压、元数据第一拍捕获、
活跃/半加载组复位、已出结果且停顿时复位、缺失 FIRST、零输出 mask、最终 bias 溢出、
缓存 129 拍越界报警以及复位后不会回放旧结果。

第一次全量测试在单拍 head 转到尾通道用例时失败：测试台在前一接受沿的同一
调度区更新了下一组元数据。已将所有新组元数据移动到下降沿，并重新跑完整测试。
失败日志保留，不用它宣称通过。XSim 的 `$fatal` 有时仍返回进程 0，
所以验收同时检查 Fatal、文件读取警告、**预期的非空输出数**和实际 PASS。
最终日志为 `data/lung_tile8_20261014/direct_run_v3.log`、`cached_run_v3.log`。

四项新增 Python 工程检查独立重算 masked 8×8 累加、符号/字节布局、
108/128 深度边界、帧描述/输出标签/末尾标记，完整 **152 项 Python 测试通过**。
两套真实 RTL 仿真是额外执行，不计入 152 项 Python 数字。

## Z7020 实际独立综合

`synth_lung_mac_tile8.tcl` 创建内存 OOC 工程，不打开原 XPR/BD。

| 缓存＋并行计算＋共享重缩放 | 初版 | 增加选择寄存器后 |
| --- | ---: | ---: |
| LUT | 3597 | 3611 |
| FF | 6029 | 6105 |
| DSP | 68 | 68 |
| RAMB36 / RAMB18 | 8 / 1 | 8 / 1 |
| BRAM Tile 等效 | 8.5 | 8.5 |
| 10 ns 内部路径 WNS | -2.814 ns | +1.273 ns |

最终 DSP 为 Z2 的 30.91%，LUT 6.79%，FF 5.74%，BRAM Tile 6.07%。
层级报告确认 **MAC 阵列 64 DSP，唯一共享重缩放 4 DSP**，不是复制 64 个完整单路核心。
BRAM 是实际综合映射：激活/keep 一块 RAMB36，权重七块 RAMB36 加一块 RAMB18；
没有把 `ram_style` 注释当成已使用 BRAM 的证据。

初版 priority select/8选1参数 mux 直接接宽乘法器，100 MHz 目标不满足；
增加带元数据的弹性 dispatch 寄存器后，规则不变，两套全量向量重新通过。
正式报告在 `data/lung_tile8_20261014/synthesis_v2/`。
这里只是综合后已约束内部路径的估算，未布局布线、未完整约束接口延迟及 OOC 时钟来源；
**不能据此宣布整机 100 MHz、端到端 FPS 或完整网络资源已经达标。**

## 缓存范围与接下来应做什么

128 深度缓存的有效逻辑容量为 `128*(72+512)/8=9344` 字节。
宽并行读端口导致实际 BRAM 占用大于简单容量除法，这属于带宽/容量取舍。
当前一组加载与计算不重叠，每个输出组仍会重新加载权重；有加载、归约、串行量化
及背压开销，**不能把 64 MAC 峰值当作端到端 64 倍加速**。

下一步优先拆分：

1. 持续权重缓存：同一层/输出块的权重一次加载，多位置复用，避免每个像素重复搬权重。
2. 特征/行窗口控制：按 XY、输入块、kernel 遍历，处理边界和尾通道；结合 DDR 保存大特征与 skip。
3. AXI/DMA 数据桥和参数/状态寄存器，然后 pool/nearest/concat、全网层间调度。
4. 整网实现/路由、bit/hwh、板端精度和延迟/带宽/内存实测。

当前输入包缓存不是整图缓存，不解决三维连通域/整例 CT 在共享 512 MB DDR 中的存储问题。
本轮不更换算法，因此未重新训练或用新病例评价 Dice；前一阶段整网定点验证仍是数学基线。
独立复核 `data/lung_tile8_20261014/verification.json` 保存非空仿真、源哈希、
实际资源及未完成标志；1182 个受保护模型/数据/来源哈希未变。

## 复现

项目根目录先生成新目录（不能覆盖已有结果）：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_tile8_vectors.py --outdir data/lung_tile8_new/vectors
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -v
```

仿真，按自己的 Vivado 安装路径调整，逐条检查编译和日志：

```powershell
Push-Location data/lung_tile8_new
& 'E:\vivado2\Vivado\2022.2\bin\xvlog.bat' --sv ../../fpga/lung_requant.v ../../fpga/lung_mac_tile8.v ../../fpga/lung_mac_tile8_cached.v ../../fpga/tb_lung_mac_tile8.sv
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_mac_tile8 -s tile_direct
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' tile_direct -runall -log direct_run.log
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_mac_tile8_cached -s tile_cached
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' tile_cached -runall -log cached_run.log
& 'E:\vivado2\Vivado\2022.2\bin\vivado.bat' -mode batch -source ../../fpga/synth_lung_mac_tile8.tcl -tclargs D:/pld_picture_process/data/lung_tile8_new/synthesis
Pop-Location
```

暂不需要用户修改 BD 或烧录板卡。先让缓存/控制与总线接口完整后再接入。
