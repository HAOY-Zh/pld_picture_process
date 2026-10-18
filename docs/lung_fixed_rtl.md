# 整网定点验证与首个肺分割 RTL 算法核心

2026-10-14：完整拟采用整数网络通过全部六个开发验证病例；卷积 MAC 与重缩放
RTL 已实际通过 Vivado 2022.2 XSim，对 Z7020 做了独立综合。
**尚未实现完整网络 RTL、AXI/DDR 调度、新比特流或板端智能分割。**
公开数据仅用于工程教学与算法演示，不作为临床诊断依据。

## 整网整数规则的验证

复用上一阶段冻结的 `w8_a8_acc32_q31_rne_v1` 规范、原第 12 轮小 U-Net，
原 18/6 患者划分不变。仅六个开发验证病例、1120 张切片，不训练现有模型、
不校准、不改变人工标注，不使用 Test/COVID，不搜索阈值或量化策略。

`python/lung_fixed_network.py` 实现原完整图：
15 个卷积、3 次池化、3 次最近邻放大、3 个重缩放 concat、原 skip 顺序。
激活/权重/偏置/Q31 舍入按上一阶段导出的整数规范执行。
输入归一化、输入量化、输出解量化、Sigmoid 和原尺寸概率恢复仍在电脑参考中；
阈值仍为 0.5，人体约束与 5 mL/20 mm 三维清理保持不变。

### 加速参考的数学边界

代码使用 FP64 direct GEMM 来加速**整数值的卷积累加**，不使用浮点训练网络来代替定点路径。
中心化激活与权重都是整数，所有乘积与任意部分和的保守上界远低于 2⁵³；
禁用 cuDNN 算法变换，逐层检查有限值、整数精确性及 INT32 范围，之后使用
真正的 INT64 乘数/移位/最近偶数舍入生成 UINT8 编码。
若卷积结果不再是精确整数则中止，不能直接四舍五入掩盖误差。

独立 NumPy 全图逐层比较全部 18 个卷积/concat 输出，CPU/CUDA 编码一致；
另检查 batch 等价性、预处理/解量化一致性和错误输入拒绝。
实际冻结模型的 15 层满范围向量也与 NumPy 整数参考一致。
这些都是**电脑数学参考检查**，不是 GPU 时间到 FPGA FPS 的换算。

| 同一六例原生分辨率与相同后处理 | 平均病例 Dice | FP | FN | 空参考误分层 |
| --- | ---: | ---: | ---: | ---: |
| 冻结 FP32 最终 | 0.9672978510 | 261212 | 260190 | 1 |
| 电脑 x86 INT8 最终 | 0.9670989970 | 271633 | 253918 | 1 |
| 拟用硬件整数规则最终 | 0.9670878194 | 271644 | 254062 | 1 |

定点最终相对 x86 INT8 为 **-0.0011178 个百分点**，相对 FP32 为 -0.0210032 个百分点。
小肺区逐切片 Dice 相对 x86 INT8 +0.000159163。
相比 x86 INT8，原始掩膜 4234 个体素不同、最终掩膜 3733 个体素不同；
未要求掩膜完全相同，也没有把细小误差或小肺区问题隐藏掉。
六例后处理均删除 0 个原定点预测中与专家重合的体素。
S3-002 第 186 层：专家 0、定点原始 5573、最终 0。
工程筛查通过，只支持进入下一硬件阶段，不是独立临床准确性结论。

产物：`data/lung_fixed_network_20261014/screen/`，
保存完整指标、真实原生掩膜、人体范围和三维决定；1182 个受保护输入哈希未变。
原 UI/模型/板端文件不替换，仍可按原入口查看此前结果。

## 实际 RTL：不是再写一份 Python

- `fpga/lung_int8_dot.v`：可复用单路卷积点积 MAC。控制器提供 `Cin×K×K` 个激活/权重，
  用 first/last 标识每个输出像素的点积；激活减零点用有符号 9 位，权重有符号 8 位，
  bias/输出配置在 first 捕获，ACC32 溢出及错误标记被报告。
- `fpga/lung_requant.v`：INT32×Q31、一次有符号最近偶数舍入、输出零点、ReLU、饱和。
  四级弹性流水：乘积 → 商/guard/sticky → 舍入加零点 → 饱和。
  任何输出 valid 而下游未 ready 时，数据保持稳定；复位清空事务。
- `fpga/tb_lung_int8_dot.sv`、`tb_lung_requant.sv`：真实 XSim 测试台。
- `python/lung_rtl_vectors.py`：仅转换数值规范为 `.mem`，不做分割、不读医学标签。

`lung_int8_dot` 目前等待一个输出被消费后才接下一个点积，不是全网络一像素/周期。
它不包含空间窗口生成、行/特征缓存、权重搬运、pool/concat RTL、AXI 或层间控制。
后续须给这些组件提供控制器，不能把它直接接 DMA 就认为能运行 U-Net。

## 真实 XSim 对拍与检查

48 组卷积用例：15 层×3 组冻结向量，以及非零零点、融合 ReLU、正负 ties 的 3 组补充。
**67952 个输出、23354608 个 MAC 输入 beat 全部逐值一致**，包括 ACC32 和 UINT8 输出。
独立重缩放 **69522 个输出**全部一致，额外覆盖 INT32 极值、shift 0..62、非法参数。
测试还覆盖输入气泡、输出背压、first 配置捕获、活跃事务/流水复位、溢出和漏 first 报警。

首次使用动态 SystemVerilog 字符串读取文件时出现 XSim 2022.2 兼容问题，导致
空向量；那次输出不作为验收。已改为固定 `vectors/` 子目录，加上文件存在、
非空计数、未知输出拒绝，以及脚本对日志中**实际输出数**的检查。
正式日志为 `data/lung_rtl_20261014/requant_run_v3.log` 和 `dot_run_v3.log`，
不能凭单独一个 PASS 字样判断成功，也不接受 0 输出通过。

## 实际 Z7020 独立综合与时序

`fpga/synth_lung_int8_dot.tcl` 创建内存工程，对 `xc7z020clg400-1` 做 OOC 综合；
不打开/编辑任何已有 XPR、BD、基础处理 IP 或板端 bit/hwh。

| 当前单路 MAC＋重缩放核心 | 第一版两级重缩放 | 优化后四级重缩放 |
| --- | ---: | ---: |
| LUT | 685 | 423 |
| FF | 190 | 366 |
| DSP | 5 | 5 |
| BRAM | 0 | 0 |
| 10 ns 时钟内部路径 WNS | -5.805 ns | +1.273 ns |

第一版时序失败不是忽略或降级标准；通过拆分商/舍入/饱和流水、guard/sticky 判定
去掉长 carry 路径，保持数学规则不变，重新通过全部向量。
5 DSP 包含约 1 个 MAC 和 4 个宽乘数重缩放；**不能复制 64 套这个完整模块**，
那会需要约 320 DSP，超过 Z2 的 220 DSP。
下一版目标是 8×8 MAC tile（64 MAC）共享少量重缩放，配合 banked BRAM/DDR，
这只是结构方向，不能用这份单路报告保证 64 路已实现或资源达标。

正式报告位于 `data/lung_rtl_20261014/synthesis_v2/`。
这里只是综合后未布局布线的内部寄存器路径估算，外部输入/输出延时未完整约束，
OOC 时钟来源/偏斜也未做板级约束；**不是整机时序签核、实测频率或帧率**。
0 BRAM 仅说明当前算术核心未带图像缓存，不代表 U-Net 不需要 BRAM/DDR。

独立复核记录 `data/lung_rtl_20261014/verification.json` 核对了六例掩膜、全部
受保护来源、非空 XSim 结果、真实资源、对应 RTL 源码哈希和未完成边界。
完整 **148 项 Python 工程测试通过**，新增 5 项完整整数图检查、3 项向量编码检查。
RTL 仿真单独实际执行，不包含在 148 项 Python 测试数字中。

## 复现与下一步

电脑项目根目录运行，必须指定新的空目录：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_fixed_network.py --device cuda --outdir data/lung_fixed_network_new/screen
.\.venv-lung\Scripts\python.exe python/lung_rtl_vectors.py --outdir data/lung_rtl_new/vectors
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -v
```

仿真示例（按自己的安装路径修改）：

```powershell
Push-Location data/lung_rtl_new
& 'E:\vivado2\Vivado\2022.2\bin\xvlog.bat' --sv ../../fpga/lung_requant.v ../../fpga/lung_int8_dot.v ../../fpga/tb_lung_requant.sv ../../fpga/tb_lung_int8_dot.sv
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_requant -s requant_check
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' requant_check -runall -log requant_run.log
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_int8_dot -s dot_check
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' dot_check -runall -log dot_run.log
Pop-Location
```

每个编译/仿真命令必须检查非零退出或 fatal；两份日志须检查预期的非空输出数。
有些 Windows 安装脚本提示 DOSKEY 缺失，但本机编译/仿真可运行；不要把启动脚本
消息与实际 RTL 数值失败混淆，也不要忽略真实 fatal/文件读取警告。
官方工具入口依据：[AMD UG900](https://docs.amd.com/r/2023.2-English/ug900-vivado-logic-simulation/xelab-xvhdl-and-xvlog-xsim-Command-Options)。

独立综合示例（输出目录不得预先存在）：

```powershell
& 'E:\vivado2\Vivado\2022.2\bin\vivado.bat' -mode batch -source fpga/synth_lung_int8_dot.tcl -tclargs D:/pld_picture_process/data/lung_rtl_new/synthesis
```

执行时另设独立工作目录/日志位置，避免 Vivado 在源码根目录生成日志。
下一步：**64 MAC tile 与共享重缩放 → banked 行/特征缓存和权重加载
→ pool/nearest/concat RTL → 全网络分层控制与 AXI/DDR → 实现/路由
→ 新 bit/hwh → 板端精度、延迟、内存与帧率实测**。
当前仍不需要用户改 BD；应等数据路径与接口准备好后再接入。

后续已完成 64 路 MAC/共享重缩放和同步 BRAM 输入包缓存，见
[并行核心与缓存](lung_tile8.md)。本页为单路阶段历史记录，原报告保留不覆盖。
