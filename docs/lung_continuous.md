# 完整 U-Net 的连续 PL 推理

## 与之前测试的区别

用户截图已显示 `Real-board dual-engine regression: PASS`。这证明两个引擎可以在同一个
bit 中完成独立回归；原始 JSON 尚未回传，不把截图写成已核验的原始报告。

新步骤只增加 PS 调度，不改变 RTL、模型、量化参数、病例划分或原有证明文件。
输入是原验证病例 `LCTSC-Train-S2-006` 的第 72 层、256×256 单通道 UINT8 编码。
只上传一次初始图像；21 个运算节点按依赖连续执行：

```text
输入 → enc1 → pool → enc2 → pool → enc3 → pool → neck
          └─────────────── skip1 ─────────────────────────┐
                         └────── skip2 ──────────┐       │
                                        skip3 ─┐│       │
neck → 放大/拼接skip3 → dec3 → 放大/拼接skip2 → dec2 → 放大/拼接skip1 → dec1 → head
```

总计 15 个卷积层（45 条通道块命令）、3 次池化、3 次融合上采样/重定标/拼接，共 51 条命令。
卷积、ReLU、Q31 舍入、池化、上采样和拼接的核心计算在 PL；PS 负责权重加载、寄存器配置和 DDR 缓冲生命周期。
下一层的物理输入地址就是前一层的实际输出地址，没有 CPU 算子回退或参考激活灌入。
原三条 skip 被保留到相应拼接完成，缓冲只在最后一个消费者结束后释放。

## 验证与安全

- `pynq/lung_continuous.py` 是调度器，不依赖 PyTorch，不读取参考激活。
- `parameters.npz` 仅有 15 层的权重、偏置、乘子、位移；参考输出独立放在 `reference.npz`。
- 测试程序只读比较每个节点的实际输出，包括 HWC8 尾部填充；比较回调拿到独立拷贝，不能改写设备输入。
- 每个设备分配保留 16 个哨兵 word；检查尚未执行的输出通道块未被覆盖。
- 两引擎串行调度，前一命令完整返回后才能发依赖命令。超时/不确定完成保留全部在用缓冲和驱动，禁止自动重试或局部复位。
- PC 模拟器根据实际上游模拟内存计算，检查完整 21 节点和 51 条命令；它不进入板端包，也不冒充真实 FPGA 结果或耗时。

原来的单引擎、逐层和双引擎回归包保持不变。新包位置：
`data/lung_continuous_20261015/lung_continuous/`。
构建/测试证据：`data/lung_continuous_20261015/verification.json`。

本机已通过完整 **219 项工程测试**；真实 CT 256×256 输入的 21 个节点在 CPU 实际内存流模拟中
全部逐 word 对齐冻结参考。计划中同时存活的设备缓冲峰值为 **2,359,680 bytes（约 2.25 MiB）**，
这是调度器的 CMA 申请量估计，不包含系统、模型文件和 Python 运行内存，也不是板端实测占用。
新上传目录共约 **6.18 MB**。连续 FPGA 推理仍待下面的实板操作确认。

以上为构建时状态；2026-10-15 的实板接收结果见下文，不回写原构建证明。

## 上板

同一份已通过双引擎回归的 bit/HWH，不需要再搭 BD 或生成 bit。
电脑 PowerShell：

```powershell
Set-Location "D:\pld_picture_process\data\lung_continuous_20261015"
& "C:\Program Files\Git\usr\bin\scp.exe" -r lung_continuous xilinx@192.168.2.99:/home/xilinx/ct_test/
```

确认上一测试已正常结束，没有其他程序使用 PL；板端 Jupyter 运行：

```python
%cd /home/xilinx/ct_test/lung_continuous
%run lung_continuous_test.py --report /home/xilinx/ct_test/lung_continuous_result.json
```

每个节点打印一次进度和 `continuous output/reference: SAME`，最后应为：

```text
Real-board continuous U-Net: PASS (21 nodes, 51 commands, all outputs bit-exact)
```

成功后保存原始报告及真正的 PL `head_codes`：
`/home/xilinx/ct_test/lung_continuous_result.json` 与 `lung_continuous_result_head.npz`。
若已经有同名报告，使用新的 `--report` 文件名，不能覆盖旧证据。
若卡住或报错，停止后续操作并提供最后输出；不盲目重跑或重新下载 bit。

## 资源与验收边界

复用当前双引擎工程的已布线资源：LUT 15,415（28.98%）、FF 17,766（16.70%）、
DSP 81（36.82%）、BRAM Tile 31.5（22.50%）；100 MHz，WNS +0.425 ns、WHS +0.013 ns。
PS 调度不会把 15 个卷积层各复制一套到 PL。特征存储使用 DDR，不把整幅图或整个人体积塞进 BRAM。

即使本次成功，也只确立**单张真实验证切片的完整整数网络核心连续推理**。
HU 转换、输入归一化、输出反量化/sigmoid/原尺寸插值、人体约束、三维清理、体积和界面仍属另外的系统链路。
不能因此宣称全病例 Dice、临床有效性、持续 FPS 或整个分割系统均已板端验证。
诊断耗时包含分配、权重装载、同步、轮询和参考比较，不是稳定吞吐率。

## 已回传的真实实板结果（2026-10-15）

用户已回传 `lung_continuous_result.json` 与 `lung_continuous_result_head.npz`。
新增 `python/verify_lung_continuous_board.py` 核验包/bit/HWH 身份、21 节点、51 条命令的顺序、
输出 word 数、45/6 次独立完成计数、卷积特征请求数、张量读数、总周期与时间范围。
实际 head 文件的 SHA256 与报告一致；65,536 个有效输出码与冻结参考逐点一致。
验收状态保存在 `data/lung_continuous_20261015/board_report_verification.json`：
`verified_returned_real_board_continuous_output`。
这是用户回传证据核验，不是远程设备实时认证，也不是全病例医学精度证明。
新增报告/输出验收与原尺寸恢复的 10 项测试通过，完整 **229 项**工程回归通过，
日志另存 `data/lung_continuous_20261015/board_acceptance_tests.log`，不覆盖旧 219 项准备证明。

`python/lung_board_preview.py` 直接读取上述实际硬件编码，在 CPU 按原顺序进行反量化、sigmoid、
原尺寸 bilinear 插值（`align_corners=False`）和 0.5 阈值，不重新执行模型。
恢复的 512×512 掩膜与此前冻结定点网络同切片原尺寸结果完全一致。
固定验证切片 Dice **0.970638**，仅代表这一张切片，不是整病例或独立测试平均值。
原图、人工掩膜、实际 FPGA 掩膜与误差图在
`data/lung_continuous_20261015/preview/comparison.png`；指标及来源记录在同目录 `metrics.json`。
人体包络输出约束也在 CPU 单独检查，该切片没有发生改变；单张切片不运行三维连通域清理。

### 实测耗时分解

以实际 HWH/设计的 **100 MHz** 换算报告硬件 busy 周期。两引擎串行，不重叠累加：

| 非重叠项 | 耗时 |
| --- | ---: |
| PL 命令活动周期，含 DDR/cache/背压等待 | 1,814.548 ms |
| 驱动调用包围区间扣除 PL 活动时间 | 1,146.141 ms |
| 21 节点输出参考比较回调 | 610.926 ms |
| 其余未单独计时的主机工作 | 2,775.728 ms |
| 整张切片诊断流程 | 6,347.342 ms |

PL 活动时间中卷积 **1,197.808 ms**，张量操作 **616.740 ms**。
驱动包围区间包含同步、MMIO/轮询等；其余工作包含权重打包/加载、配置、分配、哨兵检查、
打印等，当前报告不能准确把每一项分开，不能把 2,775.728 ms 全说成权重加载。
参考比较只占总时间约 9.6%，不能声称去掉比较即可实时。
即便忽略所有主机开销，当前串行硬件这一帧的活动周期也约 1.815 s；要达到毫秒级还需改变数据通路。
后续先测量权重准备/加载、同步与调度的细分耗时，再优化主机开销及硬件的逐字/逐通道处理和 DDR 访问。
这些数字不是持续吞吐率、纯 MAC 时间或 FPGA 对电脑 CPU/GPU 的公平加速比。
