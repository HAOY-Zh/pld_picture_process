# 卷积响应直通与池化行缓存优化

2026-10-17。独立实验目录：`data/lung_flow_20261017`。

本轮基于已实板验收的 2026-10-16 缓存命中直返版本，保持冻结 epoch12 模型、
INT8/UINT8、Q31/RNE、输入预处理、21 节点与 51 条命令及原 CPU 基线。
旧版 RTL、bit/HWH、部署包及原始验收记录不覆盖。

## 改动

1. 新增 `fpga/lung_feature_window_flow.v`：在 RESPONSE 状态可直接交付返回数据，
   接收端背压时锁存同一数据和错误标记至 OUTPUT。保留单在途读请求、
   padding 和 HWC8 窗口顺序、地址边界、first/last、坐标及错误传播。
   此前命中通常经过 ISSUE → RESPONSE → OUTPUT；现在就绪时省去独立 OUTPUT 周期。
2. `python/lung_flow_prepare.py` 在独立副本生成池化缓存优化：池化访问原输入 A 时，
   复用现有 A/B 两份 16×64bit 缓存保存上下行。命中时检查两份缓存，避免短行共享
   同一缓存线时重复填充；缺失时按上下行选择替换位置。源地址范围、分支量化参数和
   concat 行为仍由实际输入分支决定，与缓存位置选择分开。

没有增加 MAC、降低输入分辨率、改变数值精度或重训。
没有通过取消末端 B 响应、越界保护、时钟核对或在途缓冲保留来缩短计时。

## 验证与证据

准备脚本先核对已验收 IP 依赖、旧三方包及 1,182 项医学/模型输入哈希，
将实际综合的 RTL 复制为 `baseline_rtl/`，优化副本置于 `rtl/`。
卷积与张量的原 gold、输入和模型参考不重新生成。

实际 XSim 对新旧两套 RTL 使用相同的总线延迟/背压规则：

- 147 帧卷积、内存写回及 AXI-Lite 控制检查，包含精确整数输出、尾通道、padding、canary、错误及延迟最终 B。
- 512 轴、极限地址及独立坐标 oracle；缓存冲突、填充失败及命中返回背压。
- 新增响应直通定向对照：每版 192 个精确输出、背压、响应错误、重复命令与周期收益。
- 35 帧张量、379,310 个精确输出字、26 个错误条件；物理 AR/读拍数逐帧对照独立地址模型。
- 额外命令中途读错误与 B 错误，检查已提交计数、在途排空及健康重试。

完整 Python 回归及实际 RTL、路由结果以实验目录中的原始日志和 `verification.json` 为准。
RTL 周期改善不能直接作为整网实板延迟或端到端 FPS。

本轮已完成的本地验证：完整 **341 项 Python 测试通过**；新旧各 **10 组实际 XSim**
及额外中途读/B 错误注入全部通过。相同冻结输入、gold、BFM 下：

| 验证范围 | 原版周期 | 优化周期 | 减少 |
| --- | ---: | ---: | ---: |
| 147 帧缓存卷积链路 | 1,384,965 | 1,081,516 | 21.91% |
| 147 帧 AXI-Lite 配置至写回 | 1,358,445 | 1,053,649 | 22.44% |
| 8 个池化用例 | 99,837 | 41,432 | 58.50% |
| 35 个张量用例合计 | 7,786,702 | 7,727,737 | 0.76% |

8 个池化用例的逻辑读取均为 3,652 个字；真实总线读突发从 1,401 降为 339，
读拍从 22,151 降为 5,256，逐帧与独立地址模型一致。35 个张量用例中大尺寸拼接
占比很高，而本轮只改池化缓存，所以合计收益远小于池化局部收益。
非池化用例的小幅周期波动来自 BFM 基于全局周期施加背压，不能解读为拼接加速。

实际 Vivado 2022.2 整机 route 已通过 100 MHz 时序：setup WNS **+0.165 ns**，
hold WHS **+0.018 ns**。整机 **19,193 LUT、20,349 FF、112 DSP、31.5 BRAM**；
相对保留版为 LUT +130、FF -11，DSP/BRAM 不变。最终 bit 与硬件包身份见实验目录证明。
DRC 为 0 Error / 0 Critical Warning、356 条已记录 Warning，类别及数量与旧版相同。
no_clock 与 unconstrained_internal_endpoints 均为 0；没有放宽原时序约束。

本地门禁、bit 生成及最终部署包核验已完成，证明为
`data/lung_flow_20261017/verification.json`，状态 `built_routed_flow_not_board_tested`。
新包：`data/lung_flow_20261017/lung_flow`。

- bit SHA256：`20e22607b5f58a12b63ac011c5c814f996b052823a0b7f7c2bd4ba340c14bd07`。
- HWH SHA256：`402f0473d8c4ad117aaf20a5faaa73600c068bb8c70521c53d156b5bb5927368`。

新实板输出与计时仍待返回，旧版 686.368 ms 不覆盖。

## 复现入口

以下命令只用于新的独立输出目录；不要覆盖已经完成的实验或证明。

```powershell
.\.venv-lung\Scripts\python.exe python/lung_flow_prepare.py --root data/lung_flow_NEW
.\.venv-lung\Scripts\python.exe python/lung_flow_prepare.py --root data/lung_flow_NEW --simulate baseline
.\.venv-lung\Scripts\python.exe python/lung_flow_prepare.py --root data/lung_flow_NEW --simulate flow
.\.venv-lung\Scripts\python.exe python/lung_tensor_burst_faults.py --root data/lung_flow_NEW
.\.venv-lung\Scripts\python.exe python/lung_flow_deploy.py --root data/lung_flow_NEW --rtl-only
```

构建沿用 100 MHz、GP0/HP0、低有效复位及已验证的 AUX_RESET 常1连接。
`build_flow.tcl` 只从本实验的 RTL 快照打包 IP；实际 route 的 setup/hold 和 DRC 通过后才写 bit。
不修改原 XDC，不放宽数据通路时序例外。

## 新包实板测量

只有本地完整门禁通过后，`lung_flow_deploy.py` 才生成 `lung_flow/` 和最终证明。
包复用原始 FP32、精确 INT8 和 FPGA 三方入口，并绑定新 bit/HWH 哈希。
本轮不会自动连接板端或替换正在运行的在线服务。

板端需先正常结束原有 PL 任务；AXI 状态不确定时保留现场，不重载或释放缓冲。
没有其他硬件所有者时，电脑上传新目录：

```powershell
Set-Location D:\pld_picture_process\data\lung_flow_20261017
& "C:\Program Files\Git\usr\bin\scp.exe" -r .\lung_flow xilinx@192.168.2.99:/home/xilinx/ct_test/
```

板端在专用新内核中运行：

```python
%cd /home/xilinx/ct_test/lung_flow
%run lung_threeway_benchmark.py --runs 20 --warmup 3 --report /home/xilinx/ct_test/lung_flow_result.json
```

应完成 21 节点及每次 head 核验，并输出三方时间、逐命令周期、时钟、真实 head 和交叉哈希。
取回九个 `lung_flow_result*` 原始文件：

```powershell
Set-Location D:\pld_picture_process\data\lung_flow_20261017
& "C:\Program Files\Git\usr\bin\scp.exe" "xilinx@192.168.2.99:/home/xilinx/ct_test/lung_flow_result*" .
```

电脑回到项目根目录，首次运行独立验收：

```powershell
Set-Location D:\pld_picture_process
.\.venv-lung\Scripts\python.exe python/verify_lung_threeway_board.py --root data/lung_flow_20261017 --package data/lung_flow_20261017/lung_flow --stem lung_flow_result
```

实板结果返回之前，新版本始终标为 `board_tested=false`，原 686.368 ms 继续作为旧版实测，
不能将仿真收益相乘后宣称新整网加速比。新包不自动替换在线服务固定的保留版身份。

仅用于工程教学与算法演示，不作为临床诊断依据。

## 用户终端反馈：新版本三方通过，原始文件待取回

用户提供了 3 次预热、20 次正式测量的终端输出，保存为实验目录的
`board_terminal_20261017.txt`，复算与附件哈希见 `board_terminal_feedback.json`。
这属于终端反馈，尚未取得九份原始结果和实际 head，不能替代独立实板验收。
构建证明中 `board_tested=false` 是当时状态，保留不追写。

终端显示 bit SHA256 与新包一致，CPU 650 MHz 不变，加载后将 FCLK0 从 62.5 校正到 100 MHz，
每次推理前后时钟检查通过；21 个实际节点和全部20次最终输出均 SAME。

| 实现 | 本次均值 ms | 中位数 ms（由三位小数样本复算） |
| --- | ---: | ---: |
| 双核 ARM 原始 FP32 | 1547.211 | 1527.502 |
| 双核 ARM 精确 INT8 | 908.312 | 902.680 |
| 新 flow FPGA INT8 | 569.419 | 568.783 |

本次同模型 CPU INT8 / FPGA INT8 为 **1.595 倍**；原始 FP32 / FPGA INT8 为 **2.717 倍**。
FPGA 20次范围 565.013–575.663 ms；引擎活动时间约485.594 ms，其余约83.825 ms。
相对旧已验收 FPGA 686.368 ms，本轮终端均值减少约116.949 ms，即 **17.04%**，旧/新约1.205倍。
新旧为不同轮次运行，不把不同轮次的CPU波动算成硬件改进；本次CPU基线使用本次同板测量。

当前只有引擎总时间，卷积/池化/拼接的实际收益分配需取回逐命令JSON后复算。
这仍是同一张验证切片重复测速，不是全病例精度或网页/三维端到端FPS。
下一步按上文SCP取回原始文件并运行独立验收器；不需要重新构建或重跑已有成功测量。

## 原始文件回传后的独立验收（2026-10-17）

九份原始文件已取回。未经修改的 `verify_lung_threeway_board.py` 验收通过，结果见
`data/lung_flow_20261017/lung_flow_result_threeway_verification.json` 及
`lung_flow_result_verification.json`。本节更新前述“待取回”状态；原构建证明保持当时内容。

- 实际 FPGA head 的 65,536 个 UINT8 码与冻结参考逐点一致；21 节点、51 命令和时钟/包身份核验通过。
- ARM FP32 / ARM INT8 / FPGA INT8 均值分别为 1547.211 / 908.312 / 569.419 ms。
- 实际卷积引擎 403.612 ms，张量引擎 81.983 ms；其中池化合计 12.142 ms、三次拼接合计 69.840 ms。
- 当前已验收硬件性能基线改为 flow 的 569.419 ms。仍是同一切片重复测速，不能作为病例吞吐或整套网页/三维系统 FPS。
- 后续精确查表和结构化剪枝实验见 [本轮优化记录](lung_acceleration_20261017.md)。在线服务仍使用其原先绑定的包，未自动替换。
