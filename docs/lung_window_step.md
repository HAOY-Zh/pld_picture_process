# 卷积窗口递推寻址优化

## 实板依据

上一阶段突发搬运版的初测及 20 次确认共 12 份回传文件已独立验收，
证明分别为 `data/lung_tensor_burst_20261016_v3/lung_tensor_burst_result_verification.json`
和 `lung_tensor_burst_confirm_result_verification.json`。
实际 head 的 65,536 个编码完全一致，确切 bit/HWH、100 MHz、21 节点和 51 命令均匹配。
确认轮均值：双核 ARM 896.845 ms、PL 859.078 ms，均值加速 1.044 倍；
这是单个验证切片的常驻完整网络测试，不代表全数据集推理、病例后处理或渲染。

独立逐命令计数：卷积 660.871 ms，张量操作 113.559 ms，外围约 84.648 ms。
最慢卷积为 dec1.0 99.513 ms、dec2.0 88.501 ms、dec3.0 82.530 ms。
因此本阶段转向卷积数据供给，而不是继续主要削减张量操作或 Python 开销。

## 改动与不变项

新增 `fpga/lung_feature_window_step.v`，不覆盖旧窗口模块。
仍按输出像素、kernel y/x、输入通道块的原顺序产生 HWC8 窗口，
但用独立的像素原点、kernel 行原点和下一输出行原点寄存器推进地址，
避免每个邻域点重复坐标乘法及 PREP/ROW 等待。
配置阶段仍捕获全部几何参数，使用 34 位地址保存可能为负的填充原点；
只有坐标处于实际输入范围内时才发出请求，绝不读取填充范围。
一次最多一个未完成请求，保留反压期间请求和输出稳定、错误传播与完成握手。

冻结模型、量化和舍入、MAC、缓存替换策略、读写 AXI、输出写回、突发张量引擎、
CPU 基线、主机运行时、共享复位、100 MHz 和原约束不变。
原始 CT、标签、病例划分及旧构建/实板证据不变。

## 实际验证

有效实验目录 `data/lung_window_step_20261016_v2`。
最初测试台使用了 SystemVerilog 保留字，编译失败，旧目录保留且没有部署。
修正测试台后在新目录重新冻结；未修改原金标准。

新旧版本分别通过七组实际 XSim：

- 147 帧缓存卷积：68,849 个结果、371,974 个窗口、312,018 次逻辑取数完全一致。
- 同一缓存策略与阻塞内存模型：2,612 个 AXI 读突发、41,606 个读 beat 不变。
- 147 帧内存写回及寄存器启动：8,866 个写回字、最终 B 等待、边界 canary、共享复位及错误传播通过。
- 原 H512/W512/C96 地址上限、缓存容量与协议测试通过。
- 新增独立坐标 oracle 的 10 个窗口案例通过：负填充原点、最高字地址、
  窄高/宽矮体积、valid crop、部分通道、响应错误、输入配置捕获及背压稳定。

新测试台直接用 `(source_y * width + source_x) * blocks + channel_block` 计算参考地址，
不复制 RTL 的递推算法。配置输入在命令接收后立即改变，用于检查是否确实锁存。

| 同一冻结 147 帧 | 旧周期总和 | 新周期总和 | 旧/新 |
| --- | ---: | ---: | ---: |
| 缓存卷积 | 1,766,502 | 1,686,337 | 1.0475 |
| 寄存器启动至写回 | 1,731,127 | 1,660,753 | 1.0424 |

这是约 4% 的模拟周期小步优化，不是整网实板加速比。
独立缓存/卷积综合：6,603 LUT、8,251 FF、69 DSP、31.5 BRAM，
100 MHz setup WNS +1.273 ns。整机资源与最终 setup/hold 必须另看整机路由报告。

## 构建与验收

`python/lung_window_step_prepare.py` 生成独立 RTL、测试和构建脚本，
在准备阶段核对已验收的上一版实板报告、实际输出及旧来源哈希。
`python/lung_window_step_deploy.py --rtl-only --root data/lung_window_step_20261016_v2`
独立核对实际日志、所有冻结文件及周期/资源。
部署审计还要求整机路由、最终 DRC、完整工程回归、1,182 个医学输入保护和
打包 RTL 逐文件差异检查通过，才产生新部署包；只有窗口文件允许不同。
新 bit 的输出一致性和性能必须上板重测，不承诺额外实板提速。

完整环境的 274 项 Python 工程测试已通过，71.083 秒，无跳过。
记录为实验目录的 `python_tests.log`。

用户随后明确要求不再执行综合、实现或 bit，已停止进行中的整机构建；部分综合已完成，
本轮没有完成最终路由或新 bit 交付。不再自动恢复构建。
现改为[仅源码工程与本地验证交付](lung_window_step_local_vivado.md)，由用户运行后续 Vivado。

## 用户授权恢复自动构建

用户随后于 2026-10-16 明确恢复自动综合、布局布线及 bit。
已重新核验原 RTL/向量/仿真/OOC 哈希；旧工程的两个核及大部分 IP 综合完整保留，
只重启中断的内存互连及后续整机流程。
`fpga/resume_lung_window_step.tcl` 先执行实际 BD 地址/时钟/复位检查，
保存中断日志，再仅重置未完成的综合任务；不会覆盖已验收的旧 bit 或结果。
布局布线先停在 route_design，检查 setup/hold 和 DRC，全部通过后才继续生成 bit。
源码专用入口依然不启动任何硬件运行。
当前新整机资源、最终时序和 bit 仍待本轮恢复流程及独立打包审计确认。
恢复流程的完整工程回归 279 项通过，67.572 秒、无跳过，
保存在实验目录 `python_tests_resume.log`，旧 274 项日志仍保留。

## 恢复构建最终完成

自动恢复已成功完成整机综合、布局布线及按时序门控的 bit 生成。
实际 BD 检查、原依赖/实板基线、导入 RTL 逐文件差异、最终 DRC、279 项回归、
1,182 个医学输入保护及新部署清单全部通过独立审计。
构建证明为 `data/lung_window_step_20261016_v2/verification.json`，
状态 `built_routed_window_step_not_board_tested`；此状态记录构建时尚未实板测试，不追写历史证明。

- 时钟目标 100 MHz，路由 setup WNS +0.469 ns、hold WHS +0.016 ns。
- 整机 19,037 LUT（35.78%）、20,361 FF（19.14%）、112 DSP（50.91%）、31.5 BRAM（22.50%）。
- 最终 DRC 无 Error/Critical Warning；356 条 Warning 主要为 DSP 流水建议及一条无可布线负载提示。
- 时序摘要中 no_clock 与 unconstrained_internal_endpoints 均为 0。
- 原配置 multicycle 范围不变；厂商 CDC 例外有一条 Invalid endpoint，上一已验收突发版也存在，
  本轮未增加通配数据路径例外。
- 新 bit SHA256：`e14dfc6663f4becba8f934bd2f6833440806ea69ec1cff706edc36786b042ead`。
- 新 HWH SHA256：`46028ba190a8a5874e3894379690a56a6e3ee6b1e1ff83b70feecfcf02c4977c`。

旧 859 ms 突发版包与原始结果保留不变。上述 4% 是冻结 RTL 回归的周期收益，
不能代替下列新 bit 整网实际测速，也不能外推为全病例或临床处理吞吐。

电脑 PowerShell 上传新目录：

```powershell
Set-Location "D:\pld_picture_process\data\lung_window_step_20261016_v2"
& "C:\Program Files\Git\usr\bin\scp.exe" -r lung_window_step xilinx@192.168.2.99:/home/xilinx/ct_test/
```

Jupyter（前一次正常结束、没有其他 PL 任务时）：

```python
%cd /home/xilinx/ct_test/lung_window_step
%run lung_selected_benchmark.py --runs 20 --warmup 3 --report /home/xilinx/ct_test/lung_window_step_result.json
```

仍通过绑定入口检查 bit 身份及每帧前后 100 MHz，再运行 21 节点诊断、预热和正式测速。
完成后取回原始文件供独立验收：

```powershell
Set-Location "D:\pld_picture_process\data\lung_window_step_20261016_v2"
& "C:\Program Files\Git\usr\bin\scp.exe" "xilinx@192.168.2.99:/home/xilinx/ct_test/lung_window_step_result*" .
```

## 20 次实板终端反馈

用户随后提供上述确切 bit 的 20 次测速终端：21 节点诊断及全部正式测量输出均 SAME，
绑定入口、100 MHz 每帧前后检查通过。随后原始六文件已取回，
`lung_window_step_result_verification.json` 独立验收通过：实际 65,536 个 head 码与参考完全一致，
21 节点诊断、51 命令的顺序/写入字数/完成计数、计时身份和时钟检查通过。

| 指标 | 双核 ARM | 窗口递推 FPGA |
| --- | ---: | ---: |
| 均值 | 895.768 ms | 786.033 ms |
| 中位数 | 875.558 ms | 785.527 ms |
| 最小 / 最大 | 865.974 / 1016.426 ms | 778.799 / 796.897 ms |

中位数按显示到三位小数的 20 个样本计算，后续以原始 JSON 的完整精度复核。
保留全部样本，均值加速比约 1.140 倍，中位数约 1.115 倍，平均时延相对同次 CPU 减少 12.25%。
相对上一突发版 859.078 ms，本轮 PL 时延再减少 8.50%。
本轮 PL 最大值也低于 CPU 最小值，支持当前切片、当前测试条件下的明显小幅领先，
不能外推为所有病例或包含后处理/界面在内的系统性能。
原始逐命令周期复算：卷积 587.617 ms、张量 113.539 ms、外围约 84.877 ms。
下一轮隔离实验仅改变缓存命中返回时序，不改变原模型、权重、量化、窗口、AXI 主机或张量引擎。
