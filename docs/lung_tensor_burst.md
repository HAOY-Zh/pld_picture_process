# 按实板逐层周期优化张量搬运

## 已验收的输入证据

用户回传的六份 `lung_cache_fast_result*` 已完成文件级验收。
`python/verify_lung_prepared_board.py` 核对主/CPU/诊断/频率/入口报告之间的哈希关系、
确切 bit 与清单、51 命令顺序/写入数/完成计数、100 MHz 和 CPU 650 MHz，
并直接读取实际 FPGA head：65,536 个 UINT8 编码与冻结参考全部一致。
验收文件 `data/lung_cache_fast_20261016/lung_cache_fast_result_verification.json`。
这只是同一验证切片的输出与测速验收，不是全测试集准确率或临床实时性能证明。

同次双核 ARM 888.878 ms、PL 1147.477 ms；PL 活动引擎 1060.827 ms，外围约 86.650 ms。
逐层计数显示：

| 范围 | 活动时间均值 |
| --- | ---: |
| 全部卷积 | 660.884 ms |
| 全部张量操作 | 399.943 ms |
| 最大拼接 cats.2 | 190.242 ms |
| 中间拼接 cats.1 | 95.121 ms |
| 最小拼接 cats.0 | 47.561 ms |

三个拼接已占约 333 ms，且每个字此前都要等待一次独立 DDR 读/写响应。
本轮优先减少这些短事务，不增加 MAC 数量，不更改分割模型或降低精度。

## 实现与安全边界

- 从冻结 `lung_tensor_word_engine.v` 生成独立 `lung_tensor_burst_engine.v`，保持地址映射、
  pooling、nearest、aligned/unaligned concat、八路 Q31/RNE 和尾通道 mask 原代码。
- 为 A/B 各增加一个 16 字小行缓存。突发最多 16 个 64-bit 字，裁剪到逻辑输入跨度；
  不读取已分配源帧之外。行对齐与裁剪保证不跨越 4 KiB。
- 连续输出先收集最多 16 字，再交给新 `lung_axi_burst_write64.v`；4 KiB 边界或最终尾块提前截断。
  AW/W 可独立阻塞或先后握手，数据/地址在阻塞时稳定。只在物理 B 返回成功后增加 WRITTEN；
  最后一次 B 以及读主机排空前，不报告 DONE。
- 中途读错时丢弃尚未提交的输出块；B 错误使整个命令无效，保留此前成功批次的计数。
  驱动错误/超时处理不变：不释放不确定在途 CMA，不自动重载、复位或改频。
- READS 仍表示旧引擎的逻辑输入字交付数，不是预取后的物理 DDR beat 数。
  物理读/写突发由独立仿真模型逐帧核对，不用逻辑计数冒充带宽节省。

本轮 35 帧合计逻辑读取 382,152 字；预取物理读取 496,404 字，分为 31,115 次突发，
输出 379,310 字合并为 23,734 次写突发。预取增加了部分有效跨度内流量，
目标是减少短事务等待，不声称减少 DDR 字节数或增加并发 outstanding。

## 验证

实际 XSim 对拍沿用 35 帧冻结金标准，覆盖三个完整尺寸真实网络拼接层、全编码/舍入、
非对齐通道、尾填充、512 边界和不同 4 KiB 起始偏移，全部 379,310 输出字一致。
另通过 26 项配置/读写错误、13 项独立突发写协议案例；补测已提交两个批次后读错、
第二批写 B 错误、DONE 后没有晚到写入、错误后的健康重试。
旧引擎另在同一个突发兼容内存 BFM 下运行，避免混用仿真环境。

初版原型的声明顺序及缓存行裁剪组合路径存在问题，未部署。
有效实现为 `data/lung_tensor_burst_20261016_v3`，对裁剪/长度检查加入了真实流水寄存器，
未降低时钟、改舍入或放宽时序约束。
独立张量顶层综合为 5,835 LUT、4,023 FF、40 DSP、0 BRAM，100 MHz setup WNS +1.273 ns。
整机资源/时序必须以最终路由报告为准，不能直接引用独立综合作为整机占用。
完整工程回归 270 项通过（69.393 秒）；原 CT、标注、训练/测试划分、冻结参数不变。

## 构建与交付

准备/仿真：`python/lung_tensor_burst_prepare.py`；
公平旧引擎对照：`python/lung_tensor_burst_baseline.py`；
中途故障测试：`python/lung_tensor_burst_faults.py`；
最终审计与打包：`python/lung_tensor_burst_deploy.py`。
不覆盖旧 `lung_cache_fast` 包及任何回传结果。

只有整机路由、DRC、100 MHz setup/hold、依赖来源和医学输入保护检查全部通过，
才生成 `data/lung_tensor_burst_20261016_v3/lung_tensor_burst`。
包中模型、输入/参考、CPU 基线、卷积核心、驱动及运行时不变，内置确切路径/哈希校验入口。
本地验证不能替代新 bit 的板端正确性与性能测试，尚不承诺快过 ARM。

最终整机已通过审计：100 MHz、WNS +0.395 ns、WHS +0.007 ns，
DRC 无 Error/Critical Warning（有既有 DSP/互连等 Warning，并非零警告）。
整机 18,966 LUT（35.65%）、20,257 FF（19.04%）、113 DSP（51.36%）、31.5 BRAM Tile（22.50%）。
1,182 个保护医学源文件未变，已逐项核对打包依赖，卷积 RTL 与上一版完全一致。
证明 `data/lung_tensor_burst_20261016_v3/verification.json`，
状态 `built_routed_tensor_burst_not_board_tested`，不是板端性能证明。
新 bit SHA256：`804d507451f85cc9f6421a2753496dd8e57add2d6db8cda453dca83bf366465a`。

同一突发兼容 BFM 下，三个完整尺寸拼接层周期：

| 层 | 旧周期 | 新周期 | 旧/新 |
| --- | ---: | ---: | ---: |
| cats.0 | 1,576,979 | 915,115 | 1.723 |
| cats.1 | 3,075,096 | 2,050,305 | 1.500 |
| cats.2 | 6,348,522 | 4,100,075 | 1.548 |

这些是实际 RTL 周期，而非整网实板加速。下一步以新 bit 的真实整网输出与墙钟时间为准。

审计通过、目录实际存在后，电脑上传：

```powershell
Set-Location "D:\pld_picture_process\data\lung_tensor_burst_20261016_v3"
& "C:\Program Files\Git\usr\bin\scp.exe" -r lung_tensor_burst xilinx@192.168.2.99:/home/xilinx/ct_test/
```

Jupyter（无其他 PL 任务、上次健康结束时）：

```python
%cd /home/xilinx/ct_test/lung_tensor_burst
%run lung_selected_benchmark.py --runs 3 --warmup 1 --report /home/xilinx/ct_test/lung_tensor_burst_result.json
```

取回本轮原始六份结果：

```powershell
Set-Location "D:\pld_picture_process\data\lung_tensor_burst_20261016_v3"
& "C:\Program Files\Git\usr\bin\scp.exe" "xilinx@192.168.2.99:/home/xilinx/ct_test/lung_tensor_burst_result*" .
```

## 首轮实板反馈与稳定性复测（2026-10-16）

用户终端反馈加载了上述 SHA256 的新 bit，100 MHz 时钟与绑定入口检查通过，
21 个实际节点输出全部 SAME。PL 三次为 854.270 / 855.867 / 860.131 ms，
均值 856.756 ms，活动引擎约 774.352 ms；相对上一缓存版均值 1147.477 ms，耗时下降 25.34%。
同次双核 ARM 为 1016.670 / 885.330 / 880.711 ms，均值 927.570 ms，
均值 CPU/PL 为 1.083 倍，但双方中位数比仅 885.330 / 855.867 = 1.034 倍。
保留全部样本，不事后删除较慢首帧；当前只说明出现小幅领先，不能据此认定稳定加速。
本节依据用户终端文本，原始六文件与实际 head 尚待取回并由独立验收器核验，
不修改先前构建证明的历史状态，不外推为全测试集精度或完整病例处理性能。

使用已上传的同一入口、同一 bit 和模型增加预热/样本，不需要重新上传或构建。
上次正常结束、没有其他 PL 任务时，在 Jupyter 执行：

```python
%cd /home/xilinx/ct_test/lung_tensor_burst
%run lung_selected_benchmark.py --runs 20 --warmup 3 --report /home/xilinx/ct_test/lung_tensor_burst_confirm_result.json
```

CPU 与 PL 仍为先后两个阶段，并非交错配对；需同时查看全部样本、均值、中位数与波动，
尤其留意系统负载及阶段顺序对这种小幅差距的影响。该计时是常驻整网推理，
不含整病例读取、原尺寸后处理、三维清理或界面渲染。
用户随后提供了 3 次预热、20 次正式测量的确认终端输出（原始文件仍待回传）。
所有节点及 20 次最终输出均 SAME，bit 身份与每次推理前后 100 MHz 检查通过：

| 指标 | 双核 ARM | FPGA 常驻整网 |
| --- | ---: | ---: |
| 均值 | 896.845 ms | 859.078 ms |
| 中位数 | 882.278 ms | 857.905 ms |
| 最小 / 最大 | 876.187 / 1061.815 ms | 853.002 / 873.271 ms |

以上按终端显示的三位小数样本计算，与主报告均值一致到显示精度。
保留全部 20 个样本，均值加速比 1.044 倍（时延减少 4.21%），
中位数加速比 1.028 倍；本轮 FPGA 最大值也低于 CPU 最小值。
这支持当前切片、当前先后分阶段测试下的小幅领先，不代表不同切片或系统负载下均领先。
相比首轮 1.083 倍，确认结果更接近真实的小幅优势，不能宣称大幅加速。
当前 FPGA 活动引擎约 774 ms、其余约 85 ms；下一步取回逐命令计数，
核对卷积占比是否仍接近前版 661 ms，再选择卷积数据供给或输出写回优化，
不直接假定增加乘法器就能提速，也不更改冻结模型。

复测后取回初测与确认两组文件：

```powershell
Set-Location "D:\pld_picture_process\data\lung_tensor_burst_20261016_v3"
& "C:\Program Files\Git\usr\bin\scp.exe" "xilinx@192.168.2.99:/home/xilinx/ct_test/lung_tensor_burst*result*" .
```

初测和 20 次确认的 12 份原始文件随后均已取回，两组分别通过
`python/verify_lung_prepared_board.py` 独立验收，新增同名 `_verification.json`，
没有修改构建时的历史证明。实际 head 的 65,536 个编码全部一致，
逐命令计数表明确认轮卷积 660.871 ms、张量 113.559 ms。
下一阶段据此实施[卷积窗口递推寻址](lung_window_step.md)，保持其他核和冻结模型不变。
