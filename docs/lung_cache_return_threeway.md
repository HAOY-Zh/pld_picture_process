# 缓存命中直返与同板 FP32／INT8／FPGA 三方对比

2026-10-16。本轮保留冻结 epoch12 模型、训练／验证／测试划分和全部旧实板结果。
没有剪枝、蒸馏、重训、重新校准或对测试集选参。

## 已验收基线

窗口递推版返回的原始六文件已通过 `verify_lung_prepared_board.py` 独立验收。
同一验证病例 S2-006 第72层、256×256、batch1，20次均值：
双核 ARM INT8 895.768 ms，FPGA INT8 786.033 ms，1.140倍。
实际 65,536 个最终输出码与整数参考完全一致。活动时间卷积587.617ms、张量113.539ms。
该测试不是全病例吞吐或临床精度验证。

## 本轮 FPGA 优化

`fpga/lung_feature_cache_return.v` 保留原三路组相联缓存、同步 BRAM、替换与填充策略。
命中后在 CHECK 状态即可完成响应握手，省去旧的额外 RETURN 周期。
如接收端背压，则将同一数据／错误标记锁存到 RETURN，保证 valid、data、error 稳定。
miss／失败填充不建立有效行、边界裁剪、4KiB、单在途 AXI 读主机均不变。
没有组合 RAM、容量增加、数值近似或放宽 XDC 的数据路径例外。

八项实际 XSim 检查覆盖原147帧、68,849个精确输出、312,018次逻辑请求、
2,612个物理读突发、41,606个读拍、错误处理、容量冲突、512轴和独立16次命中直返／背压／复位测试。
相同 BFM 累计周期：cached链路1,686,337→1,384,965；寄存器到写回1,660,753→1,358,445。
这是仿真证据，不能当作整网实板提速。

首次 Vivado 启动参数顺序及之后的 BD 检查顺序均在综合前失败，日志保留在第一实验目录。
修正为 `validate_bd_design` 完成参数传播后再检查复位；有效构建目录为
`data/lung_cache_return_20261016_v2`，不改动旧窗口／突发版工程。
复位仍为低有效，未使用 AUX_RESET 接常1，GP0配置／HP0访存、共享100MHz时钟与地址范围不变。
构建 Tcl 在 route 完成后检查 setup／hold／DRC，通过后才生成 bit。

## 原始 FP32 CPU 基线

`pynq/lung_fp32_native.cpp`、`lung_fp32_kernel.h`、`lung_fp32.py`：

- 使用原始 FP32 checkpoint，不是 INT8 权重反量化；eval BN 折叠为浮点卷积。
- 完整21节点，FP32乘加、池化和上采样拼接，输入／输出CHW、内部HWC。
- C++ `-O3`、ARM Cortex-A9 显式 NEON FP32 内核、OpenMP双线程，无Torch／ONNX下载。
- 权重预排布及所有中间张量一次准备，计时不含编译／加载／初始化；无逐像素堆分配。
- 不启用 fast-math，四路输出共享输入块。是自编优化基线，不宣称是最佳成熟ARM推理库。
- 本机单／双线程与原始Torch21节点检查通过，当前实际切片head最大绝对误差约3.43e-5。
  固定容差 atol0.002、rtol0.0002，板端若超限则停止，不自动放宽。
- 相同规范化CT输入在INT8量化前导出，并已由Torch确认量化后与FPGA原输入逐码一致。
- FP32返回浮点logits，INT8返回整数head编码，不要求这两个不同数值模型逐位一致。

本机数字验证与ARM汇编检查不代表板端运行成功。板端程序仍会逐层对照、每次检查输出，
保存真实浮点head，后续独立验收。之前两份FP32导出是开发中间版本，最终使用v3目录。

## 三方统一演示与计时

`lung_threeway_benchmark.py` 先测双核 FP32，再通过未改动的绑定入口测双核INT8和FPGA INT8。
每组预热3次、测20次，保留全部样本和均值／中位数／范围，不挑选最好一次。
只加载一次确切bit，CPU保持650MHz、PL校准并逐帧验证100MHz。
所有原有AXI超时、在途缓冲保留、末端哨兵和21节点非计时诊断不变。
若遇不确定AXI完成，不能因新测速功能自动重载bit／释放CMA／清空模块缓存。

分别展示：CPU FP32／CPU INT8（量化收益）、CPU INT8／FPGA INT8（同模型硬件收益）、
CPU FP32／FPGA INT8（完整方案收益）。比值小于1也照实报告。
计时边界均为内存中的网络输入到网络输出；Sigmoid、原尺寸恢复、人体约束、三维清理和界面不计入。
这里没有系统端到端FPS、全病例吞吐或量化精度的新结论。

## 复现与独立验收

本轮源码与证明入口：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_cache_return_deploy.py --rtl-only
.\.venv-lung\Scripts\python.exe python/lung_cache_return_deploy.py
.\.venv-lung\Scripts\python.exe python/lung_threeway_package.py
```

上述打包入口拒绝覆盖已存在的证明／部署目录；已完成后不要重复执行。
原始结果取回后首次使用：

```powershell
.\.venv-lung\Scripts\python.exe python/verify_lung_threeway_board.py
```

它检查FP32源与真实logits、所有计时样本／统计／比值、文件交叉哈希，
并调用原封不动的INT8独立验收器复核实际head与每条命令。
FP32与INT8网格掩膜差异不是原尺寸Dice，不能混称精度结果。

## 最终构建与交付

新目录第二次八组实际XSim及完整290项回归通过，无跳过。
路由后setup WNS +0.204ns、hold WHS +0.022ns；无时钟／未约束内部端点均为0。
整机19,063 LUT（35.83%）、20,360 FF（19.14%）、112 DSP（50.91%）、31.5 BRAM（22.50%）。
最终DRC无Error／Critical Warning；356条Warning类型与旧版相同，主要为DSP流水建议和一条无可布线负载提示。
实际门禁通过后bit生成成功，硬件包、导入RTL逐文件差异和1,182个医学源保护均通过独立审计。

- bit SHA256：`567ddc0e47e3556e21b248a2d0bab85f26381c8056295f88ddd14335e695c89b`。
- HWH SHA256：`36ce4d28641805e7705a8f3f312a5c0712e3f45a17a605a39c062013a6f58f58`。
- 硬件证明：`data/lung_cache_return_20261016_v2/verification.json`。
- 最终三方包：`data/lung_threeway_20261016/lung_threeway`。
- 打包证明：`data/lung_threeway_20261016/verification.json`。

本次仅改缓存命中返回时序，旧MAC、量化、窗口、写回、张量、CPU INT8和绑定时钟入口保持原文件一致。
原始FP32参考来自最终v3导出，真实浮点模型单／双线程各21节点对照通过，ARM汇编具有NEON `vmla.f32`。
构建时新bit尚未在实板执行，历史构建证明不追写。随后实板结果见下方独立验收记录，
不能把仿真周期收益或路由完成本身称为实板加速。

电脑PowerShell上传（前一次板端运行已正常结束、没有在途PL任务）：

```powershell
Set-Location "D:\pld_picture_process\data\lung_threeway_20261016"
& "C:\Program Files\Git\usr\bin\scp.exe" -r lung_threeway xilinx@192.168.2.99:/home/xilinx/ct_test/
```

板端Jupyter：

```python
%cd /home/xilinx/ct_test/lung_threeway
%run lung_threeway_benchmark.py --runs 20 --warmup 3 --report /home/xilinx/ct_test/lung_threeway_result.json
```

FP32会先做21节点对照，随后原绑定入口完成CPU INT8、新bit身份、100MHz时钟及FPGA21节点检查与测速。
保存九个原始结果文件，包括真实FP32 logits与FPGA整数head、CPU分阶段结果、诊断、时钟及绑定证据。
不能删除或覆盖失败／中间报告；重测请另选report文件名。遇AXI超时不盲目重跑、重载或释放CMA。

取回：

```powershell
Set-Location "D:\pld_picture_process\data\lung_threeway_20261016"
& "C:\Program Files\Git\usr\bin\scp.exe" "xilinx@192.168.2.99:/home/xilinx/ct_test/lung_threeway_result*" .
```

然后执行上面的 `verify_lung_threeway_board.py` 独立验收，三方加速比以真实返回文件为准。

## 实板三方原始文件独立验收

用户返回九个原始结果文件后，`verify_lung_threeway_board.py` 验收通过。
证明为 `data/lung_threeway_20261016/lung_threeway_result_threeway_verification.json`，
状态 `verified_returned_threeway_actual_fp32_and_int8_heads`。
同时由未修改的INT8验收器生成 `lung_threeway_result_verification.json`。

同一S2-006第72层、256×256、batch1，预热3次、每组20次，保留全部样本：

| 实现 | 均值 ms | 中位数 ms | 最小／最大 ms |
| --- | ---: | ---: | ---: |
| 双核ARM原始FP32 | 1604.589 | 1593.432 | 1579.470／1721.777 |
| 双核ARM精确INT8 | 890.518 | 877.507 | 872.187／1015.116 |
| 新缓存直返FPGA INT8 | 686.368 | 685.966 | 680.794／693.725 |

- CPU FP32／CPU INT8：1.80186倍，反映本组软件实现的量化收益。
- CPU INT8／FPGA INT8：1.29744倍，中位数1.27923倍，反映同整数网络的硬件收益。
- CPU FP32／FPGA INT8：2.33780倍，中位数2.32290倍，反映量化＋FPGA的综合收益。
- FPGA相对CPU FP32时延减少57.22%，相对CPU INT8减少22.92%。
- 相对已验收窗口版786.033ms，新FPGA时延减少12.68%，模型／量化／CPU基线未改动。

原始21节点诊断及51条命令顺序、写字数、完成计数、周期求和、实际bit身份、CPU650MHz与PL100MHz均通过。
真实FPGA head的65,536个UINT8码与整数参考完全一致。
真实FP32 head相对原始Torch最大绝对误差3.4332e-5，模型网格logits符号分歧为0；
不要求FP32和INT8两种数值模型互相逐位一致，也不把网格掩膜差异称为原尺寸Dice。

逐命令周期复算：卷积489.445ms、张量113.498ms，活动合计602.942ms，外围约83.425ms。
卷积相对旧窗口版587.617ms减少16.71%；张量部分基本不变，符合仅修改卷积缓存的设计范围。
当前最贵卷积为dec1.0约72.432ms、dec2.0约66.463ms、dec3.0约63.229ms。

仍是同一张验证切片的顺序分阶段重复测速，不是全病例吞吐、完整临床流程或显示FPS；
没有新的全数据集精度结论，自编优化CPU实现也不宣称是最优成熟ARM推理库。

仅用于工程教学与算法演示，不作为临床诊断依据。
