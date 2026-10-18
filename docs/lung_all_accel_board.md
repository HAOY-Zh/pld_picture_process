# 全部已有分割加速组合上板（2026-10-17）

本轮已完成并通过独立回传验收：原模型 PL 552.647 ms，新剪枝 PL 492.401 ms，时延减少10.90%；六组完整结果及精度/速度判断见 [实板验收报告](lung_all_accel_results_20261017.md)。以下上传和运行步骤保留用于复现；旧结果与构建证明不覆盖。

本轮使用通过 27 例逐病例 Dice 门槛的 `fisher_moderate`，与原模型使用同一张 CT、同一个 exact-LUT bit，在同一开发板上测量六组常驻整网时延：

| 模型 | ARM FP32，2 线程 | ARM INT8，2 线程 | FPGA INT8 |
| --- | --- | --- | --- |
| 原模型 | 测量 | 测量 | 全部既有硬件/调度优化 |
| fisher_moderate | 测量 | 测量 | 全部既有硬件/调度优化 + 剪枝 |

FP32 来自各自真实浮点检查点，BN 融合后导出，不从 INT8 权重反量化。剪枝 FP32 已在电脑上完成单/双线程各 21 个节点的原生 C++/Torch 一致性检查。INT8 两模型沿用各自已冻结的校准与参数，并保留已通过的逐病例精度确认。

## 启用的优化

- INT8 静态量化、8 通道并行卷积及 HWC8 排布。
- 权重预打包、常驻缓冲区、原生 C++ 批量 MMIO 写入及主机调度优化。
- 卷积缓存命中返回及响应直通。
- 池化双行缓存。
- 拼接重定量化的精确查表。
- 实际读回校验的 PL 100 MHz、ARM 650 MHz；未提高 CPU 频率。
- 剪枝模型另启用结构化通道剪枝和恢复训练：82465 参数、44 块命令。

共同 bit 的 SHA256 为 `54f54656e405eb4cee0ce073d476665110ee49ef6f200ccdaf6edc814a85cafc`。无需重新运行 Vivado，也不需要先手动烧录；测速入口会校验并加载对应 bit。

基础处理的共享 bit、切片缓存与二进制网络传输属于网页链路，步骤见 [基础处理往返加速](ct_platform_fast.md)。本次六组计时从常驻网络输入到输出，不含网络往返、预后处理、三维重构和显示。

## 1. 电脑上传

当前使用 **v2**：修复 Windows 清单中的反斜杠路径在 Linux 板端无法解析的问题，保留全部文件哈希检查；模型、权重、bit 和子目录逐字节不变。6 项路径/交接回归测试通过，71 个清单项通过 POSIX 路径检查。

如果已经上传 v1，并在加载前出现 `Changed suite file: original\clock_manifest.json`，只补传两个文件：

```powershell
Set-Location D:\pld_picture_process\data\lung_all_accel_20261017_v2
& "C:\Program Files\Git\usr\bin\scp.exe" .\portable_fix\lung_accel_suite.py .\portable_fix\suite_manifest.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_all_accel/
```

该错误发生在获取硬件锁、加载 bit 和分配设备缓冲区之前。如果之前已正常释放网页服务，可直接重新运行第 3 步，无需再次释放服务或重启内核；其他错误或已进入硬件推理的失败仍按原交接要求处理。v1 包和出错前源码快照保留，未覆盖历史证明。

在 PowerShell 执行：

```powershell
Set-Location D:\pld_picture_process
& "C:\Program Files\Git\usr\bin\scp.exe" -r .\data\lung_all_accel_20261017_v2\lung_all_accel xilinx@192.168.2.99:/home/xilinx/ct_test/
```

这个目录使用新名称，不覆盖原网页服务和旧实验包。也可把同目录下的 `lung_all_accel_upload.zip` 解压到板端 `/home/xilinx/ct_test`。

## 2. 交接当前板端服务

如果网页板端服务正在运行，先停止网页提交任务，在**原来启动 `platform_service` 的 Notebook** 执行：

```python
%cd /home/xilinx/ct_test/lung_all_accel
from lung_accel_suite import validate, release_web
validate("/home/xilinx/ct_test/lung_all_accel")
release_web(platform_service)
```

只在打印 `Healthy web service released PL` 后，重启**这个 Notebook 的内核**，清除旧包导入。函数会检查服务健康、请求空闲及硬件释放；如果报忙，等待当前任务完成。若报故障，保留当前内核与错误记录，不执行后续加载。

如果当前没有任何板端服务或其他 Notebook 占用 PL，可直接用干净内核进入下一步。测速入口也会获取与网页相同的硬件所有者锁。

## 3. 一次运行全部六组

```python
%run /home/xilinx/ct_test/lung_all_accel/lung_accel_suite.py --runs 20 --warmup 3
```

无需另外导入模型或手动加载 bit。默认每条路径 3 次预热、20 次正式运行；首次本地编译、模型准备、bit 加载单独记录，不混入每帧时延。

结束时会打印 `ALL SIX MEASUREMENTS COMPLETE`、两行时延表、剪枝的 PL 时延减少比例，以及 `RETURN THIS RESULTS FOLDER`。输出目录使用 UTC 时间戳自动生成，不覆盖旧结果。对每个模型核验 FP32 全部 21 个节点、INT8 全部节点、每帧实际输出、实际时钟及命令计数。

若中途报错，保存完整错误和 `results` 目录，不通过重载 bit、关闭内核或重跑脚本处理不确定的硬件状态。入口会保留本轮所有者与已有驱动保存的硬件资源。

## 4. 回传

电脑 PowerShell：

```powershell
Set-Location D:\pld_picture_process
& "C:\Program Files\Git\usr\bin\scp.exe" -r xilinx@192.168.2.99:/home/xilinx/ct_test/lung_all_accel/results .\data\lung_all_accel_20261017_v2\returned
```

回传整个目录，包含两个模型的实际 FP32/INT8 输出、计时、时钟、节点诊断及组合结果。之后对实际时间戳目录运行独立验收：

```powershell
.\.venv-lung\Scripts\python.exe -X utf8 python\verify_lung_accel_suite_board.py --root data\lung_all_accel_20261017_v2\returned\时间戳目录 --package data\lung_all_accel_20261017_v2\lung_all_accel
```

验收分别汇报同模型 CPU INT8/PL INT8、各模型 FP32/INT8、同 bit 上的剪枝收益及最终综合收益。既往缓存/flow/LUT 的独立实现收益继续使用各自已验收记录，不能把六组组合测速解读为各项 RTL 优化独立消融。

剪枝候选仍有空肺误报增加和新增 1 张小肺切片漏分，见 [精度完整记录](lung_pruning_refinement.md)。本次是独立测速，不自动替换网页模型；板端结果回传后再判断实际收益。
