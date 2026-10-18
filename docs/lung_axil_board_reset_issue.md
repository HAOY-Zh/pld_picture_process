# 首次上板挂起与辅助复位修正（2026-10-15）

## 现象与确认范围

首次运行 `lung_axil_smoke.py` 后 Notebook 长时间忙碌，中断无反应。
分步检查中 `Overlay(...)` 返回，打印 bit 加载完成；后续单元格未见 ID 结果。
这些现象本身不能精确证明挂起发生在哪一条 Python 指令。

本地核对实际 HWH、BD 生成网表及厂商复位实现，发现独立、确定的板级连接错误：
`proc_sys_reset_0.C_AUX_RESET_HIGH=0`（低有效），`aux_reset_in` 却接 `const_zero`。
这持续请求辅助复位，`peripheral_aresetn` 连接的加速器与两个 SmartConnect 无法正常工作。
bit 下载成功、无 DRC Error、静态时序满足，以及直接驱动复位的计算核仿真，
都不能替代对真实板级复位释放的检查。

厂商依据：[AMD PG164 复位模块说明](https://docs.amd.com/api/khub/documents/5VcEWg~hYG0gT~7WNDt64g/content)。
本机 Vivado 2022.2 官方 HDL 的 ACTIVE_LOW_AUX 分支也明确反相辅助输入。

## 文件与处理

- **禁止继续加载** `data/lung_axil_board_20261015/deploy/` 内的原始 bit。
  已增加 `DO_NOT_USE.txt`；原始 bit/HWH/核验记录保持不变，保留故障证据。
- 原始构建脚本快照：`data/lung_axil_board_20261015/build_lung_axil_board_original.tcl`，
  哈希与原 manifest 中 `builder_sha256` 一致。
- 修正构建脚本将低有效 `aux_reset_in` 接常量 1；`dcm_locked` 仍为 1，
  高有效 `mb_debug_sys_rst` 仍为 0，外部复位来自 `FCLK_RESET0_N`。
- 新构建使用独立目录 `data/lung_axil_board_20261015_resetfix/`，不覆盖原证据。
- 原有计算核 RTL、147 任务仿真及模型/医学数据保持不变；此次为板级连接修复。

修正版已完成 bit/HWH 构建和审计：100 MHz，WNS +0.571 ns，WHS +0.016 ns；
LUT 10,306、FF 11,933、DSP 73、BRAM Tile 31.5。
DRC 无 Error/Critical Warning，仍有 278 条 DSP 流水线/OPMODE Warning。
真实厂商复位/ID 仿真与 187 项 Python 回归通过；1182 个模型/医学来源哈希未变。
新包位于 `data/lung_axil_board_20261015_resetfix/deploy/`，最终 manifest 包含复位连线与仿真/回归证据哈希。
状态仍是 `built_routed_single_block_smoke_not_board_tested`，不是实板 PASS。

## 新增验收

`fpga/tb_lung_board_reset.sv` 实例化真实厂商 `proc_sys_reset` IP 的正确/错误两种连线。
核对旧连线保持复位、修正后释放、外部复位重新断言/恢复、时钟锁定丢失/恢复，
并通过真实 `lung_axil_control` 完成三次 ID 读取与读响应背压检查。

```powershell
& 'E:\vivado2\Vivado\2022.2\bin\vivado.bat' -mode batch -source fpga/simulate_lung_board_reset.tcl -tclargs D:/pld_picture_process/data/lung_board_reset_sim_20261015
.\.venv-lung\Scripts\python.exe python/verify_lung_axil_resetfix.py --board data/lung_axil_board_20261015_resetfix
```

核验器在复制驱动、形成新部署 manifest 前，检查 HWH 中常量的真实类型、位宽、值，
辅助/外部复位极性，以及核心和两个互联的共享复位；还要求真实厂商复位仿真 PASS、
187 项 Python 回归 PASS，以及既有完整板级时序/资源/DRC/封装来源检查。
旧错误 HWH 必须被新核验器拒绝，不能仅检查引脚名字或脚本文字。

## 板端恢复与复测

若原 bit 已引起不可中断的挂起，先对开发板整体断电重启，再换入新文件。
不要在旧挂起内核中重新加载 overlay，也不要依靠 CLEAR 或仅复位计算核恢复在途 AXI。
替换同名 bit 与 HWH 后，先分步加载、读取 `0x4c554e47` ID，再进行原单块 smoke。
新报告使用未存在的名字，保留旧现象。最终仍需真实开发板 PASS；修复构建本身不是上板成功。

## 用户实板复测截图（2026-10-15）

用户已回传两张 Jupyter 结果截图：ID 返回 `0x4c554e47`；单块 smoke 最后显示
`Real-board single-block smoke: PASS`。可见各 case/block 均为 `same: true`、
`different_words: 0`，最后完成计数为 7（包含同地址更新测试）。
这支持小型合成输入的真实 DDR → 卷积 → DDR 写回和缓存更新测试通过，
但不等于整网、CT 分割精度或完整帧率已实板验证。

本机截图观察记录为 `data/lung_axil_board_20261015_resetfix/board_result_screenshot_record.json`。
原始板端报告 `/home/xilinx/ct_test/lung_axil_board_result_resetfix.json` 尚待回传核验，
因此没有把构建时 manifest 的 `fpga_deployed` 字段改成独立验证成功。
报告回传后应核对 bit/HWH 哈希、全部 7 条结果、完成计数与计算配置，再归档正式板端证据。
截图中的 `cpu_call_ms` 为小样例驱动调用耗时，不是完整 U-Net 单切片延迟。

下一阶段先验证冻结 U-Net 真实层的权重、逐通道量化参数和输入/输出布局，
再接 pool/nearest/concat、缓冲管理与整网调度；保留当前可工作的单块 bit 作为基线。

### 原始板端报告已回传并核验

用户随后通过 scp 回传原始 JSON。本机核验器 `python/verify_lung_board_smoke.py`
核对 bit/HWH 与不可变部署包一致、完整 7 条命令、完成计数 1～7、写回 word 数量和零差异，均通过。
合计写回 221 个 word，工程周期计数总和 56,066；不据此推断整网 FPS。
正式记录为 `data/lung_axil_board_20261015_resetfix/board_smoke_verification.json`，
状态 `verified_returned_real_board_single_block_smoke`。
这是核验用户回传的板端证据，不冒充本机直接远程读取开发板。
后续 15 个真实卷积层对拍见 [真实层上板说明](lung_model_layers.md)。
