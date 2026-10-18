# 本地 Vivado：仅创建与验证窗口递推工程

后续状态：用户已于 2026-10-16 明确要求恢复自动综合、实现及 bit 生成。
自动流程使用独立的 `fpga/resume_lung_window_step.tcl`，不向本页的源码入口添加自动构建功能。
因此下面的“停止”说明保留为当时历史；源代码入口仍可用于用户手动验证。
自动恢复现已完成，整机 100 MHz setup/hold 通过，新包通过独立审计；
具体资源、身份和上板命令见[窗口优化最终交付](lung_window_step.md#恢复构建最终完成)。

按最新要求，代理已停止此前启动的 `lung_window_step_20261016_v2` 构建及其子进程。
停止前部分综合已经完成；没有完成整机实现或生成本轮新 bit。中断目录只保留为历史过程记录，
不将它当作已验收的构建，也不自动重启。后续综合、布局布线及 bit 由用户自行决定何时运行。

## 本次交付

- 优化 RTL：`fpga/lung_feature_window_step.v`，此前新旧七组实际 XSim 对拍通过。
- 新工程入口生成器：`python/lung_window_step_project.py`，只写文件，绝不调用 Vivado。
- BD 检查器：`fpga/validate_lung_window_step.tcl`，只检查连接/参数/地址并执行 Validate Design。
- 已准备源文件目录：`data/lung_window_step_source_20261016`，含 17 个选定 RTL、PS 配置、XDC、
  `create_project.tcl`、`validate_bd.tcl`、源文件哈希清单。不包含已生成的新 bit。

生成的 `create_project.tcl` 已彻底移除自动综合/实现/bit 部分，并拒绝额外 `build` 参数。
它沿用原双引擎 BD、创建 IP 输出文件和 HDL wrapper，最后保持工程打开。
本轮没有在 Vivado 中执行这个新入口；Python/Tcl 语法静态检查不能冒充实际 BD 或时序验收。
完整纯软件回归 278 项通过，73.273 秒，无跳过；日志位于源文件目录的 `python_tests.log`。

原已验证的 859 ms 突发版 bit、模型、数据划分及原始回传文件均保留不变。

## 连接与约束审查

| 项目 | 设置 |
| --- | --- |
| 器件 | xc7z020clg400-1 / PYNQ-Z2 |
| 控制 | PS GP0 → 1入2出 SmartConnect → 卷积/张量 AXI-Lite |
| 数据 | 两个 64-bit AXI4 master → 2入1出 SmartConnect → PS HP0 |
| 寄存器 | 卷积 0x43c00000、张量 0x43c10000，各 1 KiB |
| DDR | 两个 master 均映射 0x00000000–0x1fffffff，缓冲区不得越界 |
| 时钟 | FCLK0 100 MHz，同步驱动两引擎、GP0/HP0 ACLK、互连和复位同步器 |
| 复位 | FCLK_RESET0_N → proc_sys_reset；低有效 peripheral_aresetn 共享给两引擎和互连 |
| 未使用复位 | AUX 低有效，所以必须接常量 1；debug reset 接常量 0 |

无需新 AXI DMA。现有引擎本身带 master，主机通过 AXI-Lite 配置与轮询。
静态核对了请求/返回 ready-valid、输出反压、错误传播、填充边界和最终 B 完成条件；
张量 burst 仍最多 16 beats、避免跨 4 KiB，卷积仍使用原写回协议，未改变数据布局。
这些代码/既有仿真检查不能替代新工程的实际仿真和板端协议测试。

XDC 沿用 `fpga/lung_axil_config_timing.xdc` 的已锁定配置寄存器 4/3、派生配置 3/2 周期例外。
新寻址寄存器和所有实时 AXI/MAC 数据路径仍按单周期检查，未增加 blanket false-path 或放宽时钟。
FCLK0 时钟由 PS/IP 自动生成约束，不额外重复 `create_clock`。
约束文件保持 `USED_IN_SYNTHESIS=false`、`PROCESSING_ORDER=LATE`，用于真实实现网表的路径约束。
后续时序报告若提示 collection 无对象、未约束路径或 multicycle 无效，应提供完整日志，不能直接屏蔽。

## 在本地执行（以下均由用户运行）

建议使用 Vivado 2022.2。先打开 Vivado，不要打开已经中断的旧工程。
在 Tcl Console 输入：

```tcl
cd D:/pld_picture_process
set argv [list D:/pld_picture_process/data/lung_window_step_source_20261016]
set argc [llength $argv]
source D:/pld_picture_process/data/lung_window_step_source_20261016/create_project.tcl
```

这一步创建 `project/lung_network.xpr`，不启动综合、实现或 bit。
如果目标已有 `project`，脚本会拒绝覆盖。需要重新开始时，选新的目录，重新准备：

```powershell
Set-Location D:\pld_picture_process
.\.venv-lung\Scripts\python.exe python/lung_window_step_project.py --root data/lung_window_step_source_retry
```

然后将 Tcl 命令中的目录改为新目录。源文件目录是快照；若编辑仓库原始 RTL，需要重新准备快照，
或明确更新 Vivado 工程选中的文件，不能以为修改旧文件会自动改变 IP 内部打包副本。

工程创建成功后：

1. 打开 `design_1.bd`，运行 Validate Design。也可重新执行：

   ```tcl
   open_bd_design [get_files design_1.bd]
   source D:/pld_picture_process/data/lung_window_step_source_20261016/validate_bd.tcl
   update_compile_order -fileset sources_1
   check_syntax -fileset sources_1
   report_ip_status -file D:/pld_picture_process/data/lung_window_step_source_20261016/ip_status.rpt
   ```

2. 在 GUI 选择 RTL Analysis → Open Elaborated Design，检查层级、端口宽度、未连接/多驱动信号。
   不点击 Run Synthesis、Run Implementation 或 Generate Bitstream。
3. 如要重新做纯行为仿真，可新建独立 RTL 仿真工程，只加入源目录的
   `rtl/lung_feature_window.v` 和仓库 `fpga/tb_lung_window_step.sv`，仿真顶层设为
   `tb_lung_window_step`，Run Behavioral Simulation → Run All。
   预期末尾出现 `Lung stepped window RTL: PASS (10 cases; ... )`。

当前阶段不能验证最终 LUT/DSP/BRAM 占用、布线 setup/hold 或实际提速；须等用户以后运行完整流程。
若出错，请提供 Tcl Console 第一处 ERROR 及其上下文、`vivado.log`、IP 状态报告，
若为仿真错误，再附 `xvlog.log`、`xelab.log` 和仿真运行日志。密码、token 无需提供。
