# ROI HU 统计 RTL：整机与网页接入

来源为同学 `zhenghe/CT_Studio_最终整合_20261024_源码与验收.zip` 内的 `modules/candidate_linkage`。没有整包替换当前工程。来源文件及原验收材料逐文件 SHA-256 见 `source_provenance.json`；新增的复核脚本不改变原 RTL。

已保留当前候选列表、自动 ROI、MPR、Plotly 三维与局部 MIP。2026-10-24 新统计核已完成整机 bit/HWH、实板和网页验收。MIP 仍是 CPU 显示操作；选择新 FPGA ROI 时，count/sum/sumsq/min/max 来自统计核，均值与总体标准差由电脑用这些标量计算。电脑参考路径保留独立标识。

## 当前运行入口

- 运行 `start_ct_studio_roi_demo.cmd`，打开 `http://127.0.0.1:8099/`。选择「ROI 定量分析 → FPGA ROI · 实际硬件」，可在原始或完整增强 HU 上分析。候选的自动矩形沿用同一入口。
- 原 `8097` 会话和其中的整病例增强缓存保持运行；所有旧模型、bit/HWH、结果目录保留。新版加载了同一 386 层病例，未自动重算整病例增强。
- 新部署包：`data/ct_studio_roi_statistics_20261024_01/`；板端为 `/home/xilinx/ct_test/ct_studio_roi_statistics_20261024_01/`，服务 `ct-studio-roi-statistics.service`，端口 `8766`。
- 新服务已设为开机启动，旧 `ct-studio.service` 单元与包保留、开机启动停用。本轮没有重启板卡；开机配置验收见包内 `autostart_acceptance.json`。
- 架构沿用单一所有者串行切换基础处理、LDCT、分割、ROI overlay；新 bit 是完整 ROI PS/DMA/GPIO overlay，未把所有网络复制进同一个 bit。

## 整机与实板证据

- 构建输出 `D:/PLD/data/roi_statistics_overlay_20261024_02/`，100 MHz 布线后 WNS 0.633 ns、WHS 0.020 ns，通过后生成匹配 `ct_roi_statistics.bit/.hwh`。
- 新包保留并校验 118 个旧硬件/模型文件。轻量 CNN 权重 SHA-256 仍为 `00d7d04f33219c381ad748ae4c869a046538ef7c176e96e2031202b15b0ae022`；主模型配置未改动。
- 板端启动：24 帧 ROI 极值/空帧/掩膜对拍；15 个 LDCT 自检；基础处理和肺分割实际启动验收。原始数组下载后又在电脑独立复核。
- 当前真实病例：37 次 ROI 的掩膜与 int64 统计参考一致；3 张原轻量 CNN 与整数参考逐像素一致，最大误差 0 HU；验证 window/LDCT/segmentation/ROI 切换。
- 本轮 3 张 CNN 中位处理时间 374.49 ms。37 次 ROI 中位 DMA/计算 3.98 ms、板端模块总时间 75.45 ms；不含切换和网络往返，不能把 DMA 时间当作网页总延迟。
- 网页实际按钮、保存与导出通过；79 项相关 Python 回归通过，另有独立合成浏览器联动回归。详细 JSON/日志位于新包目录，`statistics_integration.json` 是证据索引。

整机重建与新包生成应使用新的目录名：

```powershell
cd D:\PLD\integration\ct-studio-runtime-20261018
$env:PROCESSOR_ARCHITECTURE='AMD64'
& D:\Xilinx\Vivado\2018.2\bin\vivado.bat -mode batch -source hardware/roi_statistics_20261024/build_overlay.tcl -tclargs D:/PLD/data/roi_statistics_overlay_new
.\.venv-web\Scripts\python.exe python/package_ct_studio_roi_statistics.py --build D:/PLD/data/roi_statistics_overlay_new --outdir data/ct_studio_roi_statistics_new
```

新包仍须单独实板验收，不应沿用本次验收结论或盲目重载故障所有者。

## 保留的硬件增量

`fpga/pynq_z2_roi_statistics.v` 包含原 ROI1 筛选核，并按匹配的输出握手累加 HU 总和、平方和、最小值和最大值。原 32 位 AXIS 掩膜输出、计数、配置锁存和错误检查保持不变。基础 `pynq_z2_roi.v` 与当前 `hardware/roi_20261019/ct_roi_axis.v` 在仅统一模块名和空白后完全相同。

无背压时按一拍一像素处理；使用 64 位累加结果，CPU 可从少量统计字求均值、总体标准差。这是硬件结构描述，不是实板或网页端到端测速。

## 本机重新验证

输出：`D:/PLD/data/roi_statistics_review_20261024_02/review_verification.json`。

- 重新生成独立 NumPy 输入与期望值：9 帧、803,180 像素，含完整 512×512 的有符号极值、正负交替、随机 ROI/HU/肺掩膜、空区域、单像素、零值和错误 TKEEP。
- XSim：掩膜、TLAST、TKEEP、背压稳定性及 count/sum/sumsq/min/max 全部通过；回传 CSV 再由 `verify_review.py` 独立检查，差异为零。
- Vivado 2018.2，`xc7z020clg400-1`，100 MHz OOC：755 LUT、612 FF、1 DSP、0 BRAM；WNS 1.362 ns，WHS 0.124 ns。
- 该轮 OOC 不包含整机外部 I/O 时序和真实时钟树；该轮未生成整机 bit/HWH、未访问板卡。当前整机验收见上节。

复跑必须指定新的输出目录：

```powershell
cd D:\PLD\integration\ct-studio-runtime-20261018
.\hardware\roi_statistics_20261024\run_review.ps1 -OutDir D:\PLD\data\roi_statistics_review_new
```

首次沙箱内的编译失败输出保存在 `data/roi_statistics_review_20261024_01`，不用于通过结论。通过结论只来自上述 `_02` 输出。

## 接口与保留边界

1. 新统计核替换原 ROI 模块实例，内部已包含筛选核，不重复串联两个 ROI。
2. 额外 sum、sumsq、minmax、valid、RST1 identity 输出按 `docs/fpga_statistics.md` 接到同一个硬件所有者管理的只读寄存器/GPIO。
3. `pynq/roi_statistics_processor.py` 在当前帧完成后持有锁读取统计；`python/roi_statistics_contract.py` 校验 RST1、valid、状态和整数范围。HTTP 回执绑定输入、掩膜、统计、bit/HWH 与发布清单。
4. `python/ct_studio_roi_client.py` 和 `python/roi_workspace.py` 使用硬件矩汇总整段 ROI；旧 bit 继续使用原统计路径。新硬件发生错误时不会回退 CPU、自动重试或强制释放 DMA 缓冲。

现有去噪 CNN、分割模型、旧服务和旧 ROI bit 均保留。真实整病例结节服务与冻结评分参数仍按现有交接清单推进，本次 HU 统计接入不代表整病例结节检测已完成。
