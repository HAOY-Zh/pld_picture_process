# Handoff：PYNQ-Z2 医学图像处理与在线肺分割

交接日期：2026-10-16。面向接手的队友 / Codex。

本文件描述当前本地工作区的最新状态，不覆盖旧的 `project_handoff_20261010.md`。
用户后续提供的申报原文见 `docs/project_proposal.md`；新增的项目范围与工期澄清见第 14 节。
旧 handoff、README 早期段落及历史构建报告里“尚无模型 / 尚未上板”等描述是历史快照，
不能用来判断最新进度；请结合本文件、最新源码和下列独立验收记录核实。

## 1. 接手后先看这几条

1. 已完成完整 INT8 小型 U-Net 在真实 PYNQ-Z2 FPGA 上的连续推理，不是只有单个卷积算子。
   21 个图节点、51 条硬件命令；实际输出与整数参考逐码一致。
2. 当前保留版 FPGA 网络单层平均 **686.368 ms**，同板双核 ARM INT8 **890.518 ms**，
   双核 ARM 原始 FP32 **1604.589 ms**。这是同一张验证切片重复测速，不是整例或网页端到端性能。
3. 用户现在的优先目标是：保留这版网络 / bit，做“手动输入 CT → 现场 FPGA 推理 → 网页显示”，
   不能用预先计算的掩膜冒充在线计算，也不能断网后悄悄切换电脑推理。
4. 在线服务、网页、部分层输入、局部三维及可选专家标注对照已完成；最新全量 **337 项测试通过**，
   独立浏览器交互测试通过。但浏览器验证使用明确标记的测试桩，
   **用户尚未返回新网页连接真实板端的验收结果**。
5. 下一步优先做实板网页联调和真实输出核验，不要先重训、换模型、调阈值或再跑 Vivado。
6. 旧版 bit、模型、板端验收报告和旧离线网页都保留。当前新增功能只改电脑端展示 / 调度，
   不需要换 bit；板端 HTTP 服务也是单独的新入口，不能与其他 Overlay 同时使用。

仅用于工程教学与算法演示，不作为临床诊断依据。

## 2. 环境、仓库与文件交付

- 当前工作区：`D:\pld_picture_process`，Windows PowerShell。
- 私有仓库：`https://github.com/HAOY-Zh/pld_picture_process`。
- 本次检查本地 HEAD 仍为 `98fee11 Initial commit: FPGA medical image processing project`。
  工作区有大量修改及未跟踪源码。本轮没有 commit / push，也没有核验远端最新状态。
  **不能假设队友克隆仓库就能获得本文件所述全部新代码。**
- 请保留现有脏工作区，不执行 `git reset --hard`、`git checkout --` 或擅自清理目录。
- `.gitignore` 排除了 `data/`、`.venv-lung/`、CT、权重、NPZ、bit/HWH 等。
  队友需要最新版源码和单独传递的部署包 / 必要数据，不能只拿 Markdown 或旧 Git 提交。
- 电脑 Python：`.venv-lung\Scripts\python.exe`，Python 3.11、PyTorch 2.8.0+cu126；
  NumPy / SciPy / pydicom / nibabel / scikit-image / Plotly 已可用。
  依赖清单见 `python/requirements-lung.txt`，PyTorch 环境需另外准备。
- 电脑 GPU：RTX 4060 Laptop；之前离线整例很快的某些数字是 CUDA 推理，不能当作电脑 CPU 测速。
- Vivado：2022.2；器件：`xc7z020clg400-1`。
  本机安装目录：`E:\vivado2\Vivado\2022.2`。
- 板端：ARMv7l、CPU 650 MHz，已有 NumPy / g++，无 Torch / ONNX Runtime。
  新板端服务不需要安装 Torch 或联网下载库。
- 板端地址：`192.168.2.99`；SSH 用户：`xilinx`。
  Jupyter：`http://192.168.2.99:9090`，基础目录：`/home/xilinx/ct_test`。
- 可用 SCP：`C:\Program Files\Git\usr\bin\scp.exe`。
  PowerShell 中先进入目录，再用相对路径上传；不要混用 Git Bash 的 `/d/` 和 Windows 的 `D:\`。
- 数据：LCTSC 在 `D:\CT2\lctsc`；COVID CT 数据在 `D:\CT3`。
  原始医学数据不在 Git 仓库中，也没有因本轮交互开发而改动。

不要在交接、截图、提交中包含 `live_token.txt` 的内容或其他凭据。

## 3. 当前完成情况与竞赛边界

| 工作 | 当前状态 |
| --- | --- |
| CT 输入 / DDR / 显示基础链路 | 已有独立演示及板端测试 |
| 窗宽窗位、阈值分割、二值形态学 | 已有 PL 模块，腐蚀 / 膨胀 / 开 / 闭运算已分别验证 |
| 智能分析：胸部 CT 肺区提取 | 已有训练模型、人工标注评价、完整整数网络实板验证 |
| INT8 量化与 FPGA 加速 | 已有同板 CPU FP32 / CPU INT8 / FPGA INT8 原始文件独立验收 |
| 3D / MPR / 体积与色块展示 | 原离线病例网页已有；在线结果重建已完成本机测试 |
| 手动输入新 CT、现场调用 FPGA | 新 HTTP 服务与电脑网页已实现，真实板端网页联调待验收 |
| 部分层输入 / 分批计算 / 局部三维 | 已实现并通过本机及测试桩浏览器测试 |
| 所有基础模块与 AI 合并一个 Overlay | 没有完成，不能把多个独立演示说成统一硬件系统 |
| 病灶 / 感染区 / 肺叶分割 | 不是当前目标，未完成 |
| 新 2.5D / 3D 神经网络部署 | 未采用，不能把三维展示称为 3D 网络推理 |

基础处理 Overlay 与肺分割 Overlay 不应同时被不同 Notebook 操作。
在线网页的 WL / WW 目前只是浏览器显示参数，不是在调用原来基础处理的 PL 窗宽窗位模块。

## 4. 冻结算法与精度，勿混用不同阶段数字

当前任务是二值“左右肺区域并集”，不是肺叶、病灶或左右肺分别分类。

- 原小型 2D U-Net，base channels 8，122,321 个训练参数，单通道 256×256；保留 epoch 12。
- Checkpoint：`data/lung_training_20261011/early_stop_run/best.pt`。
  SHA256：`9f822eb29910325e92b4cd6912d15c6dd687ca535857ee9c27f629172be47f88`。
- 训练 18 病例 / 3002 层；开发验证 6 病例 / 1120 层。
- 预留测试实际可用 17 病例 / 2705 层，另外 7 个测试目录无 CT；冻结原 FP32 平均病例
  Dice 0.965467。不是官方 24 例完整成绩。
- COVID 外部包审计 20 份，其中 10 份原 HU / 2581 层用于评价；另 10 份是显示灰度，
  不冒充 HU。该轮 FP32 平均 Dice 0.958590、电脑 INT8 0.959075。
- 六例最终相同后处理下：FP32 0.9672978510、电脑 x86 INT8 0.9670989970、
  拟硬件整数规则 0.9670878194。最后一个是电脑整数参考全数据集验证，
  **不是所有病例都已在真实 FPGA 上做了精度评价**。
- 这些数字的病例、数值模型与后处理条件不同，不可拼成同一条“正确率提升”。
  Dice 是掩膜重合度，不是诊断正确率。
- 多轮裁剪 / 微调 / 2.5D / 蒸馏 / 层面辅助等尝试没有替换当前保留模型。
  本轮未重训、未重新校准、未根据测试集调参。

数值与后处理契约必须保持：

1. 原 HU clip 到 `[-1000,400]`，像素中心 bilinear 到 256×256。
2. `(HU+1000)/1400`，保留原 float16 缓存舍入再转 float32，使用原 Torch affine 输入量化。
3. 输入 UINT8 CHW `(1,256,256)`，scale `0.007870171219110489`、zero point 0。
4. FPGA 执行完整整数网络。权重有符号 INT8，激活 UINT8；中心化、累加、重缩放位宽按冻结契约。
   不可自行换 rounding、截断、padding 或拼接重定标规则。
5. Head UINT8 `(1,256,256)`，scale `0.3118481934070587`、zero point 72。
6. 电脑先将 head 还原 logits 并 Sigmoid，再按原尺寸 bilinear `align_corners=False`，最后阈值 0.5。
   不要先缩放硬掩膜，不要将 logits 插值代替概率插值。
7. 原尺寸 CT-only 人体包围框输出约束；它不是改变模型输入的身体裁剪。
8. 全输入结果可做原三维孤岛清理；局部截断片段跳过该整肺清理。
   不因为显示轮廓有小圈就擅自填肺掩膜内部孔洞或修改人工标签。

详细说明：`docs/lung_fpga_deployment.md`、`docs/lung_fixed_rtl.md`、
`docs/lung_heldout_evaluation.md`、`docs/lung_external_evaluation.md`。

## 5. 已实板验收的 FPGA 保留版

- 构建目录：`data/lung_cache_return_20261016_v2`。
- 三方完整部署包：`data/lung_threeway_20261016/lung_threeway`。
- 独立验收：
  - `data/lung_threeway_20261016/lung_threeway_result_verification.json`。
  - `data/lung_threeway_20261016/lung_threeway_result_threeway_verification.json`，
    状态 `verified_returned_threeway_actual_fp32_and_int8_heads`。
- 九个返回原始文件同目录，前缀 `lung_threeway_result`：主结果、CPU、diagnostic、clock、
  selection、真实 INT8 head、FP32、真实 FP32 head、threeway 汇总。
- `verification.json` 等构建时的“not_board_tested”状态是当时快照，
  不追写覆盖；是否实板验收应看后来独立返回文件的证明。

bit SHA256：

```text
567ddc0e47e3556e21b248a2d0bab85f26381c8056295f88ddd14335e695c89b
```

HWH SHA256：

```text
36ce4d28641805e7705a8f3f312a5c0712e3f45a17a605a39c062013a6f58f58
```

资源与时序：100 MHz，WNS +0.204 ns，WHS +0.022 ns；
19,063 LUT（35.83%）、20,360 FF（19.14%）、112 DSP（50.91%）、31.5 BRAM（22.50%）。
DRC 0 Error / 0 Critical Warning，仍有 356 条已记录 Warning；不要宣称零警告。

BD / 硬件要点：

- `lung_conv8_axil_0`：base `0x43c00000`，ID `0x4c554e47`。
- `lung_tensor_axil_0`：base `0x43c10000`，ID `0x54454e53`；两者 ABI `0x10000`。
- GP0 配置，HP0 共享 DDR 数据访问；现有自定义 AXI 读写主机，不是必须另加 AXI DMA。
- 共用 100 MHz 时钟 / 低有效复位；未使用的低有效 AUX reset 要保持无效高电平。
  历史上此处错误曾导致 MMIO 卡死，不要退回旧连接。
- 加载 bit 后 PYNQ 曾实际设置成 62.5 MHz；当前入口先核对 ID / idle，再只校正 FCLK0 为 100 MHz，
  并逐帧读回验证。不得改 CPU / DDR 时钟或擅自超频。
- 21 节点 / 51 命令 = 45 次卷积通道块 + 3 次池化 + 3 次融合放大拼接。
- `fpga/lung_axil_config_timing.xdc` 的例外范围受约束，不添加覆盖整个数据通路的 blanket false path。

当前同板实测，batch 1，S2-006 第 72 层，256×256，每组预热 3 次 / 测量 20 次：

| 实现 | 均值 ms | 中位数 ms |
| --- | ---: | ---: |
| Z2 双核 ARM 原始 FP32 | 1604.589 | 1593.432 |
| Z2 双核 ARM 精确 INT8 | 890.518 | 877.507 |
| FPGA INT8 | 686.368 | 685.966 |

CPU INT8 / FPGA INT8 = 1.29744×；CPU FP32 / FPGA INT8 = 2.33780×。
FPGA head 的 65,536 个码与整数参考完全一致；实际 FP32 head 相对原始 Torch 最大误差约 3.43e-5。
FP32 基线来自原始 FP32 checkpoint，不是将 INT8 权重反量化。

计时为驻留网络输入 → head，不含 DICOM、Sigmoid / 原尺寸恢复、身体约束、三维、网页及 bit 初始化。
每帧真实权重传输、输入同步、硬件等待和末端 guard 在 FPGA 计时内。
卷积约 489.445 ms、张量约 113.498 ms、其他约 83.425 ms；驻留 CMA 约 7.471 MiB。
240 层乘此单层均值约 2 分 45 秒只是网络时间估算，不是实际整例网页测试。

细节：`docs/lung_cache_return_threeway.md`。既有验收器对已存在输出有拒绝覆盖逻辑，
不要为了“再确认”直接重跑 `verify_lung_threeway_board.py` 并覆盖已验收文件。

## 6. 当前在线实现的关键文件

| 文件 | 职责 |
| --- | --- |
| `pynq/lung_live_service.py` | 真实板端服务，持有 FPGA / CMA；鉴权、单任务互斥、启动 21 节点自检 |
| `python/lung_live_data.py` | DICOM / NIfTI / HU NPZ、连续部分加载、层号解析、原量化和原生后处理 |
| `python/lung_live_web.py` | 电脑 loopback 后端、板端代理、输入 / head 哈希、任务进度、局部 / 全输入重建 |
| `python/lungseg/live_viewer.html` | 四宫格、色块、手动计算、部分层选择、MPR、局部范围及截断提示 |
| `python/lung_live_package.py` | 新目录打包；已生成后不要对同目录重复执行 |
| `python/tests/test_lung_live.py` | 在线路径单元测试；其中 FakeHardware 仅供测试 |
| `python/tests/live_browser_fixture.py` | 合成 CT + 明确测试桩的本机 UI 测试服务，不是生产入口 |
| `python/tests/live_browser_smoke.mjs` | 独立 CDP 浏览器交互验证，不依赖 npm 包 |
| `docs/lung_live_web.md` | 详细启动、输入规则、职责及安全说明 |

在线包：`data/lung_live_20261016/lung_live`，约 15 MiB，38 个文件。
同级 `host_config.json` 指向默认板端 `http://192.168.2.99:8765`；`model.json` 只供主机预 / 后处理，
没有主机模型 forward。`live_token.txt` 是两端匹配的随机秘密，不在本文件中记录。

在线包复制了验收版及其完整哈希依赖，旧运行时仍逐文件不变：
`lung_fast.py`、`lung_fast_io.cpp/.h`、`lung_axil.py`、`lung_tensor.py`、`lung_continuous.py`、
`lung_fast_benchmark.py`、`lung_selected_benchmark.py`、`lung_clock_benchmark.py` 等。
不要为了改网页去修改这些被冻结的核心文件。

留档链条：

1. `data/lung_live_20261016/preserved_release.json`：最初独立在线包 / 原保留版 / 旧网页的哈希快照。
2. `partial_input_update.json`：后续主机部分加载 / 选层更新。
3. `local3d_update.json`：最新主机局部三维更新。

后续主机代码哈希与第一份初始快照不同是有记录的增量更新，不要回退新网页使之匹配初始快照，
也不要改旧证明“补齐”。板端包及原验收版没有随这两次主机更新改变。

## 7. 在线系统职责与使用规则

```text
电脑 CT 原 HU → 原量化预处理 → HTTP 发送 1×256×256 输入码
  → Z2 ARM 调度常驻 FPGA 完整网络 → 返回实际 head 和命令 / 时钟 / 哈希回执
  → 电脑 Sigmoid / 原生恢复 / 人体约束 → 浏览器色块 / MPR
  → 连续片段局部网格，或全输入结果的三维清理及网格
```

- FPGA：完整网络前向；Z2 ARM：MMIO / CMA / HTTP；电脑：I/O、预后处理、几何和三维。
  不宣称三维重建在 FPGA，也不宣称所有处理在 PL。
- 网页手动路径指**电脑上的路径**，不是板端 SD 卡路径，也不是浏览器上传功能。
- 接受单个 `.dcm`、单序列目录、原 HU `.nii/.nii.gz` 或严格四键 HU `.npz`。
  NIfTI / NPZ 需操作者确认 HU / mm，不接受 PNG/JPEG 显示灰度冒充 HU。
- 多 CT Series 不自动挑选；按提示缩小目录或指定 Series UID。
  不支持斜切 / 剪切重采样、缺层 / 非均匀层距、Enhanced 多帧 DICOM。
- “读取层”留空全部，`60-90` 是含两端的 31 层，`72` 是单层；部分加载只接受连续范围。
- “计算层号”允许 `0-9`、`0,5,10-15`，自动去重排序。未选层不调用 FPGA，可以继续补算。
- 全部从 0 编号。源范围 `60-90` 加载后，当前数据的 `0-4` 对应源 `60-64`。
  DICOM 按空间位置而非文件名 / InstanceNumber 排序；NIfTI 按规范化后的轴位顺序。
- DICOM 部分加载只解码选中像素；压缩 NIfTI / NPZ 的解压或重排可能仍访问完整文件。
- “未计算”有单独标志，不能当成预测为背景；无人工标注的新个体不显示 Dice。
- 当前所有已计算层形成一个连续范围、至少两层时，任务正常完成后自动重建。
  单层不外挤成三维，零散有间隔的层不补洞拼体。
- 局部三维保留真实坐标 / 层距，并显示源层范围、上下红色截断面。
  网格端面封口是可视化截断，不是解剖边界，也不改变二维掩膜。
- 局部仅用人体约束后的掩膜，不执行整肺孤岛清理；“三维清理后”选项只供全输入结果。
- `volume_ready` 表示有可用网格；`full_volume_ready` 仅表示全部输入重建就绪。
  局部 `volume_scope=partial`，`volume_ml=null`，只报告 `region_volume_ml`。
  `reconstruction` 保存局部范围、源层号与偏移后的几何原点。
- “全部输入”不保证扫描覆盖完整肺，不能把截断采集称为完整器官。
- 层间停止只在当前 FPGA 推理正常完成后停，不做强行 reset；停止后不自动伪造完整体。

原离线六病例页面仍在：`data/lung_body_gate_20261014/browser/index.html`，
早期入口 `data/lung_case_browser_20261014/index.html`；它们是预先导出的软件结果，不是在线 FPGA。

## 8. 从当前包启动：下一步需要用户操作

先确认旧 FPGA 测试已正常结束，没有未恢复的 AXI 卡死；不能同时运行其他 Overlay。

电脑 PowerShell 上传（如果尚未上传；主机局部三维更新本身不要求重传）：

```powershell
Set-Location D:\pld_picture_process\data\lung_live_20261016
& "C:\Program Files\Git\usr\bin\scp.exe" -r .\lung_live xilinx@192.168.2.99:/home/xilinx/ct_test/
```

专用板端 Jupyter Notebook：

```python
%cd /home/xilinx/ct_test/lung_live
from lung_live_service import start
live_service = start(bind="0.0.0.0", port=8765)
```

等待 `FPGA LIVE READY ... 21-node startup PASS`。该调用返回后后台继续服务，保留专用内核与对象。
不要重复 start、运行旧测试、重新下载其他 bit 或清空驱动模块。

电脑 PowerShell：

```powershell
Set-Location D:\pld_picture_process
& .\.venv-lung\Scripts\python.exe .\python\lung_live_web.py --config .\data\lung_live_20261016\host_config.json
```

浏览器打开 `http://127.0.0.1:8088/`。这里 127.0.0.1 是运行电脑后端的电脑，不是开发板。
保持终端，点击“连接 FPGA”，加载真实 CT，例如：

```text
D:\CT2\lctsc\LCTSC-Train-S3-008
```

建议先“读取层”填 `124-129`，点击加载，再点“计算片段并局部重建”。
这只算 6 层，适合先核对回执和局部联动；之后再试全输入整例。
默认勾选“加载专家标注”，上述病例目录内会自动找到匹配 RTSTRUCT。
`124-129` 层的人工肺掩膜已与旧缓存核对一致；该例 `60-65` 层专家标注为空，不能据此判断显示坏了。
若用户已有不同来源，先确认 HU / 方向 / 层距，不要直接把显示图片送网络。

断网或 board error 时会停，不会自动换电脑网络推理。新服务及网页的真实上线状态需以用户输出为准，
不能因为命令已提供就写成已上传 / 已启动 / 已实板网页验收。

## 9. 当前已验证与下一步验收清单

已完成：

- 原验收切片的实际 DICOM → 新网页预处理，与原 `input.npz` 全 65,536 个输入码一致。
- 最新 325 项完整测试通过，日志：`data/lung_live_local3d_checks_20261016/python_tests.log`。
  测试里的训练 / 病例是合成夹具，不是重新训练当前保留模型。
- 浏览器证据：`data/lung_live_local3d_checks_20261016/browser_TEST_ONLY.json`。
  `partialInputTestPassed=true`、`local3DTestPassed=true`、无浏览器异常。
- `local_3d_TEST_STUB.png` 等是明确标记的合成测试图，绝不能当成真实 CT / FPGA 性能或精度图展示。
- 局部原点偏移、网格空间范围、上下截断、片段体积、切片平面移出 / 返回范围均有测试。
- 原保留版、板端包 / 服务及旧离线网页哈希核验未变；未跑 Vivado，未改真实医学标签。
- 本轮临时测试服务和独立隐藏浏览器已关闭，不保证用户生产后端已启动。

接手后按顺序做：

1. 确认用户是否已上传新 `lung_live` 并看到真实服务的启动自检成功；收集完整输出。
2. 网页先连真实 FPGA、算一层或连续 6 层，核对实际输入 / head SHA、bit 身份、51 条命令、
   100 MHz 时钟、源层号及体素恢复；不能只看轮廓像不像。
3. 检查电脑的 `data/lung_live_20261016/results/<case_uuid>/`：
   `input.json`、每个请求的 `actual_head.npz` / `receipt.json`，以及任务的 `*_masks.npz` / `*_volume.json`。
   局部 masks 只含对应片段的 raw / body；几何使用片段原点；整例另含 cleaned。
4. 必要时独立对照同一个真实输入的冻结整数参考，不要用另一个切片的旧 head 比较。
   独立验收程序不应回写或替换实际 FPGA 输出。
5. 实测部分输入 / 分批补算 / 断网报错 / 层间停止，以及全输入运行和二维三维同步。
   先做小范围，避免一开始等待完整 240 层。
6. 新验收证据另建报告，保留旧快照；分别报告网络、通信、预后处理及重建总耗时。
7. 专家标注对照已完成本机验证（见第 12 节）；真实 FPGA 同源输出对照仍需联调。
   本链路稳定后再考虑更多实际输入验证或性能优化，不先扩展病灶任务。

本机回归命令（不使用真实板端）：

```powershell
& .\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -p test_lung_live.py -v
& .\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -p "test_*.py"
```

完整测试当时耗时约 70 秒，35 项实时入口测试已包含在 325 项中，不要相加。

## 10. 必须遵守的硬件安全与数据保护

- `lung_fast._UNSAFE`、`lung_continuous._UNSAFE_RUNS` 等可能持有不确定 AXI 事务涉及的 CMA。
  超时后不要自动释放、重载 bit、回滚时钟、清空 `sys.modules` 或强行结束内核来“恢复”。
  先保留现场 / 日志，判断硬件状态，再与用户协调板级恢复。
- `FastUNet` 为单 owner 持久缓冲；输入 CPU 写后 flush，PL 中间张量保持 device-owned，
  CPU 读前 invalidate。不能向 PL 已写的中间数据 flush 过期 CPU cache。
- 生产服务只有真实 FPGA backend；FakeHardware 只能在 `python/tests/` 中使用，不能接进生产兜底。
- 板端有服务锁，但无法阻止旧 Notebook 绕过锁直接 MMIO；必须人工保证独占。
- `python/verify_lung_prepared_board.py` 是历史独立验收依赖，保持不变；其 SHA256 为
  `5d24033f0a4e823c7426dbd48b7293b947ddbdc71550018f3abf7fbb7788b5c4`。
- 已保护的医学 / 模型文件哈希在 `data/lung_fixed_network_20261014/screen/evaluation.json`
  的 `protocol.input_sha256` 中，历史核验 1,182 项。不要改标签、测试划分、checkpoint 或冻结契约。
- 生成器与独立验收器多使用独占新文件 / 新目录，不要删除旧证据只为让脚本再次运行。
- 用户之前允许清理废案，但当前任务不是新的删除授权；不要继续自动删除报告、数据或旧版。
- HTTP 仅供可信实验局域网；电脑只监听 127.0.0.1，板端独立 token，禁止浏览器跨域。
  无 TLS，不适合隐私病例、公网、多用户医疗服务，不能将端口转发公网。
- 本 handoff 不包含密钥。共享源码 / 推 Git 前检查凭据和医学数据；没有授权则不要提交或推送。

## 11. 给接手 Codex 的简短指令

> 先读本文件及 docs/lung_live_web.md、docs/lung_cache_return_threeway.md，核对工作区与实际文件。
> 保留现有完整 INT8 网络和实板验收结果，不重训、不换 bit、不回退旧模型。
> 优先协助用户启动真实板端服务和电脑网页，做小范围真实 CT 在线推理及局部三维联调；
> 要取得实际 head / receipt 后再称新网页实板验收。测试桩、旧离线 HTML 不冒充 FPGA 现场计算。
> 遇超时先保护在途 AXI / CMA，保留现场，不自动重载或释放。持续推进到确实需要用户操作时再停。
> 最新功能包含部分加载、分批补算、连续局部三维、截断范围提示和片段体积；详细规则见第 7 节。

## 12. 2026-10-16 专家标注对照补充

- 用户要求补回旧软件网页已有的专家标注对照。新增 `python/lung_live_reference.py`，
  并更新电脑端 CT 元信息、Controller、网页和部署包的主机源码哈希清单；未改板端服务或硬件。
- DICOM 病例目录默认自动寻找唯一匹配 Series UID 的 RTSTRUCT，也可填写具体标注路径；
  可选几何匹配的 NIfTI / 四键 mask NPZ。复用原 SOP / Frame / 轮廓平面检查、肺 ROI 栅格化及左右并集，
  不修改标签；NIfTI / NPZ 只无损换轴/翻转、截取同源片段，不重采样标注。
- 网页显示橙色 FPGA 预测与青色专家轮廓，两个开关独立；专家二维预览不需要运行推理。
  连续片段/完整输入重建后，三维增加同范围专家网格。未计算层可看标注，但不计 Dice。
- 汇总和每个 MPR 剖面的 Dice 只覆盖实际已计算区域，区分 partial / 全部输入；
  双方均空时 Dice 无定义，不显示 1。专家标注不参与量化、网络前向、人体约束、清理和阈值。
- 病例 `input.json` 记录专家来源/文件及掩膜哈希，另存 `expert_mask.npz` 审计副本；
  每层 receipt、正常完成的 `*_comparison.json` 和重建报告记录评价范围及指标。
- 全量 337 项回归通过：`data/lung_live_expert_checks_20261016/python_tests_v2.log`。
  首次全量的新增测试按随机文件名误取报告，修正后复跑；原失败日志保留，不覆盖。
- 独立浏览器验证专家轮廓实际像素、开关、三维叠加、局部范围与四宫格一屏布局通过，
  证据 `browser_TEST_ONLY.json`，只使用合成 CT 与明确标记的测试桩。
- 真实 `LCTSC-Train-S3-008` 的 124–129 层，6×512×512，125,458 个专家前景体素，
  新入口与旧 `reference_native.npy` 逐体素一致；同例 60–65 层专家为空，也核对一致。
  这是只读标注加载检查，没有现场 FPGA 推理或新的精度评价。
- `source_reference_audit.json` 核对 1,182 项冻结医学/模型输入与 75 项保留文件不变。
  新增留档 `data/lung_live_20261016/expert_reference_update.json`，不改原三份历史更新证明。
- 使用时重启电脑后端并刷新网页；不要求重传板端包、不换 bit、不重训。
  本轮没有连接板端、操作现有生产后端、Git 提交或推送。真实 FPGA 网页联调仍待用户实测。

## 13. 在线界面合并预测展示

- 按用户要求，将“人体约束后”和“原始 FPGA 预测”合并为单个“FPGA 肺区分割”展示项，
  实际使用原 body 掩膜。完整输入完成后仍保留“三维清理后”选项。
- 只简化电脑网页的选项；二维、三维及专家 Dice 继续使用相同 stage。
  原始 FPGA head、raw 掩膜、回执及各阶段后台评价仍保留，预后处理与板端均未修改。
- 更新页面需重启电脑网页后端并刷新浏览器。

## 14. 用户补充的申报范围与工期

- 文档于 11 月 1 日提交，11 月 10 日之前展示；按当前上下文为 2026 年。
- 用户提供了《低剂量CT增强与智能分析辅助系统》申报原文，保存于 `docs/project_proposal.md`。
  整体范围为通用基础处理、以局部噪声与结构信息控制的 LDCT 自适应增强，以及 CT 智能分析 FPGA 加速。
  不能把项目主线收窄为肺分割或配准；配准是拓展方向，不应取代申报的 LDCT 特色。
- 用户明确：LDCT 尚未开发，计划使用 CNN。`fpga/Readme.md` 和 `matlab/Readme.md` 中虽列有 LDCT，
  只是规划文字；本轮未找到 LDCT 实现。原肺分割训练中的局部亮度增强不是 LDCT 图像增强模块。
- 当前已核实的基础 PL 功能仍为窗宽窗位、阈值与形态学；直方图增强、伪彩在申报中出现，
  还需要单独核查或实现，不得直接写成已有实板功能。
- 后续设计应围绕 LDCT CNN、局部噪声/结构自适应控制、FPGA 数据流/缓存优化，以及增强前后独立评价。
  增强是否改善分割需实验验证；保留当前冻结肺分割模型及原 HU 路径作基线，不默认替换输入。
- 本次仅保存申报原文并澄清范围，未下载 LDCT 数据、训练 CNN、修改硬件或运行 Vivado。
