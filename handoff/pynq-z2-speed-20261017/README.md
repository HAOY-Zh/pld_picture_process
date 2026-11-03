# PYNQ-Z2 去噪加速：上板验证交接

目标：在 PYNQ-Z2 上验证同一 CNN 的四种执行方案，检查图像输出是否完全一致，并测量整张图的真实处理时间。

**当前状态：RTL 仿真、Vivado 布局布线以及两张真实 CT 的实板测速均已通过。**
本版沿用先前已上板的 3 层、8 通道 CNN 整数权重；不包含后来 300 轮训练的新权重。

2026-10-17 实板结果：两张512×512图分别预热1次、测5次，先取各图中位数再平均：

| 方案 | 含PS/DMA/PL/拼图的耗时 |
| --- | ---: |
| legacy48 | 760.254 ms/张 |
| fast48 | 659.752 ms/张 |
| fast58 | 559.450 ms/张 |
| compact58 | 523.226 ms/张 |

最终耗时降低31.18%，约1.453倍加速、1.91帧/秒。下载全部8份实际FPGA输出后再次独立核验：
与整数参考及此前旧版实板输出均逐像素相同，最大误差0 HU。仅代表这两张演示图的条件。
原始计时、环境与二次校验见[results/](results/)，[速度对照图](results/speed_comparison.png)。
部署包已修复模型加载器对训练模块的多余依赖（r2），使用轻量板端包即可运行。

## 同学需要下载什么

下载本目录中的 [ct_tile_speed_deploy_20261017.zip](ct_tile_speed_deploy_20261017.zip)。
它包含匹配的 `.bit` / `.hwh`、整数模型、板端 Python、两个 Notebook、两张公开 CT 演示切片及对应正常剂量参考。
无需训练、无需下载完整数据集，也无需在板上安装 PyTorch、CUDA 或 Vivado。
`SHA256SUMS.txt` 记录本次交付文件的 SHA256。

两张演示图来自项目使用的公开 **LDCT-and-Projection-data** 数据集：C067 第 180 层、C077 第 190 层（从 0 开始计数）。
包内只保留原始分辨率的 HU 数组，不包含 DICOM 文件、姓名等原始头信息或训练缓存。
这些样例用于工程演示和一致性检查，不能代替完整测试集或人工标注评价。

## 上板步骤

1. 使用 **PYNQ-Z2 / xc7z020clg400-1**，开机后进入自己的 Jupyter。此前实测环境为 PYNQ 3.1.1；其他版本需要实际确认兼容性。
2. 上传 ZIP，在 Jupyter Terminal 中解压到一个新的目录：

   ```bash
   python3 -m zipfile -e ct_tile_speed_deploy_20261017.zip .
   ```

3. 在 Jupyter 文件列表进入解压得到的 `ct_tile_speed_deploy_20261017` 文件夹，打开 **`run_speed_test.ipynb`**，按顺序运行全部单元格。
   也可以在 Terminal 中执行：

   ```bash
   cd ct_tile_speed_deploy_20261017
   python3 pynq/benchmark_ct_tile_speed.py --outdir board_speed_001 --repeats 5
   ```

4. 首先会计算一次 CPU 整数参考结果，这一步可能较慢；它不计入板端速度。之后会打印每轮计时。
5. 每个方案先预热，随后测 5 次；两张图分别运行。重复测试请换一个新输出目录，例如 `board_speed_002`。

请保留整套解压目录结构。程序会核验包内文件哈希、PL 时钟和逐像素结果；文件不匹配或结果不一致时会报错退出。
加载 Overlay 会替换板上当前 PL 逻辑，测试时不要同时运行其他 DMA / Overlay 程序。

## 四个方案分别验证什么

| 方案 | CPU 打包 | 输出块 | FPGA 返回范围 | 512×512 图的 DMA 往返 |
| --- | --- | --- | --- | ---: |
| `legacy48` | 每块分别转换 | 48×48 | 含重叠边缘 | 121 |
| `fast48` | 整帧只打包一次 | 48×48 | 含重叠边缘 | 121 |
| `fast58` | 整帧只打包一次 | 58×58 | 含重叠边缘 | 81 |
| `compact58` | 整帧只打包一次 | 58×58 | 仅有效核心，跳过无用计算 | 81 |

四种方案都由 FPGA 执行 CNN，ARM CPU 负责分块、搬运和拼图。CPU 参考程序只用于核验，不能代替 FPGA 输出。
计时包含 CPU 打包、缓存同步、DMA、PL 计算、轮询和拼图；不包含读文件、生成参考结果或显示图像。
`legacy48` 是同一个新 bitstream 的兼容路径，因此四组使用同一时钟、同一权重和同一输入。

## 需要回传的结果

请把整个 **`board_speed_001/`** 文件夹压缩回传，主要包括：

- `comparison.json`：板卡环境、时钟、模型/bit 哈希、各方案耗时、帧率、相对加速比。
- `timings.csv`：每张样例图、每个方案的统计。
- `C067_*.npy`、`C077_*.npy`：四种方案的实际 FPGA 输出，便于离线再次核对。

通过条件：程序正常结束；`all_outputs_bit_exact=true`；四种方案对同一张图输出完全一致。
是否提速以实际 `median_frame_ms` 和 `speedup_vs_legacy48` 为准；目前没有承诺整板加速倍数。
若失败，请回传完整报错、PYNQ 版本和运行到哪一步，先不要跳过一致性检查或改用 CPU 结果。

## 已完成的离线检查

- Vivado **2018.2**，PL **50 MHz**，布局布线后 WNS **3.019 ns**、WHS **0.020 ns**，时序检查通过。
- 9452 LUT（17.77%）、46.5 BRAM（33.21%）、78 DSP（35.45%）；BRAM/DSP 与旧实现相同。
- RTL 仿真 **78 帧、51607 个输出像素**，覆盖全部滤波模式、CNN 极值/边界、AXI 背压、复位及连续无复位换块。
- 同一 64×64 内部块，完整返回约 232380 周期，compact 返回约 204063 周期，少约 **12.2%**；该数值不是实板整张图的加速比。
- 仿真与构建的 RTL 哈希匹配，部署包 24 个文件哈希和 ZIP CRC 均已检查。

此前旧版曾在两张样例上实测 CNN 平均约 765.8 ms/张。这个历史数值不能当作本次新版本实测结果，请以同板同次的四组对比为准。

## 源码与工程位置

- `hardware/`：本次冻结的 Verilog、ROM、testbench、构建 Tcl、PYNQ-Z2 board files。
- ZIP 内 `pynq/`：板端驱动和四方案测速脚本。
- ZIP 内 `python/ctdenoise/`：整数数学参考，不依赖 PyTorch。
- ZIP 内 `verification/`：时序、资源、DRC 和 RTL 仿真记录。

直接上板只需 ZIP。若要重新构建，用 Vivado 2018.2 执行 `hardware/build_ct_denoise_tile.tcl`，
Tcl 参数依次为：新的绝对输出目录、`hardware/board_files/pynq-z2` 的绝对目录、`build`。
构建失败或时序不通过时不要使用生成的 bitstream。当前交接版不要求同学重新构建。
