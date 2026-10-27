# CT Studio：PYNQ-Z2 医学影像处理平台

**当前默认发布主线：`codex/ct-studio-current-20261027`。** 本次以作者刚刚实际运行、实板验收的版本为准，从自己的 `58ec1d4` 备份继续整理，没有合入队友之后的修改。

包含通用 CT 处理、100 MHz CNN 低剂量增强、U-Net 肺分割、肺区约束的 2.5D 结节检测、ROI、MPR/三维联动和性能记录。肺分割与结节检测复用同一 FPGA 计算引擎。模型、INT8 参数、bit/HWH、Python/C++、Verilog 与完整配套运行包随仓库提供。

仅用于工程教学与算法演示，不作为临床诊断依据。真实医学影像和连接密钥不公开上传。

## 新电脑快速启动

需要 Windows x64、Python 3.11、Git，首次安装需要联网下载依赖。

```powershell
git clone --branch codex/ct-studio-current-20261027 --single-branch https://github.com/HAOY-Zh/pld_picture_process.git
cd pld_picture_process
.\setup_ct_studio_current.cmd
.\start_ct_studio_preview.cmd
```

打开 **http://127.0.0.1:8099/**。电脑浏览模式可导入 CT、浏览切片/MPR 和基础处理参考；真实 FPGA 推理需要先将本次安装生成的板端包配到开发板，再运行 `start_ct_studio_current.cmd`。不会自动用电脑神经网络冒充 FPGA。

**完整步骤：[当前版安装、接板与验收说明](docs/CT_Studio_latest_20261027.md)。** 安装生成 `data/ct_studio_current_20261027`，无需向作者索取私有 data 文件夹。必须使用 `python/requirements-current.txt`，特别是 NumPy 1.26.4；不需要 GPU。

## 当前版本的改动与结果

- LDCT：原传输加后台保存；压缩通信实测更慢，未作为默认。
- 肺分割：实际网页使用 V3；默认原三维清理。
- 结节：等价搜索与有限采样缓存；原检测模型、采样坐标、跨层评分和阈值不变。
- 文件夹选择：增加字节读取、超时和安全重试；同一文件不会重复计数。当前浏览器实际导入 300 个合成 DICOM 并逐文件核对通过。大病例也可直接输入本机路径，免去额外上传副本。

同一例 351 层 CT 的整例实测：LDCT **201.7 → 183.6 s**，肺分割与重建 **199.9 → 195.1 s**，结节检测 **80.2 → 72.7 s**。本次所有完整输出一致；计时包含准备、保存和后处理，不含首次读取/启动预热及浏览器绘制。这是单例完整对照，不是多病例统计，也没有宣称每项都达到 10%。

## 目录

| 目录 | 内容 |
|---|---|
| `python/` | 当前网页、图像输入、处理调度、ROI/三维/结节主机代码、安装器与测试 |
| `pynq/` | 板端所有者、实际硬件驱动、通信与 C++ 源码 |
| `hardware/ct_studio_current_20261027/` | 当前版 Verilog、Tcl 与板卡源文件快照 |
| `runtime/ct_studio_current_20261027/` | 无连接凭据、仅合成自检的模型/bit 完整运行 ZIP 与哈希 |
| `evidence/ct_studio_current_20261027/` | 无病例身份的版本来源、环境和验收摘要 |
| `docs/` | 最新安装/部署说明及明确标注日期的历史研发记录 |

历史文档描述的是各自日期的版本；当前运行入口、依赖和行为以上面的 2026-10-27 交付说明为准。
