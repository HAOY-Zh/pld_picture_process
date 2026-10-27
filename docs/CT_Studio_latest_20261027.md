# 刚刚验收版本：完整安装与接板（2026-10-27）

本次默认发布主线为 `codex/ct-studio-current-20261027`，以我们自己的 `58ec1d4` 备份为父提交，叠加当前实际运行版本。没有 merge、pull 或 cherry-pick 队友之后的修改。历史分支仍保留。

仅用于工程教学与算法演示，不作为临床诊断依据。

## 新电脑先打开网页

前提：Windows x64，已安装 Python 3.11、Git，首次安装可联网下载依赖。

```powershell
git clone --branch codex/ct-studio-current-20261027 --single-branch https://github.com/HAOY-Zh/pld_picture_process.git
cd pld_picture_process
.\setup_ct_studio_current.cmd
.\start_ct_studio_preview.cmd
```

打开 `http://127.0.0.1:8099/`。这是当前同一套界面的电脑浏览模式：可导入本机 CT、浏览切片/MPR；基础处理与LDCT增强仅支持实际FPGA执行。真实 FPGA 增强、分割及结节推理需要下述板端配对，不会自动用电脑神经网络代替。

也可以手动安装：

```powershell
py -3.11 -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-web\Scripts\python.exe -m pip install -r python/requirements-current.txt
.\.venv-web\Scripts\python.exe -B -X utf8 python/ct_studio_current_release.py install
.\.venv-web\Scripts\python.exe -B -X utf8 python/ct_studio_current_release.py verify
```

必须使用当前依赖清单，特别是 NumPy 1.26.4。GPU/CUDA 不需要；电脑只用 Torch 做已固定的输入量化、输出恢复等主机操作。模型推理由板端执行。

安装后，完整配套目录是 `data/ct_studio_current_20261027`。里面包含模型、INT8 参数、bit/HWH、冻结评分参数、电脑配置以及整个 `ct_studio` 板端目录，普通 Git clone 就包含恢复它们所需的公开运行 ZIP。

## 接入队友自己的 PYNQ-Z2

1. 安装默认生成的板端地址为 `http://192.168.2.99:8766`，可在生成的 `host_config.json` 中只修改 `board_url` 以适配实际 IP。
2. 把本次安装生成的整个 `data/ct_studio_current_20261027/ct_studio` 复制到板子，例如 `/home/xilinx/ct_test/ct_studio_current_20261027`。
3. 确保旧任务已经结束、板端所有者健康且完全空闲，再正常退出旧所有者。不能同时运行其他控制 DMA/Overlay 的 Jupyter 程序；故障中不得强行复位、释放缓冲区或重载 bit。
4. 在板端终端执行：

```bash
cd /home/xilinx/ct_test/ct_studio_current_20261027
sudo /usr/local/share/pynq-venv/bin/python3 -B -u ct_ldct_lossless_owner.py --folder "$PWD" --expected-manifest "$(sha256sum platform_manifest.json | cut -d ' ' -f1)" --port 8766
```

等待所有真实硬件启动自检通过，看到 `LOSSLESS LDCT WIRE READY`。这个名称表示服务具备可选接口，**当前网页 LDCT 默认仍走原始传输**，没有启用本次被实测否决的压缩路径。首次自检可能约 2～3 分钟；`active (running)` 本身不等于硬件已就绪。

5. 正常关闭电脑浏览模式对应的启动终端，释放 8099，保留已保存结果，然后在电脑运行：

```powershell
.\start_ct_studio_current.cmd
```

如原 8099 仍有任务，保留它并换端口，例如 `.\start_ct_studio_current.cmd --port 8100`。新版入口会拒绝把旧主机页面误认成本版。

### 为什么不能拿新电脑配置直接连作者现有板端服务

公开仓库不包含作者的连接密钥或真实自检医学图像。每次新安装都会生成自己的密钥，并用确定性合成自检重新绑定清单，所以服务包 SHA256 会与作者私有包不同；这不代表模型或 bit 被改了。必须让电脑配置和板端目录来自**同一次安装**，不能关闭检查或手改哈希绕过配对。

### 开机启动

把 `python/install_ct_studio_current_boot.py` 复制到板子，在原所有者空闲、确认新包配对后执行：

```bash
sudo /usr/local/share/pynq-venv/bin/python3 install_ct_studio_current_boot.py --folder /home/xilinx/ct_test/ct_studio_current_20261027 --enable-next-boot
```

它只安装并启用下次开机的服务，不停止或启动正在运行的硬件。重启前核对并禁用旧的所有者自启动，以及依赖旧所有者的 `ct-studio-ps-web.service`；不要多个服务抢同一端口或 FPGA。公开副本必须在接收方板子重新通过全部自检，不能把作者私有包的验收结果直接当作副本已经上板通过。

## 当前包含什么

- 通用处理：FPGA 窗宽/窗位、HU 阈值、二值腐蚀/膨胀/开/闭。电脑参考执行、直方图均衡与伪彩映射已删除。
- 100 MHz 轻量 CNN LDCT：原始传输与电脑后台保存重叠，原算法和输出不变。
- U-Net 肺分割：实际网页接入 V3，原三维清理算法保留，整例完成后自动展示清理后的切片叠加与三维。单层或部分范围保留清理前预览。可选 `structural-preview` 是另一项清理策略实验，不属于本次数值不变的速度对照。
- 肺区约束的 2.5D 结节检测：复用同一 FPGA 计算引擎，等价搜索、有限采样缓存，冻结跨层评分不变，原契约阈值仍为 0.6418673229668569；独立最终选择策略默认使用 0.7655561641768475，并可切到原档或 0.8738163050576156。已完成的真实结果可重新筛选，无需重跑CNN。
- ROI 定量、候选区域联动、MPR/三维显示、来源绑定与真实耗时记录。
- 文件夹导入恢复：读取单文件字节后上传；上传专用等待 120 秒；网络中断最多安全重试两次，相同编号必须名称/大小/SHA 完全一致，不重复计数。直接输入本机路径无需再复制上传，通常更适合大病例。

## 已经验证的范围

同一例 351 层 CT，旧/新完整任务各一次，全部输出相同：

| 功能 | 原版 | 当前版 | 耗时减少 |
|---|---:|---:|---:|
| LDCT | 201.691 s | 183.628 s | 8.96% |
| 肺分割与重建 | 199.903 s | 195.090 s | 2.41% |
| 结节候选检测 | 80.241 s | 72.727 s | 9.36% |

包含准备、请求、保存、对应后处理；不含首次 DICOM 读取、启动自检、首次 bit 切换预热和网页绘制。结节核对了 225 个完整检测平面、5753 个排序候选和原档最终 9 个候选。当前严格档会改变候选数量，不能将此历史精度/输出结论混作新阈值评价。没有新增专家标注评价，也没有本轮实测功耗降低的证据；三项并未都达到 10%。

上传修复还通过了当前浏览器的 300 个小尺寸合成 DICOM 完整导入与字节核验，不能据此保证所有大文件或浏览器异常都已覆盖。

源码与运行资产的来源、固定库版本、合成自检及校验信息位于 `runtime/ct_studio_current_20261027`、`hardware/ct_studio_current_20261027`、`evidence/ct_studio_current_20261027`。不会读取作者的私有绝对路径或自动恢复作者病例。

## 本次界面与视频说明

基础处理与 LDCT 不再提供电脑参考执行；ROI 的独立参考选项保留。无需 FPGA 的入口仍能完成输入与切片/MPR浏览，未连接时不会自动运行电脑模型。详细阈值取舍、验收边界见 [当前网页功能调整与阈值说明](当前网页功能调整与阈值说明_20261027.md)，拍摄见 [演示视频流程与配套旁白](演示视频流程与配套旁白_20261027.md)。统一 bit 尚未实施，当前各独立模块仍按需切换。

## 三维后处理与视频脚本更新

当前源码包含已接入8099与桌面版的等价三维加速：裁去全零外围后提取同精度网格，恢复原生坐标；两例清理与双网格生成约20秒降到9～10秒。掩膜及完整网格数组一致，原模型、bit与清理规则保持不变。该计时不含推理、存盘或浏览器显示，前面的351层整例记录保留历史口径。见[完整对照](三维清理与建模等价加速_20261027.md)。

会话恢复遇到暂时网络超时时，网页仍能显示已核验的保存结果并保持未连接状态；凭据或身份拒绝仍明确报错。新版[演示脚本](演示视频流程与配套旁白_20261027.md)以低剂量CT增强与后续分析为主线，包含板端、上位机与EXE介绍，ROI接在结节候选之后讲解。公开仓库采用前述Python安装入口；本机私下配对的EXE目录含连接凭据，未整包公开上传。
