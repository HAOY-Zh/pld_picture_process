# CT Studio：LDCT 增强与后续分析集成

本版基于队友运行交付 `f2ab932`，合入 LDCT 独立交付 `8ccf920`。电脑侧已实现完整数据流；统一服务的真实板端验收待开发板可用后执行。独立 LDCT 已有的上板结果不替代本版整机验收。

## 操作者流程

1. **载入影像**：读取原始 HU 和毫米几何，可只载入一个短连续片段。
2. **LDCT 增强（可选）**：先预览当前层，或增强整个已载入范围。无板卡时明确选择“电脑整数参考”，使用主模型的实际量化权重计算，不伪造 FPGA 结果。
3. **使用增强影像分析**：全部载入切片增强完成后点击该按钮，或用公共“影像来源”选择器切换。
4. **基础处理、分割与三维**：所选来源的 HU 实际传给处理模块和 MPR。分割仍使用原有冻结模型和预后处理；电脑模式只浏览 MPR，不执行神经分割。
5. **记录与导出**：结果绑定病例、影像来源、HU 哈希、方法、模型及后端。没有配对参考时不显示 PSNR/SSIM，没有专家标注时不显示 Dice。

“原始 CT”和每份完整“LDCT 增强 CT”各有独立的处理结果、分割 Mask、三维状态和输出目录。切换来源会恢复对应结果，不能把原图 Mask 留在增强图上。增强不改变尺寸、坐标和几何；页面称为增强 CT，不将它标成真实正常剂量影像。

单层预览和未完成批次只用于查看，不把未增强切片混进增强体数据。需要分析少量切片时，可先只载入该连续范围，再完整增强。取消发生在当前层处理完成后，保留已完成输出。相同输入重新运行会生成新结果，不覆盖旧文件。

## 电脑端启动

首次在新克隆中按原交付文档准备 Python 3.11 虚拟环境：

```powershell
.\.venv-web\Scripts\python.exe -m pip install -r python/requirements-web.txt
.\.venv-web\Scripts\python.exe python/ct_studio_runtime.py install --board-url http://192.168.2.99:8765
.\.venv-web\Scripts\python.exe python/ct_studio_runtime.py verify --variant classic
.\.venv-web\Scripts\python.exe python/ct_studio_runtime.py verify --variant fast
```

运行包已经恢复的目录不再执行 install。该命令拒绝覆盖现有部署。

```powershell
.\.venv-web\Scripts\python.exe python/ct_workspace_web.py --desktop --port 8093 --output data/ct_workspace_ldct_20261018/results
```

打开 `http://127.0.0.1:8093/`。公开样例为 `data/ct_studio_demo/synthetic_ct_512.npz`，8 层、512×512，勾选 HU/mm 并关闭专家标注。样例是合成工程数据，适合检查流程，不是医疗效果评价。

默认 CPU 参考和 FPGA 主版均绑定 `configs/ldct_primary_model.json` 的 range-safe 模型及 100 MHz 发布目录；CPU 模式没有实际 PL 频率。`--ldct-release` 可显式指定另一份经过校验的发布目录。

## 生成新的统一部署包

先保留已恢复的 classic/fast 包。以下命令只创建新目录，若输出已存在，选新的目录名。

```powershell
.\.venv-web\Scripts\python.exe python/package_ct_studio.py --host data/ct_platform_20261017_v2 --variant classic --ldct-release handoff/ldct-slice-v1-20261017/releases/100mhz --outdir data/ct_studio_ldct_classic_02
.\.venv-web\Scripts\python.exe python/package_ct_studio.py --host data/ct_platform_fast_20261017_v2 --variant fast --ldct-release handoff/ldct-slice-v1-20261017/releases/100mhz --outdir data/ct_studio_ldct_fast_02
```

每个新目录有电脑配置 `host_config.json` 和板端子目录 `ct_studio/`。原模型、bit/HWH、嵌套清单和本机凭据继承自冻结包；增加 LDCT 后生成新的顶层部署清单。凭据仍只在本机忽略的 data 目录，不上传。

classic 是优先实板验收的回退平台；fast 的共享基础 bit 原本也待实板验收。两种版本都需要验证“基础处理 → LDCT → 分割 → 返回基础处理”。

| 模块 | 频率依据 |
| --- | --- |
| 当前 LDCT 主版 | 独立已验证 100 MHz，主模型不改为50 MHz |
| LDCT 回退版 | 单独保留的50 MHz包；选它需重新生成对应统一包 |
| 原六个基础 bit | HWH声明50 MHz，服务在切回时设置并读回检查 |
| 共享基础 bit | HWH声明100 MHz，尚需本版实板验收 |
| 肺分割 | 原冻结100 MHz路径，逐次核验时钟 |

## 板卡可用后的启动与验收

电脑硬件模式预后处理需要 `python/requirements-web-fpga.txt` 中固定的 Torch 2.8.0。板端仍使用其 PYNQ 环境，不复制电脑虚拟环境。

将新包的 `ct_studio/` 复制到板端**新的独立目录**，例如 `/home/xilinx/ct_test/ct_studio_ldct_classic_02/`。确认没有其他硬件所有者后，在专用 Jupyter 内核中：

```python
import os, sys
folder='/home/xilinx/ct_test/ct_studio_ldct_classic_02'
os.chdir(folder)
sys.path.insert(0,folder)
from ct_studio_service import start
platform_service=start(folder=folder)
```

若旧服务仍由原内核持有，先校验新包，再使用新模块的 `upgrade(old_service, folder=folder)` 进行健康空闲交接。它不用于故障恢复，不删除锁，不强行释放 DMA 缓冲。

新服务沿用 `/tmp/lung_live_z2.lock` 和请求锁。统一所有者负责 bit、FCLK0、DMA/CMA；LDCTProcessor 只借用资源。发生超时、部分启动、DMA/PL错误时保留资源及故障状态，不自动重试、复位或改用 CPU。

启动顺序包含原12组基础对拍、21节点分割校验、15组LDCT实际输出对拍和切回分割校验。只有全部实际检查通过后才发布 HTTP 服务，成功标志：

```text
CT STUDIO READY: basic + segmentation + LDCT actual startup checks PASS.
```

然后在电脑启动新配置：

```powershell
.\.venv-web\Scripts\python.exe python/ct_workspace_web.py --config data/ct_studio_ldct_classic_02/host_config.json --port 8093 --output data/ct_workspace_ldct_board_20261018/results
```

连接成功后才允许选择 FPGA。板端返回实际 HU、模型/bit/HWH身份、每帧前后频率、DMA/PL/拼图时间及切换时间；电脑另计往返。CPU参考、模块时间、通信和历史独立测试分别标记，不能混算。

将新的 `platform_startup_*.json/.npz` 和 `studio_ldct_startup_*.json/.npz` 一起取回，用各自验证器检查：

```powershell
.\.venv-web\Scripts\python.exe python/verify_ct_platform_board.py --package data/ct_studio_ldct_classic_02/ct_studio --report data/returned/platform_startup_UUID.json
.\.venv-web\Scripts\python.exe python/verify_ct_studio_board.py --package data/ct_studio_ldct_classic_02/ct_studio --report data/returned/studio_ldct_startup_UUID.json
```

另需用实际 CT 检查连续请求、往返时间、切换、取消、CMA/内存和网页端到端体验；界面和CPU测试通过不算这些项目通过。预测肺表面与体积需要真实分割结果，且明确已处理层范围。

## 电脑测试

```powershell
# 模拟协议/资源对象，验证新数据流、取消、故障保留、频率和版本；没有访问真实开发板。
.\.venv-web\Scripts\python.exe -m unittest discover -s python/tests -p test_ct_studio_ldct.py -v

# 此项验证电脑模式不导入Torch，独立进程运行。
.\.venv-web\Scripts\python.exe -m unittest discover -s python/tests -p test_ct_workspace_desktop.py -v
```

数值参考、图像恢复和CPU/FPGA标签的已有回归继续保留。独立 LDCT 发布包的 RTL、时序、50/100 MHz实板对照与旧模型结果位于 `handoff/ldct-slice-v1-20261017/`，没有改写。
