# 现有基础PL与智能分割的统一网页集成

> 后续加速版已准备：六种基础方法共用bit、切片输入缓存、无损二进制传输及网页即时完成通知，见 [基础处理往返加速](ct_platform_fast.md)。最终包为 `data/ct_platform_fast_20261017_v2/ct_platform_fast`，已通过RTL及100MHz布局布线验收，实板升级和测速待执行。本页六独立bit方案保留为备用。

窗宽窗位、HU范围阈值、腐蚀、膨胀、开运算、闭运算已经有现成PL模块和上板程序。
本次复用它们的原bit/HWH，新增统一服务、真实DMA请求接口及网页调用；没有重写算法或运行Vivado。
电脑预览保留为明确选择的参考，不再代替基础硬件功能。

## 已准备内容

- 板端目录：`data/ct_platform_20261017_v2/ct_platform`，54个文件。
- 其中 `basic/` 是6套现有bit/HWH，`segmentation/`是字节不变的保留原模型分割包。
- `ct_platform_service.py` 是统一所有者，基础与分割不能并发占用PL；按任务在确认空闲后切换bit。
- `platform_token.txt`是新认证凭据，不输出或提交；电脑配置与板端共享这份新凭据。
- `platform_manifest.json`固定全部文件与每个bit/HWH身份；`verification.json`证明本机打包/接口检查及旧文件未变，不冒充新实板验收。
- 早期 `data/ct_platform_20261017` 是中间打包快照，未部署，不要上传它；使用本节的v2。

## 1. 电脑上传（PowerShell）

```powershell
Set-Location D:\pld_picture_process
& "C:\Program Files\Git\usr\bin\scp.exe" -r .\data\ct_platform_20261017_v2\ct_platform xilinx@192.168.2.99:/home/xilinx/ct_test/
```

## 2. 板端Jupyter启动统一服务

先保证旧测速已正常结束、没有其他Notebook直接访问PL。
如果原 `live_service` 正在运行，在持有它的原内核执行下面代码；健康且空闲时，辅助函数会确认所有命令完成，
停止旧HTTP服务、释放正常缓冲并交还文件锁。它拒绝处理忙碌或故障所有者。
如果旧服务未启动，可在专用新内核执行。

```python
%cd /home/xilinx/ct_test/ct_platform
from ct_platform_service import start, stop_idle_live_service

if "live_service" in globals():
    stop_idle_live_service(live_service)

platform_service = start(bind="0.0.0.0", port=8765)
```

如果提示另一个服务持有锁，回到它的原内核正常交还所有者；不要为绕过锁而删锁文件或直接结束故障内核。
其他Notebook绕过文件锁直接MMIO仍无法由服务阻止，需要保持独占。

启动会执行：

1. 原分割21节点/51命令自检。
2. 每个基础PL模块两组原生512×512输入，共12组，检查真实DMA输出与独立软件参考。
3. 从基础模块切回分割，再次逐节点检查，确认切换后链路正常。

只有看到 **`CT PLATFORM READY: 12 basic PL checks + segmentation 21-node after-switch PASS.`** 才点击网页连接。
启动会产生新 `platform_startup_<uuid>.json/.npz`，保存实际输出与时钟/切换/DMA回执。
首次启动可能编译已有MMIO辅助库并多次下载原bit；均不计入单算子DMA耗时。

如发生DMA/AXI超时，服务会保留Overlay、CMA和现场并拒绝继续请求；不自动释放、复位、切换或重试。
提供完整错误和启动证据后再判断如何恢复。

## 3. 使用网页

本机8090现已更新为统一服务配置。刷新 `http://127.0.0.1:8090/` 后：

1. “连接”会核对平台清单、6套基础bit及分割模型身份。
2. 加载原生512×512 CT，默认“FPGA PL”，选择基础方法并应用。
3. 显示实际PL输出与参考逐像素一致状态，保存实际结果和回执；若不一致，单独保存实际/参考，报错，不替换结果。
4. 转到智能分割页运行当前层/连续片段；统一服务在空闲状态切回分割，自检后再处理真实输入。
5. 再回基础页计算会切到对应基础bit，连续同方法请求则复用已加载模块及缓冲。

电脑端需重新启动时：

```powershell
& .\.venv-lung\Scripts\python.exe -X utf8 python\ct_workspace_web.py --config data/ct_platform_20261017_v2/host_config.json
```

基础模块原生支持512×512。其他尺寸会明确拒绝硬件请求，不静默缩放；可手动选择电脑参考。
形态学沿用之前上板测试：PS先用HU范围生成二值输入，腐蚀/膨胀/开/闭的形态学阶段由PL执行。
窗口和范围阈值本身在PL执行。直方图、伪彩目前只提供电脑参考，没有声称它们已在PL实现；LDCT仍未实现。
原HU、专家标签、模型和神经网络输入保持不变，基础输出不自动送入分割模型。

## 4. 回传并独立验收

电脑取回启动证据：

```powershell
& "C:\Program Files\Git\usr\bin\scp.exe" "xilinx@192.168.2.99:/home/xilinx/ct_test/ct_platform/platform_startup_*" .\data\ct_platform_20261017_v2\
```

然后选择实际的JSON文件名运行，不覆盖既有验收：

```powershell
& .\.venv-lung\Scripts\python.exe -X utf8 python\verify_ct_platform_board.py --package data/ct_platform_20261017_v2/ct_platform --report data/ct_platform_20261017_v2/platform_startup_实际编号.json
```

独立核验12组共3145728个基础输出像素及切回后的65536个分割码。启动合成自检通过后，
仍应在网页上用同一真实CT跑6种基础方法、一次分割及一次来回切换，保存会话/回执再确认整条链路。

## 性能与资源边界

- 每个基础回执分开记录PS准备、DMA处理、Overlay切换和完整通信往返；往返已包含前面步骤，不将其相加。
- 当前是一个FPGA串行切换多个现有bit，不是6基础模块加U-Net同时常驻或硬件并行。
- 神经网络推理时间与切换/重新自检时间分开；首次操作或跨模块操作不能只报内核耗时当交互延迟。
- 每套bit的资源独立，不能把不同bit的资源相加后宣称同时占用，也不能把剪枝计算量减少说成物理DSP减少。
- 仍复用原模型/cache-return分割版本，不自动引入有精度风险的剪枝模型；flow/查表历史基准不当作当前服务实测。

## 本机验证状态

新增平台测试检查6方法请求与分割切换、请求/参数/bit/输出哈希、忙碌拒绝、失败保留、错误输出不替代和DMA超时保护。
最终统一回归56项全部通过（21项工作台/平台/回传核验＋35项原实时分割），日志见 `data/ct_platform_checks_20261017_v2/python_tests.log`。
这些是明确模拟硬件的本机测试，不是新上板报告。浏览器测试也使用合成CT与FakeHardware，独立保存TEST_ONLY证据。
实际板端统一链路尚未验收：本轮只读SSH连接被 `192.168.2.99:22` 关闭，因此需要按上面步骤在Jupyter启动。
# 基础处理加速更新

六种基础处理共用bit、输入切片缓存、无损二进制传输和网页即时完成通知已准备，部署见 [基础处理往返加速](ct_platform_fast.md)。最终包为 `data/ct_platform_fast_20261017_v2/ct_platform_fast`；本页原六独立bit部署保留为备用。新bit已通过RTL与100MHz布局布线验收，真实板端升级与测速仍需执行。

