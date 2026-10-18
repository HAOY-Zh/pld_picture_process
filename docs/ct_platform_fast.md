# 基础处理往返加速与部署

本轮将六种基础处理合成一个常驻bit，换方法只写模式寄存器；增加板端切片输入缓存和无损压缩二进制传输；网页任务完成后立即返回结果，不再等待下一次800ms状态轮询。隐藏的三维工作区暂停刷新，基础图像重绘复用已生成的像素缓存。

当前电脑工作台仍为 http://127.0.0.1:8090/ 。最终升级包为 `data/ct_platform_fast_20261017_v2/ct_platform_fast/`，约19.8MB，共45个哈希绑定文件。首次候选目录`ct_platform_fast_20261017`不用于本轮部署；旧的`ct_platform_20261017_v2`保留备用。

## 板端升级

电脑PowerShell执行：

```powershell
& "C:\Program Files\Git\usr\bin\scp.exe" -r "D:\pld_picture_process\data\ct_platform_fast_20261017_v2\ct_platform_fast" xilinx@192.168.2.99:/home/xilinx/ct_test/
```

在原来运行`platform_service`的同一个板端Jupyter内核中执行：

```python
%cd /home/xilinx/ct_test/ct_platform_fast
from ct_platform_fast_service import upgrade
platform_service = upgrade(platform_service)
```

`upgrade`先核对新包和凭据，再检查旧服务空闲及DMA状态，随后交接唯一FPGA所有者。原有任务运行中或硬件故障时会拒绝交接；如果报错，应保留完整输出排查，不要反复调用或重置内核。

服务会执行12组真实基础PL校验、分割网络21节点校验，最后预加载共享基础bit。看到`CT FAST PLATFORM READY`后，刷新网页，点击左侧“连接新版基础加速服务”。这一步只核对身份并切换电脑端通信契约，不再次下载bit、不清空现有CT及分割结果。

若提示`platform_service`未定义，说明当前内核没有持有旧服务，应回到原服务所在内核执行交接，不能另外启动第二个所有者。

## 加速范围与计时

| 环节 | 旧版本 | 本轮 |
| --- | --- | --- |
| 六种基础方法之间切换 | 不同bit，实测重载约2～3秒 | 同一bit，只写GPIO模式寄存器；不进行完整重配置 |
| 同一切片调参数或换基础方法 | 每次发送524288字节HU | 首次无损压缩上传；后续发送哈希引用，图像请求体为0字节 |
| 基础结果返回 | 262144字节图像再转Base64和JSON | 直接返回无损压缩二进制图像，另附绑定请求的回执 |
| 页面等待完成 | 下一次800ms轮询发现结果 | 后台完成即返回；超出等待窗口时仍保留状态查询，不重发硬件任务 |
| 页面绘制 | 隐藏三维页仍随状态刷新；重复构建像素图 | 懒加载并暂停隐藏页刷新；复用已生成图像的绘制缓存 |
| 窗映射硬件 | 保守握手封装 | 精确整数的八级商位流水线，支持背压；不改整数映射规则 |

只缓存输入，不缓存计算输出。每次请求仍实际执行PL、保存输出，并与独立参考逐像素校验。缓存最多保留8张512×512 int16切片（原HU约4MiB），以内容SHA区分输入。只有明确返回“输入缓存未命中、尚未执行硬件”时才补传完整图像；忙碌、超时、硬件故障都不会自动重发DMA。

基础处理与智能分割仍使用不同bit，二者之间切换的重配置开销继续单独计时。分割模型、分割bit、量化和预后处理保持原契约。服务初始化含自检和基础bit预加载，启动时间单列在健康回执`startup_ms`中，不把它隐入硬件FPS。

## 本地验证与待上板部分

- Vivado2022.2实际完成综合、布局布线、bit生成和时序验收：100MHz，WNS=0.842ns、WHS=0.026ns，均无失败端点，无未约束内部端点。资源为5174 LUT、6755寄存器、3 BRAM、3 DSP。
- RTL小图覆盖1858帧、475648像素，包含所有int16 HU值、奇数及极端窗参数、模式切换、连续帧和AXI背压。
- 原生512×512仿真覆盖18帧、4718592像素，包括真实`LCTSC-Test-S1-101`第44层及边界输入，六种操作均与独立参考一致。这是RTL仿真，不是新bit已上板运行的声明。
- 74项相关Python回归全部通过，日志位于`data/ct_platform_fast_final_checks_20261017/python_tests_verified.log`。初次并行构建期间一项Torch冷启动超时被保留在初始日志中，测试随后按生产一致的2线程设置复核。
- 快速协议合成浏览器验证见`data/ct_platform_fast_browser_20261017_v2/browser_TEST_ONLY.json`，包括6种PL协议方法、2种明确的电脑参考、切片/病例隔离、形态学显式输入、三维及窄屏布局。其约60～150ms的点击至显示仅衡量本机假硬件协议，不是实板测速。
- 已有130层真实分割会话在电脑端更新时保存恢复，Dice、实际头回执、HU和各阶段掩膜已对比。原结果不会标为本轮新硬件推理。

板端SSH目前没有免密权限，因此还需要执行上述Jupyter升级，才能验证新bit的真实运行与实际往返加速。不会用本地仿真或假硬件数字代替实板结果。

## 真实电脑到板端测速

升级成功、网页没有其他计算任务时，在电脑项目根目录运行：

```powershell
& .\.venv-lung\Scripts\python.exe -X utf8 python\ct_platform_roundtrip_benchmark.py --config data\ct_platform_fast_20261017_v2\host_config.json --ct D:\CT2\lctsc\LCTSC-Test-S1-101 --slice 44 --repeats 3 --out data\ct_platform_fast_actual_20261017\roundtrip.json
```

它对同一真实CT切片执行21次PL请求，包含同方法重复与六模式轮换，保存所有实际输出和回执，逐像素对照参考。每次分别打印电脑到板端往返、DMA/PL、bit切换、输入缓存命中。已有输出文件会拒绝覆盖；需要重测时使用新的输出文件名。

原网页实际回执可作范围说明：一次腐蚀DMA/PL约18.206ms，重配置约2850.838ms，往返约3016.871ms；另有同窗方法复用时往返约150ms。正式新旧对照应分别使用相同CT、相同参数、冷/热状态和计时范围，不能直接拿一次含重配置的旧请求与一次预热的新请求宣称纯计算提速。

启动时生成的`platform_startup_*.json/.npz`也可回传到电脑，用既有验证器核对：

```powershell
& .\.venv-lung\Scripts\python.exe -X utf8 python\verify_ct_platform_board.py --package data\ct_platform_fast_20261017_v2\ct_platform_fast --report <回传的platform_startup文件.json>
```

验证器会核对12组实际基础图像和切换后分割头；该启动自检与真实CT网页往返测速各有独立证据。
