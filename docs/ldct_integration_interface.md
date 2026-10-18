# 给 LDCT 模块负责人的接口约定

负责人按最新 [三人分工](三人分工与提交前推进计划.md)：LDCT负责人同时维护统一平台并负责整机集成/验收；原平台负责人转做肺结节候选，第三人做共用筛选/计数硬件。下文“我方统一服务”指平台服务层，不再表示原平台负责人继续承担集成；处理类与服务的资源职责边界保持不变。

这是后续统一网页/板端服务的**接口约定草案**，不是现有部署包已经提供的类。可直接基于现有 `run_frame_fast` 封装，无需重写 CNN、分块计算或网页。接口名称可以协商，但资源所有权、HU语义和时钟责任应保持一致。

## 1. 50 MHz / 100 MHz 怎么处理

**先交付当前已验证的50 MHz版本即可，100 MHz不是接入前提。** 当前各模块使用不同bit轮流占用PL，时钟由统一服务在切换模块时管理。

| 模块 | 当前应使用的频率 | 谁负责设置 |
| --- | ---: | --- |
| LDCT交接版 | 50 MHz，FCLK0 | 我方统一服务 |
| 肺分割已验证版 | 100 MHz，FCLK0 | 我方统一服务 |

切换顺序为：停止新请求→确认旧模块不再访问内存→释放旧模块资源→核验并加载新bit/HWH→在引擎未启动时设置目标FCLK0→读回实际频率并确认ARM频率未变→分配/绑定缓冲→运行自检→接受请求。

LDCT处理类**只读时钟、不设置时钟**，每帧开始/结束检查实际频率与清单一致，超出声明容差则报错；不能在处理过程中改频率、PLL或CPU设置。当前LDCT容差可沿用原驱动的0.5 MHz，实际值随结果回传。

现有统一服务的分割路径已经有100 MHz校验；LDCT接入时，我方还需新增按模块清单切换50/100 MHz的逻辑，并非现有服务已经完成这一适配。

如果后续做100 MHz版，请单独交付：

1. 100 MHz目标下重新综合、布局布线的匹配bit/HWH，不能仅把Python频率设置改成100。
2. 与该bit匹配的setup/hold时序、DRC和资源报告；setup与hold均通过后才使用。
3. 实际频率读回、实际输出数组及逐像素整数一致性结果，以及包含PS/DMA/PL/拼图的整帧实测。
4. 新版本号与全部哈希，保留50 MHz包。优先固定权重、输入和计时范围对照50/100 MHz；不能直接把现有523 ms除以2作100 MHz性能承诺。

未来若要把多个IP集成在同一bit中并行运行，需另外决定共同频率或多时钟域、时钟转换与验证范围；不要求本次交接先解决该问题。

## 2. 需要的板端 Python 调用接口

建议交付 `ldct_processor.py`，在板端可以如下调用。HTTP接口、网页、CT读取和三维重建由我方负责。

```python
processor = LDCTProcessor(
    dma=overlay.axi_dma_0,
    gpio=overlay.axi_gpio_0,
    tx_buffer=tx_buffer,
    rx_buffer=rx_buffer,
    model_dir=model_dir,
    read_clock_mhz=lambda: float(Clocks.fclk0_mhz),
    expected_clock_mhz=50.0,
    timeout_s=10.0,
)

result = processor.process_slice(hu, method="cnn")
enhanced_hu = result["hu"]
processor.assert_idle()
state = processor.get_state()
```

我方负责加载/核验Overlay、配置时钟、获取全局硬件锁和分配常驻CMA缓冲区。当前可传入两份4096元素、`uint32`、物理连续且未被其他请求使用的发送/接收缓冲。处理类借用这些对象，**不自行创建Overlay，不复位DMA，不释放外部缓冲，不重烧bit，不退出进程**。一次初始化后应能连续处理多张切片。

构造与模块导入不运行完整CPU参考，也不启动DMA。缓冲区通过统一服务串行使用，不要求处理类支持并发请求。

### 输入

- `hu`：有限值、非空的二维NumPy数组，单位为**原始HU**，支持`int16`或`float32`。
- 浮点输入沿用当前`np.rint`语义，检查范围后转`int16`；越界拒绝，不静默裁剪。
- 至少支持512×512，其他支持尺寸和边缘块规则写入清单；分块、halo、58×58核心和compact协议由处理类内部选择。
- `method`：字符串方法名，例如`cnn`、`gaussian5`；支持列表和硬件mode编号放入清单，未知方法应在启动DMA前拒绝。
- 不接受窗宽图、伪彩图代替HU。构造类时已知的模型与硬件版本在处理期间保持固定。

### 输出

```python
{
    "hu": enhanced_hu,             # int16，H×W，与输入位置完全对应
    "total_ms": 523.2,             # 示例字段，不是固定值
    "timing_scope": "pack+cache+DMA+PL+poll+stitch",
    "actual_pl_mhz_before": 50.0,
    "actual_pl_mhz_after": 50.0,
    "tiles": 81,                   # 此次实际块数
    "method": "cnn",
    "model_id": "明确的模型版本或检查点SHA256"
}
```

`hu`必须是独立可持有的数组，不能返回下次DMA会覆盖的缓冲视图。输出不做显示窗宽、不缩放、不改变像素位置；spacing/origin/direction由我方沿用原CT元数据。

`total_ms`从本帧打包到结果拼接完成，包含必要缓存同步、DMA、PL、轮询；不含文件读取、模型加载、bit下载、绘图或CPU参考核验。当前没有独立硬件计数器就不需要提供“纯PL耗时”；可选的分阶段计时需明确边界。

## 3. 空闲、故障与资源所有权

`get_state()`至少返回：

```python
{
    "busy": False,
    "faulted": False,
    "dma_may_access": False,
    "error": None
}
```

- 在可能启动任一DMA传输之前置`dma_may_access=True`，避免部分启动后抛异常却误判为安全。
- 只有确认两路DMA不再访问内存、传输完成且硬件状态无误，才能置False；`assert_idle()`不确定时应抛异常，不能通过复位伪造空闲。
- 超时、DMA错误、PL协议错误必须上报，保留不确定状态；不自动重试、自动复位或返回CPU结果。
- 正确性比较放在自检/测试入口，不应每帧先执行完整CPU CNN。计时和返回值中应能区分实际PL输出与电脑参考。
- 由我方统一服务在确认空闲后释放缓冲与切换bit。无需在借用资源的处理类中提供自行释放硬件的析构函数。

## 4. 交付清单中的必要信息

```json
{
  "interface_version": "ldct_slice_v1",
  "board": "PYNQ-Z2",
  "clock_domain": "fclk0",
  "clock_mhz": 50.0,
  "clock_tolerance_mhz": 0.5,
  "input_units": "HU",
  "output_units": "HU",
  "output_dtype": "int16",
  "max_internal_tile": [64, 64],
  "cnn_halo": 3,
  "preferred_cnn_core": [58, 58],
  "cnn_compact_protocol": true,
  "methods": {"cnn": 6},
  "bit_sha256": "填实际值",
  "hwh_sha256": "填实际值",
  "model_id": "填实际值"
}
```

示例仅展示字段，实际`methods`应列出该版本支持的方法，bit、模型、GPIO布局及返回协议必须匹配。保留当前已有模型参数、文件哈希、Python/PYNQ/Vivado版本、时序与验证记录即可，不需要另造一套重复配置。

## 5. 接口交付验收

1. 使用同一初始化对象连续运行两张512×512配对样例，输出与现有整数参考逐像素一致，且模型/bit哈希对应。
2. 覆盖边缘块、不整除核心尺寸、全部声称支持的方法，以及非法输入在DMA前拒绝。
3. 验证频率错误时不启动DMA，正常开始/结束频率一致；不要求故意在实板制造总线挂死，可用模拟对象验证超时/错误的所有权状态。
4. 回传实际FPGA输出NPY、逐次耗时、模型/硬件/时钟记录；完整CPU参考核验不计入速度。

优先交付50 MHz的稳定可调用接口；100 MHz升级、新训练模型和显式自适应强度可作为后续独立版本，不阻塞初版网页接入。
