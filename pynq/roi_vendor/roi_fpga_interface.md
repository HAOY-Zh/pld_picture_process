# ROI1 接口约定与给 B 的接入说明

版本 roi-filter-1.0.0，日期 2026-10-19。根据 10 月 18 日三人分工文档实现。
这是已实现的 ABI v1 提案，供 B 核对后固定到统一服务；不表示 B 已经确认或整机已联调。

## 模块语义

```text
keep = (y0 <= y < y1 AND x0 <= x < x1)
       AND (!use_lung_mask OR lung_mask == 1)
       AND (!use_hu OR low <= HU <= high)
```

输入原生单层 HU，支持 H、W 分别为 1—512。首版已对完整 512×512 做 RTL 仿真。
输出原生同尺寸 uint8 0/1 掩膜，ROI 外为 0，count 为该掩膜求和。
不裁剪硬件输出，不在 PL 做缩放。截图中的 ROI 局部放大由 B 的电脑界面用原图
索引范围显示；本包仅提供独立的 CPU 显示样例。

HU 按 np.rint 舍入到 int16，先检查有限值和转换后范围，拒绝溢出回绕。
软件参考和 FPGA 必须使用同一份舍入后的 HU。不能传窗宽窗位后的显示灰度。
肺 Mask 接口为同网格 uint8 0/1；边界适配器可明确接收 0/255 并规范化，拒绝
任意非零以及混合 0/1/255 编码。bool 可显式转换为 0/1。None 表示关闭肺条件，
不能用全零数组冒充未计算的肺分割。

ROI 为 `(y0,y1,x0,x1)`、左闭右开，允许空范围；越界、反向和非整数报错。
HU 阈值为 int16 整数，包含两个端点；use_hu=False 可以不传 hu_range。
HU 筛选和肺 Mask 是独立开关。空结果 count=0，HU min/max/mean 为 null。

## 流 ABI

32 位 AXI4-Stream，一个 beat 一个像素，按 C-order 逐行扫描。两路 DMA 均为 Simple。

| 项目 | 定义 |
| --- | --- |
| 输入 word [15:0] | 有符号 int16 HU 的二进制补码 |
| 输入 word [16] | 当前像素肺 Mask，0/1；关闭时填 0 |
| 输入 word [31:17] | 必须为 0 |
| 输出 word [0] | 筛选结果 0/1 |
| 输出 word [31:1] | 0 |
| TKEEP | 0xF，输入输出一致约定 |
| TLAST | 整帧末像素一次；不是视频逐行 TLAST |
| 每方向帧长 | H×W 个 uint32，即 H×W×4 字节 |
| 512×512 每方向帧长 | 1,048,576 字节，不包含额外计数 trailer |
| DMA 长度寄存器 | 构建脚本设置为 23 位 |

count/status 走 GPIO，不追加到输出流，因此 DMA 接收长度与输入相同。
配置在首个输入握手锁存，帧内更改不会影响当前帧。计数在每帧开始清零；
最终 count/status 在最后一个输出握手后发布。在 busy=1 时，count 可能仍是上一帧，
不能读取并当作当前结果。不同帧之间有结束握手保护；帧内无背压时每周期处理一像素。

输入 TLAST 错误、TKEEP 错误、保留位非零会置错误位。错误帧不能当作合法结果使用。
如果未发 TLAST 且 DMA 卡住，模块不自行复位或回收资源，统一服务按故障流程处理。

## 配置和状态 GPIO

GPIO 每通道 32 位，第二通道 DATA 寄存器偏移 0x8。地址由同一次 HWH 读取，
不硬编码物理基址。下面是 AXI GPIO 的 DATA，不是另建的 AXI-Lite 寄存器 IP。

| IP 名称 | channel1 | channel2 |
| --- | --- | --- |
| roi_geometry | 输出：低16位 W，高16位 H | 输出：bit0 use_hu，bit1 use_lung_mask，其余0 |
| roi_bounds | 输出：低16位 x0，高16位 x1 | 输出：低16位 y0，高16位 y1 |
| roi_hu_status | 输出：低16位 low，高16位 high，均int16补码 | 输入：status |
| roi_count_identity | 输入：uint32 count | 输入：0x524F4931，ASCII ROI1 |

status：bit0 busy，bit1 done，bit2 配置错误，bit3 TKEEP 错误，bit4 TLAST/帧长错误，
bit5 输入保留位错误，bit[31:24] ABI主版本1，其余为0。成功完成为 `0x01000002`。
done 和错误保持到下一帧第一个输入握手；不能仅凭旧 done 立即返回，必须等本次 DMA 完成。

## B 注入资源的调用方式

```python
from ct_roi_processor import CtRoiProcessor

# 这些对象必须属于 B 的唯一资源管理层；示例不自行加载 Overlay。
processor = CtRoiProcessor(
    dma=overlay.axi_dma_0,
    geometry_gpio=overlay.roi_geometry,
    bounds_gpio=overlay.roi_bounds,
    hu_status_gpio=overlay.roi_hu_status,
    count_identity_gpio=overlay.roi_count_identity,
    input_buffer=manager_input_uint32_buffer,
    output_buffer=manager_output_uint32_buffer,
    clock_hz=manager_confirmed_clock_hz,
)
result = processor.process_slice(
    hu_slice_i16,
    roi_yx=(10, 120, 20, 180),
    hu_range=(-500, 300),
    use_hu=True,
    lung_mask=lung_mask_u8_or_none,
)
```

两块借用缓冲区必须是不重叠的一维连续 uint32，容量至少 H×W。适配器不分配或释放
CMA，不加载 Overlay，不改 FCLK，不启动 HTTP 服务。B 在切 bit 后重建绑定对象，
串行化所有模块访问并验证 bit/HWH 哈希。identity 仅为 ABI 自检，不能替代资产哈希。
适配器从接收 DMA 先启动，使用 nbytes 限定本帧长度；返回数组拥有独立内存，后续 DMA
不会覆盖。可向构造器注入 `wait_dma(dma, timeout_seconds)`，以复用 B 的等待策略。
默认等待检查 DMA 错误和 timeout，完成后调用 PYNQ wait 进行接收缓存失效处理。

超时、DMA 错误或输出协议不符时抛异常并将实例标为 faulted，保留借用内存；不会自动
重试、释放可能仍被 DMA 使用的缓冲区、切 bit 或改成 CPU。恢复由 B 的管理层执行。

## 结果与三维汇总

结果含 filtered_mask、valid_count、backend、status、module_version、algorithm_id、
params_version、实际 ROI/阈值/模式、clock_hz 和 timing。clock_hz 是管理层报告值，
不是本模块测得频率。FPGA 时间 `dma_compute_seconds` 包含 DMA 传输和等待，不拆出
不存在的纯 PL 实测值。CPU HU 统计来自返回掩膜，明确标为 CPU_FROM_RETURNED_MASK。

`python/ct_roi_reference.py::roi_analyze` 仅在 PC 上遍历指定 z 范围，process_slice 回调
由 B 注入真实服务请求。它不新增网络协议，不把整卷送入板卡。输出原生 ZYX 全尺寸
掩膜，范围外为0。只有明确传入正交规则等间距几何并置 regular_orthogonal=True 才
计算面积/体积，否则物理量为 null。筛选体积不命名为精准结节体积。

空 z 范围返回 SUCCESS/count=0、backend=NOT_EXECUTED，表明没有调用硬件；与执行失败
以及实际执行但 count=0 不同。体积内禁止混用 CPU_REFERENCE 和 FPGA 后端结果。

## 暂未接入的部分

- A 的候选初筛规则尚未收到，未发明额外“结节规则”；当前 algorithm_id 仅为手动筛选。
- 当前本机团队仓库是较早快照，缺少 10/18 文档里的统一服务代码；GitHub 拉取因连接
  失败未取得新基线。此包为独立模块，没有修改公共网页或服务。
- 没有真实板端测试证据；已有脚本需要 B 的独占管理会话实际执行。
- ABI v1 由本包实现，尚待 B 确认兼容，不能声称整机已经联调通过。
