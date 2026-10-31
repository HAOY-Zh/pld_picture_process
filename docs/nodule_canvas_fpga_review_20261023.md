# 结节统一计算版：实板结果与下一轮优化审查

后续进展：回传已独立验收；直接输入打包和精确查表迁移的本地检查及真实全网 RTL 仿真已通过，新独立测速包已交付。见 [直接输入与精确查表加速](肺结节直接输入与精确查表加速_20261023.md)。下文保留审查时的发现和当时状态。

本页根据用户粘贴的 3 次实板输出及本地代码、冻结包进行审查。完整实板结果文件尚未回传，不能把控制台汇总当作已完成独立文件验收。

| 同一完整检测平面 | ARM 双核 INT8 | PL INT8 |
|---|---:|---:|
| 上一版分块方式 | 2570.148 ms | 2240.628 ms |
| 统一计算方式 | 1692.721 ms | 1778.761 ms |

PL 时延减少 20.61%，原/新比值为 1.260；ARM 时延减少 34.14%，原/新比值为 1.518。新 PL 比新 ARM 耗时多 5.08%。全部控制台样本通过有效头部及候选一致性断言。这不是本次优化后已经获得 PL 相对 ARM 的加速。

## 1. 优先迁移已有的精确拼接查表硬件

当前 `pynq/nodule_benchmark.py:hardware` 加载冻结父包中的 flow-v1 bit。其 SHA256 为：

`20e22607b5f58a12b63ac011c5c814f996b052823a0b7f7c2bd4ba340c14bd07`

肺分割最终 exact-LUT bit 位于 `data/lung_lut_20261017/lung_lut/`，SHA256 为：

`54f54656e405eb4cee0ce073d476665110ee49ef6f200ccdaf6edc814a85cafc`

本地逐文件比较两套 RTL，差异只有 `lung_tensor_burst_engine.v`。结节包并没有迁入分割最终版的这项加速。当前 flow 引擎已包含窗口递推、缓存/响应直通、池化双行缓存等优化，不应重复把它们列作尚未实施的新策略。

原 flow 对拼接分支的每个输出编码执行 Q31/RNE 重缩放。exact-LUT 每条对齐拼接命令开始时，用同一重缩放器生成两个分支各 256 项的完整映射表，再以编码查表。保留全部 UINT8 取值的原舍入与饱和结果；不是 INT4、减少量化级数或修改阈值。

结节网络的 `cat1`、`cat0` 具备相应通道对齐条件。已有 bit 的路由记录为 100 MHz、setup WNS +0.034 ns、hold WHS +0.017 ns，80 DSP、31.5 BRAM tile；另有肺分割实板输出验收。但这些证据不能代替结节参数下的新整网验证。

下一步应生成独立对照包，绑定已验收 bit/HWH 和原结节参数，不覆盖原父包；验证两次拼接的所有 256 个输入编码，并跑实际结节全网 RTL/板端逐值对照。若只迁移该硬件优化，不需要重新训练。收益须从实际张量计数和整层延迟核实，不能套用肺分割的加速比。

## 2. 恢复直接生成 FPGA 输入格式的本地路径

上一版 `nodule_fast_batch.py` 使用 C++ 直接把 HU 查表、stride-2 展开及 HWC8 排布写入输入 CMA。

统一计算版 `nodule_canvas_pl.py` 先用 NumPy 查表、补边、stack 生成 45 通道 CHW；随后继承的 `lung_fast.py:predict_codes` 又调用 `feature_words`，分配 HWC8 临时数组、逐通道重排并复制到 CMA。ARM 网络直接接受 CHW，因此这段额外重排主要影响 PL 路径。

可新增支持任意对齐矩形的 C++ 打包函数，直接写入输入 CMA。保持 HU 全范围查表、原点补零、通道顺序和三个尾通道完全一致。验收需要全 INT16 值、边界、只读/带跨度输入和真实 448×448 逐字节比较；其后再对全部节点和最终候选核对。无需更改网络或重新标定量化。

## 3. 用已有实板记录定位剩余耗时

当前 `result.json` 已保存分阶段明细，无须重跑测速：

```python
import json
from pathlib import Path
r = json.loads(Path('/home/xilinx/ct_test/nodule_canvas_speed_01/result.json').read_text())
for name in ('canvas_ARM', 'canvas_PL'):
    print(name, r['results'][name]['mean_ms'])
for i, row in enumerate(r['results']['canvas_PL']['samples'], 1):
    d = row['details']
    print('PL sample', i, d['timing_ms'])
    print('hardware-active ms', d['pl_command_cycles'] / (r['clock_mhz'] * 1000))
    print('commands', [(x['node'], x['cycles']) for x in d['commands']])
```

硬件活动周期包含总线等待和写应答，不是纯 MAC 时间，也不能用模拟周期代替这份实板计数。

- 先区分采样、量化展开、HWC8 转换/输入同步、网络执行等待、保护区/结果同步、解码。
- 若硬件周期已经很短，继续加 MAC 的端到端收益受限，应优先优化 PS–PL 输入/读回和控制。
- 若拼接占明显比例，优先 exact-LUT。
- 若卷积自身占主导，再依据逐层计数设计像素重叠流水、窗口行缓存和读请求流水；这些需要新 RTL、仿真、时序及物理板验证，不能预报确定倍数。
- 当前运行时会 invalidate 全部中间分配以检查保护区。可评估保护区/输出范围同步，但必须先确认板端 PYNQ 的范围同步语义，保留全部保护检查与超时内存保留，不能简单删掉同步或保护区以提高数字。

电脑回传完整现有记录：

```powershell
& "C:\Program Files\Git\usr\bin\scp.exe" -r xilinx@192.168.2.99:/home/xilinx/ct_test/nodule_canvas_speed_01 "D:\pld_picture_process\data\"
```

回传后运行 `python/verify_nodule_canvas.py --result data/nodule_canvas_speed_01` 独立核对实际输出、身份、统计和加速比，再依据原始分阶段数据确定实现顺序。

## 4. 与肺分割比较的正确口径

`docs/lung_all_accel_results_20261017.md` 的最终剪枝模型：同模型 ARM INT8 797.458 ms、PL INT8 492.401 ms，硬件比值约 1.620；原 FP32 到最终 PL 的 3.262 倍包含模型压缩和数值格式变化。

这些主要是网络执行记录。当前结节平面的 1778.761 ms 还包含在线 CT 采样、输入准备、候选解码和平面合并。需要同时给出网络与完整平面两个口径，不能用其总时间直接推断卷积引擎没有加速，也不能用不同网络或 FP32/INT8 混合比值掩盖当前 PL 总时间略慢于 ARM 的事实。

本次仅完成审查和记录，尚未交付或实测以上新增优化。此前统一计算版的 75 例精度一致性证明仍属于该已交付版本；新执行路径必须另行核对。
