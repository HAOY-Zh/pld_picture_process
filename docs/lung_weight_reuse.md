# 持续权重复用：一次加载，多像素计算

2026-10-15。新增的是 FPGA RTL，不是重新训练或修改肺分割算法。
本页保留权重复用阶段记录；后续窗口地址控制已实现并验证，见 [硬件窗口说明](lung_feature_window.md)。
仅用于工程教学与算法演示，不作为临床诊断依据。

## 解决的问题

上一阶段的组输入缓存每处理一个像素/输出通道块，都接收相同的卷积权重。
现在 `fpga/lung_mac_tile8_reuse.v` 把权重、bias、零点、重缩放参数保存在独立接口中，
同一输出通道块可以连续用于多个像素。每个像素仍调用原来的 64 路 MAC 和唯一共享重缩放器。
原计算核心和定点规则没有变，不需要重新训练、调整分割阈值或重跑测试集选参数。

这是替代上一阶段 `lung_mac_tile8_cached` 的另一种外围组织，不是把两者叠加使用。
权重仅存一个输出通道块；切换块/层时再加载。特征窗口当前由测试台提供，尚未自动从图像生成。

## 接口与状态

全部端口在同一个时钟域，低有效同步复位；数据仅在 valid 与 ready 同时为 1 时接受。

| 接口 | 内容与约定 |
| --- | --- |
| `w_*` | 512 位权重、FIRST/LAST；第一拍保存 8 路 bias/乘数/移位及输入/输出零点、output keep、ReLU |
| `weights_loaded` | 完整权重组加载后为 1；半加载/复位时为 0；不代表加载内容一定合法，应检查输出错误 |
| `s_*` | 64 位 UINT8 激活、8 位输入 keep、像素组 FIRST/LAST；**不再传权重** |
| `m_*` | 原 INT32 累加诊断值、UINT8 结果、通道 lane、LAST 和错误；支持背压 |

权重字节顺序仍为 `output_lane*8+input_lane`，lane 0 在最低字节。
读取是同步 BRAM，带弹性寄存器；气泡和输出停顿不能使权重与激活错位。
每个像素输入长度必须等于权重组长度，输入 keep 可以逐拍变化，边界激活必须用输入零点填充。

第一拍参数捕获后，其余加载拍的配置变化不影响驻留参数。权重加载与当前像素结果不能交叠：
当前像素的 LAST 输出被消费后，才接受下一组权重。下一组权重可以提前保持 valid 等待，
但不能阻止当前像素剩余输入前进；空闲时权重加载优先。也可以持续送下一像素并由 ready 背压。

默认深度 128，最大冻结模型输入组为 108 拍；参数允许 1..256。
复位只清有效位、长度和指针，不清 RAM；复位后必须重新加载，不能读旧内容。
缺失/重复 FIRST、早/晚 LAST、超过缓存深度都会标错；越界读返回零而不是读未初始化/回绕 RAM。
错误结果不得用于分割；未收到 LAST 的未完成事务需要生产者补齐或复位，并非超时自动恢复。
仍只允许一组像素计算在途，未实现双缓冲权重或多像素流水重叠。

## 全量验证与传输量

`python/lung_weight_reuse_vectors.py` 校验旧冻结向量哈希，将顺序改为
病例向量→输出通道块→XY，让同块权重连续复用。人工标注、医学 CT、模型权重不参与调参。
跨像素权重/配置不一致、错误标记、缺少结果等情况会拒绝转换，不静默去重。

- 48 组冻结/尾通道算子用例，139 个权重块。
- 8700 个像素组、367338 个激活拍、68046 个输出。
- 权重加载由 367338 拍降为 **5781 拍，减少 98.426%**。
- 只计激活/keep 与权重，估算载荷从 26815674 字节变为 3676026 字节。
  不含 FIRST/LAST、参数、总线封装和对齐；**不是实测 DDR 带宽或 FPS**。

真实 XSim 在 `reuse_run_v3.log` 逐输出比较 acc/code/lane/LAST/error；另外检查输入气泡、
输出背压、尾通道、稀疏输出、连续单拍组、参数第一拍捕获、半加载/半像素复位、
停顿时复位、加载越界、输入长短错误、非法 FIRST、当前输出未消费时禁止重载、
提前请求重载时仍能收完当前像素，以及随后使用新权重。

首轮测试台在改变非第一拍元数据时没有撤销 valid，导致重复发送权重，虽计算值相同但错误位正确报警。
已修正驱动；失败日志保留，不作为通过证据。最终版本补充独立加载者等待的无死锁检查，重新跑全量。
验收同时检查 Fatal/文件警告及精确非空 PASS 数量，不仅看 XSim 进程退出码。
155 项 Python 工程回归通过；真实 RTL 仿真额外执行，不计入 Python 数字。

`python/verify_lung_weight_reuse.py` 从新打包激活/驻留权重独立重算全部 68046 个累加与重缩放结果，
核验正式仿真/综合报告、旧核心哈希及 1182 个受保护医学/模型来源哈希。
最终审计记录为 `data/lung_weight_reuse_20261015/verification_v2.json`；早期审计/报告保留作历史。

## 资源与边界

正式 OOC 综合针对 `xc7z020clg400-1`，不打开/修改原 XPR 或 BD。
最终资源与时序以 `data/lung_weight_reuse_20261015/synthesis_v2/` 为准。

| 最终核心＋持续权重缓存 | 使用量 | Z2 占比 |
| --- | ---: | ---: |
| LUT | 3633 / 53200 | 6.83% |
| FF | 6191 / 106400 | 5.82% |
| DSP | 68 / 220 | 30.91% |
| BRAM Tile 等效 | 7.5 / 140（7 RAMB36＋1 RAMB18） | 5.36% |
| 10 ns 内部综合后 WNS / TNS | +1.273 ns / 0 ns | 非布局布线结果 |

128×512 位有效权重容量为 8192 字节，宽读端口导致实际 BRAM 占用大于简单按容量估算。
计算阵列仍是 64 DSP，唯一共享重缩放仍是 4 DSP，不是每个 MAC 各复制量化器。

只报告综合后内部 10 ns 约束；接口输入/输出延迟和 OOC 时钟来源尚未完整约束，未布局布线。
不能据此宣布整机 100 MHz、完整网络可放入芯片或端到端加速倍数。

下一步：特征/行窗口地址与边界控制→AXI/DMA 流桥与参数寄存器→pool/nearest/concat 和全网调度
→整机实现/路由→bit/hwh 与板端精度、延迟、带宽和内存实测。
暂时不要添加此模块到现有 BD 或把宽内部端口 Make External；它不是可直接接 32 位 DMA 的 AXI IP。
整例三维清理仍需考虑 PYNQ 共享 512 MB DDR，而不是因这一小块缓存已验证就认为整例体数据能放下。

## 复现

在项目根目录运行，依赖上阶段冻结向量/模型本地数据，源码仓库不会分发这些数据。
选新的输出目录，不覆盖已有证据；按自己的 Vivado 安装路径调整。

```powershell
.\.venv-lung\Scripts\python.exe python/lung_weight_reuse_vectors.py --outdir data/lung_weight_reuse_new/vectors
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -v
Push-Location data/lung_weight_reuse_new
& 'E:\vivado2\Vivado\2022.2\bin\xvlog.bat' --sv ../../fpga/lung_requant.v ../../fpga/lung_mac_tile8.v ../../fpga/lung_mac_tile8_reuse.v ../../fpga/tb_lung_weight_reuse.sv
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_weight_reuse -s weight_reuse
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' weight_reuse -runall -log reuse_run_v3.log
& 'E:\vivado2\Vivado\2022.2\bin\vivado.bat' -mode batch -source ../../fpga/synth_lung_weight_reuse.tcl -tclargs D:/pld_picture_process/data/lung_weight_reuse_new/synthesis_v2
Pop-Location
.\.venv-lung\Scripts\python.exe python/verify_lung_weight_reuse.py --root data/lung_weight_reuse_new
```
