# 硬件图像窗口控制与卷积链路

2026-10-15：PL 地址控制器现在能从特征读取接口生成卷积窗口，接入已验证的权重复用/64 路核心。
本页保留窗口控制阶段记录；后续特征缓存与 AXI4 读通路已独立实现并验证，见 [缓存/AXI 说明](lung_cached_axi.md)。
不更改模型、CT、人工标注、分割阈值、显示界面、BD 或 bit/hwh。
仅用于工程教学与算法演示，不作为临床诊断依据。

## 已完成的硬件

`fpga/lung_feature_window.v` 接收尺寸和布局命令，按 Y/X、kernel Y/X、输入通道块遍历。
支持 stride=1 的 1×1 无 padding、3×3 padding=1，以及 3×3 valid（padding=0）卷积。
H/W=1..512，输入通道=1..96；每次读取 8 通道，最后一个通道块用 keep 屏蔽尾槽。
padding 位置输出输入零点，而不是数字 0，**不发内存请求**。

`fpga/lung_conv8_window.v` 将窗口接入不改动的 `lung_mac_tile8_reuse`。
每个命令计算一个输出通道块的整张特征图，跨像素复用权重，输出包含 X/Y、lane、像素 LAST、帧 LAST。
最后一个结果被下游消费后才发 done，禁止帧中途重载权重或接受新命令。
读错误会补真实零值、标记受影响的像素结果，并让帧 error 置位；不能把该帧当有效分割。

测试台现在提供普通特征存储与读取延迟，**不再提前准备窗口并送给硬件**。
Python 生成的窗口仅作为 scoreboard 独立参考，实际送给卷积核的窗口来自 RTL。

## 特征布局与读取接口

一拍是 64 位的 HWC8 特征字，lane 0 为最低字节：

```text
word_address = base + ((y * width + x) * ceil(channels / 8) + channel_block)
```

这里 `base/req_addr` 是 **64 位字索引，不是物理 DDR 字节地址**；后续 AXI 桥需要转换、
检查实际分配缓冲区及物理地址范围。当前 32 位逻辑字地址支持地址末尾等于 2^32，超过则拒绝。
原软件 CHW 特征必须转成此布局；这是数据排列变化，不改变数值。
输出通道块起始编号由外围调度器保存，通过 lane 加起始编号得到实际通道。

| 接口 | 含义 |
| --- | --- |
| `c_valid/c_ready` | 捕获 base、H/W、Cin、kernel3、padding；非法配置以 done+error 结束，不发读/结果 |
| `req_valid/req_ready/req_addr` | 发一个特征字读取，valid 未接受时地址必须稳定 |
| `rsp_valid/rsp_ready/rsp_data/rsp_error` | 对应读取的返回；可以有可变延迟；返回 valid 保持到 ready |
| 窗口数据/keep/FIRST/LAST | kernel-major、C8-minor 顺序，与驻留权重顺序一致 |
| `m_x/m_y/m_lane/m_last/m_frame_last` | 输出位置和通道，背压时与 acc/code/error 同时保持 |
| `busy/done/error` | 在途、单拍完成通知、帧错误状态；新命令重新初始化帧错误 |

一个时钟域、低有效同步复位，**只有一个内存读请求在途**。当前按小步状态机等待响应，
没有读超时、乱序返回或多笔 outstanding。复位必须与内存响应服务共享取消语义；
未来 AXI 桥必须排空/隔离旧事务再重启，不能只复位地址计数器后让旧响应混入新帧。
内存不返回响应时需要外围超时/复位处理，本模块不会凭空完成。

## 缓存和性能边界

本轮完成的是**窗口地址/边界控制和读取协议**，不是三行缓存、DDR 控制器或完整 U-Net。
没有把整张特征图放到 BRAM；读取服务在仿真中位于测试台，实机后续放片上特征缓存/DDR。
最大 512×512×96 的 HWC8 特征占 24 MiB；整网还要考虑跳连、输入输出和 Linux 共享 DDR。

相邻 3×3 窗口会重复读取特征，这个正确性优先的单请求实现不能当高带宽设计。
下一阶段需要特征/行缓存与突发读取，以及读写总线桥；不能将一条随机 word 请求直接接到
原 32 位 DMA 的顺序 AXI-stream 输入上。也不能把所有内部宽端口 Make External。

## 数值、地址与协议验证

正式向量：45 组冻结算子向量＋8 个额外用例，含非方形、1×1/单像素、薄图、Cin=96、
Cin=11/Cout=13 尾通道、非零输入零点、3×3 valid。

- 147 个输出通道块任务，**68849 个输出全部逐项一致**。
- **371974 个窗口拍**的数据、keep、FIRST/LAST/帧 LAST 和 XY 全部一致。
- **312018 次真实模拟内存请求**地址全部一致。
- 59956 个 padding 拍不读内存；6055 拍权重加载。

`tb_lung_conv8_window.sv` 除黄金值比较外，还测试可变内存延迟、请求/窗口/输出背压、
命令及权重参数捕获、帧中重载/命令拒绝、非法尺寸/通道/padding/地址范围、
在途内存请求共享复位、停顿结果复位、内存错误标记、帧完成等待最终输出握手。

`tb_lung_feature_window_bounds.sv` 独立测试 512×1×96、1×512×96 全部地址，
最高 word 地址 0xffffffff，以及 512×512×96 的第一行和下一行前缀/地址溢出拒绝。
**没有宣称完整 512×512×96 全部窗口已仿真，也没有用这一边界测试代表医学精度。**

159 项 Python 工程测试通过；两套真实 XSim 额外执行，不计入 Python 数字。
`verify_lung_window.py` 从 HWC8/权重存储重新构造 CHW/OIHW 卷积，独立复算所有输出、
窗口、坐标和地址，并验证非空精确 PASS 数量、正式资源报告及 1182 个受保护来源哈希。
原权重复用/数学规则源文件未变；未重新训练或访问独立测试集/COVID 进行参数选择。
正式证据：`data/lung_window_20261015/verification.json`、`window_run.log`、`bounds_run.log`。

## Z7020 独立综合

| 窗口控制＋权重复用＋卷积 | 使用量 | Z2 占比 |
| --- | ---: | ---: |
| LUT | 4020 / 53200 | 7.56% |
| FF | 6439 / 106400 | 6.05% |
| DSP | 70 / 220 | 31.82% |
| BRAM Tile 等效 | 7.5 / 140（7 RAMB36＋1 RAMB18） | 5.36% |
| 10 ns 内部综合后 WNS / TNS | +1.273 ns / 0 ns | 未布局布线 |

层级报告：MAC 阵列 64 DSP、共享重缩放 4 DSP、窗口地址控制 2 DSP。
BRAM 仍只存一个权重块，不含整张特征图或未来行缓存。
OOC 报告没有完整接口延迟及最终时钟来源约束，不能据此宣布整机 100 MHz、FPS、DDR 带宽或网络资源达标。
正式报告在 `data/lung_window_20261015/synthesis/`，未打开原 XPR/BD，未生成 bit/hwh。

下一步：读写总线桥与特征/行缓存→输出打包和层间缓冲→pool/nearest/concat 与全网调度
→完整实现/路由→比特流及板端逐输出/分割精度、内存、带宽和延迟实测。
现阶段用户不需要修改 BD 或烧录。

## 复现

在根目录执行，选择新目录不覆盖已有证据，依赖本地冻结整数合同/参数与医学来源记录。
工具路径按自己的安装调整；每步检查错误日志而不只看 XSim 退出码。

```powershell
.\.venv-lung\Scripts\python.exe python/lung_window_vectors.py --outdir data/lung_window_new/vectors
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -v
Push-Location data/lung_window_new
& 'E:\vivado2\Vivado\2022.2\bin\xvlog.bat' --sv ../../fpga/lung_requant.v ../../fpga/lung_mac_tile8.v ../../fpga/lung_mac_tile8_reuse.v ../../fpga/lung_feature_window.v ../../fpga/lung_conv8_window.v ../../fpga/tb_lung_conv8_window.sv ../../fpga/tb_lung_feature_window_bounds.sv
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_conv8_window -s window
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' window -runall -log window_run.log
& 'E:\vivado2\Vivado\2022.2\bin\xelab.bat' tb_lung_feature_window_bounds -s bounds
& 'E:\vivado2\Vivado\2022.2\bin\xsim.bat' bounds -runall -log bounds_run.log
& 'E:\vivado2\Vivado\2022.2\bin\vivado.bat' -mode batch -source ../../fpga/synth_lung_conv8_window.tcl -tclargs D:/pld_picture_process/data/lung_window_new/synthesis
Pop-Location
.\.venv-lung\Scripts\python.exe python/verify_lung_window.py --root data/lung_window_new
```
