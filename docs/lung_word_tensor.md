# 第二阶段：字级并行拼接 RTL

目标是降低整网实际硬件周期，而不是继续削减已经很小的主机开销。
不改分割模型、训练/测试划分、权重、量化尺度或舍入规则。

## 当前实板基线

用户 100 MHz 校准测速截图：双核 ARM 平均 890.895 ms；现有 PL 平均 1924.060 ms，
引擎活动平均约 1842.582 ms，其他约 81.478 ms。全部 21 节点和各次 head 相同，
每个 forward 前后均回读 100 MHz，CPU 保持 650 MHz。
PL 仍比双核 ARM 慢约 2.16 倍，约 95.8% 时间位于硬件活动中。
这些是用户截图记录，原始主/时钟 JSON 和实际 head 尚未取回，不冒充文件级验收。

## 改动及兼容边界

新增 `fpga/lung_tensor_word_engine.v`：当 concat 的 A 通道数是 8 的倍数时，
按 HWC8 整字选择分支和地址，每次读取 64 bit，八路 `lung_requant` 并行完成原有 Q31/RNE。
覆盖冻结网络三个 fused nearest+concat 层；先 nearest 映射，再按原顺序拼接 decoder 与 skip。

A 通道非 8 对齐时继续使用原逐字节映射/舍入路径；pool、nearest 的行为不变。
B 尾通道仍强制输出零点，不使用输入填充字节计算出来的无效结果。
八路仅在全部 ready/valid 时共同握手，不允许一个字中部分通道先前进。
仍等待最终 B 响应和读主机排空后才 DONE；错误使整个命令无效。
AXI 仍然单拍读写，这次没有声称增加 burst、并行 outstanding 或卷积读算写重叠。

旧 RTL、旧顶层和旧 BD 脚本没有修改。
`python/lung_word_prepare.py` 在新实验目录生成机械派生顶层（只替换 engine 实例类型）、
测试平台及构建脚本，记录原件和派生文件哈希。原卷积 IP 的每个打包源文件与旧归档逐个核对。
新 bit 保持寄存器 ABI/IP 名称、地址、DDR 和复位连接，沿用冻结运行时和加载后时钟校准入口。

## 已完成的真实 RTL 对照

新旧模块分别通过同一批 XSim 测试：35 帧、379310 个输出字，24 类错误场景。
包括原来的 pool/nearest/未对齐 concat、全 256 编码、正负 ties、shift 0/62、饱和、
96 通道、512 边长、脏输入尾通道、三个完整尺寸冻结网络拼接层；
另检查背压时 AXI 稳定、延后最终 B、错误 ID/RESP、错误 LAST 排空、越界哨兵、配置锁及共享复位恢复。

三个完整尺寸拼接层的 RTL 周期对照：

| 层 | 输出 64-bit 字 | 旧周期 | 新周期 | 旧/新周期比 |
| --- | ---: | ---: | ---: | ---: |
| cats.0 | 49152 | 5125103 | 1576979 | 3.250 |
| cats.1 | 98304 | 10092550 | 3075096 | 3.282 |
| cats.2 | 196608 | 20359097 | 6348522 | 3.207 |

这是**同一仿真总线阻塞模型下单个拼接层的周期比较**，不是板卡实测，也不是整个 U-Net 提速 3 倍。
仿真日志和解析后的结果位于 `data/lung_word_20261015/sim_baseline/`、`sim_word/`。
独立综合为 5230 LUT、3649 FF、40 DSP、0 BRAM，100 MHz 内部 setup 余量 +1.273 ns；
整机仍须以新工程路由报告为准，不能把 OOC 资源直接当成整机资源。

## 整机路由与部署审计结果

新工程已生成独立 bit/HWH，并通过来源、仿真、完整回归、DRC 与路由审计：
100 MHz 下 WNS +0.113 ns、WHS +0.017 ns；最终 DRC 无 Error/Critical Warning。
整机 18439 LUT（34.66%）、19748 FF（18.56%）、113 DSP（51.36%）、31.5 BRAM Tile（22.50%）。
工程 254 项测试通过，1182 个受保护医学源文件未变化；所有原卷积/控制/AXI/舍入依赖与旧打包归档逐个核对一致。
这是整机路由和本地验收，不是新 bit 的物理板端测试。新 bit 的实际帧时延、输出一致性仍待下述上板步骤。
完整证明 `data/lung_word_20261015/verification.json`，状态 `built_routed_word_parallel_tensor_not_board_tested`。

## 上传与实板验证

### 先修正 Jupyter 缓存入口（当前步骤）

用户最新截图：CPU 902.694 ms、PL 1925.784 ms、引擎约 1842.255 ms，仍未加速，
且与旧 bit 的 100 MHz 周期几乎不变。已在本机复现：同一内核先运行 `lung_fast`，
再切换 `lung_word`，旧入口里的普通 `import lung_fast_benchmark` 可复用旧目录模块。
新 wrapper 验证了新清单，但旧 benchmark 用自己的 `__file__` 找回旧 bit。
截图不能单独证明实际 bit 身份；该次结果仍需取回主报告中的 `bit_sha256` 核实，不能算新 RTL 的时延验收。

后续用户已回传绑定入口的完整终端：确认为上述新 bit 哈希，100 MHz、21 节点和各次 head 全部一致，
PL 1686.432 ms、双核 ARM 891.049 ms，相对旧 bit 的 1924.060 ms 耗时下降约 12.35%。
仍未快过 ARM，原始 JSON/head 待取回核验；下一阶段见[缓存/窗口优化](lung_cache_fast.md)。

增量修复只增加 `lung_selected_benchmark.py`、`selection_manifest.json`，不重做 bit、不改旧包和旧报告。
新入口从确切文件路径加载两个入口，下载前核对路径和 bit/HWH 哈希，完成后核对主报告身份；
额外保存 `_selection.json`。不清除驱动模块缓存，保留故障时的 CMA 所有者；发现未恢复的故障所有者时拒绝下载。
本机实包绑定检查及模拟构造器测试不是板端性能证明。
本次完整工程回归 262 项全部通过（72.406 秒），日志 `data/lung_selected_20261015/python_tests.log`；
其中新增 8 项覆盖缓存复现、绑定恢复、完整入口记录、错误路径/哈希拦截及故障 CMA 所有者保留。

电脑 PowerShell（已有 `lung_word` 包，只补两个文件）：

```powershell
Set-Location "D:\pld_picture_process\data\lung_selected_20261015\deploy"
& "C:\Program Files\Git\usr\bin\scp.exe" lung_selected_benchmark.py selection_manifest.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word/
```

Jupyter（上次正常 PASS 且没有其他 PL 任务运行时）：

```python
%cd /home/xilinx/ct_test/lung_word
%run lung_selected_benchmark.py --runs 3 --warmup 1 --report /home/xilinx/ct_test/lung_word_pinned_result.json
```

应打印 `PINNED benchmark` 指向 `lung_word/lung_fast_benchmark.py`，以及 `VERIFIED bit SHA256`：
`0c7dc65f4c76825b5abfe5333488d54b59dd5123c06c749e42ff091c525337fa`。
不要覆盖先前 `lung_word_result*` 证据。若入口拒绝哈希、路径或驱动版本，不绕过检查。

本轮完成后取回六个文件：

```powershell
Set-Location "D:\pld_picture_process\data\lung_selected_20261015"
& "C:\Program Files\Git\usr\bin\scp.exe" xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result_clock.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result_selection.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result_diagnostic.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result_cpu.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_pinned_result_head.npz .
```

### 完整包首次上传（保留来源说明）

只有整机路由、DRC、时序、完整工程回归和来源审计全部通过，
`python/lung_word_deploy.py` 才会创建 `data/lung_word_20261015/lung_word/`。
该目录里的模型和输入参考与旧版相同，新副本的分层 manifest 重新绑定新 bit/HWH；旧目录完全不覆盖。

电脑 PowerShell：

```powershell
Set-Location "D:\pld_picture_process\data\lung_word_20261015"
& "C:\Program Files\Git\usr\bin\scp.exe" -r lung_word xilinx@192.168.2.99:/home/xilinx/ct_test/
```

首次上传后仍需补上上述两个入口修复文件，使用新入口。不要再用旧缓存易错入口测速：

```python
%cd /home/xilinx/ct_test/lung_word
%run lung_selected_benchmark.py --runs 3 --warmup 1 --report /home/xilinx/ct_test/lung_word_pinned_result.json
```

会在路径和哈希匹配后实际下载新 bit、恢复并读回 100 MHz、重新测双核 CPU、完整核对 21 节点，再测新整网。
只有输出匹配才保存主 JSON 和实际 head；另存 CPU/诊断/频率阶段证据。
此时若 `CPU/PL < 1`，仍然只能报告 PL 更慢，不能用单层仿真周期比代替整网加速。

旧可疑运行的原始证据可另取回，仅用于实际 bit 身份和逐节点周期核验：

```powershell
Set-Location "D:\pld_picture_process\data\lung_word_20261015"
& "C:\Program Files\Git\usr\bin\scp.exe" xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_result.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_result_head.npz xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_result_clock.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_result_diagnostic.json xilinx@192.168.2.99:/home/xilinx/ct_test/lung_word_result_cpu.json .
```

如发生无法中断的 AXI 超时，不盲目重试/重载或释放 CMA；需要恢复共享板级系统。
旧 `lung_fast` 目录仍保留，仅在上次正常完成、无在途事务的条件下可作为回归对照。
本阶段卷积核未变，不能保证这一项就使整网快过双核 ARM；后续还需要优化卷积的数据通路。
