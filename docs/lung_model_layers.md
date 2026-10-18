# 冻结 U-Net 真实卷积层上板对拍

## 当前状态与范围

已回传的 `lung_axil_board_result_resetfix.json` 经独立核验：bit/HWH 与本机修复版哈希一致，
完整 7 次命令、完成计数、写回数量、同地址更新与零差异结果均通过。
正式接收记录：`data/lung_axil_board_20261015_resetfix/board_smoke_verification.json`。
构建时的 manifest 保持历史状态，不覆盖为实板结果。

下一阶段复用同一 bit，不新增 IP、不重搭 BD、不重新训练或校准。
`python/lung_model_layer_package.py` 使用冻结整数 U-Net 的真实权重和逐通道量化参数，
从既有验证病例 **LCTSC-Train-S2-006 的第 72 张（零起始）切片**生成真实激活输入。
选择规则固定为“第一个既有验证病例的中间层”，不读取人工掩膜，不使用独立测试病例选样或调参。
原缓存 FP16 图像按现有流程先转 FP32，再通过同一 Torch 输入量化器量化。

导出全部 15 个卷积层的真实空间/通道尺寸、权重、偏置、Q31 乘子、移位和期望输出。
每层的 Torch 整数数值参考均与独立 NumPy INT64 累加 + Q31 RNE 逐编码一致。
合计 **45 个输出通道块、3,866,624 个有效输出编码**。

这是 **15 个独立卷积层测试**：每层输入由电脑端完整冻结图生成。
池化、最近邻放大、concat 重量化、层间缓冲调度和完整肺分割尚未在这项测试中由 PL 完成。
因此不能把本测试 PASS 称为完整 FPGA U-Net PASS，也不评价医学精度或完整帧率。

## 文件与上板运行

生成包：`data/lung_model_layers_20261015/lung_model_layers/`，包含：

- `layers.json`：层描述、来源和传输文件哈希；
- `vectors.npz`：真实激活/权重/逐通道参数及软件期望；
- `lung_model_layers_test.py`：板端测试；
- 原有 `lung_axil.py`、`lung_axil_smoke.py`：冻结驱动及打包辅助函数。

不需要向板端安装 PyTorch。已验证的 bit/HWH 保留在 `/home/xilinx/ct_test/`。
在电脑 PowerShell 上传整个新目录（无 Windows 通配符问题）：

```powershell
Set-Location "D:\pld_picture_process\data\lung_model_layers_20261015"
& "C:\Program Files\Git\usr\bin\scp.exe" -r lung_model_layers xilinx@192.168.2.99:/home/xilinx/ct_test/
```

停止其他 PL 程序，保持同一块开发板只运行一项测试。在 Jupyter 运行：

```python
%cd /home/xilinx/ct_test/lung_model_layers
%run lung_model_layers_test.py --report /home/xilinx/ct_test/lung_model_layers_result.json
```

报告名必须不存在，不覆盖已有证据。测试先校验数据包和 bit/HWH 哈希，再加载 overlay，
读 ID/ABI，逐层打印装载/启动与比较结果。预期结束语：

```text
Real-board frozen model convolution layers: PASS
```

每块验证整个输出分配区：已写块应等于参考、未写块保持哨兵、尾通道为输出零点，
并检查 16 个后缀哨兵 word 防止越界写。偏置/乘子/移位逐输出通道装入真实硬件。
板端期望数组仅用于比较，绝不复制到硬件输出缓冲区冒充 PL 计算。
超时/中断仍保留可能在途的缓冲区；不要直接关闭内核、重发命令或只清状态，先报告现象并可靠恢复整板。

## 本机验证

新增测试覆盖真实报告身份/顺序/数量错误拒绝、层元数据、带符号偏置、尾通道参数、
多块调用、越界哨兵检测与超时不释放缓冲。主机调用测试使用明确的测试替身，不是实板证明。
旧 187 项回归日志和板级 manifest 保持不变，新完整回归日志独立保存到
`data/lung_model_layers_20261015/python_tests.log`。

本轮完整 **198 项测试通过**。`python/verify_lung_model_layers.py` 又独立复核原 CT 输入量化、
15 层全部权重/参数/尺寸、池化/放大/拼接产生的逐层输入及全部 3,866,624 个输出编码，均一致。
该图关系复核在电脑上执行，仍不表示这些非卷积算子已经在 PL 实现。
独立核验记录为 `data/lung_model_layers_20261015/verification.json`，
状态 `prepared_verified_real_model_layer_fixtures_not_board_tested`。

生成命令：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_model_layer_package.py --outdir data/lung_model_layers_new
```

输出目录必须全新；原权重、人工标注、训练/验证划分、原始 CT、已经工作的 bit 和旧证明都不覆盖。
