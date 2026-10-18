# COVID CT 外部测试

仅用于工程教学与算法演示，不作为临床诊断依据。当前推理在电脑 GPU/CPU，
不是 PYNQ-Z2/PL 推理。固定原 2D epoch 12 和现有 INT8 模型，不训练或重新校准。

## 原始数据与预声明范围

本机下载在 `D:\CT3`，原始压缩包不修改或解压覆盖：

- `Covid-19.zip`：国内教程镜像，包含 CT 和若干标注；不是官方原 ZIP。
- `Lung_Mask.zip`：直接取得的官方肺部人工标注，MD5 必须为
  `972b3d9b4c7b64d2c518b936bd98cb47`。**实际评价使用这个包，不使用感染标注。**

来源：[官方 20 例数据](https://zenodo.org/records/3757476)、
[作者的输入编码说明](https://github.com/JunMa11/COVID-19-CT-Seg-Benchmark#guidelines)。
十份 Coronacases 保存原 HU；另十份 Radiopaedia 已被调整肺窗并归一化到 0–255，
不能当原 HU，也不能无损反推原 HU。因此在读外部像素前固定主测试为
`coronacases_001..010`；其他体数据检查配对、几何和像素，但不推理、不混入均值。
这不是按 Dice 挑病例；报告必须同时给出完整包的审计与编码排除数量。

## 空间、标签和隔离

- 不按文件名顺序直接叠图。检查 coded sform/qform、一致性、可逆有限 affine、
  无斜切/倾斜，以及 CT/标签同一个物理体素网格。无法确认则报告数据失败。
- 用 NiBabel 根据 affine 做**仅轴置换与翻转**，转为 LPS 轴序再 ZYX 排列。
  原 LCTSC 的 24 开发 / 17 独立测试病例 DICOM 方向均为
  `[1,0,0,0,1,0]`；此排列匹配原输入方向，不盲目 `rot90`。
- 标签必须是 0/1/2，原始尺寸左右肺并集；不做插值、形态学修改或伪标签替换。
  方向解释依据 [NiBabel 文档](https://nipy.org/nibabel/image_orientation.html)。
- 全部主测试 CT 层都评，包括人工掩膜为空的层。仍使用原 HU 裁剪归一化、256 输入、
  float16 缓存；输出 Sigmoid 后双线性概率恢复原尺寸，再 `>=0.5`。
- 固定 FP32 与 INT8 原权重、代码、旧开发/测试清单和两份 ZIP 的 SHA256；
  校验 ZIP/gzip CRC、缓存 SHA256。输出新目录，拒绝覆盖旧报告。
- 与此前 24 开发 / 17 测试缓存比较精确归一化 CT 哈希，并检查外部完全重复体数据。
  **NIfTI 不含原始 PatientID/SOP，不能据此声称已经证明患者身份完全独立**；
  哈希不重复也无法检测所有裁剪/重采样/编码不同的重复病例。
- 有数据缺失时明确报告覆盖子集；推理/指标失败中止，不静默排除困难病例。
  看过分数后再调参，这批数据就不再是新的盲测。

## 执行

在电脑项目根目录运行，不在板卡 Jupyter 执行。依赖已加入 `requirements-lung.txt`；
需要现有 PyTorch 环境和 NiBabel 5.x。

```powershell
.\.venv-lung\Scripts\python.exe python/lung_external.py prepare --ct D:/CT3/Covid-19.zip --masks D:/CT3/Lung_Mask.zip --dataset data/lung_learning_20261009/dataset.json --prior-test data/lung_training_20261011/heldout_2d_evaluation/test_dataset.json --checkpoint data/lung_training_20261011/early_stop_run/best.pt --int8-model data/lung_training_20261011/int8_ptq_validation_v2/int8_model.pt --outdir data/lung_external_20261012/prepared
```

准备后先查看各 `cache/<case>/alignment.png`：青色人工肺边界应与 CT 对齐，
不是人工复核标注质量或据此按效果挑选病例。再执行：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_external.py evaluate --prepared data/lung_external_20261012/prepared --outdir data/lung_external_20261012/evaluation --device cuda
```

输出 `evaluation.json`、`case_metrics.csv`、每个病例两种原尺寸预测以及对比图。
青色人工、黄色 FP32、绿色 INT8；误差图红色多分、蓝色漏分、灰色正确。
选择最大肺区、最差非空层、量化差异最大层及空层错误，不能只展示容易的层。
汇报病例体积 Dice/IoU、小区逐层 Dice 与空层误分，不能把 Dice 当诊断正确率。
FP32 GPU 与 INT8 CPU 的时间不是同硬件加速比，更不是板端 FPS。

## 尚未完成

外部入口不包含 RTL/HLS、DDR/BRAM 规划、PL 智能分割或资源/时序实测；
外部精度验证不能证明模型已经能在 PYNQ-Z2 上运行。

## 2026-10-12 实际数据审计

两份压缩包齐全，官方肺部标注 MD5 匹配。全部 20 份 CT 与 20 份官方肺部标注
完成解码、ZIP/gzip CRC 和物理网格配对检查，无数据失败。主测试 10 份 Coronacases
共 **2581 层**，均为 int16 HU；另外 10 份为 UINT8 显示灰度，按预声明不作 HU 推理。
`coronacases_002` 等存在高 HU 值（最高 9567），仍沿用原裁剪规则，不据此剔除。
缓存与旧 24 开发 / 17 测试缓存无完全相同的归一化输入哈希；不等于患者身份已获证明。

镜像 CT 的空间单位头字段为 unknown，官方肺部标注为 mm；转换后两者体素坐标
和 affine 数值一致，按配对官方标注的毫米网格解释。没有只靠未知单位头声称确认物理单位。
十份最大肺区切片的人工叠加图均已查看，没有发现明显错位；不代表逐层临床标注质控。
53 项工程回归通过。源 ZIP、旧权重、旧训练/验证划分均未改动。

## 2026-10-12 实际外部测试结果

主测试 **10/10 份体数据、2581 层全部完成**，包含空参考层，无按成绩排除。
冻结原 2D epoch 12、概率阈值 0.5 与原静态 INT8 模型；没有训练或重新校准。

| 指标 | FP32 | INT8 核心 + FP32 后处理 |
| --- | ---: | ---: |
| 平均体数据 Dice | 0.958590153 | 0.959075389 |
| 平均体数据 IoU | 0.921254842 | 0.922127966 |
| 最低体数据 Dice，003 | 0.899158865 | 0.900577190 |
| 最高体数据 Dice，002 | 0.973330485 | 0.973486436 |
| 小肺区逐层 Dice，80 层 | 0.576827029 | 0.571993061 |
| 中区逐层 Dice，332 层 | 0.915626779 | 0.915313918 |
| 大区逐层 Dice，1744 层 | 0.961607167 | 0.962038208 |
| 空参考层出现前景 | 74 / 425 | 74 / 425 |
| 空参考层误分像素 | 72625 | 71429 |

平均体数据 Dice 的量化变化为 +0.000485236，不能据此称量化提升分割能力；
小区逐层 Dice 反而略降。两种模型共有相似失败：003 的肺内较亮区域有明显漏分，
空参考层误分也仍多。因此结论是核心量化对这批数据的整体影响较小，
**不是外部泛化问题已解决，也不是临床可靠性已达标**。
与 LCTSC 的 0.965467 是不同人群/标注的数据测试，不能将均值直接当同条件模型排名。

原尺寸预测不同像素合计 201316，占本批输出像素约 0.029754%；这不是对人工标注的
错误率。十份 FP32 和 INT8 保存掩膜重新计算指标与报告一致；所有冻结输入、代码、
旧权重与源 ZIP 的 SHA256 核对不变，评价正常退出。

产物：`data/lung_external_20261012/{prepared,evaluation}`。前者包含全 20 份审计、
排除原因、原始尺寸参考与 HU、归一化输入和人工叠加图；后者包含全部十份预测、
逐体数据 CSV、逐层指标及最大肺区/最差层/空层失败图。
示例 `evaluation/coronacases_003/comparison_slice_080.png` 展示明显漏分，不只展示成功图。
没有新增比特流、板卡推理或 Git 提交/推送。
