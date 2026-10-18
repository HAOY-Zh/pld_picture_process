# 轻量 2D 肺分割模型：电脑训练阶段

本阶段只在电脑 CPU/GPU 上训练和评价，不是在 PYNQ 上推理，不是 FPGA 部署完成。
所有结果仅用于工程教学与算法演示，不作为临床诊断依据。

## 最新独立测试结果（2026-10-11）

此前各实验分数是六病例开发验证；现在另已冻结原 2D 第 12 轮模型和阈值 0.5，
完成全部 17 个可用预留病例（2705 层）的独立测试，平均病例 Dice **0.965467**、
IoU **0.933475**。24 个官方测试目录均已尝试，7 个只有 RTSTRUCT、没有 CT，
因此这是可用子集成绩，不是官方全 24 例成绩。没有重新训练或用测试分数挑权重。
小肺区逐切片 Dice **0.614803**，空参考切片有前景 **67/1174**，仍存在失败层。
完整流程、逐病例表、误差图和限定见 [独立测试说明](lung_heldout_evaluation.md)。
后文实验历史保留原时间点的结论；量化和智能分割 PL 推理仍未完成。

后续更新：已完成电脑 INT8 静态量化的开发验证，平均 Dice 0.966576→0.966412；
见 [量化说明](lung_int8_quantization.md)。这不是新独立测试或 PL 部署；
历史段落中的“量化未完成”指该段记录时点，当前 RTL 位精确验证仍未完成。

## 2026-10-11：第一次短训练的实际结果

- Windows 文件占用问题已修复：缓存读取/评价使用受控 `.npy` 映射，
  `SliceDataset` 通过 context manager 在正常或异常退出时关闭自己的映射。
  返回的 tensor 由独立 NumPy 拷贝构造，不共享已经关闭的映射。
- 22 项工程测试全部通过，CUDA smoke 通过；新增测试覆盖异常关闭、
  标签缺失导致部分打开、重复关闭、关闭后访问，以及 tensor 拷贝继续有效。
- RTX 4060 Laptop GPU 上完成两轮训练，模型参数 122321 个。
- 固定 18 个训练病例（3002 层）、6 个验证病例（1120 层）；人工标签与划分没有修改。
- Epoch 1：训练 loss 1.2387，原始分辨率平均病例验证 Dice 0.6423。
- Epoch 2：训练 loss 0.7175，原始分辨率平均病例验证 Dice 0.9591。
- `best.pt`、`last.pt`、训练历史、模型摘要及六病例评价已保存。

同一六例开发验证、同一原始参考标签上的结果：

| 验证病例 | 短训练模型 Dice | 固定 CPU 规则基线 Dice |
| --- | ---: | ---: |
| LCTSC-Train-S2-006 | 0.9375 | 0.7978 |
| LCTSC-Train-S2-007 | 0.9713 | 0.8518 |
| LCTSC-Train-S2-008 | 0.9682 | 0.9390 |
| LCTSC-Train-S3-002 | 0.9600 | 0.4088 |
| LCTSC-Train-S3-003 | 0.9699 | 0.3112 |
| LCTSC-Train-S3-008 | 0.9479 | 0.2665 |
| 病例平均 | 0.9591 | 0.5959 |

Dice 是掩膜重合指标，不是像素准确率；以上不是官方独立测试集结果。
不能只看较大的肺切片：S2-006 第 25 层小面积参考肺区 Dice 仅 0.0957，
第 132 层参考为空但模型产生 272 个假阳性像素。这些失败图也已保存。
六病例的体素合并 Dice 较高，不代表所有切片都好，也不证明临床有效性。

结果目录（本地 `data/`，不在 GitHub）：

```text
data/lung_training_20261011/short_run/
  best.pt, last.pt, history.json, model_summary.json, run_config.json
data/lung_training_20261011/short_run_evaluation/
  evaluation.json
  <case>/metrics.json, prediction_native.npy, comparison_slice_*.png
```

以上保留第一次短训练记录；后续早停续训结果见下一节。
当前没有改动 FPGA 模块，也没有声称 GPU 结果就是 PL 推理结果。

## 2026-10-11：早停续训的实际结果

- 从原 `short_run/last.pt` 的第 2 轮恢复，累计上限 30 轮；实际在第 13 轮早停。
- 验证最高分出现在第 12 轮；重新评价使用 `best.pt`，不是第 13 轮的 `last.pt`。
- 自第 5 轮以后最高分仍有小幅增加，程序照常保存；但不足早停阈值 0.001，
  因此等待计数没有重置。第 13 轮验证平均病例 Dice 回落至约 0.9623。
- 学习率在第 9 轮验证后从 0.001 降至 0.0005，第 10 轮开始使用；
  第 13 轮结束后调度器记录的下轮学习率为 0.00025，但早停后没有再训练下一轮。
- 网络参数、256×256 输入、18/6 患者划分、人工标注及预测阈值 0.5 均不变。
- 27 项工程测试全部通过；覆盖早停阈值、状态恢复、学习率保持、
  历史最佳权重继承、已早停断点保护、分组边界和按切片数加权的分组均值。

同一六验证病例、同一原始参考掩膜的前后比较：

| 指标 | 两轮短训练 | 早停选中模型（第 12 轮） |
| --- | ---: | ---: |
| 平均病例整体 Dice | 0.9591 | 0.9666 |
| 49 张小肺区切片的平均逐切片 Dice | 0.4991 | 0.4629 |
| 516 张空参考切片中出现任意预测前景的数量 | 205 | 40 |
| 空参考切片上的假阳性像素总数 | 22917 | 11351 |

分组采用原尺寸专家参考肺面积：小区 1..1000 像素。逐切片均值与病例整体 Dice 是不同统计。
结论是整体 Dice 和空层误分改善，但小肺区没有改善，不能声称所有切片都更准确。
S2-006 第 25 层的 306 个参考肺像素全部漏分（Dice 0），该失败图仍然保留。
下一步应针对小区域做受控实验，例如更高输入分辨率或训练采样/损失调整，
而不是单纯增加轮数、改人工标签、删除困难验证切片或立即固定部署模型。
六个验证患者不足以证明泛化；独立测试、INT8 精度评价与肺分割 PL 实现尚未执行。

实际产物（本地，不随 Git 上传）：

```text
data/lung_training_20261011/early_stop_run/
  best.pt, last.pt, history.json, training_curves.png, training_summary.json
  run_config.json, model_summary.json
data/lung_training_20261011/early_stop_evaluation/
  evaluation.json
  <case>/metrics.json, prediction_native.npy, comparison_slice_*.png
```

重新评价最佳权重的命令（本次已执行；复跑请选择新的空输出目录）：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py evaluate --dataset data/lung_learning_20261009/dataset.json --checkpoint data/lung_training_20261011/early_stop_run/best.pt --outdir data/lung_training_20261011/early_stop_evaluation --device cuda
```

## 早停、学习率与训练曲线

新增控制不改变网络、人工标签、患者划分、输入尺寸或预测阈值。
默认将 `--epochs 30` 作为累计轮数上限，而不是必须训练满 30 轮：

- `--early-stopping-patience 8`：连续 8 轮没有明显验证改善时停止；0 可禁用早停。
- `--min-delta 0.001`：验证平均病例 Dice 必须比早停参考最佳值增加超过 0.001，
  才重置等待计数；`best.pt` 仍按真实最高分保存，不忽略小于这个幅度的提升。
- `--lr-patience 3 --lr-factor 0.5 --min-lr 0.00001`：连续 4 次验证没有明显改善时，
  使用 `ReduceLROnPlateau` 将学习率减半，下限 1e-5；新学习率从下一轮生效。
- 每轮保存 `training_curves.png`、`history.json`、`training_summary.json`。
  曲线分别是训练 loss、验证平均病例 Dice、不同参考肺面积的逐切片 Dice、学习率。
  训练 loss 与验证 Dice 是不同指标，不能直接比较数值大小。
- 验证另外记录参考面积 1..1000、1001..10000、>10000 像素的逐切片 Dice，
  以及空参考切片数、有预测前景的空切片数、空切片假阳性像素数。
  分组由未缩放的专家参考确定；空层不混入有肺切片 Dice 均值。
  “出现任何前景”的计数不等于临床假阳性率。

从已完成的两轮续训到新目录：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py train --dataset data/lung_learning_20261009/dataset.json --resume data/lung_training_20261011/short_run/last.pt --outdir data/lung_training_20261011/early_stop_run --epochs 30 --device cuda
```

旧 checkpoint 没有控制状态：早停计数从原历史初始化，学习率监控从续训开始，
不会倒推修改旧学习率。旧第 1、2 轮没有记录分组分数与学习率，曲线中不补造这些数据。
新 checkpoint 同时保存学习率调度器和早停计数，续训不会重置；控制参数必须保持相同。
已经触发早停的 checkpoint 不会被静默重启。
新目录会继承原最佳权重；如果 `last.pt` 不是最佳轮，则要求旁边的 `best.pt` 匹配。
即使续训所有轮数都没有超过旧成绩，也不会丢失可评价的最佳模型。

早停和验证集模型选择降低训练过度的风险，但不能证明不存在过拟合。
当前只有六个验证患者，最终仍需固定模型后进行未参与调参病例的独立评价。
训练控制说明：<https://docs.pytorch.org/docs/2.8/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html>

## 512×512 分辨率对照实验流程

实际结果：完成 13 轮，按相同标准选中第 10 轮。六病例平均 Dice 0.963092，
49 张小肺区平均逐切片 Dice 0.427707，空参考切片出现前景 41/516，FP 像素 11301。
原 256 早停最佳模型对应为 0.966576、0.462885、40/516、11351。
因此这个实验在相同轮数预算内没有实现目标，不替换原模型；
这不证明更高分辨率在所有模型、预算或病例上都无效。负面结果和失败图全部保留。

这是一次单独的分辨率实验，不把更高分辨率预先当成更准确的模型。
模型结构、通道数、批量大小 8、随机种子、损失、HU 归一化、患者划分、
预测阈值 0.5 和最佳模型选择标准保持不变。为匹配原实验实际完成的 13 轮，
新模型从相同随机种子初始化，训练上限设为 13 轮，同时保留早停与学习率调度。
它不是从已经早停的 256×256 权重继续训练，也不声称已完成充分收敛。

`rebuild-resolution` 重读原 DICOM 像素，并检查患者、物理切片顺序、Frame、
形状与间距。输入不是低分辨率缓存放大的结果；原尺寸参考 `.npy` 文件逐病例
复制并用 SHA256 校验，没有重新栅格化或改变人工答案。某病例失败会禁止训练，
不会悄悄删除病例或更改划分。新分辨率产生新指纹，禁止误用旧尺寸的断点。

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py rebuild-resolution --dataset data/lung_learning_20261009/dataset.json --outdir data/lung_learning_20261011_512 --size 512
.\.venv-lung\Scripts\python.exe python/lung_unet.py smoke --device cuda --size 512 --base-channels 8 --batch-size 8
.\.venv-lung\Scripts\python.exe python/lung_unet.py train --dataset data/lung_learning_20261011_512/dataset.json --outdir data/lung_training_20261011/resolution_512_run --epochs 13 --device cuda
.\.venv-lung\Scripts\python.exe python/lung_unet.py evaluate --dataset data/lung_learning_20261011_512/dataset.json --checkpoint data/lung_training_20261011/resolution_512_run/best.pt --outdir data/lung_training_20261011/resolution_512_evaluation --compare-evaluation data/lung_training_20261011/early_stop_evaluation/evaluation.json --device cuda
```

原缓存、原模型和原结果均保留；这些命令的输出目录不能重复覆盖。
比较时同时报告六病例 Dice、49 张小肺区切片平均 Dice、空层误分和假阳性像素数。
不要只挑小区域最好的一层，或通过调验证集阈值掩盖其他指标变差。

同一网络从 256 到 512，参数数量不变，卷积 MAC 数为原来的 4 倍。
这不等于实测速率恰好慢 4 倍，也不等于 FPGA 资源恰好增加 4 倍；
GPU 时间、片上缓存、DDR 带宽和 PL 架构仍需分别测量/设计。
额外核对：24 个病例的新 512 训练标签与原尺寸参考逐像素一致。
49 张小肺区验证切片的旧 256 训练标签均仍包含前景，没有整张标签消失；
因此不能把现有漏分简单归因为“缩小丢掉了所有小肺区标注”。
512 输入的合成 FP32 GPU smoke 已通过，峰值分配显存约 1556.5 MiB；
这个数不是实际 AMP 训练的显存，也不是 FPGA 资源或医学精度指标。
当前 39 项工程测试通过，新增覆盖原 CT 重读、参考文件哈希、划分不变、失败阻止训练、
切片顺序/身份/几何校验、分辨率改变后的模型算量和旧预测/参考的一致性。

`evaluate --compare-evaluation` 在原来的规则对比图之外，另存
`model_comparison_slice_*.png`：青色是专家，黄色是旧模型，绿色是当前模型，
最后一列是当前模型的多分（红）/漏分（蓝）。会保留旧模型原来的失败切片索引，
不只展示新模型最好的切片。上一份报告与预测的 TP/FP/FN 必须能在当前参考上复算一致，
阈值也必须一致；否则拒绝比较。原有 `comparison_slice_*.png` 的黄色仍是规则基线。

## 256×256 小肺区采样对照实验

实际完成 13 轮，按同一平均病例 Dice 标准选中第 13 轮；未触发早停，不能声称充分收敛。
六病例平均 Dice 0.965430，小肺区平均逐切片 Dice 0.521037；
空参考切片出现前景 52/516，空层 FP 像素 47698。
相比原模型，小区局部改善，但整体略降、空层误分增加，不直接替换原基线。

保持原 256 缓存、模型、损失、预测阈值、种子与训练控制策略，单独改变训练采样。
`--small-lung-weight 4` 将原尺寸参考肺面积 1..1000 像素、且训练标签仍有前景的
训练切片赋予 4 倍抽样权重。权重函数拒绝验证患者；不会读取官方测试图像或更改人工掩膜。
原训练集实际有 149 张符合条件的切片，占 3002 张的约 4.96%；
加权后它们的期望抽样占比约 17.28%，不是直接把实际概率乘 4。

每轮仍抽取 3002 个样本，使用有放回抽样：同一图可以多次出现，某些图该轮可能未抽到。
默认权重 1 仍保持原来的无放回均匀打乱，旧断点按权重 1 兼容；
续训时不得静默从权重 1 改成 4。若小区训练标签已经缩小为全空，则只记录数量、不额外提升其权重。
实际 149 张训练小区标签均还有前景，没有此类丢失。

从相同种子重新初始化，不接着旧模型改变采样后冒充同一实验：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py train --dataset data/lung_learning_20261009/dataset.json --outdir data/lung_training_20261011/small_sampling_run --epochs 13 --small-lung-weight 4 --device cuda
.\.venv-lung\Scripts\python.exe python/lung_unet.py evaluate --dataset data/lung_learning_20261009/dataset.json --checkpoint data/lung_training_20261011/small_sampling_run/best.pt --outdir data/lung_training_20261011/small_sampling_evaluation --compare-evaluation data/lung_training_20261011/early_stop_evaluation/evaluation.json --device cuda
```

最佳权重仍按同一平均病例 Dice 选择；不能改选只对小区最好的轮数再隐去整体退步。
改变采样分布后，训练 loss 不能与均匀采样 loss 直接作为精度高低比较；验证集分布不变。
训练采样变化不增加推理网络参数或每切片卷积 MAC，但可能改变漏分/误分之间的权衡，
必须同时报告小区、整体和空层结果，再决定是否采用。

## 本轮两组对照的结论

| 指标 | 原 256 模型 | 512 分辨率实验 | 256 加权采样实验 |
| --- | ---: | ---: | ---: |
| 选中轮次 | 12 | 10 | 13 |
| 六患者平均整体 Dice | 0.966576 | 0.963092 | 0.965430 |
| 小肺区平均切片 Dice（49 张） | 0.462885 | 0.427707 | 0.521037 |
| 空参考切片出现前景（共 516 张） | 40 | 41 | 52 |
| 空参考切片误分像素总数 | 11351 | 11301 | 47698 |
| 网络参数量 | 122321 | 122321 | 122321 |
| 每切片卷积 MAC | 665845760 | 2663383040 | 665845760 |

整体 Dice 是逐患者整卷 Dice 的平均，小区 Dice 是逐切片平均，两者不能混用。
三组使用相同患者划分、原尺寸人工参考、阈值 0.5 与模型选择标准；
两个新实验从同一随机种子重新初始化，单次最多 13 轮，不代表多种子或充分收敛比较。
原最佳权重仍在 `data/lung_training_20261011/early_stop_run/best.pt`，未被覆盖。
两个新实验的权重、曲线、完整评价与失败层图片分别保存在
`resolution_512_run` / `resolution_512_evaluation` 和
`small_sampling_run` / `small_sampling_evaluation`，均位于同一日期目录下。

S2-006 索引 25 的困难切片，原模型 TP=0、FN=306，Dice=0；
加权模型 TP=280、FP=249、FN=26，Dice=0.670659。
这说明该层漏分减少，但不是完全正确；不能以这一张改善替代整体验证。
配对图 `small_sampling_evaluation/LCTSC-Train-S2-006/model_comparison_slice_025.png`
依次展示人工参考（青）、旧模型（黄）、加权模型（绿）和当前误差（红多分、蓝漏分、灰正确）。

结论：512 实验没有达到本轮目标；加权采样是有代价的局部改善。
继续优化时要同时限制空层假阳性并保持小肺区检出，不能只追求单个分组 Dice。
这些仍是电脑 GPU 开发验证；官方测试图像未用于调参，量化与智能分割 PL 推理尚未完成。

## 相邻三切片 2.5D 肺部分割实验

实际完成至第 11 轮触发早停，仍按最高平均患者整体 Dice 选中第 8 轮；
六病例完整评价与训练验证指标一致，原缓存指纹和 18/6 患者名单均复核一致。

| 指标 | 原 2D 基线 | 本轮 2.5D |
| --- | ---: | ---: |
| 六患者平均整体 Dice | 0.966576 | 0.961381 |
| 49 张小肺区平均切片 Dice | 0.462885 | 0.227280 |
| 516 张空参考切片中出现前景的张数 | 40 | 13 |
| 空参考切片误分像素总数 | 11351 | 23665 |

结论：本轮未实现改善肺提取的目标，不替换原 2D 基线。
空层误分张数少不等于误分像素少；2.5D 的全部空层 FP 来自 S3-002，
需结合面积与失败图判断，不能只报告 40→13。S2-006 索引 25 仍完全漏分（TP=0、FN=306）。
第 3 轮小区 Dice 曾为 0.624051，但其整体 Dice 0.960659、空层 FP 108784；
不能为宣称成功而事后改选这轮。本轮负面结果不证明所有 2.5D 架构都无效，
也不能仅凭它断言原因是过拟合、层间距或数据数量。
所有权重、日志、曲线、逐层指标与配对图保留在下面命令指定的目录，未上传 GitHub。

任务仍是左右肺并集提取，不是五肺叶分割，也不是完整 3D 卷积网络。
原 2D 模式保持默认：`train --context-slices 1`。
`--context-slices 3` 将同一患者同一序列的前一层、当前层、后一层 CT 作为三个通道，
只预测当前层的人工肺掩膜。首尾用最近边缘层重复，不跨患者取邻居，不使用相邻人工标签作输入。
水平翻转对三通道和中心标签同步进行；训练与评价使用同一个 `context_batch` 函数。

三通道模式在训练/评价前重读 CT 文件头，检查患者/序列/Frame、SOP 唯一性、
物理顺序、形状、方向、像素间距、均匀层间距及无剪切。检查失败不开始训练或输出评价。
各病例仍使用原始相邻物理层，不做跨患者统一毫米间距重采样；层间距会记录在报告，
所以三层对应的毫米范围可能因病例而异。人工标注与原缓存不改写，也不生成三倍体积的新缓存。
实际检查的 24 个病例层间距为 1.25..3 mm，均通过上述几何检查。

本轮保持 256 输入、原 18/6 患者划分、随机种子、损失、批量 8、阈值 0.5，
恢复默认均匀无放回采样；不同时叠加上次小肺区权重 4。
从同一种子重新初始化，最多 13 轮，最佳模型仍按平均患者整体 Dice 选择，
不从旧 2D 权重续训。新网络首层通道数不同，所以相同种子不代表参数完全相同。
旧 2D 基线来自两轮短训练后续训的历史实验；新实验保留同一训练控制设置，
但不宣称学习率轨迹、充分收敛或多随机种子统计完全匹配。

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py smoke --device cuda --size 256 --base-channels 8 --batch-size 8 --context-slices 3
.\.venv-lung\Scripts\python.exe python/lung_unet.py train --dataset data/lung_learning_20261009/dataset.json --outdir data/lung_training_20261011/context_25d_run --epochs 13 --context-slices 3 --small-lung-weight 1 --device cuda
.\.venv-lung\Scripts\python.exe python/lung_unet.py evaluate --dataset data/lung_learning_20261009/dataset.json --checkpoint data/lung_training_20261011/context_25d_run/best.pt --outdir data/lung_training_20261011/context_25d_evaluation --compare-evaluation data/lung_training_20261011/early_stop_evaluation/evaluation.json --device cuda
```

旧断点没有 `context_slices` 时按 1 通道兼容；续训禁止改变通道模式，
历史最佳断点也必须匹配。评价自动读取权重中的输入模式，不依赖手工指定，
报告和模型配对图会明确标记 2D/2.5D。当前 39 项测试含邻层顺序、边缘重复、
跨病例隔离、同步增强、中心目标、几何失败阻止训练、3 通道训练/续训/评价及旧 2D 兼容。

基础通道 8 时，三通道输入只改变首层卷积：参数从 122321 增至 122465；
256 输入的卷积 MAC 从 665845760 增至 675282944，约增加 1.42%，不是整个网络算量变成 3 倍。
每个样本 CT 输入数据从一层变为三层；实际带宽、GPU 时间和 PL 缓存仍需单独核算。
GPU 合成 FP32 smoke 峰值分配显存约 393.5 MiB，不代表实际 AMP 训练或 PL 资源。
这仍是电脑 GPU 实验，没有改动现有 PL 算法、生成比特流或完成 3D 浏览/重建界面。
未来可提供 2D/2.5D 模式选择；独立 3D 模式和扩展要求需要另行实现与验证。

## 模型与数据

- 小型 U-Net 类结构，通道 8/16/32/64，三次下采样，最近邻上采样与跳跃拼接。
- 操作为 3×3/1×1 卷积、BatchNorm、ReLU、MaxPool、最近邻上采样、拼接。
- 默认输入为 256×256，另有 512×512 对照实验；HU 裁剪至 [-1000,400] 并归一化到 [0,1]。
- 预测目标是专家左肺、右肺掩膜的并集，不是病灶检测或左右肺分类。
- 训练标签用最近邻缩放；评价恢复到原始尺寸后与未缩放标签比较。
- 保留所有切片，包括没有肺的层，避免只展示有肺层而忽略假阳性。
- Loss 为 BCEWithLogits + soft Dice；训练使用成对水平翻转，预测阈值固定 0.5。
- 量化、BN 折叠、算子/缓存设计和 PL 卷积加速尚未完成。

现有 24 个训练候选按患者与机构固定分成 18 个训练、6 个验证病例。
此前仔细查看过的 S2-001、S3-001 保留在训练组，不作为新的独立验证证据。
同一患者或同一 Frame of Reference 跨病例重复时拒绝缓存，防止切片泄漏。
17 个官方测试候选只记录名字，训练和验证程序不读取其图像。
缺少 S1 的训练 CT，不能声称当前验证覆盖了三家机构。

## 训练环境

工作站使用 Python 3.11，项目环境为 `.venv-lung`，PyTorch 2.8.0 CUDA 12.6。
环境复用已有 numpy/scipy/pydicom/matplotlib，新训练依赖只装入 `.venv-lung`。
不需要把安装包复制到板卡或解压。用户提供主安装包：

```text
D:\pytorch\torch-2.8.0+cu126-cp311-cp311-win_amd64.whl
```

依赖齐全后，项目根目录安装：

```powershell
.\.venv-lung\Scripts\python.exe -m pip install --no-index "D:\pytorch\torch-2.8.0+cu126-cp311-cp311-win_amd64.whl"
.\.venv-lung\Scripts\python.exe python/lung_unet.py smoke --device cuda
```

`smoke` 只验证前向、反向和参数更新，不是医学精度验证。

PyTorch 官方版本安装说明：<https://pytorch.org/get-started/previous-versions/>
混合精度 API 说明：<https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>

## 准备缓存

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py prepare --input-root "D:\CT2\lctsc" --audit data/lung_validation_20261009/download_audit/audit.json --prepared-root data/lung_validation_20261009 --outdir data/lung_learning_20261009
```

只运行一次；已经生成的目录不能覆盖。任何病例转换失败会记录在 `dataset.json`，
将缓存标为 incomplete 并禁止训练，不会悄悄丢弃效果差的病例或改患者划分。
已有匹配缓存可加 `--resume-cache` 重试失败病例，不改变原患者划分。
每个病例输出 `reference_alignment.png`，需要检查轮廓对齐。左右肺标签有重叠时
保留原标签、记录数量；由于任务目标是两肺并集，重叠体素只计算一次。不能将此
策略解释为所有标注都已人工审核。

## 先短训练，再正式训练

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py train --dataset data/lung_learning_20261009/dataset.json --outdir data/lung_training_20261011/short_run --epochs 2 --device cuda
```

短训练用于检查完整流程与学习趋势，不能作为已收敛的最终模型。
每轮验证在原始分辨率上计算每个病例的整体 Dice，再取病例均值。
保存 `last.pt`、验证集表现最好的 `best.pt`、训练历史及模型算量摘要。
这里的算量摘要不等于 LUT/BRAM/DSP 资源报告，也不证明可以实时部署。

可以恢复可信的项目权重，输出到新目录，`--epochs` 为累计目标轮数：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py train --dataset data/lung_learning_20261009/dataset.json --resume data/lung_training_20261011/short_run/last.pt --outdir data/lung_training_20261011/continued_run --epochs 30 --device cuda
```

恢复时检查患者划分指纹、通道数与 seed，保存优化器/混合精度状态。
按每轮 seed 固定打乱和增强，但 GPU 运算不承诺逐位可复现。
只用可信的本项目 checkpoint，加载采用 `weights_only=True`。

## 与规则基线和专家参考比较

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py evaluate --dataset data/lung_learning_20261009/dataset.json --checkpoint data/lung_training_20261011/short_run/best.pt --outdir data/lung_training_20261011/short_run_evaluation --device cuda
```

默认评价六个验证病例；`--case LCTSC-Train-S3-001` 可以看某个训练病例，但其分数
必须明确标为训练集演示，不作为泛化精度。
上面短训练和评价已经执行过，原目录非空，不能直接覆盖；复跑时使用新的输出目录。
生成原图+专家轮廓、规则输出、模型输出和误差图，同时显示较差切片，避免只看
最好的一层。红色为多选、蓝色为漏选、灰色为正确选中的肺。

模型时间包含缓存输入加载、推理、恢复原始尺寸和结果拷回；规则基线时间仅计算
算法，两者时间范围不同，不直接声称加速倍数，也不是 FPGA 或系统端到端 FPS。
最终还需固定模型后做独立测试、INT8 精度验证、PL 部署和一致条件下性能对比。

## 工程测试

```powershell
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -v
```

测试检查患者划分、标签缩放、数据泄漏保护、模型梯度、权重读取与切片索引。
不安装 PyTorch 的环境会跳过模型测试，因此完整测试需使用 `.venv-lung`。
