# 轻量 2D 肺分割模型：电脑训练阶段

本阶段只在电脑 CPU/GPU 上训练和评价，不是在 PYNQ 上推理，不是 FPGA 部署完成。
所有结果仅用于工程教学与算法演示，不作为临床诊断依据。

## 模型与数据

- 小型 U-Net 类结构，通道 8/16/32/64，三次下采样，最近邻上采样与跳跃拼接。
- 操作为 3×3/1×1 卷积、BatchNorm、ReLU、MaxPool、最近邻上采样、拼接。
- 输入整张切片缩放至 256×256；HU 裁剪至 [-1000,400] 并归一化到 [0,1]。
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
每个病例输出 `reference_alignment.png`，需要检查轮廓对齐。左右肺标签有重叠时
保留原标签、记录数量；由于任务目标是两肺并集，重叠体素只计算一次。不能将此
策略解释为所有标注都已人工审核。

## 先短训练，再正式训练

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py train --dataset data/lung_learning_20261009/dataset.json --outdir data/lung_training_20261009/short_run --epochs 2 --device cuda
```

短训练用于检查完整流程与学习趋势，不能作为已收敛的最终模型。
每轮验证在原始分辨率上计算每个病例的整体 Dice，再取病例均值。
保存 `last.pt`、验证集表现最好的 `best.pt`、训练历史及模型算量摘要。
这里的算量摘要不等于 LUT/BRAM/DSP 资源报告，也不证明可以实时部署。

可以恢复可信的项目权重，输出到新目录，`--epochs` 为累计目标轮数：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py train --dataset data/lung_learning_20261009/dataset.json --resume data/lung_training_20261009/short_run/last.pt --outdir data/lung_training_20261009/continued_run --epochs 30 --device cuda
```

恢复时检查患者划分指纹、通道数与 seed，保存优化器/混合精度状态。
按每轮 seed 固定打乱和增强，但 GPU 运算不承诺逐位可复现。
只用可信的本项目 checkpoint，加载采用 `weights_only=True`。

## 与规则基线和专家参考比较

```powershell
.\.venv-lung\Scripts\python.exe python/lung_unet.py evaluate --dataset data/lung_learning_20261009/dataset.json --checkpoint data/lung_training_20261009/short_run/best.pt --outdir data/lung_training_20261009/short_run_evaluation --device cuda
```

默认评价六个验证病例；`--case LCTSC-Train-S3-001` 可以看某个训练病例，但其分数
必须明确标为训练集演示，不作为泛化精度。
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
