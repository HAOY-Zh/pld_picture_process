# 肺区域自动分割：离线开发与验证

本阶段在电脑 CPU 上运行，不烧录 FPGA，也不修改已有 Vivado 工程。
它是数据与算法验证基础设施，不是已经完成的智能分割系统。
仅用于工程教学与算法演示，不作为临床诊断依据。

## 要分割的是什么

胸部 CT 同时包含肺、心脏、胸壁、骨骼与体外背景。
本任务输出与 CT 对齐的肺区域二值掩膜，而不是直接检测肺癌或分割病灶。
显示时可以把掩膜轮廓叠加到原图上，原图不用变成黑白图。
左右肺人工标注分别保存；当前自动基线仅预测左右肺的并集，不声称区分左右肺。

LCTSC 提供 CT 和人工勾画、经复核的 RTSTRUCT。该数据集的轮廓规范对肿瘤、
肺门等区域有特定取舍，因此“参考肺”不是任何条件下都通用的解剖学定义。
数据应保留官方来源、许可与引用记录。

官方数据集：<https://www.cancerimagingarchive.net/collection/lctsc/>

## 当前已实现

入口为 `python/lung_segmentation.py`，有四个子命令：

- `inspect`：递归读取 DICOM 头，列出 CT 序列 UID、RTSTRUCT 路径和 ROI 名称。
- `audit`：按病例清点下载；缺少 CT/标注、引用切片缺失等暂时排除，保存清单，不删除原始数据。
- `prepare`：读取一个完整 CT 序列，转换 HU；将人工左右肺轮廓转换成掩膜。
- `evaluate`：运行独立于人工标注的自动基线，输出 Dice/IoU、错误图和 CPU 耗时。

准备阶段的重要限制与保护：

- 按图像方向法向量上的物理位置排序，不按文件名或 InstanceNumber 排序。
- 按 SOP Instance UID 关联轮廓与切片，并检查 Frame of Reference。
- 用 ImagePositionPatient、ImageOrientationPatient 和 PixelSpacing 转换坐标。
- 支持 CLOSED_PLANAR 与 CLOSEDPLANAR_XOR；同一 ROI 混用类型时拒绝处理。
- 普通闭合轮廓做并集，XOR 轮廓做异或；多边形按像素中心填充，包含边界。
- 不做缩放、重采样或 HU 的 int16 量化；原始 HU 保存为 float32。
- 初版仅支持单帧 CT、均匀且无剪切的网格；增强型 CT、缺片、变向等明确报错。
- 初版要求轮廓提供一个对应 CT 的 SOP 引用；没有引用时不猜测切片。
- 无轮廓的切片暂按空掩膜保存；必须目视检查是否为完整标注，而非漏标。
- 多个序列/标注不能擅自混合；无法唯一选择时需要显式指定。
- 输入文件不会修改，输出目录必须新建或为空，避免覆盖之前的实验结果。

像素边界的离散填充可能与治疗规划软件有少量差异。先检查叠加图，必要时与
原始轮廓在查看器中的显示对比；不能只因转换脚本能运行就认定标注对齐正确。

坐标与轮廓规范：

- DICOM Image Plane：<https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html>
- DICOM ROI Contour：<https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.6.html>

## 先在电脑运行，不是在板卡 Jupyter 运行

当前电脑已有 numpy、scipy、pydicom、matplotlib，可以直接运行。
如果以后换到新 Python 环境，可安装：

```powershell
python -m pip install -r python/requirements-lung.txt
```

下面路径是规划示例，不表示下载文件已经存在。将数据放到
`D:\pld_picture_process\data\lctsc` 或替换为实际下载路径即可。
TCIA 下载的目录层级可能不同；`inspect` 会递归搜索，不要求固定文件名。

### 第一步：清点数据

在项目根目录 PowerShell 运行：

```powershell
python python/lung_segmentation.py inspect --input "data/lctsc"
```

批量排查下载失败病例（路径替换为实际目录）：

```powershell
python python/lung_segmentation.py audit --input "D:/CT2/lctsc" --outdir "data/lctsc_download_audit"
```

生成 `audit.json`、Excel 可打开的 `case_inventory.csv` 与 `download_audit.md`。
“可进入准备步骤”只意味着文件头和引用切片检查通过，不表示已通过轮廓对齐或
医学准确率验证。缺失病例只是暂时排除，补下载后可重新清点。

确认下载了 CT **和** RTSTRUCT。只有 CT 没有标注时，不能进行人工参考评价。
整个集合包含很多序列，第一次建议选一个训练病例，不要把全体病例堆成一套体数据。

### 第二步：转换一个病例

输入目录如果只有一个 CT 序列与一个匹配 RTSTRUCT：

```powershell
python python/lung_segmentation.py prepare --input "data/lctsc/某个病例" --outdir "data/prepared/lctsc_train_001"
```

输入目录包含多个 CT 序列时，复制 `inspect` 输出的真实 UID：

```powershell
python python/lung_segmentation.py prepare --input "data/lctsc" --series "实际的CT序列UID" --rtstruct "实际的RTSTRUCT完整路径" --outdir "data/prepared/lctsc_train_001"
```

常见 `Lung_L` / `Lung_R`、`Lung L` / `Lung R` 可以自动匹配。
其他命名可以加 `--left-roi "准确名称或编号" --right-roi "准确名称或编号"`。
不会把名字中带 Lung 的肿瘤 ROI 自动当成肺。

左右肺掩膜有重叠时，默认拒绝准备并报告数量。只有先目视核对轮廓对齐后，才可
显式加 `--allow-reference-overlap`，保留原始两份标签、记录重叠数量，并仅评价
左右肺并集；不会删除像素或重写人工标注。该开关不证明标注正确。

输出：

- `case.npz`：HU、左右肺参考掩膜、坐标/间距/切片 UID；数组顺序为 `(slice,row,column)`。
- `metadata.json`：ROI 来源、参考规范及转换注意事项；不另存患者姓名。
- `alignment_preview.png`：青色线为人工肺边界，绿色线为基线预测。

首先检查青色轮廓是否正确覆盖肺。默认展示人工肺面积最大的切片；一张预览不能
替代所有切片的标注检查，后续也应检查肺尖、肺底和异常病变层面。

### 第三步：验证基线

```powershell
python python/lung_segmentation.py evaluate --case "data/prepared/lctsc_train_001/case.npz" --outdir "data/results/lctsc_train_001_baseline"
```

默认流程为：

```text
CT HU < -400
    → 二维 8 邻域连通区域标记
    → 排除连接到图像边缘的空气区域
    → 面积至少 25 mm² 的候选中，保留最大的两个
    → 一次 3×3 闭运算（各阶段零填充，与已有 PL 规则一致）
    → 二维填孔
    → 肺候选掩膜
```

该流程逐层独立处理，不是 3D 连通域算法，也不是训练好的模型。
默认阈值和面积只是待验证起点，不是临床参数。
处理逻辑完全不读取参考掩膜；参考只用于评价与选择显示切片。

可调参数：`--threshold -400 --min-area 25 --max-components 2 --close-iterations 1`。
`--close-iterations 0` 表示跳过闭运算，避免 SciPy 的零迭代收敛语义。
`--slice 50` 仅改变预览层，评价仍覆盖整个体数据。

输出：

- `preview.png`：人工轮廓、候选空气、排除外部空气后的候选、筛选结果、预测轮廓、错误图。
- `prediction.npz`：整个 CT 体的预测掩膜与间距。
- `metrics.json`：整体 Dice/IoU、每层指标、体积和 CPU 算法耗时。
- `candidate_mask.npy`：预览层中形态学清理前的自动候选，uint8 0/255。
- `reference_mask.npy`：同层人工参考，uint8 0/255。
- `exported_slice.json`：导出层号、尺寸及编码，防止误把不同层做对比。

错误图：灰色是正确选中的肺像素；红色是多选的区域；蓝色是漏选的肺区域。
Dice 和 IoU 越接近 1 越重合，但无论数值多高，都不等于可用于临床。
预测和参考都为空的层指标记为 `null`，不以 1 分灌高逐层均值。
总体指标是全部体素合并计算，不是逐层 Dice 的简单平均。
CPU 时间只包含基线计算，不包含读文件、画图、DMA、FPGA 和界面，不可称为系统帧率。

## 基线的已知失败风险

- 高密度病变、肺不张等可能超过阈值，导致漏选。
- 气管可能被保留，尤其肺尖区域；“最大两个区域”不等于可靠的左右肺识别。
- 裁剪视野、胸壁断裂、异常连接可能使肺与外部空气连通，被误删除。
- 小肺尖可能被最小面积条件删除；肺门、肺底和病变层需要重点检查。
- 填孔能补回部分内部高密度结构，但无法正确解决所有与边缘相连的病变。
- LCTSC 的排除区域规则与简单“填满肺内部”的目标可能不同。

因此不会输出一个没有医学依据的 PASS 门槛，也不会用人工参考修补预测来虚报效果。

## 智能分析与 FPGA 的后续路线

1. 下载完后先验证数据转换，选几个训练病例跑基线并记录失败层。
2. 确认要求中“智能分析”是否需要学习型模型。若需要，主路线采用轻量 2D
   U-Net 类分割模型，以上标注转换与评价代码继续复用。当前尚未实现训练代码。
3. 按患者固定训练/验证/测试划分，同一患者所有切片只能属于同一组；官方训练
   与测试划分需要保留。不能在测试集上调阈值、选模型或挑最好看的病例。
4. 先建立浮点模型的可复现基准，再研究 INT8 定点量化与精度变化；不能先假定
   模型一定能放进 Zynq-7020，需检查算力、权重大小、片上缓存及 DDR 带宽。
5. 确定耗时热点后再实现 PL 核心加速：已有阈值/形态学能用于规则基线，学习型
   路线需有真正的核心计算加速，不能只在 PS 推理再把画图称为 FPGA 分割。
6. 文件解析、界面与调度交给 PS/电脑；PL 做选定的核心算法。最终画面显示原图、
   分割掩膜/轮廓、参数、分割质量、端到端耗时与资源占用。

目前的独立 bitstream 不能直接当作同时驻留的完整流水线。频繁切换 bitstream
会破坏连续处理目标；最终需整合可选择/旁路的模块或单独设计模型加速通路。

`candidate_mask.npy` 可作为现有独立形态学 PL 测试输入，但只在尺寸为 512×512 时
适用。这只是复用已有测试，不表示该离线脚本已经在 PL 完成肺分割。

## 工程测试

```powershell
python -m unittest discover -s python/tests -v
```

测试用合成 DICOM 和掩膜验证坐标、HU、SOP 引用、XOR 孔洞、空掩膜指标、CLI 与
防覆盖行为。也已在原有 Mayo 单切片上跑通基线，但它没有人工肺标注，不能报告
医学分割准确率。真实 LCTSC 数据与模型训练验证须等待下载与病例划分完成。
