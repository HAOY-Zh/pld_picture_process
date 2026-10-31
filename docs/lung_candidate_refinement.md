# 肺结节候选改进实验：兼顾LDCT、U-Net与PYNQ-Z2

日期：2026-10-18。本实验保留既有LDCT增强、冻结肺分割模型、旧v3算法和网页默认入口。新增的是候选算法的软件参考与小型特征评分器，没有新增CNN，没有生成新bit，也没有完成板端性能验证。

## 系统位置

原始HU → 已有U-Net肺掩膜 → 肺内候选初筛 → 局部特征筛选与三维去重 → 候选JSON → ROI/MPR/三维定位。

LDCT作为增强对照分支保留。增强图进入分析链之前需独立验证；本实验不把增强显示图替代原始HU，也不重新训练U-Net。输入肺掩膜必须覆盖全部待分析层，且与原生CT网格一致。

## 改了什么

- 局部中心：在物理毫米尺度的局部邻域中测量连通亮区，比较原峰值与有限位移的加权中心。最大位移2 mm，不靠参考标注修正坐标。
- 补充尺度：旧算法先取各尺度最大响应，邻近细血管可能压掉较大尺度中心。本实验在所有开发病例统一额外保留2 mm尺度的独立峰值，不按病例或标注补点。
- 结构筛选：放宽前置候选池，球状比0.04、结构分数40、粗区域直径2.5 mm、延伸比6；后续用多项特征综合排序，减少单个强阈值误删。
- 局部特征：18项，包括原响应、三维曲率、区域尺寸、局部对比度、延伸形状及长轴两端延续程度。肺血管连接是一个评分特征，不是一票否决。
- 去重：结合毫米距离与三维包围盒重叠，按评分抑制重复区域，不通过传递式合并将一串相邻结构连成一个巨大候选。
- 评分对照：固定规则、线性评分及32棵深度为2的浅层规则树。线性模型为18个权重加1个偏置；浅层树运行时是特征比较与加法。没有第三套神经网络硬件核。

候选粗区域、包围盒和等效直径均不是专家精确病灶轮廓；分数也不是已校准的患病概率。保留的候选在效果图中全部显示，不用隐藏大量候选来冒充误报降低。

## 评价边界

原固定划分不变：90例开发、45例留出。本轮先使用之前已查看的24例开发扫描，其中28个参考结节；旧v3命中24个，平均277.375个误报/例。

小型评分器按扫描UID分为4折，同一CT的候选不能跨训练和验证；阳性候选的训练权重按参考结节平衡，避免一个大结节的多个候选压过小结节。原始PatientID不可得，因此不能宣称已独立证明患者级分组。

阈值和方案仍在这24例开发结果上选择，因此四折OOF结果属于开发比较，不是独立测试或官方LUNA16成绩。选择条件之一是逐参考检查是否丢掉旧v3已命中的结节；另外完整保存不同检出率下的误报曲线和所有方案。

在读取评分结果前，另按固定SHA256顺序选择6例此前未参与的开发扫描（4例有参考、2例无参考），清单位于 `data/lung_candidate_rules_20261018/next_development_plan.json`。该检查使用固定模型与阈值，不据结果再次调参。45例正式留出仍不执行推理。

## 代码入口

- `python/lung_candidate_refined.py:extract_refined_regions`：完整实验候选入口，接原始HU、已有肺掩膜、geometry及导出的评分模型。
- `python/lung_candidate_rules.py`：局部测量、宽松候选池、去重。
- `python/lung_candidate_feature_rank.py`：小型规则树训练与评分导出。
- `python/lung_candidate_rank_runtime.py`：仅NumPy依赖的特征表评分，可作为后续PS侧对接参考；不负责从CT计算特征。
- `python/lung_candidate_rules_run.py prepare`：重用已核对的肺掩膜，生成原24例紧凑特征。
- `python/lung_candidate_scale_patch.py`：统一补充独立2 mm尺度候选。
- `python/lung_candidate_rules_evaluate.py`：分组交叉验证、评分导出、开发阈值比较。
- `python/lung_candidate_rules_check.py`：固定模型在预选6例新开发扫描上的检查。

```python
import json
from lung_candidate_refined import extract_refined_regions

rank_model = json.load(open("selected_rank_model.json", encoding="utf-8"))
result, debug = extract_refined_regions(
    original_hu_zyx, existing_lung_mask_01, geometry, rank_model
)
```

`geometry`包含 `spacing_zyx_mm`、`origin_lps_mm`、`basis_lps_zyx`。中心索引允许浮点，供亚体素定位；包围盒仍为整数、零起始、ZYX左闭右开。调用者负责关联CT病例UID、原像素哈希和肺掩膜身份；不能将结果叠加到不同CT上。

四折输出使用对应折未见过该扫描的模型；`selected_rank_model.json`使用24例重新拟合，供新开发检查。不能用全量拟合模型在这24例上重跑后，冒充OOF效果。

## 复现命令

在项目根目录使用已有 `.venv-lung`，输出路径必须是新目录；不要覆盖本次结果。

```powershell
.\.venv-lung\Scripts\python.exe python/lung_candidate_rules_run.py prepare --outdir data/lung_candidate_rules_repeat/features
.\.venv-lung\Scripts\python.exe python/lung_candidate_scale_patch.py --features data/lung_candidate_rules_repeat/features --outdir data/lung_candidate_rules_repeat/features_scale2
.\.venv-lung\Scripts\python.exe python/lung_candidate_rules_evaluate.py --features data/lung_candidate_rules_repeat/features_scale2 --outdir data/lung_candidate_rules_repeat/evaluation
.\.venv-lung\Scripts\python.exe python/lung_candidate_rules_check.py --model data/lung_candidate_rules_repeat/evaluation/selected_rank_model.json --plan data/lung_candidate_rules_20261018/next_development_plan.json --outdir data/lung_candidate_rules_repeat/new_development_check
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -p "test_lung_candidate*.py" -v
```

复现第一步会读取此前归档的v1/v2数据，并核对压缩包哈希；不需要将大JSON恢复到松散目录。首次特征提取的精确执行源码另外保存在对应 `executed_sources` 目录。后续代码整理不回写旧结果或旧协议。

## FPGA与资源边界

本实验没有调用FPGA，CPU候选耗时不能写成PL耗时。新增局部测量和补充滤波会增加软件计算，不能仅凭评分器小就宣称整条链更快。后续应分别测量滤波、局部特征、评分、传输和整条链延迟。

板端应按切片或局部块处理，不能把电脑端整卷HU及所有调试数组原样搬进PYNQ-Z2内存。评分模型本身较小，但缓存、DMA、时钟、已有LDCT/U-Net模块和overlay切换仍须整机核算。优先研究滤波与局部统计的流式PL设计，候选关联和小型评分可先放PS；最终分配取决于实际资源与测速。

当前软件为便于对照，补充2 mm候选时重新计算了一遍该尺度。后续PL设计应直接复用多尺度滤波中已有的2 mm响应，同时保留其独立峰值，避免把软件参考里的重复滤波照搬为另一套硬件。新增需求是保留这条尺度信息，并不要求增加第三个CNN。

## 已完成的24例开发对照

| 方案 | 命中参考结节 | 平均误报/例 | 相对旧v3新增漏检 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 旧v3 | 24/28 | 277.375 | — | 同一24例的原始记录 |
| 局部特征与中心修正，未补充尺度 | 26/28 | 283.708 | 0 | 保住旧命中的操作点仍未减少误报；保留失败对照 |
| 补充2 mm独立峰值，采用固定组合规则 | 25/28 | 154.000 | 0 | 本轮选择的实验操作点，减少约44.48%的误报 |

小型分类器也做了同样的整例分组对照，本轮没有优于上述规则的综合取舍。因此实际选定文件 `evaluation_scale2/balanced_rank_model.json` 的 `kind` 为 `rule`，文件2817字节（含病例身份和说明），运行不需要额外训练模型或sklearn。规则使用局部对比度、延续程度与形状的连续惩罚；阈值为240.23485316759772，不能逐病例手调。

这组规则修复了 UID尾号 `109002525524522225658609808059` 中原先漏掉的约13.60 mm参考结节。仍有3个参考结节未检出：不能把“新候选池里曾出现过正确点”写成最终检出成功。特别是原约4.54 mm结节，独立2 mm尺度能生成距参考中心约0.29 mm的候选，但减少误报的最终操作点仍将其过滤。若追求这24例全部28个参考命中，本轮比较中需要约1892个误报/例，不能只报满召回而忽略误报代价。

所有开发操作点均保存在 `evaluation_scale2/tradeoffs.json`。仅要求总检出数相同可能出现“多找回一个、同时新漏一个”，报告另列 `lost_baseline_nodules` 和 `gained_nodules`，没有用总数掩盖病例替换。

`oof_comparisons/` 保存当前点逐例候选与指标；图固定选原来展示的首例及此前全部漏检病例，而非按改善程度选图。输出统计逐项核对为25/28、3696个误报，与评价一致。完整实际调用另从原始ZIP重新读取首例，95个候选的中心、包围盒、分数和模型身份与导出包一致，见 `standalone_parity.json`。

图上橙框是候选粗区域在当前层的截面，青色是参考中心/直径。三维点匹配成功不等于粗框覆盖参考中心所在的每一层；后续ROI应根据中心、毫米尺度和上下文边距确定范围，不能把粗框当精准结节轮廓。

上述文件均位于 `data/lung_candidate_rules_20261018/`。六例新开发检查单列如下，不与开发选参结果混写成最终测试集成绩。

## 六例新开发检查：已完成

使用提前固定的6例及同一规则/阈值，未按新结果调整参数。四例含参考结节、两例无参考结节，共5个主参考结节。

| 方案 | 命中参考结节 | 误报总数 | 平均误报/例 |
| --- | ---: | ---: | ---: |
| 原v3 | 4/5 | 1861 | 310.167 |
| 本轮固定规则 | 4/5 | 674 | 112.333 |

误报减少约63.78%，没有新增漏检，也没有在这6例中修复原来的那一处漏检。样本很小，不能由此声称已达到成熟检出性能。报告位于 `new_development_check/report.json`，逐例候选、指标与同层对比图均保留。

这6例使用冻结原FP32 U-Net重新计算肺掩膜，并对同一掩膜同时生成旧v3与新候选；不是现有INT8/PL掩膜的端到端验证。接入板端分割后，仍须检查量化/后处理引起的掩膜变化是否影响候选结果。LDCT、U-Net原模型和网页默认行为未修改。

## 保持结果一致的计算剪除

当前固定规则的对比度因子不超过1.5，延续与形状因子均不超过1。因此候选规则分数上限为 `1.5 × blob_score`（实现按实际存储的float32 log-score精确构造）。上限也低于阈值的点，无需进行局部高斯、连通区域和形状测量。此优化不依赖标注，也不改变规则阈值。

首例验证中，局部特征测量从12,720个候选减至793个，跳过11,927个、约93.77%；最终仍为同样95个候选，中心、包围盒、分数及模型身份与未剪除导出结果逐项一致。记录为 `optimized_parity.json`。仅减少局部测量数量，不代表整条链加速93.77%。完整CPU入口该次约31.57秒；此前未剪除入口是在并发开发实验下测得，不能用两次时间计算公平加速比，更不是PL时间。

6例质量检查的执行源码已保存；这项剪除是在质量检查后加入的等价计算优化。当前共有19项候选回归通过，含上界不会删除可通过候选的检查、导出评分与训练实现对拍、输入不修改、坐标、去重和纯NumPy评分运行时隔离。没有重新训练神经网络、访问45例留出或生成FPGA文件。

可以先保留这版作为后续ROI/三维联动的实验候选入口，但154或112个误报/例仍多，不能当成最终低误报检测器。下一轮应扩大90例内部的开发验证、复核剩余漏检，并测量整机时延，再冻结算法做PL设计。
