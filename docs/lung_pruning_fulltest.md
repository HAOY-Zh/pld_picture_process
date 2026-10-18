# 冻结剪枝候选的全病例测试

本轮固定 `balanced_recovered`，完整评价17个LCTSC可用留出病例（2705层）和10个COVID原HU病例（2581层）。
共27例、5286层，包含空肺层。原18个训练和6个开发验证病例的用途不变，不混入本轮测试均值。

24个LCTSC测试目录均复查，另外7个仍只有RTSTRUCT、没有CT；COVID另外10份已窗宽化UINT8，按原协议排除。
没有根据模型得分排除病例。缺失和排除清单保存在本轮 `coverage.json`。

旧数组缓存已经清理，本轮从原始CT/RTSTRUCT与官方肺标注ZIP重建。
27例归一化输入及原生专家掩膜均与历史SHA256字节一致，外部HU缓存也匹配原哈希。
缓存与恢复证据在 `data/lung_prune_test_cache_20261017`，没有覆盖旧报告或修改专家标签。

原模型和冻结剪枝候选分别重新执行FP32与FPGA整数计算参考，主要比较后者。
两模型均保持HU裁剪、256输入、FP16缓存、sigmoid后原生双线性概率恢复、0.5阈值及相同后处理。
原始、CT-only身体约束和原三维清理后的结果分别报告。

这次不训练、不重新校准、不选检查点、不调阈值或后处理。
病例均值、小肺/空肺、全部逐切片差异和二维边界指标分开报告；非空但零真阳性的误报不算成功检出。
原模型此前已看过这批数据，所以这是冻结候选的确认，不称为全新盲测。
电脑执行的硬件整数参考不代表实板病例推理或板端速度。

正式输出在 `data/lung_prune_fulltest_20261017_v2`：

- `protocol.json`：推理前冻结的模型、病例、参数、代码和源数据哈希。
- `evaluation.json`、`case_metrics.csv`：分数据集汇总及全部病例。
- 每病例四份 `*_native.npz`：原模型/剪枝模型FP32/定点的原尺寸原始与最终预测。
- 每病例 `fixed_final_slice_metrics.csv`：所有层的面积、重叠与新增/恢复漏分。
- `verification.json`：全部病例完成后的覆盖及哈希核验。
- `returned_mask_audit.json`：所有324组掩膜的TP/FP/FN/Dice独立重算。
- `full_test_report.md`、`paired_fixed_dice.png`、`worst_case_native_review.png`：审计后生成的完整报告、病例配对图和失败切片。

相关入口：`python/lung_prune_test_cache.py`、`python/lung_prune_fulltest.py`、
`python/lung_prune_fulltest_audit.py`、`python/lung_prune_fulltest_report.py`。
所有输出拒绝覆盖。首次启动因旧缓存缺失而中止的目录保留，实际推理在重建后新建的v2目录中完成。

## 已完成结果

27/27例、5286/5286层全部完成，324组掩膜的TP/FP/FN/Dice、小/中/大肺分组及空肺误报独立重算通过。
候选、原模型、原数据和处理参数均核验不变，没有训练或重新校准。

以下比较硬件整数参考经过相同后处理的最终结果，Dice先按病例计算再平均：

| 数据集 | 病例/层数 | 原模型平均Dice | 剪枝平均Dice | 变化（百分点） |
| --- | ---: | ---: | ---: | ---: |
| LCTSC可用留出子集 | 17/2705 | 0.965322 | 0.963894 | -0.143 |
| COVID原HU外部集 | 10/2581 | 0.959748 | 0.960707 | +0.096 |

两组平均值均在用户接受的下降不超过1个百分点范围内，但均值没有反映全部风险：

- LCTSC-Test-S2-204：Dice 0.962151 → 0.920850，下降4.130个百分点。
- coronacases_010：Dice 0.947068 → 0.931456，下降1.561个百分点。
- LCTSC空肺假阳性像素10,091 → 119,108；COVID为36,326 → 200,765。
- 小肺切片平均Dice：LCTSC 0.612106 → 0.607917；COVID 0.588545 → 0.560591。
- 退化也存在于FP32路径，不能只归因于定点量化误差。没有通过更改标注、阈值或清理参数来提高测试成绩。

**当前候选继续保留为性能实验，不建议直接替换原模型。** 此前六个开发验证病例上的平均Dice略升，
不能推导为全量病例均改善。后续若据这些测试结果重新调整模型，这些病例就成为开发反馈，不再称为新的盲测。
硬件工作量减少32.24%仍是结构估算，真实板端提速需独立上板确认。

完整可读报告：[27例全量结果](../data/lung_prune_fulltest_20261017_v2/full_test_report.md)。
独立复算：[掩膜审计](../data/lung_prune_fulltest_20261017_v2/returned_mask_audit.json)。
可视化：[全部病例配对图](../data/lung_prune_fulltest_20261017_v2/paired_fixed_dice.png)、
[退化病例原生切片](../data/lung_prune_fulltest_20261017_v2/worst_case_native_review.png)。
`data/`属于本机实验产物并被Git忽略，上述链接在本机可用，不能假定GitHub上也包含这些文件。
