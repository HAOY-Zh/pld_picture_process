# 肺结节成熟算法调研与缓存 SVM 第二阶段实验

日期：2026-10-18。主实验目录：`data/lung_candidate_svm_20261018_v2/`。

本轮根据用户要求先检索成熟方法，再采用能复用现有候选的第二阶段分类。没有重新读取原始 CT、运行 U-Net、计算 LoG/Hessian 或提取局部图像特征；没有修改 LDCT、U-Net、原候选模块、网页默认值或板端部署。45 例留出集未运行。

## 1. 原算法与文献方法的关系

原版本是自定义组合的候选原型：原始 HU + 原 U-Net 肺掩膜 → 逐层多尺度二维 LoG → 独立 sigma=2 mm 补峰 → 三维 Hessian/局部结构特征 → 中心修正 → 手工软评分 → 三维去重。sigma 是滤波尺度，不能直接当作结节直径。

它已使用三维信息，但没有完整复现一套公开论文 CAD 系统。本轮新增的是监督学习的误报筛除层，不把原方法重新命名成论文算法。

| 方法及原始来源 | 对本项目的判断 |
| --- | --- |
| [Li 等，2003，选择性点/线/面增强](https://pubmed.ncbi.nlm.nih.gov/12945970/) | 有明确的形状增强依据。我们已有 Hessian 特征可衔接，但当前比例公式和整条流程不能称为其完整复现 |
| [Gu 等，2019，多尺度点状增强与加权 SVM](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0210551) | 采用 RBF 加权 SVM 区分血管相关候选；优先借鉴其分类原则。原文还包括分组、壳层/表面特征及特征选择，本轮未复现这些部分，也没有下载现成训练权重 |
| [Setio 等，2016，多视角二维 CNN 筛误报](https://geertlitjens.nl/publication/seti-16/) | 从候选周围不同方向截取二维图块，再融合分类。可作为后续质量路线；会新增 CNN 训练、量化和调度适配，本轮未实施 |

本轮选择：**原候选生成器 + 既有结构描述 + 加权 RBF SVM 第二阶段**。属于文献启发的适配实验，不能写成论文完整复现或论文精度已重现。

## 2. 实际增加的代码

- `python/lung_candidate_svm.py`：13 维描述子、训练用加权 RBF SVM、纯 NumPy 推理导出。只在调用训练函数时导入 sklearn。
- `python/lung_candidate_svm_experiment.py`：原 24 例缓存上的按扫描分组四折实验；固定参数后只回放另外 6 例候选缓存。
- `python/lung_candidate_svm_apply.py`：可选调用入口，接收完整 v4 候选结果，校验原规则身份后筛除候选；不移动中心、修改 bbox 或读取标注。
- `python/lung_candidate_svm_audit.py`：独立复算 30 份输出的指标、模型身份、几何和定位误差。
- `python/tests/test_lung_candidate_svm.py`：4 项测试通过，涵盖两种缓存描述一致、模型导出一致、无 Torch 推理导入、原结果不被修改及身份错误拒绝。

描述子包括原规则得分、Hessian blob 分数/比例、估计尺度、粗区域等效直径/延伸、平均/最高 HU、三个方向 bbox 毫米尺寸、轴比和粗区域对 bbox 的填充比例。粗区域来自响应，不是专家结节轮廓；相关体积不是精确结节体积。

## 3. 实验设计与泄漏边界

先从既有缓存逐项恢复原规则的最终候选，并核对中心顺序、原分数与之前导出的包完全一致。保留原规则的安全上界提前剪枝；SVM 只筛已经保留下来的候选，不需要给所有被剪枝点补算局部特征。

1. 原 24 例按之前冻结的 4 折分组，一整套 CT 不跨训练/验证。原始 PatientID 不可得，所以只能声称扫描分组，不能声称已证明患者级独立。
2. 标准化只拟合训练折；忽略 excluded 候选，不把它们当普通负样本。正负类加权，另按参考结节平衡同一结节的重复候选。
3. 固定只比较 6 组：C=0.1/1/10，gamma=`scale`/0.1。没有继续根据 6 例结果追加网格。
4. 在 24 例 OOF 分数上选尽量减少 FP、同时保留原规则已检出全部 25 个结节的阈值。因此开发集“不新增漏检”是选择条件，不能当作独立证据。
5. 参数和全 24 例拟合模型写入磁盘后，才读取另外 6 例缓存评分。它们此前已被观察过，仍是 development check，不是新盲测。

原规则本身的阈值此前也曾在这 24 例上选择。本轮 OOF 仅隔离 SVM 的训练，不能描述成整个系统的严格嵌套盲测。

选中 C=0.1、gamma=`scale`，阈值 -0.6901288676397581。SVM 输出为未校准 margin，可为负数，不是患病概率。选择阈值时检查每个原检出结节是否仍有命中，不只比较 TP 总数。

## 4. 实际结果

| 范围 | 指标 | 原规则 | 加 SVM |
| --- | --- | ---: | ---: |
| 24 例开发、28 个参考 | 检出 | 25/28 | 25/28 |
| 同上 | Sensitivity | 89.29% | 89.29% |
| 同上 | FP/scan | 154.00 | 30.67 |
| 同上 | 候选/scan | 157.00 | 32.50 |
| 同上 | FP 总数 | 3696 | 736 |
| 另外 6 例开发检查、5 个参考 | 检出 | 4/5 | 4/5 |
| 同上 | Sensitivity | 80.00% | 80.00% |
| 同上 | FP/scan | 112.33 | 25.83 |
| 同上 | 候选/scan | 116.17 | 28.00 |
| 同上 | FP 总数 | 674 | 155 |

两组 FP 分别减少约 80.1% 和 77.0%。仍有原先的 4 个漏检；第二阶段只能删除现有候选，不能恢复前面没有生成或已被规则删除的结节。每例仍有约 26～31 个误报，尚未达到少量候选的最终展示目标。

独立检查 `output_audit.json`：30 份包、60 次前后指标复算通过，66 个保护文件及 11 个原候选代码/要求文件未变；六例运行适配器的候选选择、中心、框全部一致。不同 BLAS 线程数的 margin 差异最大约 1.93e-14，未改变任何候选选择。

各参考结节最近命中中心的距离也未变化：24 例 TP 平均/中位数约 1.389/0.668 mm；六例约 2.329/2.077 mm。该项是点定位误差，不是精确轮廓评价。

## 5. FPGA 与现有系统的适配

根据本次实现作出的工程判断：优先把重复滤波、固定局部统计等部分作为 PL 候选，SVM/排序/去重先留 PS 或电脑。当前结节生成仍是 CPU 原型，不能写成上述算子已经上 PL。

SVM 文件 357234 字节（约 349 KiB）、893 个支持向量，13 维输入；推理只需 NumPy，不添加 CNN。对当前小批候选，先在 PS 做 RBF 计算更容易集成；是否需要定点或 PL 化应由实际 Z2 测量决定。RBF 还包含指数运算，不能因为名字不是 CNN 就声称完全免费。

本次 PC 六例的纯 margin 计算约 1.2～4.9 ms，仅包括已准备描述子的 SVM 评分，不含原始候选生成、网络、DMA、通信或完整检测，不能用它宣称整例检测毫秒级。未做任何新增 bit、资源估算或板端加速比。

已有 U-Net 提供肺搜索区域，候选/SVM 输出继续交给 ROI、MPR 和三维显示，接口仍为中心、粗 bbox、连续分数。本轮肺掩膜沿用以前的原 FP32 U-Net 缓存，正式 INT8/PL 肺掩膜上的候选效果仍需另行确认。LDCT 不作为本轮候选输入，也没有声称它提升了结节检测。

## 6. 复现与可选调用

训练环境安装 `python/requirements-candidate-dev.txt`（需 sklearn）；评分接口只依赖 NumPy。缓存实验输出必须使用新的目录：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_candidate_svm_experiment.py --outdir data/lung_candidate_svm_new_run
.\.venv-lung\Scripts\python.exe python/lung_candidate_svm_audit.py --experiment data/lung_candidate_svm_new_run
```

推荐实际实验为 `data/lung_candidate_svm_20261018_v2/`：`report.json`、`development_oof.json`、`six_development_check.json`、`selected_svm.json`、`selected_fold_models/`、`output_audit.json`。24 例 OOF 包绑定各自真正的折内模型；六例包绑定全 24 例拟合后冻结的模型。

完整 v4 结果的可选调用示例：

```powershell
.\.venv-lung\Scripts\python.exe python/lung_candidate_svm_apply.py `
  --candidates data/lung_candidate_rules_20261018/new_development_check/1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866/candidates.json `
  --model data/lung_candidate_svm_20261018_v2/selected_svm.json `
  --primary-rule data/lung_candidate_rules_20261018/evaluation_scale2/balanced_rank_model.json `
  --output data/lung_candidate_svm_manual/candidates.json
```

如果上游只提供旧 OOF 精简包、缺少结构描述，接口会拒绝；不要用零值补齐缺失特征。它与完整运行结果的结构字段不同，开发 OOF 包应通过保存的特征表回放。

首轮 `data/lung_candidate_svm_20261018/` 保留为过程记录，数值结果相同，但其 OOF 输出模型身份标记不完整，已标 superseded；不要作为接口交付。v2 已修正实际模型绑定，并由独立审计核验。

目前结论：值得保留为新的候选方案，尚不自动部署或替换网页。下一步优先在其余开发病例检验原候选与第二阶段的效果，再冻结整体协议，最后才运行封存 45 例。不要根据反复观察这 30 例来声称独立泛化达标。
