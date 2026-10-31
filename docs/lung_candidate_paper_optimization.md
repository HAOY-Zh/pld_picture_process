# 肺结节候选：论文方法借鉴与低误报优化（2026-10-18）

本轮完成了可运行的候选分类实验及独立审计，**未替换网页默认算法，也没有修改 LDCT、U-Net 或 FPGA bit**。新模型在低误报区间有改善，但在 6 例复核中新增一个漏检，尚不满足高检出率版本的替换条件。

## 借鉴了什么

主要参考 Gu 等人的 [Automatic lung nodule detection using multi-scale dot nodule-enhancement filter and weighted support vector machines in chest computed tomography（2019）](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0210551)。该文通过三维梯度、特征选择、负样本欠采样和加权 SVM 降低血管分叉造成的误报。

本轮是结合现有系统的改编，不是论文完整复现；未使用作者训练权重。

| 环节 | 本轮实际实现 | 与论文的区别 |
|---|---|---|
| 初筛 | 沿用现有 LoG、Hessian 与规则 v4 的最终候选 | 没有重新实现论文的血管树分组、Otsu 和完整区域生长流程 |
| 三维局部描述 | 原来的 13 维描述，补充局部对比度、灰度波动、占比、伸长、血管延续性、中心偏移和背景，共 20 维 | 复用已有数值缓存，不将专家坐标作为输入 |
| 球壳梯度 | 候选周围 3 层球壳，42 个成对反向的固定方向；统计灰度、梯度强度、向心一致性、方向不对称及亮点比例，共新增 21 维 | 这是物理毫米尺度下的局部采样实现，非论文球壳梯度公式的逐项复刻 |
| 特征选择 | 每个训练折内生成 64 个随机子集，各含 12 维；以三折病例分组的 3-NN Gmean 评分，保留 16 维 | 借鉴 RSFS 思想；使用“出现与未出现时的平均得分差”，非原论文完整随机游走统计 |
| 困难负样本 | 仅训练时保留全部正候选；负候选由 75% 近正样本的困难负例与 25% 随机负例组成 | 固定混合策略；没有在推理时按每病例数量截断 |
| 分类 | 按类别及每个参考结节的正候选总权重加权训练 RBF SVM | 小规模开发探索，非论文的完整嵌套参数搜索 |

推理过程不读取专家标注，不新增 CNN。上述方向是用更多三维信息区分结节与血管分叉，并非再次调整 HU 阈值。

## 数据与实验约束

- 24 例开发数据、28 个参考结节；沿用已有四折病例分组划分，外折验证病例不参与该折训练或特征选择。
- 参数与算法选择会使用这 24 例的汇总结果，因此这里的 OOF 指标仍属于开发探索，不能称为无偏的嵌套交叉验证结果。
- 原有另 6 例开发复核包含 5 个参考结节。这 6 例以前已经看过，因此也不是新的独立测试集。本轮先冻结模型，再补测这些既有候选的局部特征；未根据复核结果重新拟合或调阈值。
- 45 例封存留出数据没有运行、没有调参。
- 原始 PatientID 不可用，当前仅保证 scan UID 分组及像素重复检查，不能证明完整的患者级独立性。

先比较了 13/20 维评分器，再比较 41 维描述，最后比较论文式特征选择与负样本筛选。球壳特征直接全部输入分类器没有稳定胜出，不能只展示最好看的结果。各轮完整参数、折模型与分数均保留。

最终候选选择标准为：**在保住开发集原有 25 个检出结节、且该工作点的误报不超过上一版 SVM 的前提下，选择开发 CPM 较高的方案**。这个约束是在开发探索之后、读取本轮 6 例复核之前确定的，详见 `selection_freeze.json`，不是事先注册的临床研究方案。

选中：41 维中保留 16 维，负正候选数比上限 20，SVM C=1。全开发集拟合模型含 186 个支持向量，JSON 104,514 字节；原 13 维 SVM 为 893 个支持向量、357,234 字节。模型变小不能直接等同于板端或端到端加速。

## 实际效果

下面是分别在各数据组扫阈值得到的**本地经验 FROC 工作点**。它们描述曲线，不是某个冻结阈值在新病例上必然达到的误报数量。

| 数据组 | 方法 | 每例 1 个误报时检出率 | 每例 4 个误报时检出率 | 每例 8 个误报时检出率 | 本地 CPM |
|---|---|---:|---:|---:|---:|
| 24 例开发 / 28 结节 | 上一版 SVM | 32.1%（9/28） | 46.4%（13/28） | 64.3%（18/28） | 0.306 |
| 24 例开发 / 28 结节 | 新论文式方案 | 39.3%（11/28） | 64.3%（18/28） | 64.3%（18/28） | 0.449 |
| 6 例已查看开发复核 / 5 结节 | 上一版 SVM | 20%（1/5） | 40%（2/5） | 80%（4/5） | 0.286 |
| 6 例已查看开发复核 / 5 结节 | 新论文式方案 | 60%（3/5） | 60%（3/5） | 60%（3/5） | 0.400 |

CPM 使用 0.125、0.25、0.5、1、2、4、8 FP/scan 七点线性插值的平均值。匹配采用严格“中心距离 < 参考直径/2”，同一结节多个命中只计一次、取最高分；重复命中和排除标注不作为误报，前级漏检始终保留在检出率分母。

本地计算与官方脚本的 ROC 转换方式交叉检查一致，但**未执行完整 LUNA16 十折挑战协议、官方每例 100 个候选限制和 bootstrap**，所以不能写成“官方 LUNA16 CPM”。没有通过 top-K 截断降低误报。

真正将开发阈值冻结后应用到 6 例，结果如下：

| 档位 | 方法 | 6 例检出 | 实际误报/例 |
|---|---|---:|---:|
| 开发集 4 FP/scan 对应阈值 | 上一版 SVM | 3/5 | 4.67 |
| 开发集 4 FP/scan 对应阈值 | 新论文式方案 | 3/5 | 3.50 |
| 保住开发集 25 个既有检出的阈值 | 上一版 SVM | 4/5 | 25.83 |
| 保住开发集 25 个既有检出的阈值 | 新论文式方案 | 3/5 | 23.67 |

新方案低误报档保留同样的 3 个检出，平均误报减少 25%；高检出率档新增漏检，**不通过默认替换门槛**。漏掉的是 UID 尾号 `121824995088859376862458155637` 中约 5.19 mm 的参考结节：已有候选被分类器打低分，并非坐标变换错位。不得根据这一复核结节单独放宽阈值来包装成绩。

这些数据远不足以证明接近成熟论文效果。下一轮优先扩充尚未使用的开发病例，增加小结节和困难负例训练，再验证；不继续反复针对这 6 例搜索。对前级根本未产生候选的结节，二阶段分类器无法恢复，需另做初筛召回率审查。

## 代码与交付

- `python/lung_candidate_froc.py`：本地 FROC、经验工作点及 CPM。
- `python/lung_candidate_cached.py`：校验并复用既有开发描述，拒绝封存病例。
- `python/lung_candidate_lowfp.py`：小型 SVM / 树评分器及 NumPy 推理。
- `python/lung_candidate_shell.py`、`lung_candidate_shell_prepare.py`：新增球壳梯度及候选局部测量缓存。
- `python/lung_candidate_paper.py`、`lung_candidate_paper_experiment.py`：论文式特征选择、训练负样本筛选和加权 SVM。
- `python/lung_candidate_lowfp_experiment.py`：不含特征选择的对照实验。
- `python/lung_candidate_lowfp_check.py`：先冻结选择，再对既有 6 例做局部补充测量。
- `python/lung_candidate_lowfp_apply.py`：原始 HU + 完整 v4 候选结果的可选推理适配器，校验模型身份；输出保持候选中心和框不变。
- `python/lung_candidate_lowfp_audit.py`：独立复算分数、阈值、几何与 FROC，验证各折病例隔离。

完整结果在 `data/lung_candidate_lowfp_20261018/`：

| 路径 | 用途 |
|---|---|
| `baseline_development_froc.json` | 原规则与旧 SVM 的低误报基线 |
| `search/` | 13/20 维初轮对照；精确源码快照及 `source_resolution_audit.json` 保留了后续接口重构前的源码身份 |
| `shell_features/` | 24 例共 3,768 个既有候选的新局部描述；未重跑 U-Net/LoG |
| `shell_search/` | 直接使用 41 维特征的对照 |
| `paper_search/` | 特征选择与负样本筛选的九组对照，以及每个训练折的选择结果 |
| `frozen_check/model.json`、`runtime_contract.json` | 本轮冻结模型及模型身份契约 |
| `frozen_check/report.json` | 曲线、冻结阈值结果、局部特征与分类计时 |
| `frozen_check/miss_review.json`、`deployment_gate.json` | 新漏检审查与不替换默认版本的结论 |
| `frozen_check/development_oof/`、`six_check/` | 两档输出；逐包绑定实际折模型或全开发模型 SHA |
| `frozen_check/output_audit.json` | 60 个输出包、97 个受保护文件及 11 个原模块源码核对通过 |

已通过 21 项相关测试和 60 个输出包的独立核对。测试命令（在项目根目录）：

```powershell
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -p test_lung_candidate_*.py
```

`output_audit.json` 已生成，审计脚本拒绝覆盖旧审计结果。重做实验须指定新的输出目录。不要为了出图重复跑整套 CT 推理。

可选推理示例，仅用于实验集成；`profile='4.0'` 指开发阈值来源，并不保证输入病例正好四个误报：

```python
import json
from pathlib import Path
from lung_candidate_data import sha_file
from lung_candidate_lowfp_apply import apply

root = Path('data/lung_candidate_lowfp_20261018/frozen_check')
model = json.loads((root / 'model.json').read_text('utf-8'))
contract = json.loads((root / 'runtime_contract.json').read_text('utf-8'))
primary_path = Path('data/lung_candidate_rules_20261018/evaluation_scale2/balanced_rank_model.json')
primary = json.loads(primary_path.read_text('utf-8'))
# hu: 产生 result 时使用的原始 HU；result: 完整 v4 结果，含原始峰值和结构描述。
experimental = apply(hu, result, model, primary, sha_file(primary_path), contract, profile='4.0')
```

旧 SVM 的高检出率路线继续使用 `lung_candidate_svm_apply.py` 和旧模型。网页未自动切换到本轮模型。当前新增处理均为 CPU 参考实现，局部计时不包含 CT 读取、原始候选检测或网页传输，不能作为 FPGA 提速成绩。

## FPGA 适配边界

三维局部平滑、差分梯度、固定采样和统计归约可作为后续 PL 优化对象；SVM 支持向量和计算量较小，可先放 PS，保留现有 FPGA 肺分割接口。是否真正加速必须再测 ARM、DMA、PL 和端到端时间。目前没有新增 RTL、综合资源结论或板端验证，不应给出已达到的 FPS。
