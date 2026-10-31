# 肺结节可疑区域提取：首版软件模块

日期：2026-10-18。当前输出是候选中心、粗区域包围盒和排序分数，用于提示进一步查看的位置；不提供良恶性诊断或精准结节轮廓。本轮按用户要求直接开发模块，没有修改三人分工文档。

## 当前能力与入口

已实现原生 CT 网格上的肺内多尺度亮斑初筛、三维局部结构过滤、物理距离去重和候选结果导出。使用既有冻结肺分割提供搜索区域，不新增结节神经网络。

推荐调用入口：`python/lung_candidate_structure.py` 的 `extract_suspicious_regions`。

```python
from lung_candidate_structure import extract_suspicious_regions

result, debug = extract_suspicious_regions(
    volume_hu,             # 原生 (Z,H,W)，真实 HU，不能传窗宽图
    predicted_lung_mask,   # 同网格 0/1；调用者已完成对应层的肺预测
    geometry,              # spacing_zyx_mm / origin_lps_mm / basis_lps_zyx
)
candidates = result['candidates']

# 交给后续自动ROI模块时，可移除调试字段，保留坐标与输入身份
from lung_candidate_structure import roi_candidate_packet
packet = roi_candidate_packet(result)
```

函数不读取病例标注、CSV、训练模型或网络服务。输入掩膜必须已就绪，不能把未计算层当作背景。现有掩膜如为 0/255，应先验证其二值编码，再明确转为 0/1。

模块依赖 NumPy、SciPy。离线验证脚本额外使用本机原有 PyTorch、绘图与 CT 读取依赖；未安装新环境或改动冻结模型。

## 算法链路

1. **搜索区域**：使用既有肺预测及二维人体范围约束；对搜索掩膜单独填孔并按物理尺度外扩 3 mm。保留原肺输出，不用扩展掩膜冒充更好的肺分割。
2. **多尺度 LoG**：在原生分辨率计算 1、2、4、6 mm 尺度响应。物理二阶导数除以对应间距平方，按尺度平方归一化；高斯导数核显式消除直流项。反射边界，先滤 CT 后施加搜索掩膜，避免将掩膜边缘变成人工亮斑。
3. **初筛**：响应至少 12，HU 范围为 [-950,2000]。用于响应计算的 HU 临时裁剪为 [-1000,600]；原始 HU、几何和后续 HU 统计不改变。
4. **跨层合并与局部粗区域**：三维局部极值、毫米距离合并、局部连通区域、尺寸/延伸程度筛选。初筛粗区域不是专家结节轮廓，等效直径也不是已验证的真实结节直径。
5. **三维结构**：按候选尺度做三维物理高斯平滑，计算六项 Hessian 二阶差分及特征值。用三个方向的负曲率一致性减少长血管等误报。
6. **当前开发操作点**：球状响应比至少 0.2、三维结构分数至少 180、粗区域等效直径至少 2.5 mm、延伸比不超过 4，再按物理距离去重。此参数在首组 12 例开发病例中选择，尚不是正式冻结的最终方案。

LoG 的可分离计算为：

```text
R = -sigma_mm² × [ convolve_y(D2y, convolve_x(Gx, I))
                  + convolve_y(Gy, convolve_x(D2x, I)) ]
```

其中 D2y、D2x 已按物理间距换算。当前是浮点软件参考；系数、边界和三维统计已独立成模块，便于下一步开展定点/FPGA设计，**本轮没有实现新 RTL、生成新 bit 或进行板端提取验证**。

## 输出契约：供候选展示和后续 ROI 自动框选消费

每个 `candidate` 包含：

| 字段 | 定义 |
| --- | --- |
| `candidate_id` | 当前病例和算法版本内唯一 ID；修改参数后不能据此复用旧 ROI |
| `center_index_zyx` | 原生 CT 中心索引，顺序 ZYX |
| `center_lps_xyz_mm` | 同一中心的 DICOM LPS 世界坐标，XYZ，毫米 |
| `bbox_zyx` | `[z0,z1,y0,y1,x0,x1]`，从 0 开始，左闭右开 |
| `score` / `score_kind` | 三维曲率结构排序分数，不是患病概率 |
| `voxel_count` / `region_volume_ml` | 响应粗区域体素数和体积，不是专家病灶真值 |
| `equivalent_region_diameter_mm` | 由粗区域体积计算的等效直径 |
| `mean_hu` / `min_hu` / `max_hu` | 在粗区域原始 HU 中计算的统计 |
| `structure_features` | 三维曲率特征，供复核和调试 |
| `source_candidate_id` | 原宽松初筛中的对应 ID，便于追溯 |

离线运行保存的 `candidates.json` 还包含病例 UID、原生尺寸、geometry、原始像素哈希、肺 Mask 哈希、参数、后端和版本等身份字段。单独调用函数时，调用者负责关联输入身份；不能将不同 CT 的候选直接叠加。

世界坐标关系为：`world_xyz = origin_lps_mm + basis_lps_zyx @ index_zyx`。basis 已包含方向和间距，不能再乘一次 spacing。

`debug` 只用于算法调试：其中 `search_mask` 是完整原生形状；`seed_mask`、`response`、`scale_index` 是裁剪数组，偏移由 `bbox_zyx` 给出，不得直接当全图坐标使用。可选的 ROI 外扩边距由消费方按毫米定义；候选算法不把参考标注当作自动框选提示。

## 数据、划分和评价边界

- 数据来自已核验的 `D:\CT4\luna16-1.zip`，135 例、167 个参考结节。输入字节和几何与原审计逐例核对，完整原卷按需读取，包含超过网页一次加载体素上限的开发病例。
- 冻结清单：`data/lung_candidate_split_20261018/split.json` 及相应 SHA256。90 例开发：58 例含 111 个参考结节、32 例无参考结节；45 例留出：29 例含 56 个参考结节、16 例无参考结节。
- 以扫描 UID 分组，禁止同一体数据的切片跨组，重复原始像素会拒绝划分。但镜像没有原始 PatientID，不能声称已独立证明不同扫描全部来自不同患者。
- 已查看的首例只放开发集。首组 12 例（8 例有参考、4 例无参考）用于参数开发；另组 12 例开发检查与首组不重叠，按固定元数据规则选取。第二组按首组确定的同一参数运行，不逐例调整。
- 本轮运行入口只允许开发集，不提供执行 45 例留出的选项。日后确认算法和参数后，再单独制定留出评价入口。
- 参考中心/直径只在候选产生后参与评分。匹配采用候选中心距小于参考直径一半；重复命中和官方排除项不计作额外误报，排除项负直径按官方参考脚本的 10 mm 缺省直径处理。
- 已取得并验证官方 `evaluationScript.zip` 的 MD5 `02680d438a80dc26eeff0c12e1942642`，只读取其规则与排除标注，没有执行旧版 Python 2 评价代码。本轮是自用开发统计，不能标为官方 FROC/CPM。

## 已运行记录

| 记录 | 范围 | 结果与含义 |
| --- | --- | --- |
| `data/lung_candidates_pilot_20261018_v1` | 首组 12 例，宽松 LoG 初筛 | 9/10 参考结节命中，平均约 19971.42 个误报/例；保留作初筛对照 |
| `data/lung_candidates_pilot_20261018_v2` | 同组，宽松三维结构 | 9/10，平均约 8980.17 个误报/例；保存所有结构特征 |
| `data/lung_candidates_pilot_20261018_v3` | 同组，首轮选定操作点 | 9/10，平均约 360.58 个误报/例；本组参与选参，不是盲测 |
| `data/lung_candidates_checkdev_20261018_v3` | 另一组 12 例开发检查 | 15/18，约 83.33%；平均约 194.17 个误报/例；使用相同参数完整运行 |

首组选参比较的全部 96 个操作点保存在 v2 的 `operating_points_exploratory.json`；还有几种更紧的参数会进一步降低候选数并损失检出，没有用这些参数冒充“同样高召回”。选定操作点保留 2.5 mm 的粗区域门槛，未采用试验中 3.5 mm 的更紧尺寸门槛。

已定位首组漏检：病例 UID 尾号 `144883090372691745980459537053` 的一个约 4.54 mm 参考结节，搜索和粗筛有覆盖，但局部峰值中心偏离参考匹配半径。其失败记录与图仍保留，没有手工补点。

第二组 18 个参考结节在峰值、跨层、尺寸、初步形状阶段均被保留；最终结构筛选降为 15 个。三处漏检参考直径约 13.60、16.98 和 6.29 mm，不能只归因于小结节。后续应优先复核结构分数、形状假设与阈值，不据这些结果宣称三维过滤没有精度代价。

首组 v3 重用了未变化的 Hessian 特征，缓存筛选耗时不等于完整提取时间。第二组完整运行的 `candidate_ms` 包含软件初筛、跨层处理和三维结构；肺分割另记 `lung_ms`。这些是电脑参考耗时，不是板端或网页端到端时间。

较宽松版本的原执行源码已保存于 `data/lung_candidate_sources_20261018_pilot_v1_v2`，防止后续调用入口整理覆盖原实验的实现依据。

第二组完整执行源码保存在 `data/lung_candidate_sources_20261018_check_v3`。两组共有 24 例、28 个参考结节，命中 24 个；这是两组开发结果，不是正式测试集或完整 LUNA16 成绩。当前误报仍多，适合候选算法开发与接口联调，尚不能写成成熟的低误报检出系统。

多尺度候选思路参考 [LoG 肺结节候选研究](https://arxiv.org/abs/1907.08328)。本工程是逐层 LoG 加三维结构筛选的实现，不是对该论文算法、数据或检出率的完整复现。

## 复现命令

在本机项目根目录，用已有 `.venv-lung`。以下输出目录必须不存在，重新运行时改用新目录，不覆盖旧结果。

```powershell
# 固定划分已经执行；此命令只展示首次生成方式，不能再次覆盖同目录
.\.venv-lung\Scripts\python.exe python/lung_candidate_run.py freeze-split --audit data/luna16_ingest_20261018/audit.json --outdir data/lung_candidate_split_20261018

# 完整运行当前软件模块；默认只使用预先指定的12例开发扫描
.\.venv-lung\Scripts\python.exe python/lung_candidate_run.py run-dev --split data/lung_candidate_split_20261018/split.json --checkpoint data/lung_training_20261011/early_stop_run/best.pt --excluded data/lung_candidate_planning_20261017/annotations_excluded.csv --outdir data/lung_candidates_pilot_repeat --cohort pilot --with-structure --device cuda --batch-size 16

# 另一组12例开发检查；本轮已经执行，复现请换输出目录
.\.venv-lung\Scripts\python.exe python/lung_candidate_run.py run-dev --split data/lung_candidate_split_20261018/split.json --checkpoint data/lung_training_20261011/early_stop_run/best.pt --excluded data/lung_candidate_planning_20261017/annotations_excluded.csv --outdir data/lung_candidates_check_repeat --cohort check-development --with-structure --device cuda --batch-size 16

# 工程回归，不是医学效果评价
.\.venv-lung\Scripts\python.exe -m unittest discover -s python/tests -p test_lung_candidates.py -v
```

将 `--cohort` 改为 `all-development` 可运行全部 90 例开发扫描；本轮只报告实际完成的两组数据，不能把 24 例写成全部 90/135 例。后续收紧/增加规则前先复核这些漏检，并扩大开发验证；正式留出集继续封存。

## 直接查看与对接

- 简短的首例效果图：`data/lung_candidate_roi_delivery_20261018/first_case_z078.png`、`first_case_z117.png`。
- 24 例精简候选包：`data/lung_candidate_roi_delivery_20261018/manifest.json`，逐例 JSON 共约 8.5 MB，不含 CT 像素或参考结节标注。
- 真实首例的独立调用/交付样例：`data/lung_candidate_module_smoke_20261018/candidates_for_roi.json`。
- 独立调用对拍：同目录 `verification.json`。从原 ZIP 重读首例，使用已计算肺 Mask 直接运行提取函数，323 个候选与缓存操作点逐项一致；约 13.79 秒为本次电脑完整候选计算时间，不含肺分割、网络、网页或板端处理。
- 综合开发记录：`data/lung_candidate_roi_delivery_20261018/development_summary.json`。接入方只消费候选包和相同 CT，不把 `metrics.json` 或参考 CSV 输入自动框选。

当前 v3 和开发检查的输出病例目录保留 `candidates.json`、`metrics.json` 和参考位置切片叠加图。旧 v1/v2 的大候选明细、完整结构特征和 packed masks 经过逐文件 SHA256 对拍后压缩归档，清理后不再保留松散副本；报告、参数探索记录、指标和图继续留在原目录。橙色框是候选粗区域，青色为仅供开发复核的参考中心/直径；无参考病例仍保留误报记录。真实 CT、模型、结果和私有运行资产不上传普通 Git 源码。

## 2026-10-18 缓存清理后的复现

当前模块的直接调用、上述完整 `run-dev --with-structure` 命令、v3 结果及精简 ROI 交付包均不依赖旧 v1/v2 松散明细。冻结划分和原执行源码不变。没有重新运行正式留出集。

需要复核旧版特征筛选时，将 `data/project_cleanup_audit_20261018/retired_candidate_payloads.zip` 解压到项目根目录；压缩包内路径已经以 `data/` 开头。请仅在目标文件不存在时恢复，不覆盖新实验结果。清理清单 `verified_cache_plan.json` 中每个 `lossless_archived_old_candidate_payload` 条目都记录了原路径、原始 SHA256 和 ZIP 成员名，可据此校验恢复字节。归档包含原始明细，不是仅保存统计摘要。

不解压时，旧 `lung_candidate_refine.py` 和 `lung_candidate_select.py` 不能直接读取已归档的 v1/v2 候选或特征。需要运行当前提取算法时，优先使用前面的完整命令；不要通过删除身份核对来跳过旧缓存依赖。

仅用于工程教学与算法演示，不作为临床诊断依据。

## 后续实验入口

2026-10-18新增了不增加CNN的规则改进分支，独立入口为 `python/lung_candidate_refined.py`。旧v3与本文原始结果保留。新分支比较了局部中心、三维重叠去重、独立尺度峰值和小型特征评分；本轮选中固定规则，24例开发对照为25/28、154个误报/例。完整边界、复现方法及后续新病例检查见 [肺结节候选改进实验](lung_candidate_refinement.md)。这不是已上板版本或正式留出集成绩。
