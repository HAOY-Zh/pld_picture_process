# Handoff：2026-10-17 FPGA 加速增量

先读 [2026-10-16 完整交接](project_handoff_20261016.md)，本文件只补充之后的决策和加速实现。

## 当前授权与范围

- 用户先要求只读筛查，之后明确要求“根据这个思路优化一下 FPGA 加速”，已经授权实施与必要本地验证。
- 本轮选择卷积响应直通和池化上下行缓存复用，保持原模型、量化、分辨率、CPU 基线及全部预后处理。
- 原已实板验收的缓存直返版仍是保留版：100 MHz、686.368 ms/层；新实验未返回实板结果前不替换它。
- 新实验为 `data/lung_flow_20261017`，说明见 [加速优化与核验](lung_flow_acceleration.md)。
- 未连接板端、上传文件、重载 Overlay、启动在线服务、Git 提交或推送。

## 已完成的源码与验证

- `fpga/lung_feature_window_flow.v`：同步读响应在就绪时直接交付；背压时锁存原数据、错误及原坐标。
- `python/lung_flow_prepare.py`：从旧已验收 IP 依赖复制独立快照，生成池化缓存位置选择，保留输入分支的地址与量化语义。
  池化复用原有 A/B 两份缓存，命中任一份都复用，缺失时按上下行选择替换位置；拼接逻辑未优化。
- `fpga/tb_lung_window_flow.sv`：新旧窗口独立 oracle、同步响应、背压/错误和直接周期对照。
- `python/tests/test_lung_flow.py`：池化上下行、短行共线、原网络几何、尾通道、非对齐及非池化流量对照。
- 完整 341 项 Python 回归通过；新旧各 10 组实际 XSim 通过，另有中途读/B 故障测试通过。
- 原卷积 gold、张量 gold、模型和医学文件未重新生成；准备及后续门禁核查 1,182 项受保护输入和旧部署包哈希。
- `rtl_gain_verification.json`：147 帧卷积缓存链路周期减少 21.91%；配置到写回减少 22.44%；
  8 个池化用例周期减少 58.50%，物理读拍 22,151→5,256。完整 35 张量用例只减少 0.76%，
  因大尺寸拼接占主导且本轮未改拼接；非池化微小周期变化来自全局周期驱动的 BFM 背压。
- 上述是相同冻结用例下的仿真对照，绝不是新整网实板时延、FPS 或三维端到端加速比。

## 生成与实板核验

`build_flow.tcl` 使用实验内完整 RTL 快照、原 100 MHz 与 GP0/HP0/复位；route 后检查 setup/hold、DRC 再生成 bit。
`python/lung_flow_deploy.py` 核验实际仿真/实现/完整测试与输入保护后，生成独立 `lung_flow/` 包。
本轮实际 route 已通过 100 MHz：WNS +0.165 ns、WHS +0.018 ns；
19,193 LUT、20,349 FF、112 DSP、31.5 BRAM，DSP/BRAM 与旧版相同。
bit/HWH 已生成，最终独立包核验通过：`data/lung_flow_20261017/verification.json`，
状态 `built_routed_flow_not_board_tested`。部署目录 `data/lung_flow_20261017/lung_flow`。
bit SHA256 `20e22607b5f58a12b63ac011c5c814f996b052823a0b7f7c2bd4ba340c14bd07`；
HWH SHA256 `402f0473d8c4ad117aaf20a5faaa73600c068bb8c70521c53d156b5bb5927368`。
新包保留旧三方入口，真实上板需按说明运行 21 节点/20 次输出对照并返回九份原始文件，
再用未修改的 `verify_lung_threeway_board.py` 独立验收。
真实新板端结果返回前，`board_tested=false`，不把仿真百分比套到旧 686 ms 后宣称提速。

用户随后返回终端输出：新bit身份、650MHz CPU、每次100MHz FCLK0、21节点及20次head全部通过。
三方均值 FP32 1547.211ms / ARM INT8 908.312ms / FPGA INT8 569.419ms，
同INT8加速1.595倍、FP32综合2.717倍；相对旧FPGA686.368ms减少17.04%。
终端附件已保存到实验目录 `board_terminal_20261017.txt`，独立复算见 `board_terminal_feedback.json`。
**原始九文件尚未取回，未运行独立实板验收器；不将终端通过改写成已验收实际head/JSON。**
无需重跑本次成功测量；先SCP取回 `lung_flow_result*`，再调用原三方验收器。

## 后续方案讨论的边界

- 用户明确希望按正式题目与评分、创新性、现有适配性、可展示性、FPGA 适配性和实际意义决定方向，
  不必拘泥申报书。10-16 交接第14节将 LDCT 固定为唯一主线的先前建议已被这条最新偏好修正。
- 已读正式任务书：`C:\Users\lenovo\Desktop\917c3753-238b-7b23-705e-3a233f8ec6be.docx`。
  医学影像至少三类基础功能和一种智能分析、核心计算 PL；无固定 FPS/分辨率/同一个 bit 硬规定。
  拓展包含量化小网络、MPR/体积、ROI/阈值交互并由 FPGA 更新掩膜和定量、配准、同屏对照和 CPU/资源/DDR 评价。
  **11月1日提交文档和演示视频**；现场时间按用户为11月10日前，正式文件写11月中上旬待通知。
- LDCT 尚未实现。“LDCT 增强提高肺分割/三维精度”只能是待验证假设，需要真实 LDCT+独立人工肺掩膜；
  PSNR/SSIM 则另需空间对应参考 CT。LCTSC 加图像域模拟噪声只能说明指定模拟条件下的鲁棒性。
- LUNA16 自带肺 mask 为自动生成且官方不允许作为分割真值；Tang 50例人工肺叶标注是候选，
  尚未核查低剂量采集信息、空间/标签定义及可用性，不能写成已有 LDCT 肺分割评价集。
- 现有3D是肺掩膜表面重建，显示增强不会直接提高其准确度；不把单次 CT 的相邻切片逐层刚性配准来强行平滑。

## 原始回传验收及后续优化（同日补充）

- 九份 flow 原始文件已经取回，未经修改的三方独立验收器通过。65,536 head码逐点正确、21节点/51命令/100MHz时钟及哈希通过。
  FPGA 569.4185807 ms，ARM INT8 908.3124915 ms，原始FP32 1547.2113297 ms。前文“终端反馈待取回”是历史状态。
- 当前已验收性能基线为 flow 569.419 ms。在线服务没有自动改用该bit，仍遵守自己的固定包身份。
- 用户新授权继续优化及剪枝，并明确接受平均Dice最多下降1个百分点，前提是明显提速；已实际完成本轮实验。
- 详细源码、证据及上板命令见 [优化记录](lung_acceleration_20261017.md)。
- 精确拼接查表：`python/lung_lut_prepare.py` / `lung_lut_deploy.py`，独立 `data/lung_lut_20261017`。
  新旧各10组真实XSim通过、4096表项独立oracle通过、中途R/B故障通过、完整345项Python回归通过。
  张量用例周期 -22.66%，卷积周期不变。100MHz首轮setup -0.231ns被门禁阻止；原约束下物理优化后WNS +0.034ns/WHS +0.017ns。
  最终17268 LUT / 18455 FF / 80 DSP / 31.5 BRAM；相对flow减少32DSP。新包 `data/lung_lut_20261017/lung_lut` 未实板测试。
  bit SHA256 `54f54656e405eb4cee0ce073d476665110ee49ef6f200ccdaf6edc814a85cafc`；物理优化证明 `physopt_verification.json`。
- 结构化剪枝按8通道块移除，保留原分辨率/预后处理。18原训练患者微调和校准，6原开发患者选择；未用Test/COVID调参。
  首轮decoder8和balanced虽平均Dice达标，但空肺假阳性/小肺退化明显，均保留为失败复核，不替换原模型。
- 一次有界恢复实验 `data/lung_prune_recover_20261017`：balanced继续8epoch，按协议选第1epoch。
  定点平均Dice 0.96801087 vs原0.96708782，小肺Dice 0.45493368 vs0.46267922，空肺FP 0 vs18；最差单病例Dice差-0.00128782。
  参数122321→77249，硬件卷积输入工作量-32.24%，51→42命令。新模型尚未做留出/COVID独立泛化确认。
- 严格“每病例不新增有标注空预测层”门槛仍不通过。逐层审核显示3层由非空变空，其中仅1层原先有真阳性；另2层原来也零真阳性。
  同时恢复了2张原完全空预测的阳性切片。全部记录及原生CT图保存在恢复实验 `small_slice_audit.json` / `newly_empty_review.png`。
- 已准备独立 `data/lung_prune_board_20261017/lung_pruned`，使用已验收flow bit隔离剪枝收益。
  显式标记 `performance_experiment_only=true`、`conservative_quality_review_passed=false`；不会自动部署或替换原模型。
  已做15层独立INT64参考、21节点真实上游内存模拟及重复、1/2线程原生C++INT8，实际候选新权重/宽度的XSim另存 `rtl_probes/`。
- 板端查表包用原三方入口；剪枝包用 `lung_selected_benchmark.py`，不要用旧51命令的continuous测试入口。
  原始新head/时钟/JSON回传前，不发布新实板加速倍数。未连接板端、未覆盖在线服务、未Git提交/推送。
- 候选实际XSim已通过：15层×2输入，72帧、14,050输出码、60,350窗口；原生C++1/2线程和独立上游内存调度也通过。
  最终 `data/lung_acceleration_20261017/verification.json` 交叉核验通过，1,236个受保护原始文件未变。

## 冻结剪枝候选的全量确认（同日后续）

- 用户要求把全部可用测试病例跑完，已完成17例LCTSC（2705层）+10例COVID原HU（2581层），共27例5286层。
  7个缺CT的LCTSC目录仍只有RTSTRUCT；10份已窗宽化外部数据沿用原排除规则。没有按效果剔除病例。
- 旧测试数组已清理，从原始CT/RTSTRUCT/官方肺标注ZIP在 `data/lung_prune_test_cache_20261017` 重建。
  27例模型输入与原生专家掩膜SHA256均匹配历史值，外部HU缓存也一致。首次无缓存启动失败现场保留；正式测试在v2新目录。
- 实际推理原FP32、剪枝FP32、原FPGA整数参考、剪枝FPGA整数参考，均保持256、阈值0.5、预后处理。
  候选仍为冻结的balanced_recovered，没有重新训练、校准、选检查点或根据测试得分改后处理。
- 最终 `data/lung_prune_fulltest_20261017_v2/evaluation.json` 全27例完成；`returned_mask_audit.json` 独立重算324组
  raw/body/final的TP/FP/FN/Dice、小中大分组与空肺误报全部通过；`full_test_report.md` 和两张病例/失败图已生成并查看。
- 同口径定点最终Dice：LCTSC原0.965322414→剪枝0.963893762（-0.143个百分点）；COVID原0.959748021→剪枝0.960706921（+0.096个百分点）。
  虽均值符合用户容忍，S2-204下降4.130个百分点（0.962151→0.920850），COVID010下降1.561个百分点（0.947068→0.931456）。
  两组空肺FP像素分别10091→119108、36326→200765；COVID小肺Dice 0.588545→0.560591。
- 当前判断：**不建议直接替换原模型**，候选仍为performance_experiment_only。前文“尚未测试留出/COVID”已由本节更新。
  原模型/候选/标注与数据不变，未上板、未部署、未Git提交/推送。接下来需要另行验证实板提速；若用本次测试反馈调模型，不能再称为同批新盲测。
- 完整范围及报告入口见 [全病例测试](lung_pruning_fulltest.md)。入口为 `lung_prune_test_cache.py`、`lung_prune_fulltest.py`、
  `lung_prune_fulltest_audit.py`、`lung_prune_fulltest_report.py`，新增3项统计与后处理等价测试通过。

## 剪枝速度回传验收（同日后续）

- 用户回传六份 `data/lung_prune_board_20261017/lung_pruned_result*` 原始文件，现有独立核验器通过，
  新增证据 `lung_pruned_result_verification.json`。21节点/42命令、20次正式测量和65,536个实际head码通过。
- 剪枝CPU INT8 696.910317 ms，PL 445.934570 ms；对原flow的908.312492/569.418581 ms，时延分别减少23.27%/21.69%。
  同模型CPU/PL为1.562808倍，原flow为1.595158倍。CPU原生二进制、主调度与I/O代码不变，变快来自执行更小网络。
- bit仍为flow的 `20e22607...14bd07`；CPU650 MHz、PL100 MHz。没有使用查表bit，也没有新RTL或超频。
  引擎活动时间：卷积309.756153 ms，池化12.141847 ms，拼接54.047895 ms；壁钟剩余69.988675 ms。
- 已完成的响应直通、池化缓存和主机调度优化属于两边共有基线；本次新增收益主要来自结构化剪枝。
  精确拼接查表bit仍未实板验收，下一步应先在原模型上单测，再考虑与剪枝组合，不能提前相乘收益。
- 以上更新前文“未上板/速度待验收”的历史状态。27例准确性风险与性能实验定位不变，未替换在线服务，未Git提交/推送。
  完整归因表见 [优化记录](lung_acceleration_20261017.md)。

## 最终文档的对照要求（用户明确强调）

- 用户要求同时体现PL对CPU的加速，以及剪枝、INT8量化、PL编程策略各自的贡献；后续撰写必须执行，不能只报端点综合倍数。
- 已在 [优化记录](lung_acceleration_20261017.md) 的“最终报告必须保留的对照与贡献归因”中写入实验矩阵、精度矩阵、归因规则和待测项。
- 核心对照：同模型同INT8的ARM/PL；同检查点FP32/INT8；同bit原模型/剪枝恢复模型；同原模型下各PL版本；最终组合实际测量。
  剪枝FP32同板时间、查表bit及剪枝查表组合、多切片持续运行尚缺；不要求新增PL FP32实现，不从比例推算缺项。
- 四种配置的27例全量精度来自既有工作站/整数参考评估，实际PL当前只单切片输出核验；不能把这两类证据混称全病例上板。
- 相同bit剪枝不减少已实例化的DSP/BRAM；查表112→80 DSP属于另一硬件版本。原FP32 CPU/最终INT8 PL为综合收益，
  必须同时列最终同模型CPU INT8/PL INT8的硬件加速比。当前精度风险、失败候选和测试反馈已被使用的事实继续保留。

## 用户确认的推进顺序与剪枝门槛

- 用户同意：先原模型查表bit实板测速，再尝试一个保守剪枝候选，经6例开发验证筛选与冻结、27例重复确认，最后组合上板。
- 门槛以原模型INT8为基线：各数据集平均Dice及每个病例Dice均下降不超过0.01；小肺和空肺误报另外复核。
  时延减少约15%为目标，不是保证；不把已看过的27例称为新的独立盲测。
- 查表包35文件哈希及四个入口核验通过，模型/参数/输入/参考与flow不变；尚无查表回传。下一步板端运行
  `/home/xilinx/ct_test/lung_lut/lung_threeway_benchmark.py`，20次测量、3次预热，结果前缀 `lung_lut_result`。
- 保守候选的只读形状预检：仅neck.0/neck.3为48通道，参数96913、MAC625737728、填充工作量10350592、47命令；
  相比原模型填充工作量仅减少5.71%。尚未训练/校准/导出该候选，不能声称会有15%实板收益。
  最新详细状态见 [优化记录](lung_acceleration_20261017.md)。

## 折中剪枝候选（用户希望不那么保守）

- 已只读核算六个配置，推荐 `enc3.0=24, neck.0/neck.3=48, dec3.0/dec3.3=24`，其余输出通道保留原宽。
  特别保留128分辨率dec2双卷积16通道，相比旧balanced恢复其8→16宽度；末端dec1不剪。
- 推荐候选参数82465、MAC552599552、填充工作量9207808（减少16.12%）、44命令。
  只瓶颈减少5.71%，旧balanced减少32.24%，故该方案属于中间档。数字不是实测加速或精度保证。
- 仍未训练/量化/导出新候选；先测原模型查表bit的顺序不变，后续优先该折中候选，质量门槛不变。

## 查表回传验收及折中候选已训练（同日继续）

- 九份 `data/lung_lut_20261017/lung_lut_result*` 已回传，未修改的三方核验器通过，新增两个独立verification。
  CPU FP32 / CPU INT8 / PL INT8为1609.982237 / 908.074623 / 553.080675 ms。
  同INT8 CPU/PL 1.641848倍，原FP32/PL综合2.910936倍；相对flow569.419 ms整网时延减少2.869%。
  拼接69.840→52.718 ms，卷积403.625 ms基本不变；DSP112→80。没有替换在线服务。
- 新增独立训练入口 `python/lung_prune_moderate.py`，结果 `data/lung_prune_moderate_20261017/moderate`。
  原18训练/6验证，固定8轮，从原检查点剪裁，保留dec2/dec1输出宽度，训练小肺/空肺加权、冻结BN统计并使用教师约束。
  FP32各epoch均未通过全部额外空肺保护条件，按预先规则选平均Dice最高的第5轮，不能说成严格质量合格。
- 训练/量化已完成：开发集fixed平均Dice0.967721050，原0.967087819；最差病例-0.002614551；小肺平均+0.035097268。
  空肺FP4337 vs原18，S3-008的有标注空预测层2 vs原1；既有额外保守review仍未通过，selected_for_board_experiment=None。
- 已独立freeze检查点、INT8权重、Q31参数及37个实验文件，未改旧模型/代码/数据。筛选保护条件新增4项测试通过，既有全病例统计3项测试通过。
- 新增独立 `lung_prune_moderate_fulltest.py` 和 `lung_prune_moderate_fulltest_report.py`，
  正在 `data/lung_prune_moderate_fulltest_20261017` 做全部27例/5286层确认。候选FP32/fixed全部重新推理；
  原模型预测复用上一轮已独立审核掩膜并校验哈希，重新执行后处理及重叠指标；原二维边界在最终掩膜逐点一致后复用。
  原评估器/报告器保持冻结不变。完成后用原 `lung_prune_fulltest_audit.py` 独立重算324组mask阶段，再运行本轮报告器。
  当前尚未得到新候选的全病例最终结论或实板速度，后续以完整evaluation和audit为准。

## 折中剪枝全量确认已完成：不替换默认模型

- `data/lung_prune_moderate_fulltest_20261017/evaluation.json` 已完成27例/5286层，原独立auditor重算324组mask阶段全部PASS。
  新候选FP32/fixed全部重新推理，原模型掩膜复用已验收数据并重算重叠指标，原边界在相同最终掩膜条件下复用。
- LCTSC原fixed平均0.965322414→本轮0.966411103（+0.109个百分点），17例都满足每病例下降不超过0.01，
  最差S2-204为0.962151247→0.954862315（-0.729个百分点），比旧balanced的0.920850显著恢复。
- COVID原fixed平均0.959748021→本轮0.959739979（几乎不变），但COVID010为0.947067780→0.929141322，
  下降1.793个百分点，未达到用户同意的每病例1个百分点门槛；旧balanced该例0.931456，本轮还略差。
- 空肺FP：LCTSC10091→31854，COVID36326→193855；小肺Dice：LCTSC0.612106→0.637139，COVID0.588545→0.525038。
  COVID010的FP32也下降1.614个百分点，失败不是仅由量化造成。开发集均值改善不能代表外部泛化达标。
- 采用判断：当前折中版继续保留为失败/取舍实验，不替换原模型，不改变标准，不再按该批测试追加调参。
  未制作该候选新上板包、未实板测速；16.12%仅为工作量减少，不是实测速度。原模型查表bit的553.081 ms验收有效。
- 完整报告/图在上述fulltest目录，入口为 `full_test_report.md`、`paired_fixed_dice.png`、`worst_case_native_review.png`；
  核验为 `returned_mask_audit.json` 和 `presentation_verification.json`。检查点7d0d0e61...01ccbec2，evaluation d2d061b2...d459e0c3。
- 两轮剪枝的初始化/预算/种子不同，不能把差异全部归因于保护dec2。新增4项选择测试、原3项统计测试均通过。
  旧source/模型/结果保持原哈希，未部署在线服务，未Git提交/推送。此前“全病例仍在运行”的描述由本节更新。
- 两张结果图已查看。COVID010 z300出现人体外空气误分，原模型也有、新候选更大；仅是事后定位线索。
  更精细的CT-only人体约束若后续尝试，需作为独立后处理对照，不能混入本轮剪枝结果或用专家标签定规则。

## CT Studio 初版整合网页（用户确认三个工作区）

- 用户要求把基础处理和智能分割整理成初步网页，确认“共享CT输入＋基础处理／智能分割与三维／性能记录”布局。
- 新入口 `python/ct_workspace_web.py`，新基础参考 `ct_basic_processing.py`，独立前端在 `python/lungseg/workspace/`。
  保留原live/离线网页、板端服务、bit、模型和冻结脚本不变；没有部署板端或自动切换Overlay。
- 基础页支持窗宽窗位、HU范围阈值、四种3×3形态学、直方图均衡和伪彩。本轮是明确标识的PC_REFERENCE，
  不冒充PL；参考输出独立保存，不修改原HU或送入神经网络。基础PL统一服务仍待后续接入。
- 分割页通过同源嵌入原页面，复用同一个Controller会话；共享加载/标注、单击轴位同步、原实际FPGA调用、MPR与局部/完整重建语义。
  原在线服务仍绑定cache-return原模型，不能因历史查表553ms或剪枝446ms而声称当前网页已用新bit/剪枝模型。
- 性能页区分本次真实回执与历史验收基准，历史表读取验收证明并重查原始结果哈希。没有真实推理时不填入历史速度。
  基础预览不算PL，通信往返已含板端推理，不将两者相加；会话JSON可导出。
- 10项工作台后端测试＋35项原live回归通过；独立Edge的8种处理/切片同步/三维/导出/病例清空/移动端检查通过。
  截图和证据在 `data/ct_workspace_checks_20261017_v2`，明确合成输入/FakeHardware，不能当实际医学/板端证据。
- 本机生产网页已后台启动 `http://127.0.0.1:8090/`，仅本机监听，无自动板端连接。
  日志和PID在 `data/ct_workspace_20261017/`，结果在其 `results/`，配置复用 `data/lung_live_20261016/host_config.json`。
  若需重启先核对保存PID对应命令，避免结束用户其他进程；电脑网页退出不会复位/释放板端硬件。
- 使用与修改位置见 [CT Studio说明](ct_workspace.md)。界面/服务可独立迭代，不需要重训或重新生成bit；真实板端联调尚需用户服务在线。

## 基础PL已补齐统一集成接口（更新前节仅参考的初版状态）

- 用户指出基础模块早已上板，明确要求集成；上版遗漏真实基础PL接入，现已补齐，不再以软件预览替代硬件功能。
- 找到并复用6套现成bit/HWH：`fpga/pynq_z2_{window_level,threshold,erosion,dilation,opening,closing}_axis` 的impl_1 bit和hw_handoff HWH。
  HWH检查32位simple DMA、23bit长度、窗口双16bit GPIO，以及阈值为真实range核心。没有重新跑Vivado或改RTL。
- 新板端 `pynq/ct_platform_service.py` 单一所有者共用原 `/tmp/lung_live_z2.lock`，串行调度6基础bit与原分割bit。
  正常空闲后才释放完成的缓冲并切换；忙碌/poisoned/超时保留owner/CMA，不自动重试、复位、重载或释放。
  `stop_idle_live_service(live_service)` 可明确交还健康旧服务；故障旧服务拒绝关闭。
- 新接口：`/v1/platform`身份、`/v1/basic`原生512×512 int16 HU、原 `/v1/infer`神经网络。
  窗映射/范围阈值在PL，形态学沿用既有PS阈值准备+PL形态学；开/闭两阶段在PL。
  直方图/伪彩仍只有明确电脑参考，没有宣称这些已有PL模块。
- 启动自检包含原分割21节点、基础6模块×2组真实DMA输出、切回分割再次21节点检查；
  保存 `platform_startup_<uuid>.json/.npz` 实际输出，可用 `python/verify_ct_platform_board.py` 独立核验12×512×512像素及65536分割码。
- 查RTL时发现此前新写的窗映射参考对奇数WW高端饱和不正确，已按现有RTL显式HU>=high→255修正，仅改本轮参考代码和相应测试，未改既有已验收包。
- 最终新上板目录 `data/ct_platform_20261017_v2/ct_platform`，54文件，6套基础bit/HWH＋保留原模型分割；
  `python/ct_platform_package.py` 校验原文件哈希，旧在线包/模型/bit不变。早期ct_platform_20261017是未部署中间快照，不要上传。
- `python/ct_platform_client.py` 校验请求ID、HU输入、参数、bit/HWH、输出哈希及实际DMA/时钟/guard证据。
  网页默认FPGA PL，明确可选电脑参考，不会自动回退；真实输出与参考不一致会保存actual/reference并报错，神经网络输入不变。
  基础回执分开显示DMA、PS准备、Overlay切换、通信往返。当前是串行切bit，不是全部模块同时常驻。
- 本机8090已在确认空闲与进程所有权后重启，配置为 `data/ct_platform_20261017_v2/host_config.json`，尚未连接板端。
  日志 `data/ct_workspace_20261017/web.pl.stdout.log` / `web.pl.stderr.log`。无板端自动重载或部署。
- 本轮只读SSH尝试返回 `Connection closed by 192.168.2.99 port 22`，无法代替用户在Jupyter启动新服务，须按 [集成步骤](ct_platform.md) 上传并执行。
  新链路实板验收未完成，不能把既有单模块验证或模拟浏览器结果写成统一服务实际通过。
- 新本地测试包含9项协议/故障隔离、2项回传核验，及10项原工作台测试；
  浏览器6硬件协议方法＋2明确软件参考、切片同步、切回分割/三维、导出、移动端均通过，证据在 `data/ct_platform_checks_20261017_v2`，明确TEST_ONLY。
  原网页和分割服务源文件保持不变，未更换剪枝模型、未Git提交或推送。
- 最终统一回归56项全通过（21项新平台/工作台/验收＋35项原live）；日志在ct_platform_checks_20261017_v2/python_tests.log。
  v2板端包54文件、41420009字节，已再次核对原bit/HWH及当前服务/参考源码副本一致。最终本地验证证明在该checks目录verification.json。


## CT workspace interaction correction

See [input, window display and timing correction](ct_workspace_interaction_20261017.md). Threshold and binary morphology are explicit separate actions; a same-slice saved mask is required. Fixed left CT window; applied windows appear on the right. Browser click-to-display time is separate from DMA, overlay switch and HTTP time. 58 tests and synthetic browser checks passed. Existing 130-slice actual session was verified and restored without new FPGA inference. Board package and model unchanged.

## 基础处理实际往返加速

- 最终包：`data/ct_platform_fast_20261017_v2/ct_platform_fast`，45个绑定文件、19833610字节。六功能共享bit，100MHz，WNS 0.842ns、WHS 0.026ns；5174 LUT、6755寄存器、3 BRAM、3 DSP。分割bit和模型保持原版。
- 实现同切片输入缓存（上限8张）、无损压缩二进制传输、任务完成即时返回、隐藏三维页暂停、基础图像绘制缓存。缓存只复用输入，每次仍实际运行PL并逐像素校验。
- 首传DMA状态单独跟踪，BTT一旦可能已写入立即恢复严格Idle检查；错误、超时保持唯一所有者及CMA，不自动重试。
- RTL验证：小图475648像素加原生512×512的4718592像素（含真实CT）；74项Python回归及快速协议合成浏览器流程通过。初次并行构建有Torch冷启动超时记录，按生产2线程复核后通过。
- 8090已更新并保留130层真实分割、Dice和三维；最终启动脚本在`data/ct_platform_fast_final_checks_20261017/resume_web.py`，常规新会话仍用`python/ct_workspace_web.py`。
- 尚未切换真实板端。SSH返回`Permission denied (publickey,password)`；需用户上传最终v2目录，在原内核运行`platform_service = upgrade(platform_service)`，READY后网页点“连接新版基础加速服务”。电脑已提供真实往返测速脚本，不能把假硬件网页60～150ms当成实板成绩。
- 详细步骤及公平测速口径见 [基础处理加速部署](ct_platform_fast.md)。最终本地证明在`data/ct_platform_fast_final_checks_20261017/verification.json`。未Git提交或推送。
