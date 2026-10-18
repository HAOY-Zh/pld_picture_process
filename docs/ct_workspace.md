# CT Studio 初步网页工作台

> 2026-10-17 交互修正：阈值分割与二值形态学已拆开；左侧 CT 使用固定显示参考，窗设置应用后只在结果区显示；基础性能同时列出点击至显示、后台、通信往返、bit切换及PL/DMA耗时。详见本文末尾的修正说明。

> 更新：6个已有基础PL模块现已接入统一板端服务与网页接口，默认执行位置为FPGA PL。
> 统一部署和启动步骤见 [基础PL与分割集成](ct_platform.md)。本页后续“仅参考/待接入”段落记录初版历史状态。

统一入口为 `python/ct_workspace_web.py`，默认地址 `http://127.0.0.1:8090/`。
基础处理、智能分割与三维、性能记录三个工作区共享同一份CT和可选专家标注。
原来的 `lung_live_web.py` 及独立/离线网页保持不变，可以继续使用。

## 启动与使用

在电脑PowerShell的项目根目录运行：

```powershell
Set-Location D:\pld_picture_process
& .\.venv-lung\Scripts\python.exe -X utf8 python\ct_workspace_web.py
```

默认读取新的 `data/ct_platform_20261017_v2/host_config.json`，输出写到新的
`data/ct_workspace_20261017/results/`，不混入旧网页结果。
可以用 `--config`、`--port`、`--output` 覆盖。不要同时启动两个使用相同端口的进程。
本轮已后台启动8090；日志为 `data/ct_workspace_20261017/web.stdout.log` 与 `web.stderr.log`，启动PID另存 `web.pid`。
后续若重启电脑或修改Python代码，需要重新运行命令；修改CSS/JS后刷新页面即可，修改HTML模板后需重启后端。

1. 打开网页，输入本机CT文件/病例目录。可只读取连续片段，例如已有示例
   `D:\CT2\lctsc\LCTSC-Train-S3-008` 的 `124-129` 层。
2. “标注与输入选项”中可指定专家标注或禁用。DICOM自动保留HU与几何；NIfTI/NPZ需要确认HU/mm。
3. 加载后即可在基础处理页选择轴位、窗宽窗位、处理方法，点击“应用处理”。
   “在分割页查看”会同步该轴位切片；两个工作区无需重复加载CT。
4. 智能分割需要原板端服务在线。板端启动方法仍见 [实时FPGA网页说明](lung_live_web.md)。
   电脑网页不会自动连接、下载或切换bit，也不会自动将在线服务升级到最新查表版。
5. 性能页显示本次实际回执和历史核验基准；可导出当前会话JSON。原始输出、处理参数和哈希保存在结果目录。

## 初版已经实现的内容与边界

| 工作区 | 功能 | 执行位置与状态 |
| --- | --- | --- |
| 基础处理 | 窗宽窗位、HU范围阈值、3×3腐蚀/膨胀/开/闭 | 默认真实PL接口，需启动统一板端服务；另可明确选择电脑参考 |
| 显示参考 | 直方图均衡、伪彩 | 电脑参考，无对应PL模块 |
| 智能分割与三维 | 当前层/指定层/整例推理、连续局部重建、MPR联动、专家叠加、实际范围Dice | 原真实FPGA服务负责前向；电脑负责原有预后处理与重建 |
| 性能记录 | 本次网络壁钟、往返、已算层数、基础预览耗时、历史CPU/PL对照、会话导出 | 本次数据与历史基准分开；无实际推理时显示空值 |

基础参考沿用已有基础PL测试的int16 HU四舍五入、整数窗映射和各形态学阶段3×3零填充规则。
形态学的输入是当前HU范围阈值掩膜，与原 standalone 基础PL测试一样，不使用专家标注。
参考预览不修改原HU、神经网络量化参数、已算分割掩膜或三维结果。
直方图/伪彩属于显示处理，不声称实现LDCT去噪或提高分割精度。

**基础PL统一网页接口现已实现。** 原基础功能分别使用独立bit，统一服务以单一所有者串行调度基础DMA及分割网络。
网页默认FPGA PL，电脑参考需明确选择；没有自动软件替代。统一上板包已打好，实际启动自检和真实CT网页联调仍待板端执行。
不是已完成新的整合实板验收，现有单模块上板验证事实保持有效。详细步骤见 [统一服务说明](ct_platform.md)。

统一服务的分割仍绑定保留的原模型/cache-return bit。历史表中的flow、精确查表及剪枝版速度不是当前在线服务的声明。
没有选择剪枝候选，也没有更换bit/模型。专家标注保持只用于显示/评价；无标注时不显示Dice，未算层不作为背景。

## 后续修改的位置

| 文件 | 负责内容 |
| --- | --- |
| `python/lungseg/workspace/index.html` | 布局、三个工作区及输入控件 |
| `python/lungseg/workspace/style.css` | 颜色、尺寸、桌面/窄屏布局 |
| `python/lungseg/workspace/app.js` | 基础参数、切片、API交互、性能展示与导出 |
| `python/ct_basic_processing.py` | 独立基础软件参考与参数验证 |
| `pynq/ct_platform_service.py` | 统一硬件所有者、基础DMA、空闲切换、启动真实输出自检 |
| `python/ct_platform_client.py` | 真实硬件请求、回执/bit/参数/输出绑定核验 |
| `python/ct_platform_package.py` | 原bit/HWH与保留分割包打包，不重新综合 |
| `python/ct_workspace_web.py` | 共享会话、基础任务、HTTP入口、分割页适配 |
| 原 `lung_live_web.py` / 板端服务 | 原分割通信和计算，初版不修改 |

增加后端执行方式应与处理方法分开；增加新算子不需要改三维和医学标签处理。
替换布局或配色不需要重训网络、重新量化或生成bit。界面改动不代表硬件性能改变。

## 验证记录

```powershell
& .\.venv-lung\Scripts\python.exe -X utf8 -m unittest discover -s python/tests -p test_ct_workspace.py -v
& .\.venv-lung\Scripts\python.exe -X utf8 -m unittest discover -s python/tests -p test_lung_live.py -v
```

本轮10项工作台测试及35项原实时链路回归通过，覆盖边界、病例隔离、并发阻挡、认证、真实传输调用结构与原有局部三维语义。
独立无头Edge浏览器还检查8种处理、共享病例、切片同步、分割/三维、专家对照、回执、导出、新病例清空和390px窄屏。
浏览器夹具使用合成CT和明确的FakeHardware，仅位于 `python/tests/`；不会作为生产后端或自动回退。
证据与截图在 `data/ct_workspace_checks_20261017_v2/`，全部标记TEST_ONLY，不能用于真实FPGA测速或医学效果展示。
本轮未执行真实板端推理；统一网页仍需用户连接实际服务后做小范围联调。

后续统一PL接口验证新增9项平台和2项回传核验测试，与原10项工作台测试合计21项。
浏览器对6个硬件协议方法及2个明确电脑参考方法的检查证据见 `data/ct_platform_checks_20261017_v2/`，
同样是合成影像/模拟硬件，不能当作新实板验收；新上板步骤及回传核验见 [统一服务](ct_platform.md)。
