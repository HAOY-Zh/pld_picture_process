# Windows 上位机交付

新电脑和全新板端项目请先看 [从零部署与团队协作](../docs/从零部署与团队协作_20261027.md)：该路线使用 GitHub 公开资产并生成新配对，不需要下面的私有免安装包。下面说明的是复用原配对和便携环境的路线。

接收方使用维护者私下提供的完整 ZIP，解压后双击 `CT Studio 上位机.exe`。电脑须能访问板子 `192.168.2.99:8766`；网页默认使用本机 8099，遇到占用时在 8100–8109 中选择空闲端口。不需要自行安装 Python 或配置环境。

本目录是启动器源码。GitHub 源码 ZIP 不包含内置 Python、当前板端令牌和已安装配对资产，不能当作现有板子的免安装软件包。公开运行资产的安装器会生成新凭据，不能直接代替原配对。

同学修改源码后，直接双击源码根目录的 `start_ct_studio_dev.cmd`，首次选择完整私有包即可使用其环境和配套资产运行当前源码，支持没有 `.git` 的 Download ZIP 目录。离线检查使用 `start_ct_studio_dev.cmd -Diagnose`。完整步骤与检测入口见 [同学交接说明](../docs/同学交接说明_20261026.md)。成品 exe 不会自动加载外部修改。

维护者在 Windows 上用现有 Python 执行，三个输入均为本机目录：

```powershell
python desktop/build_private_release.py `
  --paired-dir "<已安装配对目录>/ct_studio_integration_20261025" `
  --runtime-dir "<已有免安装软件目录>/runtime" `
  --output "data/private_handoff/CT_Studio_Final" --zip
```

`--paired-dir` 必须包含五份配置/安装清单、原 `platform_token.txt` 和完整 `ct_studio/`。`--runtime-dir` 使用经过检查的 Python 3.11 Windows x64 便携环境，包含原应用依赖；构建器不会下载依赖。编译器默认使用 Windows 自带 .NET Framework 的 `csc.exe`，也可通过 `--csc` 指定。

构建器只导出 Git 管理的 `python/`、`pynq/`、LDCT 配置与启动脚本，另外复制明确指定的配对目录和便携环境；不会复制病例结果、训练目录或 Vivado 临时工程。输出必须使用新名字，避免覆盖旧交付。配置路径会在副本中改为相对路径，模型和硬件资产不变，不连接或安装板端。

重打包脚本需要 Git 工作副本及 Git 命令；上面的源码开发启动入口不需要 Git。提交新增源码文件后再打包，避免未被 Git 管理的新模块漏进交付。

完成时自动编译 exe、检查环境与资产，写入源码提交、逐文件哈希及验证范围；`--zip` 另外打包并检查 ZIP CRC。离线诊断通过不代表新的电脑与板端组合已完成实板推理验收。交付包包含私有令牌，应私下传给接收方，不能上传 GitHub。

`Launcher.cs` 管理窗口、后台启动、日志和浏览器；`desktop_bootstrap.py` 验证文件、配套身份、已有网页会话及端口。板端忙或故障时，不发送取消、重启、重装或推理重试。

启动决策的离线检查：

```powershell
python -B -X utf8 -m unittest discover -s desktop -p "test_*.py"
```
