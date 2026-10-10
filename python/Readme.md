用于智能分析算法的离线训练与验证。

主要内容包括：

CT 数据集预处理
智能分割模型训练
轻量化网络实验
INT8 / INT4 量化
Dice / IoU 等精度评价
FPGA 部署所需权重和参数导出

注意：Python 主要作为离线开发工具，最终作品中的核心图像处理与主要计算任务应由 FPGA PL 完成或加速。

新增肺分割离线开发入口：`lung_segmentation.py`。
支持 LCTSC CT/RTSTRUCT 清点、人工肺轮廓转换、规则基线与 Dice/IoU 评价。
这是 CPU 数据/算法验证流程，不是已部署的 FPGA 分割，也不是训练好的模型。
详细步骤见 [肺分割开发说明](../docs/lung_segmentation.md)。

轻量学习型模型入口：`lung_unet.py`，包括患者级缓存、训练与原始分辨率评价。
使用电脑的 `.venv-lung` 环境，不是板卡程序；INT8/PL 部署尚未完成。
步骤见 [轻量模型训练说明](../docs/lung_unet_training.md)。

在项目根目录运行测试：

```powershell
python -m unittest discover -s python/tests -v
```
