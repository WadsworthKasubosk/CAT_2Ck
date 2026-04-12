# 模型权重文件说明

由于权重文件较大（>118 MB），不随 TEST 包直接分发。

## 可用权重文件

以下权重文件位于 `CTAI_model/net/` 目录：

| 文件名 | 大小 | 模型类型 | 说明 |
|--------|------|---------|------|
| `baseline_weights.pth` | 118.5 MB | 标准 U-Net | Baseline 模型 |
| `baseline_unet_weights.pth` | 118.5 MB | 标准 U-Net | 同上（备份） |
| `attention_weights.pth` | 119.9 MB | Attention U-Net | 带注意力门控的 U-Net |
| `best_model_weights_tiny.pth` | 119.9 MB | Attention U-Net | 在 tiny 数据集上的最佳模型 |

当前部署使用的权重：`CTAI_flask/core/net/model.pth`（118.5 MB，标准 U-Net）

## 如何更换权重

将目标 .pth 文件拷贝到 `CTAI_flask/core/net/model.pth` 即可。

**注意：**
- `baseline_weights.pth` 和 `model.pth` 使用 `Unet(1,1)` 架构
- `attention_weights.pth` 和 `best_model_weights_tiny.pth` 使用 `AttentionUnet(1,1)` 架构
- 更换为 Attention U-Net 需要同时修改 `CTAI_flask/app.py` 中的模型加载代码
