# CTAI 直肠癌 CT 肿瘤分割系统

## 项目简介

基于深度学习的直肠癌 CT 图像肿瘤自动分割系统。使用 Attention U-Net + Deep Supervision 架构，针对极小数据集（37 张切片/单患者）做了全面的工程优化。

## 项目结构

```
CTAI_model/
├── config.py                 # 全局配置（所有超参数集中管理）
├── train.py                  # 完整训练脚本
├── inference.py              # 推理模块（滑窗 + TTA + 后处理）
├── evaluate.py               # 全面评估脚本
├── requirements.txt          # 依赖列表
├── README.md                 # 本文件
├── core/
│   ├── losses.py             # 损失函数（Dice/Focal/Tversky/Combined）
│   ├── ema.py                # EMA 指数移动平均
│   ├── net/
│   │   ├── unet.py           # 标准 U-Net（保留原有）
│   │   └── attention_unet.py # Attention U-Net + Deep Supervision
│   ├── predict.py            # 部署推理（Flask 后端调用）
│   ├── process.py            # DICOM 预处理 + 可视化叠加
│   └── get_feature.py        # 肿瘤特征提取
├── data/
│   └── dataset.py            # 数据管道（增强 + 肿瘤中心裁剪）
├── net/
│   └── attention_unet.py     # 训练用模型（含 Deep Supervision）
└── data_set/
    └── make.py               # 旧版数据加载（保留兼容）
```

## 快速开始

### 1. 安装依赖

```bash
cd CTAI_model
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用默认配置训练
python train.py

# 自定义参数
python train.py --epochs 100 --lr 1e-3 --batch_size 4 --model attention_unet

# 不启用深监督
python train.py --deep_supervision False
```

### 3. 评估模型

```bash
# 基础评估
python evaluate.py --weights checkpoints/best_model.pth

# 启用 TTA（更准但更慢）
python evaluate.py --weights checkpoints/best_model.pth --tta
```

## 核心技术方案

### 数据策略（应对 37 张极小数据集）

| 策略 | 说明 |
|------|------|
| 肿瘤中心裁剪 | 80% 概率以肿瘤质心为锚点做 256×256 裁剪 |
| 数据重复 50x | 每 epoch 含肿瘤切片重复 50 次，配合随机增强每次不同 |
| 强数据增强 | 弹性变换 + 几何 + 光度（albumentations） |
| CT 窗宽窗位 | 腹部标准窗 center=40, width=400，聚焦软组织对比度 |

### 模型架构

- **Attention U-Net**: 在 skip connection 加入注意力门控，突出病灶区域
- **Deep Supervision**: 解码器每层都输出预测，多尺度监督防止梯度消失
- **Bottleneck Dropout**: 0.3 dropout 防止过拟合

### 损失函数

```
Warm-up（前 10 epoch）: 0.5×BCE + 0.5×Dice
正式阶段:              0.5×Dice + 0.3×Focal + 0.2×Tversky
```

- **Dice Loss**: 对类别不平衡天然鲁棒
- **Focal Loss** (γ=2, α=0.95): 聚焦难样本，降低背景权重
- **Tversky Loss** (α=0.3, β=0.7): 更惩罚漏检

### 推理优化

- **滑窗推理**: 256×256 patch, stride=128, 重叠区域取概率平均
- **TTA**: 8 种几何变换的概率平均
- **后处理**: 连通域筛选 + 形态学清理

### 训练优化

- **AdamW** + CosineAnnealingWarmRestarts
- **EMA**: 指数移动平均权重用于评估
- **梯度裁剪**: max_norm=1.0
- **早停**: 30 epoch 无改善自动停止

## 当前局限性

1. **数据量不足**: 37 张切片来自单一患者，模型泛化能力有限
2. **预期 Dice**: 在当前数据下，合理预期 Dice 为 0.15~0.35
3. **单一期相**: 仅使用动脉期 CT，未利用多期相信息

## 未来改进路线

### 短期（有更多数据后）

- [ ] 扩充至 5+ 患者、200+ 切片
- [ ] 引入 5-fold 交叉验证
- [ ] 添加 nnU-Net 自动化配置

### 中期

- [ ] 多期相 CT 融合（动脉期 + 静脉期）
- [ ] 3D U-Net 利用切片间连续性
- [ ] 半监督学习利用无标注数据

### 长期

- [ ] 集成 SAM (Segment Anything Model) 做交互式分割
- [ ] 联邦学习跨医院训练
- [ ] 临床验证与审批
