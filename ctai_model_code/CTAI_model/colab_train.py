# -*- coding: utf-8 -*-
"""
# CTAI 直肠癌 CT 肿瘤分割 —— Google Colab GPU 训练

## 使用方法:
1. 把 `CTAI_model/` 文件夹和 `直肠癌数据/` 文件夹上传到 Google Drive 根目录
2. 打开 Colab → 运行时 → 更改运行时类型 → GPU (T4)
3. 按顺序运行下面的 Cell

预计训练时间: T4 GPU 约 6-8 小时（100 epoch）
"""

# ============================================================
# Cell 1: 挂载 Google Drive + 安装依赖
# ============================================================
# 在 Colab 中运行这段代码:

"""
from google.colab import drive
drive.mount('/content/drive')

!pip install -q albumentations SimpleITK tqdm

# 检查 GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
"""

# ============================================================
# Cell 2: 设置路径 + 开始训练
# ============================================================

"""
import os
os.chdir('/content/drive/MyDrive/CTAI_model')

# 全量 GPU 训练（参数已针对 T4 优化）
!python train.py \
    --data_dir "/content/drive/MyDrive/直肠癌数据/" \
    --repeats 5 \
    --batch_size 8 \
    --epochs 100 \
    --eval_interval 5 \
    --train_mode mixed \
    --mixed_ratio 0.3 \
    --lr 3e-4 \
    --deep_supervision True \
    --use_ema True \
    --num_workers 2
"""

# ============================================================
# Cell 3: 训练完成后评估
# ============================================================

"""
!python evaluate.py \
    --weights checkpoints/best_model.pth \
    --tta \
    --output_dir evaluation_full
"""

# ============================================================
# 也可以用下面的 Kaggle 版本（每周 30 小时免费 GPU）
# ============================================================

"""
Kaggle 步骤:
1. kaggle.com → New Notebook
2. Settings → Accelerator → GPU T4 x2
3. 上传数据集 → Add Data
4. 运行同样的训练命令（路径改为 /kaggle/input/...）
"""
