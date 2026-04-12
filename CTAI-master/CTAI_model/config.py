# -*- coding: utf-8 -*-
"""
CTAI 直肠癌 CT 肿瘤分割 —— 全局配置
所有超参数集中管理，支持命令行覆盖
"""

import os
import argparse
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """训练配置（dataclass 形式，支持 IDE 自动补全）"""

    # ==================== 数据 ====================
    # 默认路径：Kaggle 环境自动检测，本地开发用 tiny 数据集
    data_dir: str = ""  # 留空则自动检测
    crop_size: int = 256
    train_mode: str = "tumor_only"        # "tumor_only": 只用含肿瘤切片; "mixed": 混合模式
    repeats: int = 50                     # 含肿瘤切片每 epoch 重复次数
    mixed_ratio: float = 0.2              # mixed 模式下背景切片占比（相对肿瘤切片数）
    num_workers: int = 0                  # DataLoader 工作线程数（Windows 建议 0）

    # 训练/验证划分
    val_ratio: float = 0.15             # 按患者划分，15% 患者做验证
    val_max_slices: int = 60            # 快速评估时最多采样的含肿瘤切片数
    full_eval_interval: int = 50        # 每 N epoch 做一次全量评估

    # CT 窗宽窗位（腹部 CT）
    ct_window_center: float = 40.0
    ct_window_width: float = 400.0

    # ==================== 模型 ====================
    model: str = "attention_unet"         # "unet" 或 "attention_unet"
    in_channels: int = 1
    out_channels: int = 1
    deep_supervision: bool = True         # 是否启用深监督
    dropout_rate: float = 0.3             # bottleneck dropout

    # ==================== 训练 ====================
    epochs: int = 200
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0               # 梯度裁剪 max_norm

    # ==================== 损失函数 ====================
    loss: str = "combined"                # "dice", "focal", "tversky", "combined"
    dice_weight: float = 0.5
    focal_weight: float = 0.3
    tversky_weight: float = 0.2
    focal_gamma: float = 2.0
    focal_alpha: float = 0.95
    tversky_alpha: float = 0.3            # FP 惩罚权重
    tversky_beta: float = 0.7             # FN 惩罚权重（更高 = 更惩罚漏检）
    loss_warmup_epochs: int = 10          # 前 N 个 epoch 只用 BCE+Dice

    # ==================== 学习率调度 ====================
    scheduler: str = "cosine_warm_restarts"
    T_0: int = 50                         # CosineAnnealingWarmRestarts 周期
    eta_min: float = 1e-6

    # ==================== EMA ====================
    use_ema: bool = True
    ema_decay: float = 0.999

    # ==================== 评估 ====================
    eval_interval: int = 5               # 每 N 个 epoch 评估一次
    eval_patch_size: int = 256
    eval_stride: int = 128               # 滑窗步长（50% 重叠）

    # ==================== 推理 ====================
    threshold: float = 0.5
    min_area: int = 30                   # 连通域最小面积
    max_area: int = 5000                 # 连通域最大面积
    use_tta: bool = True                 # Test Time Augmentation

    # ==================== 断点恢复 ====================
    resume: str = ""                     # checkpoint 路径，非空则恢复训练
    save_interval: int = 5              # 每 N 个 epoch 自动保存 checkpoint

    # ==================== 早停 ====================
    patience: int = 10                   # 连续 N 次评估无提升则停止（实际 epoch = N * eval_interval）

    # ==================== 保存路径 ====================
    save_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    vis_dir: str = "vis_results/"

    # ==================== 随机种子 ====================
    seed: int = 42

    def __post_init__(self):
        """自动检测运行环境并创建输出目录"""
        # 自动检测 Kaggle / Colab / 本地环境
        if not self.data_dir:
            if os.path.isdir("/kaggle/input"):
                # Kaggle: 数据集名称需匹配上传时的名称
                candidates = [
                    "/kaggle/input/rectal-caner-data/rectal_data",
                    "/kaggle/input/rectal-caner-data/直肠癌数据",
                    "/kaggle/input/rectal-caner-data",
                    "/kaggle/input/rectal-cancer-data/rectal_data",
                ]
                for c in candidates:
                    if os.path.isdir(c):
                        self.data_dir = c
                        break
                if not self.data_dir:
                    # 尝试找任何包含 dcm 文件的输入目录
                    import glob
                    dcm_files = glob.glob("/kaggle/input/**/*.dcm", recursive=True)
                    if dcm_files:
                        # 推断数据根目录（向上找到包含患者文件夹的层级）
                        self.data_dir = os.path.dirname(os.path.dirname(os.path.dirname(dcm_files[0])))
                if not self.data_dir:
                    self.data_dir = "/kaggle/input/"
                # Kaggle 输出目录
                self.save_dir = "/kaggle/working/checkpoints/"
                self.log_dir = "/kaggle/working/logs/"
                self.vis_dir = "/kaggle/working/vis_results/"
            elif os.path.isdir("/content/drive"):
                # Google Colab
                self.data_dir = "/content/drive/MyDrive/直肠癌数据/"
            else:
                # 本地 Windows
                self.data_dir = "c:/Users/da983/CAT_2Ck/直肠癌数据_tiny/"

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)


def get_config_from_args() -> TrainConfig:
    """从命令行参数解析配置（未指定的使用默认值）"""
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="CTAI 肿瘤分割训练配置")

    for k, v in cfg.__dict__.items():
        arg_type = type(v)
        if arg_type == bool:
            parser.add_argument(f"--{k}", type=lambda x: x.lower() in ('true', '1', 'yes'),
                                default=v, help=f"{k} (default: {v})")
        else:
            parser.add_argument(f"--{k}", type=arg_type, default=v, help=f"{k} (default: {v})")

    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


if __name__ == "__main__":
    cfg = TrainConfig()
    print("=" * 50)
    print("CTAI 训练默认配置:")
    print("=" * 50)
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
