# -*- coding: utf-8 -*-
"""
CTAI 损失函数模块
针对极度不平衡的医学图像分割任务设计
- DiceLoss: 对类别不平衡天然鲁棒
- FocalLoss: 降低易分类样本权重，聚焦难样本（AMP 安全版本）
- TverskyLoss: 可调 FP/FN 惩罚比，适合漏检代价高的场景
- CombinedLoss: 多损失加权组合 + 平滑过渡 warm-up 机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss: 1 - 2*|A∩B| / (|A|+|B|)
    对前景/背景比例不敏感，适合极度不平衡数据
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 1, H, W] 未经 Sigmoid 的模型输出
            targets: [B, 1, H, W] 二值标签 (0/1)
        """
        # 在 float32 下计算，避免 AMP float16 精度问题
        logits = logits.float()
        probs = torch.sigmoid(logits)
        # 展平为 [B, N]
        probs_flat = probs.contiguous().view(probs.size(0), -1)
        targets_flat = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss（AMP 安全版本）
    使用 F.binary_cross_entropy_with_logits 替代手动 sigmoid+log，
    避免 float16 下的数值溢出/NaN 问题。

    gamma=2.0: 标准值，对置信度 >0.9 的样本权重降至 0.01
    alpha=0.95: 前景权重极高，因为前景仅占 0.081%
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.95):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 强制 float32 计算，防止 AMP 半精度溢出
        logits = logits.float()
        targets = targets.float()

        # 使用数值稳定的 BCE（内部用 log-sum-exp 技巧，不会 NaN）
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # 计算 p_t（预测正确的概率）
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # focal 权重: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # alpha 权重: 前景用 alpha，背景用 1-alpha
        alpha_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss: 1 - TI，其中 TI = TP / (TP + alpha*FP + beta*FN)
    alpha < beta: 更惩罚漏检(FN)，适合肿瘤检测（宁可多报不能漏报）
    alpha=0.3, beta=0.7 是医学分割常用设置
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha  # FP 惩罚系数
        self.beta = beta    # FN 惩罚系数
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 强制 float32
        logits = logits.float()
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(probs.size(0), -1)
        targets_flat = targets.contiguous().view(targets.size(0), -1)

        # TP, FP, FN
        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_index.mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数 + 平滑过渡 Warm-up 机制

    前 warmup_epochs 个 epoch: 只用 BCE + Dice（稳定训练起步）
    过渡 5 个 epoch: 从 BCE+Dice 线性过渡到 Dice+Focal+Tversky
    之后: 完全使用 Dice + Focal + Tversky 的加权组合

    设计意图：
    - BCE+Dice 在初期帮助模型快速学会大致的分割区域
    - 平滑过渡避免损失函数突变导致 NaN
    - Focal+Tversky 在后期帮助模型精细化边界、减少漏检
    """
    def __init__(self, config):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=1.0)
        self.focal_loss = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
        self.tversky_loss = TverskyLoss(alpha=config.tversky_alpha, beta=config.tversky_beta)

        # BCE 用于 warm-up 阶段
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.dice_weight = config.dice_weight
        self.focal_weight = config.focal_weight
        self.tversky_weight = config.tversky_weight
        self.warmup_epochs = config.loss_warmup_epochs
        self.transition_epochs = 5  # 平滑过渡持续 5 个 epoch
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """每个 epoch 开始时调用，用于 warm-up 切换"""
        self.current_epoch = epoch

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        返回 dict: {'total': ..., 'dice': ..., 'focal': ..., 'tversky': ..., 'bce': ...}
        方便日志记录各项损失的走势
        """
        # 强制 float32 确保数值安全
        logits = logits.float()
        targets = targets.float()

        dice = self.dice_loss(logits, targets)

        transition_end = self.warmup_epochs + self.transition_epochs

        if self.current_epoch < self.warmup_epochs:
            # ===== Warm-up 阶段: BCE + Dice =====
            bce = self.bce_loss(logits, targets)
            total = 0.5 * bce + 0.5 * dice
            return {
                'total': total,
                'dice': dice.item(),
                'bce': bce.item(),
                'focal': 0.0,
                'tversky': 0.0,
                'phase': 'warmup'
            }
        elif self.current_epoch < transition_end:
            # ===== 平滑过渡阶段 =====
            # alpha: 0→1 线性过渡（warmup结束时=0, transition结束时=1）
            alpha = (self.current_epoch - self.warmup_epochs) / self.transition_epochs

            # warmup 部分: BCE + Dice
            bce = self.bce_loss(logits, targets)
            warmup_loss = 0.5 * bce + 0.5 * dice

            # combined 部分: Dice + Focal + Tversky
            focal = self.focal_loss(logits, targets)
            tversky = self.tversky_loss(logits, targets)
            combined_loss = (self.dice_weight * dice +
                             self.focal_weight * focal +
                             self.tversky_weight * tversky)

            # 线性插值
            total = (1.0 - alpha) * warmup_loss + alpha * combined_loss

            # NaN 保护: 如果 combined 部分出 NaN，退回 warmup 损失
            if torch.isnan(total) or torch.isinf(total):
                total = warmup_loss
                print(f"  [WARNING] 检测到 NaN/Inf，退回 warmup 损失")

            return {
                'total': total,
                'dice': dice.item(),
                'bce': bce.item(),
                'focal': focal.item() if not torch.isnan(focal) else 0.0,
                'tversky': tversky.item() if not torch.isnan(tversky) else 0.0,
                'phase': f'transition(α={alpha:.2f})'
            }
        else:
            # ===== 正式阶段: Dice + Focal + Tversky =====
            focal = self.focal_loss(logits, targets)
            tversky = self.tversky_loss(logits, targets)
            total = (self.dice_weight * dice +
                     self.focal_weight * focal +
                     self.tversky_weight * tversky)

            # NaN 保护
            if torch.isnan(total) or torch.isinf(total):
                bce = self.bce_loss(logits, targets)
                total = 0.5 * bce + 0.5 * dice
                print(f"  [WARNING] 检测到 NaN/Inf，退回 BCE+Dice 损失")

            return {
                'total': total,
                'dice': dice.item(),
                'focal': focal.item() if not torch.isnan(focal) else 0.0,
                'tversky': tversky.item() if not torch.isnan(tversky) else 0.0,
                'bce': 0.0,
                'phase': 'combined'
            }


def get_loss_fn(config):
    """
    根据配置返回损失函数实例
    """
    if config.loss == "dice":
        return DiceLoss()
    elif config.loss == "focal":
        return FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    elif config.loss == "tversky":
        return TverskyLoss(alpha=config.tversky_alpha, beta=config.tversky_beta)
    elif config.loss == "combined":
        return CombinedLoss(config)
    else:
        raise ValueError(f"未知的损失函数类型: {config.loss}")
