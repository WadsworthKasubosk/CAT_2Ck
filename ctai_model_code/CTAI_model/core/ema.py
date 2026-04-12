# -*- coding: utf-8 -*-
"""
EMA (Exponential Moving Average) 模型权重平滑
训练时维护模型参数的指数移动平均，评估和推理时使用 EMA 权重
通常能带来 0.5~2% 的分割精度提升，几乎无额外开销
"""

import torch
from copy import deepcopy


class EMA:
    """
    用法:
        ema = EMA(model, decay=0.999)
        for epoch in range(epochs):
            for batch in loader:
                loss.backward()
                optimizer.step()
                ema.update(model)       # 每步更新影子权重
            # 评估时切换到 EMA 权重
            ema.apply_shadow(model)
            evaluate(model)
            ema.restore(model)          # 评估完恢复训练权重
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        """
        Args:
            model: 要跟踪的模型
            decay: 衰减系数，越大越平滑（0.999 = 保留 99.9% 历史，合入 0.1% 新值）
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化影子权重为当前模型权重的深拷贝
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """更新影子权重: shadow = decay * shadow + (1 - decay) * current"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: torch.nn.Module):
        """将影子权重应用到模型（评估用），同时备份当前训练权重"""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """恢复训练权重（评估完毕后调用）"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
