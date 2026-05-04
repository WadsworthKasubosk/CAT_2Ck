# -*- coding: utf-8 -*-
"""
6 列级联推理可视化（从左到右）：
  1. 原始 CT (gray)
  2. YOLO 检测框 (红色 + "tumor 0.87")
  3. UNet 概率热图 (jet, 仅框内有色; 框外保持灰度原图)
  4. UNet 二值化 mask (白)
  5. 最终融合 (原图 + 红框 + 半透明绿色 mask) ← 给导师看的主图
  6. GT 对比 (GT 蓝色 + 预测绿色; 重叠区青色)，标题写 Dice

中文字体 fallback：SimHei → Microsoft YaHei → DejaVu Sans（缺中文则 warn）。
"""
from __future__ import annotations

import os
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.font_manager import FontProperties, findfont
import numpy as np

# ---- 中文字体一次性配置 ----
_CN_FONT_CONFIGURED = False


def _configure_cn_font():
    global _CN_FONT_CONFIGURED
    if _CN_FONT_CONFIGURED:
        return
    candidates = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
    chosen = None
    for name in candidates:
        try:
            path = findfont(FontProperties(family=name), fallback_to_default=False)
            if os.path.basename(path).lower() not in ("dejavusans.ttf",):
                chosen = name
                break
        except Exception:
            continue
    if chosen:
        plt.rcParams["font.family"] = chosen
        plt.rcParams["axes.unicode_minus"] = False
    else:
        warnings.warn("未找到中文字体（SimHei/YaHei/Noto），中文可能显示为方块。")
    _CN_FONT_CONFIGURED = True


# ---- 工具：构造 RGB 灰度底 ----

def _gray_to_rgb(img01: np.ndarray) -> np.ndarray:
    """[H,W] float32 in [0,1] → [H,W,3] uint8."""
    g = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


# ---- 主入口 ----

def save_cascade_overlay(
    image_np: np.ndarray,
    gt_mask: np.ndarray,
    detections: list,
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    dice: float,
    save_path: str,
    dpi: int = 200,
    title_prefix: str = "",
):
    """6 列可视化主函数。

    Args:
        image_np: [H,W] float32 in [0,1]，已做窗宽窗位（与 CascadeInference 输入同口径）
        gt_mask:  [H,W] uint8 in {0,1}，可为 None（会跳过第 6 列的 GT 对比）
        detections: list of {'bbox':[x1,y1,x2,y2], 'score':float, 'class_id':int}
        prob_map: [H,W] float32，仅 ROI 内非零
        pred_mask: [H,W] uint8 in {0,1}
        dice:    float，已计算的 Dice 分数（用于标题）
        save_path: 输出 PNG 绝对路径
    """
    _configure_cn_font()

    H, W = image_np.shape
    base_rgb = _gray_to_rgb(image_np)

    fig, axes = plt.subplots(1, 6, figsize=(30, 5.2))

    # ========== 1. 原始 CT ==========
    axes[0].imshow(image_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("① 原始 CT")
    axes[0].axis("off")

    # ========== 2. YOLO 检测框 ==========
    axes[1].imshow(image_np, cmap="gray", vmin=0, vmax=1)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none",
        )
        axes[1].add_patch(rect)
        axes[1].text(
            x1, max(0, y1 - 5), f"tumor {d['score']:.2f}",
            color="white", fontsize=9, weight="bold",
            bbox=dict(facecolor="red", alpha=0.85, edgecolor="none", pad=1.5),
        )
    axes[1].set_title(f"② YOLO 检测框 (n={len(detections)})")
    axes[1].axis("off")

    # ========== 3. UNet 概率热图（jet, 仅 ROI 内）==========
    axes[2].imshow(image_np, cmap="gray", vmin=0, vmax=1)
    masked = np.ma.masked_where(prob_map <= 0, prob_map)
    if masked.count() > 0:                        # 至少有一个像素非零
        im = axes[2].imshow(masked, cmap="jet", alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("③ UNet 概率热图")
    axes[2].axis("off")

    # ========== 4. UNet 二值 mask ==========
    axes[3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("④ UNet 二值 mask")
    axes[3].axis("off")

    # ========== 5. 最终融合（原图 + 红框 + 绿色半透明 mask）==========
    fused = base_rgb.copy()
    if pred_mask.any():
        green = np.zeros_like(fused); green[..., 1] = 255
        m3 = (pred_mask > 0)[..., None]
        fused = np.where(m3, (0.5 * fused + 0.5 * green).astype(np.uint8), fused)
    axes[4].imshow(fused)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.2, edgecolor="red", facecolor="none",
        )
        axes[4].add_patch(rect)
    title5 = f"⑤ 级联最终输出"
    if title_prefix:
        title5 = f"{title_prefix}\n{title5}"
    axes[4].set_title(title5, fontweight="bold", fontsize=12)
    axes[4].axis("off")

    # ========== 6. GT 对比 (蓝 GT + 绿 Pred) ==========
    cmp = base_rgb.copy()
    if gt_mask is not None:
        gt_b = (gt_mask > 0)
        pr_b = (pred_mask > 0)
        # 蓝 = 仅 GT；绿 = 仅 Pred；青 = 重叠
        only_gt = gt_b & ~pr_b
        only_pr = ~gt_b & pr_b
        both = gt_b & pr_b
        cmp[only_gt] = [0, 0, 255]
        cmp[only_pr] = [0, 255, 0]
        cmp[both] = [0, 255, 255]
        title6 = f"⑥ GT(蓝) vs Pred(绿)  Dice={dice:.3f}"
    else:
        title6 = "⑥ 无 GT"
    axes[5].imshow(cmp)
    axes[5].set_title(title6)
    axes[5].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
