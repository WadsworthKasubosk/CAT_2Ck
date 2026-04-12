# -*- coding: utf-8 -*-
"""
CTAI 推理模块
- 滑窗推理: 256×256 patch, stride=128, 重叠区域取平均
- TTA: 8 种几何变换的概率平均
- 后处理: 连通域筛选 + 形态学清理
"""

import os
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm


# ============================================================
# 滑窗推理
# ============================================================

def sliding_window_inference(model, image_tensor, patch_size=256, stride=128, device='cpu'):
    """
    滑窗推理: 对大图用小 patch 做推理，重叠区域取概率平均

    Args:
        model: 已加载的模型（eval 模式）
        image_tensor: [1, H, W] 的 float32 tensor
        patch_size: patch 大小
        stride: 步长（stride < patch_size 则有重叠）
        device: 推理设备

    Returns:
        prob_map: [H, W] 概率图 (0~1)
    """
    model.eval()
    _, h, w = image_tensor.shape

    # Padding 确保能整除
    pad_h = (patch_size - h % patch_size) % patch_size if h % stride != 0 else 0
    pad_w = (patch_size - w % patch_size) % patch_size if w % stride != 0 else 0

    if pad_h > 0 or pad_w > 0:
        image_tensor = torch.nn.functional.pad(
            image_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='constant', value=0
        ).squeeze(0)

    _, new_h, new_w = image_tensor.shape

    # 累加概率图和计数图
    prob_sum = torch.zeros(new_h, new_w, device=device)
    count_map = torch.zeros(new_h, new_w, device=device)

    with torch.no_grad():
        for y in range(0, new_h - patch_size + 1, stride):
            for x in range(0, new_w - patch_size + 1, stride):
                patch = image_tensor[:, y:y + patch_size, x:x + patch_size]
                patch_input = patch.unsqueeze(0).to(device)  # [1, 1, ps, ps]

                output = model(patch_input)
                # 处理 deep supervision（推理时只用 final）
                if isinstance(output, list):
                    output = output[0]

                prob = torch.sigmoid(output).squeeze()  # [ps, ps]
                prob_sum[y:y + patch_size, x:x + patch_size] += prob
                count_map[y:y + patch_size, x:x + patch_size] += 1.0

    # 避免除零
    count_map = torch.clamp(count_map, min=1.0)
    prob_map = prob_sum / count_map

    # 移除 padding
    prob_map = prob_map[:h, :w]

    return prob_map.cpu().numpy()


# ============================================================
# TTA (Test Time Augmentation)
# ============================================================

def tta_inference(model, image_tensor, patch_size=256, stride=128, device='cpu'):
    """
    8 种几何变换的 TTA 推理
    每种变换: 推理 → 反变换 → 累加概率 → 取平均

    变换列表:
    0: 原图
    1: 水平翻转
    2: 垂直翻转
    3: 水平+垂直翻转
    4: 90° 旋转
    5: 180° 旋转
    6: 270° 旋转
    7: 90° + 水平翻转
    """
    prob_maps = []

    for aug_idx in range(8):
        # 正向变换
        aug_tensor = _apply_augmentation(image_tensor, aug_idx)

        # 滑窗推理
        prob = sliding_window_inference(model, aug_tensor, patch_size, stride, device)

        # 反向变换
        prob = _reverse_augmentation(prob, aug_idx)

        prob_maps.append(prob)

    # 8 次推理的概率平均
    return np.mean(prob_maps, axis=0)


def _apply_augmentation(tensor, aug_idx):
    """对 [1, H, W] tensor 做几何变换"""
    if aug_idx == 0:
        return tensor
    elif aug_idx == 1:
        return torch.flip(tensor, dims=[2])      # 水平翻转
    elif aug_idx == 2:
        return torch.flip(tensor, dims=[1])      # 垂直翻转
    elif aug_idx == 3:
        return torch.flip(tensor, dims=[1, 2])   # 水平+垂直翻转
    elif aug_idx == 4:
        return torch.rot90(tensor, k=1, dims=[1, 2])   # 90°
    elif aug_idx == 5:
        return torch.rot90(tensor, k=2, dims=[1, 2])   # 180°
    elif aug_idx == 6:
        return torch.rot90(tensor, k=3, dims=[1, 2])   # 270°
    elif aug_idx == 7:
        t = torch.rot90(tensor, k=1, dims=[1, 2])
        return torch.flip(t, dims=[2])            # 90° + 水平翻转
    return tensor


def _reverse_augmentation(prob_np, aug_idx):
    """对 [H, W] numpy 概率图做反向几何变换"""
    if aug_idx == 0:
        return prob_np
    elif aug_idx == 1:
        return np.flip(prob_np, axis=1).copy()
    elif aug_idx == 2:
        return np.flip(prob_np, axis=0).copy()
    elif aug_idx == 3:
        return np.flip(prob_np, axis=(0, 1)).copy()
    elif aug_idx == 4:
        return np.rot90(prob_np, k=-1).copy()
    elif aug_idx == 5:
        return np.rot90(prob_np, k=-2).copy()
    elif aug_idx == 6:
        return np.rot90(prob_np, k=-3).copy()
    elif aug_idx == 7:
        prob_np = np.flip(prob_np, axis=1).copy()
        return np.rot90(prob_np, k=-1).copy()
    return prob_np


# ============================================================
# 后处理管道
# ============================================================

def postprocess(prob_map: np.ndarray, threshold=0.5, min_area=30, max_area=5000):
    """
    后处理管道:
    1. 阈值二值化
    2. 连通域分析，去除过小区域 (< min_area)
    3. 去除过大区域 (> max_area，防止误报整个器官)
    4. 形态学闭运算，填充小孔
    5. 形态学开运算，去除毛刺
    """
    # Step 1: 二值化
    binary = (prob_map > threshold).astype(np.uint8)

    # Step 2+3: 连通域筛选
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)

    for i in range(1, num_labels):  # 跳过背景 label=0
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == i] = 1

    # Step 4: 闭运算（填充小孔）
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_close)

    # Step 5: 开运算（去除毛刺）
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel_open)

    return filtered


# ============================================================
# 完整推理流程
# ============================================================

def run_inference(model, dataset, config, device='cpu', output_dir='inference_results'):
    """
    对数据集中的所有图像执行完整推理流程

    Args:
        model: 已加载的模型
        dataset: CTFullImageDataset 实例
        config: TrainConfig 配置
        device: 推理设备
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'prob_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'binary'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlay'), exist_ok=True)

    model.eval()
    results = []

    for idx in tqdm(range(len(dataset)), desc="推理中"):
        image_tensor, mask_tensor, person_id, slice_id = dataset[idx]

        # 推理
        if config.use_tta:
            prob_map = tta_inference(model, image_tensor,
                                    config.eval_patch_size, config.eval_stride, device)
        else:
            prob_map = sliding_window_inference(model, image_tensor,
                                               config.eval_patch_size, config.eval_stride, device)

        # 后处理
        pred_binary = postprocess(prob_map, config.threshold, config.min_area, config.max_area)

        # GT
        gt_mask = mask_tensor.squeeze().numpy()

        # 计算指标
        dice = _dice_score(pred_binary, gt_mask)
        precision, recall = _precision_recall(pred_binary, gt_mask)
        fg_ratio_pred = pred_binary.sum() / pred_binary.size
        fg_ratio_gt = gt_mask.sum() / gt_mask.size

        result = {
            'person_id': person_id,
            'slice_id': slice_id,
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'fg_ratio_pred': float(fg_ratio_pred),
            'fg_ratio_gt': float(fg_ratio_gt),
        }
        results.append(result)

        # 保存概率图
        np.save(os.path.join(output_dir, 'prob_maps', f'{person_id}_{slice_id}.npy'), prob_map)

        # 保存二值化结果
        cv2.imwrite(
            os.path.join(output_dir, 'binary', f'{person_id}_{slice_id}.png'),
            (pred_binary * 255).astype(np.uint8)
        )

        # 保存可视化叠加图
        _save_overlay(image_tensor.squeeze().numpy(), gt_mask, prob_map, pred_binary,
                      os.path.join(output_dir, 'overlay', f'{person_id}_{slice_id}.png'),
                      dice)

    # 保存评估汇总
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印汇总
    tumor_results = [r for r in results if r['fg_ratio_gt'] > 0]
    if tumor_results:
        avg_dice = np.mean([r['dice'] for r in tumor_results])
        avg_precision = np.mean([r['precision'] for r in tumor_results])
        avg_recall = np.mean([r['recall'] for r in tumor_results])
        print(f"\n含肿瘤切片 ({len(tumor_results)} 张):")
        print(f"  平均 Dice:      {avg_dice:.4f}")
        print(f"  平均 Precision: {avg_precision:.4f}")
        print(f"  平均 Recall:    {avg_recall:.4f}")

    return results


# ============================================================
# 辅助函数
# ============================================================

def _dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)


def _precision_recall(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision, recall


def _save_overlay(image_np, gt_mask, prob_map, pred_binary, save_path, dice):
    """保存可视化叠加图: [原图] [GT] [概率热图] [预测] [叠加]"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('GT Mask')
        axes[1].axis('off')

        im = axes[2].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('预测概率图')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        axes[3].imshow(pred_binary, cmap='gray')
        axes[3].set_title('预测二值化')
        axes[3].axis('off')

        # 叠加对比
        overlay = np.stack([image_np] * 3, axis=-1)
        overlay = (overlay * 255).astype(np.uint8)
        overlay[gt_mask > 0, 1] = np.clip(overlay[gt_mask > 0, 1].astype(int) + 100, 0, 255)
        overlay[pred_binary > 0, 0] = np.clip(overlay[pred_binary > 0, 0].astype(int) + 100, 0, 255)
        axes[4].imshow(overlay)
        axes[4].set_title(f'叠加 (Dice={dice:.3f})')
        axes[4].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] 保存可视化失败: {e}")
