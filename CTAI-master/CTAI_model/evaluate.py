# -*- coding: utf-8 -*-
"""
CTAI 模型评估脚本
- 逐切片指标: Dice, IoU, Precision, Recall, Specificity
- 全局指标: 平均 Dice, HD95 (可选)
- 输出: 汇总表格 + JSON + CSV
"""

import os
import sys
import json
import csv
import numpy as np
import torch
from tqdm import tqdm

from config import TrainConfig
from data.dataset import CTFullImageDataset
from inference import sliding_window_inference, tta_inference, postprocess


def dice_score(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)


def iou_score(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)


def precision_recall_specificity(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return precision, recall, specificity


def hausdorff_95(pred, gt):
    """HD95 指标（需要 scipy）"""
    try:
        from scipy.ndimage import distance_transform_edt
        pred, gt = pred.astype(bool), gt.astype(bool)
        if not pred.any() or not gt.any():
            return float('inf') if (pred.any() != gt.any()) else 0.0

        # 前景到对方边界的距离
        dist_pred = distance_transform_edt(~pred)
        dist_gt = distance_transform_edt(~gt)

        d_pred_to_gt = dist_gt[pred]
        d_gt_to_pred = dist_pred[gt]

        hd95 = max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))
        return hd95
    except ImportError:
        return float('nan')


def evaluate_model(model, config, device='cpu', use_tta=False, output_dir='evaluation'):
    """
    全面评估模型性能

    Returns:
        results: 逐切片结果列表
        summary: 全局汇总字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    dataset = CTFullImageDataset(config.data_dir, config)
    print(f"评估数据集: {len(dataset)} 张切片")

    model.eval()
    results = []

    for idx in tqdm(range(len(dataset)), desc="评估中"):
        image_tensor, mask_tensor, person_id, slice_id = dataset[idx]
        gt_mask = mask_tensor.squeeze().numpy()
        has_tumor = gt_mask.sum() > 0

        # 推理
        if use_tta:
            prob_map = tta_inference(model, image_tensor,
                                    config.eval_patch_size, config.eval_stride, device)
        else:
            prob_map = sliding_window_inference(model, image_tensor,
                                               config.eval_patch_size, config.eval_stride, device)

        pred_binary = postprocess(prob_map, config.threshold, config.min_area, config.max_area)

        # 计算指标
        d = dice_score(pred_binary, gt_mask)
        iou = iou_score(pred_binary, gt_mask)
        prec, rec, spec = precision_recall_specificity(pred_binary, gt_mask)
        hd95 = hausdorff_95(pred_binary, gt_mask) if has_tumor else 0.0

        results.append({
            'person_id': person_id,
            'slice_id': slice_id,
            'has_tumor': has_tumor,
            'dice': float(d),
            'iou': float(iou),
            'precision': float(prec),
            'recall': float(rec),
            'specificity': float(spec),
            'hd95': float(hd95) if not np.isinf(hd95) else -1.0,
            'fg_ratio_pred': float(pred_binary.sum() / pred_binary.size),
            'fg_ratio_gt': float(gt_mask.sum() / gt_mask.size),
        })

    # ==================== 汇总 ====================
    tumor_results = [r for r in results if r['has_tumor']]
    bg_results = [r for r in results if not r['has_tumor']]

    summary = {
        'total_slices': len(results),
        'tumor_slices': len(tumor_results),
        'bg_slices': len(bg_results),
    }

    if tumor_results:
        summary['tumor_avg_dice'] = float(np.mean([r['dice'] for r in tumor_results]))
        summary['tumor_avg_iou'] = float(np.mean([r['iou'] for r in tumor_results]))
        summary['tumor_avg_precision'] = float(np.mean([r['precision'] for r in tumor_results]))
        summary['tumor_avg_recall'] = float(np.mean([r['recall'] for r in tumor_results]))
        summary['tumor_avg_specificity'] = float(np.mean([r['specificity'] for r in tumor_results]))

        hd95_vals = [r['hd95'] for r in tumor_results if r['hd95'] >= 0]
        if hd95_vals:
            summary['tumor_avg_hd95'] = float(np.mean(hd95_vals))

    # 全局 Dice（纯背景切片：都为空=1.0，否则=0.0）
    all_dice = []
    for r in results:
        if not r['has_tumor']:
            all_dice.append(1.0 if r['fg_ratio_pred'] == 0 else 0.0)
        else:
            all_dice.append(r['dice'])
    summary['global_avg_dice'] = float(np.mean(all_dice))

    # ==================== 打印 ====================
    print("\n" + "=" * 70)
    print("  评估报告")
    print("=" * 70)
    print(f"  总切片:     {summary['total_slices']}")
    print(f"  含肿瘤:     {summary['tumor_slices']}")
    print(f"  纯背景:     {summary['bg_slices']}")
    print("-" * 70)

    if tumor_results:
        print(f"  [含肿瘤切片指标]")
        print(f"    Dice:        {summary['tumor_avg_dice']:.4f}")
        print(f"    IoU:         {summary['tumor_avg_iou']:.4f}")
        print(f"    Precision:   {summary['tumor_avg_precision']:.4f}")
        print(f"    Recall:      {summary['tumor_avg_recall']:.4f}")
        print(f"    Specificity: {summary['tumor_avg_specificity']:.4f}")
        if 'tumor_avg_hd95' in summary:
            print(f"    HD95:        {summary['tumor_avg_hd95']:.2f} px")

    print(f"\n  [全局平均 Dice]: {summary['global_avg_dice']:.4f}")
    print("=" * 70)

    # ==================== 保存 ====================
    # JSON 报告
    report = {'summary': summary, 'per_slice': results}
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  JSON 报告: {os.path.join(output_dir, 'evaluation_report.json')}")

    # CSV 逐切片结果
    csv_path = os.path.join(output_dir, 'per_slice_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV 详情: {csv_path}")

    return results, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CTAI 模型评估")
    parser.add_argument('--weights', type=str, default='checkpoints/best_model.pth',
                        help='模型权重路径')
    parser.add_argument('--tta', action='store_true', help='启用 TTA')
    parser.add_argument('--output_dir', type=str, default='evaluation', help='输出目录')
    args = parser.parse_args()

    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    from train import build_model
    model = build_model(config).to(device)

    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"已加载权重: {args.weights}")
    else:
        print(f"[WARNING] 权重文件不存在: {args.weights}，使用随机初始化权重评估")

    results, summary = evaluate_model(model, config, device, use_tta=args.tta,
                                      output_dir=args.output_dir)
