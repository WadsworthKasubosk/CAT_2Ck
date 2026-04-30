# -*- coding: utf-8 -*-
"""
用已有 U-Net 权重跑一遍验证集，输出 Dice / Precision / Recall。
不需要重新训练，几分钟出结果。

用法:
  python eval_unet.py --weights ../实验成果/模型权重/baseline_unet_weights.pth --data ../直肠癌数据_tiny
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
import SimpleITK as sitk

# 把模型代码目录加到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CTAI_model'))
from net.unet import UNet
from net.attention_unet import AttentionUNet


def dcm_to_tensor(dcm_path, wc=40, ww=400):
    img = sitk.ReadImage(dcm_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    lo, hi = wc - ww / 2, wc + ww / 2
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    # ROI 裁剪 (与原项目 process.py 一致)
    arr = arr[270:430, 200:300]
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor, arr


def load_mask_dcm(mask_path):
    m = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    if m.ndim == 3:
        m = m[0]
    m = m[270:430, 200:300]
    return (m > 0).astype(np.uint8)


def calc_metrics(pred, gt):
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    tp = (pred_bool & gt_bool).sum()
    fp = (pred_bool & ~gt_bool).sum()
    fn = (~pred_bool & gt_bool).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return dice, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='模型权重路径')
    parser.add_argument('--data', required=True, help='数据目录 (含患者子文件夹)')
    parser.add_argument('--model', default='unet', choices=['unet', 'attention'],
                        help='模型类型: unet 或 attention')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # 加载模型
    if args.model == 'attention':
        model = AttentionUNet(in_channels=1, out_channels=1)
    else:
        model = UNet(in_channels=1, out_channels=1)

    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f'模型已加载: {args.weights}')

    # 扫描数据
    patients = sorted([d for d in os.listdir(args.data)
                       if os.path.isdir(os.path.join(args.data, d))])

    all_dice, all_prec, all_rec = [], [], []

    for pid in patients:
        pdir = os.path.join(args.data, pid)
        scans = [d for d in os.listdir(pdir)
                 if os.path.isdir(os.path.join(pdir, d)) and '_mask' not in d]
        for sn in scans:
            sdir = os.path.join(pdir, sn)
            mdir = os.path.join(pdir, sn + '_mask')
            if not os.path.isdir(mdir):
                continue
            for dcm_path in sorted(glob.glob(os.path.join(sdir, '*.dcm'))):
                mask_path = os.path.join(mdir, os.path.basename(dcm_path))
                if not os.path.isfile(mask_path):
                    continue

                gt = load_mask_dcm(mask_path)
                if gt.sum() == 0:
                    continue  # 跳过无标注

                tensor, _ = dcm_to_tensor(dcm_path)
                with torch.no_grad():
                    out = model(tensor.to(device))
                    pred = (torch.sigmoid(out) > args.threshold).cpu().numpy()[0, 0].astype(np.uint8)

                d, p, r = calc_metrics(pred, gt)
                all_dice.append(d)
                all_prec.append(p)
                all_rec.append(r)

    print(f'\n{"="*50}')
    print(f'评估样本数: {len(all_dice)}')
    print(f'Dice:      {np.mean(all_dice):.3f} ± {np.std(all_dice):.3f}')
    print(f'Precision: {np.mean(all_prec):.3f} ± {np.std(all_prec):.3f}')
    print(f'Recall:    {np.mean(all_rec):.3f} ± {np.std(all_rec):.3f}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
