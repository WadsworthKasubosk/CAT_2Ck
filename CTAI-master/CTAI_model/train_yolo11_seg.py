# -*- coding: utf-8 -*-
"""
YOLO11-seg 训练脚本
使用 ultralytics 库训练肿瘤分割模型
"""

from ultralytics import YOLO
import argparse
import os


def train(args):
    # 加载 YOLO11-seg 预训练模型
    # 可选: yolo11n-seg (最小), yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg
    model = YOLO(args.model)

    print(f"[INFO] 模型: {args.model}")
    print(f"[INFO] 数据集: {args.data}")
    print(f"[INFO] Epochs: {args.epochs}")
    print(f"[INFO] 图像尺寸: {args.imgsz}")
    print(f"[INFO] Batch: {args.batch}")

    # 训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,       # 早停
        save=True,
        save_period=10,               # 每10个epoch保存一次
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=args.lr,
        lrf=0.01,                     # 最终学习率 = lr0 * lrf
        warmup_epochs=3,
        cos_lr=True,                  # 余弦退火
        close_mosaic=10,              # 最后10个epoch关闭mosaic
        # 数据增强（医学图像适度增强）
        hsv_h=0.0,                    # 医学图像不做色相变换
        hsv_s=0.0,                    # 不做饱和度变换
        hsv_v=0.2,                    # 轻微亮度变化
        degrees=15.0,                 # 旋转
        translate=0.1,
        scale=0.3,
        flipud=0.5,                   # 上下翻转
        fliplr=0.5,                   # 左右翻转
        mosaic=0.5,
        mixup=0.0,                    # 医学图像不做mixup
        overlap_mask=True,
        mask_ratio=4,                 # mask下采样比例
        single_cls=True,              # 单类别（肿瘤）
    )

    print(f"\n训练完成！最佳模型: {results.save_dir}/weights/best.pt")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO11-seg 肿瘤分割训练')
    parser.add_argument('--model', type=str, default='yolo11n-seg.pt',
                        help='预训练模型 (yolo11n-seg, yolo11s-seg, yolo11m-seg)')
    parser.add_argument('--data', type=str, default='./datasets/rectal_tumor_seg/data.yaml')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', type=str, default='0', help='cuda device or cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--project', type=str, default='runs/segment')
    parser.add_argument('--name', type=str, default='rectal_tumor')
    args = parser.parse_args()

    train(args)
