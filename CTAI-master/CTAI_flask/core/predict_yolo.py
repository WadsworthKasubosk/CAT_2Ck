# -*- coding: utf-8 -*-
"""
YOLO11-seg 推理模块
输入: PNG 图像路径
输出: 二值 mask 保存到 tmp/mask/
"""

import os
import cv2
import numpy as np


def predict_yolo(image_path, file_name, model):
    """
    使用 YOLO11-seg 模型进行肿瘤分割推理

    Args:
        image_path: 输入图像路径（PNG，已由 process.py 生成）
        file_name: 文件名（不含后缀）
        model: 已加载的 YOLO 模型对象

    Returns:
        None (mask 保存到 tmp/mask/{file_name}_mask.png)
    """
    # 读取图像（YOLO 需要 BGR 3通道）
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像不存在: {image_path}")

    # 如果是灰度图，转为3通道
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]

    # YOLO11-seg 推理
    results = model.predict(
        source=img,
        conf=0.25,           # 置信度阈值
        iou=0.45,            # NMS IoU 阈值
        imgsz=512,           # 推理尺寸
        retina_masks=True,   # 高分辨率 mask（与原图同尺寸）
        verbose=False,
    )

    # 合并所有检测到的肿瘤 mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # [N, H, W]

        for i, mask in enumerate(masks):
            # 确保 mask 与原图尺寸一致
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            # 二值化并合并
            binary = (mask > 0.5).astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, binary)

        num_detections = len(masks)
        conf_scores = results[0].boxes.conf.cpu().numpy()
        print(f"[INFO] YOLO11-seg 推理完成: {file_name}, "
              f"检测到 {num_detections} 个肿瘤区域, "
              f"置信度: {conf_scores}")
    else:
        print(f"[INFO] YOLO11-seg 推理完成: {file_name}, 未检测到肿瘤区域")

    # 保存 mask
    mask_path = f'./tmp/mask/{file_name}_mask.png'
    cv2.imwrite(mask_path, combined_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    fg_count = (combined_mask > 0).sum()
    print(f"[INFO] Mask 已保存: {mask_path}, 前景像素数: {fg_count}")
