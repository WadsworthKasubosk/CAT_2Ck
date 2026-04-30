import os

import SimpleITK as sitk
import cv2
import numpy as np


def data_in_one(inputdata):
    inputdata = inputdata.astype(np.float32)
    if inputdata.max() == inputdata.min():
        return np.zeros_like(inputdata, dtype=np.float32)
    return (inputdata - inputdata.min()) / (inputdata.max() - inputdata.min())


def pre_process(data_path):
    """
    读取 DICOM，保存为 PNG 供 YOLO 推理
    返回: (image_png_path, file_name)
    """
    image = sitk.ReadImage(data_path)
    image_array = sitk.GetArrayFromImage(image)  # [1, H, W] or [H, W]
    if image_array.ndim == 3:
        image_array = image_array[0]

    # 归一化为 uint8
    vis_array = data_in_one(image_array) * 255
    vis_array = vis_array.astype(np.uint8)

    file_name = os.path.split(data_path)[1].replace('.dcm', '')

    # 保存原图 PNG（灰度，供前端显示）
    image_png_path = f'./tmp/image/{file_name}.png'
    cv2.imwrite(image_png_path, vis_array, (cv2.IMWRITE_PNG_COMPRESSION, 0))

    # 同时保存一份 3 通道版本供 YOLO 推理
    img_3ch = cv2.cvtColor(vis_array, cv2.COLOR_GRAY2BGR)
    yolo_input_path = f'./tmp/image/{file_name}_3ch.png'
    cv2.imwrite(yolo_input_path, img_3ch)

    return yolo_input_path, file_name


def last_process(file_name):
    """读取分割 mask，用半透明叠加方式标注肿瘤区域到原图上"""
    image = cv2.imread(f'./tmp/image/{file_name}.png')
    mask = cv2.imread(f'./tmp/mask/{file_name}_mask.png', 0)

    if mask is None:
        print(f"[WARNING] mask 文件不存在: ./tmp/mask/{file_name}_mask.png")
        if image is not None:
            cv2.imwrite(f'./tmp/draw/{file_name}.png', image)
        return

    # 确保 mask 与 image 尺寸一致
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # 半透明绿色叠加：只在 mask>0 区域混合
    overlay = image.copy()
    alpha = 0.3
    mask_bool = mask > 0
    # 在前景区域混合绿色
    for c in range(3):
        green_val = [0, 255, 0][c]
        overlay[:, :, c] = np.where(
            mask_bool,
            np.clip(image[:, :, c] * (1 - alpha) + green_val * alpha, 0, 255).astype(np.uint8),
            image[:, :, c]
        )

    # 绘制轮廓边缘线
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0] if len(result) == 2 else result[1]
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(f'./tmp/draw/{file_name}.png', overlay)
    fg_pct = mask_bool.sum() / mask.size * 100
    print(f"[INFO] 标注完成: {file_name}, 前景像素占比: {fg_pct:.1f}%")
