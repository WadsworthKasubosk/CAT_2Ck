# -*- coding: utf-8 -*-
"""
将原始 DCM + mask PNG 数据转换为 YOLO11-seg 训练格式

实际数据结构:
  直肠癌数据/
    1002/
      arterial phase/
        10001.dcm
        10001_mask.png    <-- 二值 mask (0/255), 512x512
        10002.dcm
        10002_mask.png
        ...
      venous phase/
        20001.dcm
        20001_mask.png
        ...

输出:
  datasets/rectal_tumor_seg/
    images/train/  images/val/
    labels/train/  labels/val/
    data.yaml
"""

import os
import random
import numpy as np
import cv2
import SimpleITK as sitk


def cv2_imread(path, flags=cv2.IMREAD_GRAYSCALE):
    """cv2.imread 中文路径兼容"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def cv2_imwrite(path, img, params=None):
    """cv2.imwrite 中文路径兼容"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img, params or [])
    if ok:
        buf.tofile(path)
    return ok


def dcm_to_uint8(dcm_path, window_center=40, window_width=400):
    """读取 DCM 并应用窗宽窗位，输出 uint8 图像"""
    image = sitk.ReadImage(dcm_path)
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    arr = np.clip(arr, lower, upper)
    arr = ((arr - lower) / (upper - lower) * 255).astype(np.uint8)
    return arr


def mask_to_yolo_seg(mask_binary, img_h, img_w):
    """
    将二值 mask 转为 YOLO 分割标注格式
    返回: list of "class_id x1 y1 x2 y2 ... xn yn" (归一化坐标)
    """
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        cnt = cnt.squeeze()
        if cnt.ndim != 2 or len(cnt) < 3:
            continue
        points = []
        for x, y in cnt:
            points.append(f"{x / img_w:.6f}")
            points.append(f"{y / img_h:.6f}")
        line = "0 " + " ".join(points)
        lines.append(line)
    return lines


def convert_dataset(data_dir, output_dir, val_ratio=0.15, seed=42):
    """
    扫描数据目录，转换为 YOLO 格式。
    适配实际结构: patient/phase/{id}.dcm + {id}_mask.png
    """
    random.seed(seed)

    # 收集所有患者目录
    patients = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
    random.shuffle(patients)

    val_count = max(1, int(len(patients) * val_ratio))
    val_patients = set(patients[:val_count])
    train_patients = set(patients[val_count:])

    print(f"总患者: {len(patients)}, 训练: {len(train_patients)}, 验证: {len(val_patients)}")

    # 创建输出目录
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    total_images = 0
    total_with_tumor = 0
    skipped_no_mask = 0

    for pid in patients:
        split = 'val' if pid in val_patients else 'train'
        patient_dir = os.path.join(data_dir, pid)

        # 遍历所有扫描期相 (arterial phase, venous phase, ...)
        phase_dirs = [d for d in os.listdir(patient_dir)
                      if os.path.isdir(os.path.join(patient_dir, d))]

        for phase_name in phase_dirs:
            phase_dir = os.path.join(patient_dir, phase_name)

            # 找所有 DCM 文件
            dcm_files = sorted([f for f in os.listdir(phase_dir) if f.endswith('.dcm')])

            for dcm_fname in dcm_files:
                dcm_path = os.path.join(phase_dir, dcm_fname)
                stem = os.path.splitext(dcm_fname)[0]  # e.g. "10001"
                mask_fname = f"{stem}_mask.png"
                mask_path = os.path.join(phase_dir, mask_fname)

                if not os.path.isfile(mask_path):
                    skipped_no_mask += 1
                    continue

                # 读取 DCM -> uint8 (窗宽窗位)
                img = dcm_to_uint8(dcm_path)
                h, w = img.shape[:2]

                # 转 3 通道
                img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # 读取 mask PNG (中文路径兼容)
                mask = cv2_imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"[WARN] 无法读取 mask: {mask_path}")
                    continue
                mask_binary = (mask > 0).astype(np.uint8) * 255

                # 转 YOLO 标签
                label_lines = mask_to_yolo_seg(mask_binary, h, w)
                has_tumor = len(label_lines) > 0

                # 生成安全文件名 (去空格)
                safe_phase = phase_name.replace(' ', '_')
                out_name = f"{pid}_{safe_phase}_{stem}"

                # 保存图像 (输出目录用英文，不需要中文路径兼容)
                img_out = os.path.join(output_dir, 'images', split, f"{out_name}.png")
                cv2.imwrite(img_out, img_3ch)

                # 保存标签
                lbl_out = os.path.join(output_dir, 'labels', split, f"{out_name}.txt")
                with open(lbl_out, 'w') as f:
                    f.write("\n".join(label_lines))

                total_images += 1
                if has_tumor:
                    total_with_tumor += 1

    print(f"\n转换完成:")
    print(f"  总图像: {total_images}")
    print(f"  含肿瘤: {total_with_tumor}")
    print(f"  无肿瘤(负样本): {total_images - total_with_tumor}")
    print(f"  跳过(无mask): {skipped_no_mask}")
    print(f"  训练集: {len([f for f in os.listdir(os.path.join(output_dir, 'images', 'train')) if f.endswith('.png')])} 张")
    print(f"  验证集: {len([f for f in os.listdir(os.path.join(output_dir, 'images', 'val')) if f.endswith('.png')])} 张")

    # 生成 data.yaml
    abs_out = os.path.abspath(output_dir)
    yaml_content = f"""# CTAI 直肠肿瘤分割数据集 (YOLO11-seg 格式)
path: {abs_out}
train: images/train
val: images/val

names:
  0: tumor

nc: 1
"""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"\ndata.yaml 已保存: {yaml_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DCM+mask -> YOLO11-seg 格式转换')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join('C:', os.sep, 'Users', 'da983', 'CAT_2Ck', '直肠癌数据'))
    parser.add_argument('--output_dir', type=str, default='./datasets/rectal_tumor_seg')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    args = parser.parse_args()
    convert_dataset(args.data_dir, args.output_dir, args.val_ratio)
