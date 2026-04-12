# -*- coding: utf-8 -*-
"""
CTAI 数据管道
- 智能扫描：自动识别 DICOM + mask 配对，统计前景占比
- 两种模式：tumor_only（只含肿瘤切片）/ mixed（过采样肿瘤 + 少量背景）
- 肿瘤中心裁剪：80% 概率以肿瘤质心为锚点做 RandomCrop
- 强数据增强：albumentations 驱动，弹性变换 + 几何 + 光度
- CT 窗宽窗位：腹部 CT 标准窗 center=40, width=400
"""

import os
import numpy as np
import SimpleITK as sitk
import cv2
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False
    print("[WARNING] albumentations 未安装，将跳过数据增强。pip install albumentations")


# ============================================================
# CT 窗宽窗位预处理
# ============================================================

def apply_ct_window(image: np.ndarray, center: float = 40.0, width: float = 400.0) -> np.ndarray:
    """
    腹部 CT 窗宽窗位调整
    center=40, width=400 → HU 范围 [-160, 240]
    归一化到 [0, 1]
    """
    min_val = center - width / 2.0   # -160
    max_val = center + width / 2.0   # 240
    image = image.astype(np.float32)
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)
    return image


def simple_normalize(image: np.ndarray) -> np.ndarray:
    """简单 min-max 归一化（备用，当 DICOM 不含 HU 信息时）"""
    image = image.astype(np.float32)
    vmin, vmax = image.min(), image.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(image)
    return (image - vmin) / (vmax - vmin)


# ============================================================
# 数据扫描与统计
# ============================================================

def scan_data_directory(data_dir: str):
    """
    扫描数据目录，返回结构化样本列表
    每个样本: {
        'dcm_path': str,
        'mask_path': str,
        'person_id': str,
        'slice_id': str,
        'has_tumor': bool,
        'fg_ratio': float  (前景像素占比)
    }
    """
    samples = []

    if not os.path.isdir(data_dir):
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return samples

    for patient_dir in sorted(os.listdir(data_dir)):
        patient_path = os.path.join(data_dir, patient_dir)
        if not os.path.isdir(patient_path):
            continue

        phase_dir = os.path.join(patient_path, 'arterial phase')
        if not os.path.isdir(phase_dir):
            continue

        for fname in sorted(os.listdir(phase_dir)):
            if not fname.endswith('.dcm'):
                continue

            dcm_path = os.path.join(phase_dir, fname)
            mask_path = dcm_path.replace('.dcm', '_mask.png')

            if not os.path.exists(mask_path):
                continue

            # 读取 mask 统计前景
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            fg_pixels = (mask > 0).sum()
            total_pixels = mask.size
            fg_ratio = fg_pixels / total_pixels
            has_tumor = fg_pixels > 0

            samples.append({
                'dcm_path': dcm_path,
                'mask_path': mask_path,
                'person_id': patient_dir,
                'slice_id': fname.replace('.dcm', ''),
                'has_tumor': has_tumor,
                'fg_ratio': fg_ratio,
            })

    return samples


def print_data_report(samples: list):
    """打印数据统计报告"""
    total = len(samples)
    tumor_slices = [s for s in samples if s['has_tumor']]
    bg_slices = [s for s in samples if not s['has_tumor']]

    print("=" * 60)
    print("  数据集统计报告")
    print("=" * 60)
    print(f"  总切片数:       {total}")
    print(f"  含肿瘤切片:     {len(tumor_slices)}")
    print(f"  纯背景切片:     {len(bg_slices)}")

    if tumor_slices:
        fg_ratios = [s['fg_ratio'] for s in tumor_slices]
        print(f"  前景占比 (含肿瘤):")
        print(f"    平均:  {np.mean(fg_ratios)*100:.4f}%")
        print(f"    最小:  {np.min(fg_ratios)*100:.4f}%")
        print(f"    最大:  {np.max(fg_ratios)*100:.4f}%")

    patients = set(s['person_id'] for s in samples)
    print(f"  患者数:         {len(patients)}")
    for pid in sorted(patients):
        count = sum(1 for s in samples if s['person_id'] == pid)
        print(f"    {pid}: {count} 张切片")
    print("=" * 60)


# ============================================================
# 数据增强管道（albumentations）
# ============================================================

def get_train_transforms(crop_size: int = 256):
    """强数据增强（训练时使用）"""
    if not HAS_ALBUM:
        return None

    return A.Compose([
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15, scale_limit=0.2, rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT, p=0.8
        ),
        # 弹性变换（模拟组织形变）
        A.ElasticTransform(
            alpha=200, sigma=20, border_mode=cv2.BORDER_CONSTANT, p=0.3
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        # 光度变换
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=(0.01, 0.05), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        # 最终裁剪到目标大小（由 dataset __getitem__ 中的肿瘤中心裁剪完成）
    ])


def get_val_transforms():
    """验证/推理时不做增强"""
    return None


# ============================================================
# 肿瘤中心裁剪
# ============================================================

def tumor_center_crop(image: np.ndarray, mask: np.ndarray, crop_size: int,
                      tumor_prob: float = 0.8) -> tuple:
    """
    以肿瘤区域为中心的随机裁剪
    - 如果有肿瘤，80% 概率以肿瘤质心为锚点（+ 随机偏移）
    - 20% 概率完全随机裁剪
    - 如果纯背景，完全随机裁剪
    """
    h, w = image.shape[:2]
    ch, cw = crop_size, crop_size

    # 如果图像比 crop 还小，先 padding
    if h < ch or w < cw:
        pad_h = max(ch - h, 0)
        pad_w = max(cw - w, 0)
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        h, w = image.shape[:2]

    has_tumor = mask.sum() > 0

    if has_tumor and np.random.random() < tumor_prob:
        # 以肿瘤质心为中心裁剪
        ys, xs = np.where(mask > 0)
        cy, cx = int(ys.mean()), int(xs.mean())

        # 添加随机偏移（±crop_size//4）
        offset_y = np.random.randint(-ch // 4, ch // 4 + 1)
        offset_x = np.random.randint(-cw // 4, cw // 4 + 1)

        top = cy + offset_y - ch // 2
        left = cx + offset_x - cw // 2
    else:
        # 完全随机裁剪
        top = np.random.randint(0, h - ch + 1)
        left = np.random.randint(0, w - cw + 1)

    # 边界安全
    top = max(0, min(top, h - ch))
    left = max(0, min(left, w - cw))

    image_crop = image[top:top + ch, left:left + cw]
    mask_crop = mask[top:top + ch, left:left + cw]

    return image_crop, mask_crop


# ============================================================
# 患者级 Train/Val 划分
# ============================================================

def split_by_patient(samples: list, val_ratio: float = 0.15, seed: int = 42):
    """
    按患者 ID 划分 train/val，确保同一患者的切片不会同时出现在两个集合中。
    返回 (train_samples, val_samples)
    """
    patients = sorted(set(s['person_id'] for s in samples))
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * val_ratio))
    val_patients = set(patients[:n_val])
    train_patients = set(patients[n_val:])

    train_samples = [s for s in samples if s['person_id'] in train_patients]
    val_samples = [s for s in samples if s['person_id'] in val_patients]

    print(f"[Split] 患者划分: {len(train_patients)} 训练 / {len(val_patients)} 验证")
    print(f"        验证患者: {sorted(val_patients)}")
    print(f"        切片数: {len(train_samples)} 训练 / {len(val_samples)} 验证")
    return train_samples, val_samples


# ============================================================
# PyTorch Dataset
# ============================================================

class CTTumorDataset(Dataset):
    """
    CT 肿瘤分割数据集

    Args:
        data_dir: 数据根目录
        config: TrainConfig 配置对象
        mode: "train" 或 "val"
        train_mode: "tumor_only" 或 "mixed"
    """

    def __init__(self, data_dir: str, config, mode: str = "train", samples: list = None):
        """
        Args:
            data_dir: 数据根目录（当 samples 为 None 时使用）
            config: TrainConfig
            mode: "train" 或 "val"
            samples: 预先划分好的样本列表（优先使用，跳过扫描和 split）
        """
        super().__init__()
        self.config = config
        self.mode = mode
        self.crop_size = config.crop_size
        self.ct_center = config.ct_window_center
        self.ct_width = config.ct_window_width

        # 扫描数据（或使用外部传入的 samples）
        if samples is not None:
            all_samples = samples
        else:
            all_samples = scan_data_directory(data_dir)
        if not all_samples:
            raise RuntimeError(f"在 {data_dir} 中未找到任何有效样本！")

        if samples is None:
            print_data_report(all_samples)

        # 按模式筛选
        tumor_samples = [s for s in all_samples if s['has_tumor']]
        bg_samples = [s for s in all_samples if not s['has_tumor']]

        if config.train_mode == "tumor_only":
            self.samples = tumor_samples
            print(f"[{mode}] tumor_only 模式: {len(self.samples)} 张含肿瘤切片")
        elif config.train_mode == "mixed":
            # 含肿瘤切片 + 少量背景切片（比例 5:1）
            n_bg = max(1, int(len(tumor_samples) * config.mixed_ratio))
            if len(bg_samples) > n_bg:
                bg_selected = list(np.random.choice(bg_samples, n_bg, replace=False))
            else:
                bg_selected = bg_samples
            self.samples = tumor_samples + bg_selected
            print(f"[{mode}] mixed 模式: {len(tumor_samples)} 肿瘤 + {len(bg_selected)} 背景")
        else:
            self.samples = all_samples
            print(f"[{mode}] all 模式: {len(self.samples)} 张切片")

        # 训练模式：重复含肿瘤切片 N 次（配合随机增强，每次不同）
        if mode == "train" and config.repeats > 1:
            original_len = len(self.samples)
            self.samples = self.samples * config.repeats
            print(f"[{mode}] 数据重复 {config.repeats}x: {original_len} → {len(self.samples)} 样本/epoch")

        # 数据增强
        if mode == "train":
            self.transforms = get_train_transforms(self.crop_size)
        else:
            self.transforms = get_val_transforms()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 读取 DICOM 图像
        try:
            sitk_image = sitk.ReadImage(sample['dcm_path'])
            image_array = sitk.GetArrayFromImage(sitk_image)  # [1, H, W] or [D, H, W]
        except Exception as e:
            print(f"[ERROR] 读取 DICOM 失败: {sample['dcm_path']}: {e}")
            # 返回零图
            dummy = torch.zeros(1, self.crop_size, self.crop_size)
            return dummy, dummy, sample['person_id'], sample['slice_id']

        # 取第一个 slice
        if image_array.ndim == 3:
            image_2d = image_array[0]
        else:
            image_2d = image_array

        # CT 窗宽窗位归一化
        image_2d = apply_ct_window(image_2d, self.ct_center, self.ct_width)

        # 读取 mask
        mask = cv2.imdecode(
            np.fromfile(sample['mask_path'], dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            mask = np.zeros_like(image_2d, dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)

        # 确保尺寸一致
        if image_2d.shape != mask.shape:
            mask = cv2.resize(mask, (image_2d.shape[1], image_2d.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # 训练模式：肿瘤中心裁剪 + 数据增强
        if self.mode == "train":
            # 肿瘤中心裁剪
            image_2d, mask = tumor_center_crop(image_2d, mask, self.crop_size)

            # albumentations 增强
            if self.transforms is not None:
                augmented = self.transforms(image=image_2d, mask=mask)
                image_2d = augmented['image']
                mask = augmented['mask']

        # 确保数据类型
        image_2d = image_2d.astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)  # 增强后重新二值化

        # 转为 tensor: [1, H, W]
        image_tensor = torch.from_numpy(image_2d).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor, sample['person_id'], sample['slice_id']


# ============================================================
# 全图数据集（用于评估和推理，不裁剪不增强）
# ============================================================

class CTFullImageDataset(Dataset):
    """返回原始完整 512×512 图像，用于滑窗推理评估"""

    def __init__(self, data_dir: str, config, samples: list = None):
        self.config = config
        self.samples = samples if samples is not None else scan_data_directory(data_dir)
        if not self.samples:
            raise RuntimeError(f"在 {data_dir} 中未找到任何有效样本！")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        sitk_image = sitk.ReadImage(sample['dcm_path'])
        image_array = sitk.GetArrayFromImage(sitk_image)
        if image_array.ndim == 3:
            image_2d = image_array[0]
        else:
            image_2d = image_array

        image_2d = apply_ct_window(image_2d, self.config.ct_window_center,
                                   self.config.ct_window_width)

        mask = cv2.imdecode(
            np.fromfile(sample['mask_path'], dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            mask = np.zeros_like(image_2d, dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)

        if image_2d.shape != mask.shape:
            mask = cv2.resize(mask, (image_2d.shape[1], image_2d.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        image_tensor = torch.from_numpy(image_2d.astype(np.float32)).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor, sample['person_id'], sample['slice_id']


if __name__ == "__main__":
    # 测试数据加载
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TrainConfig

    cfg = TrainConfig()
    dataset = CTTumorDataset(cfg.data_dir, cfg, mode="train")
    print(f"\n数据集大小: {len(dataset)}")

    # 取一个样本看看
    img, msk, pid, sid = dataset[0]
    print(f"图像 shape: {img.shape}, mask shape: {msk.shape}")
    print(f"图像范围: [{img.min():.3f}, {img.max():.3f}]")
    print(f"mask 前景: {(msk > 0).sum().item()} 像素")
    print(f"患者: {pid}, 切片: {sid}")
