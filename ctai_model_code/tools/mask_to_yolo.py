# -*- coding: utf-8 -*-
"""
DICOM + mask PNG  →  YOLO11 detect 训练数据集

执行规则（与上层项目约定一致）：
  1. 严格按 ../splits.json 划分 train/val/test 患者
  2. 同时纳入 arterial / venous 两个时相，作为独立样本
  3. CT 切片 → 窗宽窗位 (40/400) → uint8 PNG
  4. mask 连通域 → bbox（过滤 area < min_area）→ YOLO 归一化标注
     注意：使用 detect 格式（class x_c y_c w h），不是 seg 多边形
  5. 背景切片（无任何连通域）→ 输出**空 .txt** 作为 hard negative
  6. **train 集**含肿瘤切片**物理复制 3 份**（_aug0/_aug1/_aug2）→ 在数据集层面做重采样
     **val/test 集**保持自然分布，不重采样
  7. 文件命名：{pid}_{a|v}_{slice_idx}[_augN].png
     phase 用 a / v 缩写，避免空格

输出目录 ctai_model_code/yolo_dataset/：
    images/train/*.png   labels/train/*.txt
    images/val/*.png     labels/val/*.txt
    images/test/*.png    labels/test/*.txt
    rectal_tumor.yaml    （ultralytics dataset config）
    MANIFEST.md          （生成元数据）

用法：
    cd ctai_model_code/
    python tools/mask_to_yolo.py                # 默认全跑
    python tools/mask_to_yolo.py --force        # 覆盖已存在的输出
    python tools/mask_to_yolo.py --dry_run      # 不写文件，只统计

依赖：opencv-python, SimpleITK, numpy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import SimpleITK as sitk

# 把上一层 ctai_model_code/ 加到 sys.path 以引用 data_split
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)  # ctai_model_code/
sys.path.insert(0, PROJECT_ROOT)

from data_split import load_splits  # noqa: E402

EXPECTED_FINGERPRINT = "84706e8b75c8f403"
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "yolo_dataset")
SPLITS_PATH = os.path.join(PROJECT_ROOT, "splits.json")
TRAIN_TUMOR_REPLICAS = 3   # 含肿瘤切片在 train 集物理复制份数
MIN_AREA = 30              # 连通域最小像素阈值
CT_WC = 40.0               # 窗宽窗位（与 dataset.py 保持一致）
CT_WW = 400.0
CLASS_ID = 0               # 单类：tumor


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def cv_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def cv_imwrite(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"imencode 失败: {path}")
    buf.tofile(path)


def apply_ct_window_uint8(arr_int16, wc=CT_WC, ww=CT_WW):
    """与 data/dataset.py 完全一致的窗宽窗位转换，最终输出 uint8 PNG。"""
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    arr = arr_int16.astype(np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)              # → [0, 1]
    return (arr * 255.0).round().astype(np.uint8)


def dcm_to_uint8_png(dcm_path):
    """读 DCM → 应用窗宽窗位 → uint8 ndarray (H, W)"""
    sitk_img = sitk.ReadImage(dcm_path)
    arr = sitk.GetArrayFromImage(sitk_img)
    if arr.ndim == 3:
        arr = arr[0]
    return apply_ct_window_uint8(arr), arr.shape


def mask_to_bboxes_yolo(mask_path, img_shape, min_area=MIN_AREA):
    """读 mask → connectedComponentsWithStats → 归一化 YOLO 标注列表。

    返回:
        bboxes: list[(class, x_c, y_c, w, h)]，全部归一化 0~1
        empty:  bool，True 表示无任何前景（→ hard negative）
    """
    mask = cv_imread(mask_path)
    h_img, w_img = img_shape
    if mask is None:
        return [], True
    # 尺寸不一致时与 dataset.py 一致地 nearest 缩放
    if mask.shape != img_shape:
        mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    bin_mask = (mask > 0).astype(np.uint8)
    if not bin_mask.any():
        return [], True

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    bboxes = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        x_c = (x + w / 2.0) / w_img
        y_c = (y + h / 2.0) / h_img
        nw = w / w_img
        nh = h / h_img
        # 边界 clamp（理论上不会越界，保险起见）
        x_c = float(np.clip(x_c, 0.0, 1.0))
        y_c = float(np.clip(y_c, 0.0, 1.0))
        nw = float(np.clip(nw, 0.0, 1.0))
        nh = float(np.clip(nh, 0.0, 1.0))
        bboxes.append((CLASS_ID, x_c, y_c, nw, nh))

    if not bboxes:
        # 有 mask 但所有连通域都 < min_area → 也算空（前景太碎不入训）
        return [], True
    return bboxes, False


def write_yolo_label(path, bboxes):
    """空 list → 0 字节文件（hard negative）；非空 → 每行一个 bbox"""
    with open(path, "w", encoding="utf-8") as f:
        for cls, xc, yc, w, h in bboxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def discover_slices(data_dir):
    """枚举所有 (pid, phase, slice_id, dcm_path, mask_path)"""
    items = []
    for pid in sorted(os.listdir(data_dir)):
        pdir = os.path.join(data_dir, pid)
        if not os.path.isdir(pdir) or not pid.isdigit():
            continue
        for phase_full, phase_short in [("arterial phase", "a"), ("venous phase", "v")]:
            phase_dir = os.path.join(pdir, phase_full)
            if not os.path.isdir(phase_dir):
                continue
            for f in sorted(os.listdir(phase_dir)):
                if not f.endswith(".dcm"):
                    continue
                sid = f[:-4]
                dcm = os.path.join(phase_dir, f)
                mask = os.path.join(phase_dir, f"{sid}_mask.png")
                if not os.path.exists(mask):
                    continue
                items.append((pid, phase_short, sid, dcm, mask))
    return items


def build_dataset(data_dir, splits, out_dir, dry_run=False):
    """主转换循环。返回 stats dict。"""
    train_ids = set(splits["splits"]["train"])
    val_ids   = set(splits["splits"]["val"])
    test_ids  = set(splits["splits"]["test"])

    if not dry_run:
        for sub in ["images/train", "images/val", "images/test",
                    "labels/train", "labels/val", "labels/test"]:
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    items = discover_slices(data_dir)
    print(f"[scan] 总切片 {len(items)} 张")

    # 计数器
    stats = {
        "train": {"images_written": 0, "tumor_slices_unique": 0, "bg_slices": 0},
        "val":   {"images_written": 0, "tumor_slices_unique": 0, "bg_slices": 0},
        "test":  {"images_written": 0, "tumor_slices_unique": 0, "bg_slices": 0},
        "skipped_unknown_pid": 0,
        "skipped_dcm_read_fail": 0,
    }

    t0 = time.time()
    for idx, (pid, phase, sid, dcm, mask) in enumerate(items, 1):
        if pid in train_ids:
            split = "train"
        elif pid in val_ids:
            split = "val"
        elif pid in test_ids:
            split = "test"
        else:
            stats["skipped_unknown_pid"] += 1
            continue

        try:
            img_uint8, img_shape = dcm_to_uint8_png(dcm)
        except Exception as e:
            stats["skipped_dcm_read_fail"] += 1
            print(f"  [WARN] DCM 读取失败 {dcm}: {e}")
            continue

        bboxes, empty = mask_to_bboxes_yolo(mask, img_shape)
        is_tumor = not empty

        # 写多少份？
        if split == "train" and is_tumor:
            n_copies = TRAIN_TUMOR_REPLICAS
        else:
            n_copies = 1

        # 统计（unique 只算一次）
        if is_tumor:
            stats[split]["tumor_slices_unique"] += 1
        else:
            stats[split]["bg_slices"] += 1

        if dry_run:
            stats[split]["images_written"] += n_copies
            if idx % 500 == 0:
                print(f"  [{idx}/{len(items)}] (dry-run)")
            continue

        for k in range(n_copies):
            suffix = f"_aug{k}" if n_copies > 1 else ""
            base = f"{pid}_{phase}_{sid}{suffix}"
            img_path = os.path.join(out_dir, "images", split, f"{base}.png")
            lbl_path = os.path.join(out_dir, "labels", split, f"{base}.txt")
            cv_imwrite(img_path, img_uint8)
            write_yolo_label(lbl_path, bboxes)
            stats[split]["images_written"] += 1

        if idx % 500 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            eta = (len(items) - idx) / rate
            print(f"  [{idx}/{len(items)}] {rate:.1f} slices/s  ETA {eta:.0f}s")

    return stats, len(items)


def write_yaml(out_dir):
    """ultralytics 数据集 yaml"""
    yaml_path = os.path.join(out_dir, "rectal_tumor.yaml")
    abs_root = os.path.abspath(out_dir).replace("\\", "/")
    content = (
        f"# CTAI 直肠肿瘤检测数据集 (YOLO11 detect 格式)\n"
        f"# 由 ctai_model_code/tools/mask_to_yolo.py 生成；改动请重新生成而不是手改\n"
        f"path: {abs_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: tumor\n"
        f"\n"
        f"nc: 1\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[yaml] {yaml_path}")
    return yaml_path


def write_manifest(out_dir, stats, n_items, splits_payload, args):
    """生成 MANIFEST.md，记录生成时的所有关键参数。"""
    splits_path_abs = os.path.abspath(SPLITS_PATH)
    splits_hash = hashlib.sha256(open(splits_path_abs, "rb").read()).hexdigest()[:16]

    manifest = os.path.join(out_dir, "MANIFEST.md")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("# YOLO 数据集生成清单\n\n")
        f.write(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- 生成脚本：`ctai_model_code/tools/mask_to_yolo.py`\n")
        f.write(f"- 源数据目录：`{splits_payload['data_dir']}`\n")
        f.write(f"- splits.json fingerprint：`{splits_payload['fingerprint']}`\n")
        f.write(f"- splits.json sha256[:16]：`{splits_hash}`\n")
        f.write(f"- 患者总数：{splits_payload['total_patients']}"
                f" (train={splits_payload['counts']['train']} / "
                f"val={splits_payload['counts']['val']} / "
                f"test={splits_payload['counts']['test']})\n")
        f.write(f"- 总扫描切片数：{n_items}（每患者 arterial+venous）\n\n")

        f.write("## 处理参数\n\n")
        f.write(f"- CT 窗宽窗位：center={CT_WC}, width={CT_WW}\n")
        f.write(f"- 连通域最小面积：{MIN_AREA} 像素（小于此值视为噪点丢弃）\n")
        f.write(f"- 图像格式：PNG, uint8, 与原 DICOM 同分辨率\n")
        f.write(f"- 标注格式：YOLO detect (class x_center y_center w h，归一化)\n")
        f.write(f"- 类别：1 类（0=tumor）\n")
        f.write(f"- 文件命名：`{{pid}}_{{a|v}}_{{slice_idx}}[_augN].png`\n\n")

        f.write("## 数据增强 / 重采样\n\n")
        f.write(f"- **train 集**含肿瘤切片物理复制 **{TRAIN_TUMOR_REPLICAS}** 份"
                f"（文件名后缀 _aug0/_aug1/_aug2），背景切片 1 份\n")
        f.write(f"- 背景切片对应 .txt 为 **0 字节空文件**，作为 hard negative 进训\n")
        f.write(f"- **val / test 集**保持自然分布，不重采样\n\n")

        f.write("## 各 split 实际产出\n\n")
        f.write("| split | unique 含瘤切片 | 背景切片 | 写入 PNG 总数 | 正/负比 |\n")
        f.write("|---|---|---|---|---|\n")
        for sp in ["train", "val", "test"]:
            s = stats[sp]
            tot = s["images_written"]
            tumor_written = (s["tumor_slices_unique"] *
                             (TRAIN_TUMOR_REPLICAS if sp == "train" else 1))
            bg = s["bg_slices"]
            ratio = f"{tumor_written}:{bg}" if bg else "N/A"
            f.write(f"| {sp} | {s['tumor_slices_unique']} | {bg} | {tot} | {ratio} |\n")

        f.write("\n## 异常计数\n\n")
        f.write(f"- skipped_unknown_pid: {stats['skipped_unknown_pid']}\n")
        f.write(f"- skipped_dcm_read_fail: {stats['skipped_dcm_read_fail']}\n")
        f.write("\n（以上两项应为 0；非 0 表示数据集与 splits.json 或 DICOM 文件存在异常）\n")
        f.write("\n## 复现命令\n\n```bash\ncd ctai_model_code/\npython tools/mask_to_yolo.py --force\n```\n")
    print(f"[manifest] {manifest}")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true",
                   help="覆盖已存在的输出目录")
    p.add_argument("--dry_run", action="store_true",
                   help="不写文件只统计")
    args = p.parse_args()

    payload = load_splits(SPLITS_PATH)
    if payload["fingerprint"] != EXPECTED_FINGERPRINT:
        raise SystemExit(
            f"[FATAL] splits.json fingerprint mismatch: "
            f"got {payload['fingerprint']!r}, expected {EXPECTED_FINGERPRINT!r}"
        )
    print(f"[splits] fingerprint OK = {payload['fingerprint']}")
    print(f"[splits] data_dir = {payload['data_dir']}")

    if not os.path.isdir(payload["data_dir"]):
        raise SystemExit(f"[FATAL] 数据目录不存在: {payload['data_dir']}")

    if os.path.isdir(args.out):
        if args.force:
            print(f"[force] 删除已存在的 {args.out}")
            shutil.rmtree(args.out)
        elif not args.dry_run:
            raise SystemExit(
                f"[ABORT] 输出目录已存在: {args.out}（加 --force 覆盖，或 --dry_run 仅统计）"
            )

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    stats, n_items = build_dataset(
        payload["data_dir"], payload, args.out, dry_run=args.dry_run
    )

    print("\n=== 统计 ===")
    for sp in ["train", "val", "test"]:
        s = stats[sp]
        print(f"  {sp:5s}  unique_tumor={s['tumor_slices_unique']:4d} | "
              f"bg={s['bg_slices']:4d} | written_total={s['images_written']:5d}")
    print(f"  skipped_unknown_pid    = {stats['skipped_unknown_pid']}")
    print(f"  skipped_dcm_read_fail  = {stats['skipped_dcm_read_fail']}")

    if not args.dry_run:
        write_yaml(args.out)
        write_manifest(args.out, stats, n_items, payload, args)
        print(f"\n[done] 数据集就绪: {args.out}")
    else:
        print("\n[dry_run] 未写任何文件")


if __name__ == "__main__":
    main()
