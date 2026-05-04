# -*- coding: utf-8 -*-
"""
CascadeInference 管道自检 — 不依赖真训权重，3 张 tiny 数据跑通契约。

定位：Step 4 的 TDD 验收。**先于 inference_cascade.py 完成**，作为可执行的契约。

8 项 assert（每项必须通过）：
  1. 输入 image_np: float32, shape=[H,W], 范围 [0, 1]
  2. mask: uint8, shape == image_np.shape, max <= 1
  3. prob_map: float32, shape == image_np.shape
  4. 每个 detection.bbox 在 [0,W] × [0,H] 范围内
  5. 所有扩边 ROI 之外的 prob_map 严格等于 0
  6. 多框重叠时 mask 用 np.maximum 合并（不相加）
  7. 0 框场景下 mask / prob_map 全零、detections 是空 list
  8. 6 列可视化图能生成、文件大小 > 50 KB

执行：
    cd ctai_model_code/
    python tools/sanity_check_cascade.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np
import SimpleITK as sitk
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)         # ctai_model_code/
MODEL_PKG = os.path.join(PROJECT_ROOT, "CTAI_model")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODEL_PKG)

# 来自被测代码（这两行如果 import 失败就证明 Step 4 没写完）
from CTAI_model.inference_cascade import CascadeInference  # noqa: E402
from CTAI_model.vis_utils import save_cascade_overlay      # noqa: E402

# ---- 测试数据：从 直肠癌数据_tiny/1002 取 3 张含瘤切片 ----
TINY_DIR = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "直肠癌数据_tiny", "1002", "arterial phase")
)


def _load_tiny_slices(n=3):
    """返回 list of (image_np float32 [0,1], gt_mask uint8 0/1, slice_id)"""
    files = sorted(f for f in os.listdir(TINY_DIR) if f.endswith(".dcm"))
    out = []
    for f in files:
        if len(out) >= n:
            break
        sid = f[:-4]
        dcm = os.path.join(TINY_DIR, f)
        mp = os.path.join(TINY_DIR, f"{sid}_mask.png")
        if not os.path.exists(mp):
            continue
        # CT 窗宽窗位转 [0,1]
        arr = sitk.GetArrayFromImage(sitk.ReadImage(dcm))
        if arr.ndim == 3:
            arr = arr[0]
        arr = arr.astype(np.float32)
        lo, hi = 40 - 200, 40 + 200
        arr = np.clip(arr, lo, hi)
        img = ((arr - lo) / (hi - lo)).astype(np.float32)
        # mask
        m = cv2.imdecode(np.fromfile(mp, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != img.shape:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
        gt = (m > 0).astype(np.uint8)
        if gt.sum() > 0:                       # 只挑含瘤切片
            out.append((img, gt, sid))
    if len(out) < n:
        raise SystemExit(f"[fatal] 在 {TINY_DIR} 取不到 {n} 张含瘤切片")
    return out


# ---- 测试辅助：用 monkey-patch 给 cascade 注入受控 YOLO 输出 ----

def _mock_yolo_factory(boxes_with_score):
    """boxes_with_score: list of (x1, y1, x2, y2, score)。返回可注入 _run_yolo 的函数。"""
    def _mock(image_uint8_rgb):
        # 与真实 _run_yolo 的返回口径一致：list of dict
        out = []
        for x1, y1, x2, y2, score in boxes_with_score:
            out.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "class_id": 0,
            })
        return out
    return _mock


def _outside_all_boxes_mask(image_shape, dets, pad):
    """构造所有扩边 ROI 之外的 bool mask（True = 框外）。"""
    H, W = image_shape
    inside = np.zeros((H, W), dtype=bool)
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
        inside[y1:y2, x1:x2] = True
    return ~inside


# ---- 8 项 assert ----

def main():
    print("=" * 70)
    print("  CascadeInference — sanity check (no real weights required)")
    print("=" * 70)

    samples = _load_tiny_slices(n=3)
    H, W = samples[0][0].shape
    print(f"[setup] tiny 切片 {len(samples)} 张, 形状 {H}x{W}")

    # 构造 cascade（mock_mode 跳过权重加载）
    cascade = CascadeInference(
        yolo_weights=None,           # 触发 mock 路径
        unet_weights=None,           # 用随机初始化 UNet（只做 shape/dtype 验证）
        device="cpu",
        roi_pad=20,
        unet_input_size=256,
        conf_thres=0.25,
    )
    assert cascade.unet is not None, "UNet 必须能在无权重时构造（用于 sanity check）"

    # ---------- assert 1: 输入契约 ----------
    img0 = samples[0][0]
    assert img0.dtype == np.float32, f"image_np dtype={img0.dtype} 必须 float32"
    assert img0.ndim == 2, f"image_np 必须是 2D, got ndim={img0.ndim}"
    assert 0.0 <= img0.min() and img0.max() <= 1.0, \
        f"image_np 范围 [{img0.min()}, {img0.max()}] 超出 [0,1]"
    print("[1/8] 输入契约 OK (float32, [H,W], [0,1])")

    # ---------- assert 7（先做）：0 框场景 ----------
    cascade._run_yolo = _mock_yolo_factory([])
    res = cascade.predict(img0)
    assert isinstance(res, dict) and {"detections", "mask", "prob_map"} <= set(res), \
        f"返回 dict 必须含 detections/mask/prob_map，got keys={list(res.keys())}"
    assert res["detections"] == [], f"0 框时 detections 必须 [], got {res['detections']}"
    assert res["mask"].sum() == 0, "0 框时 mask 必须全零（禁止 fallback 到全图 UNet）"
    assert res["prob_map"].sum() == 0, "0 框时 prob_map 必须全零"
    assert res["mask"].dtype == np.uint8 and res["mask"].shape == (H, W)
    assert res["prob_map"].dtype == np.float32 and res["prob_map"].shape == (H, W)
    print("[7/8] 0 框场景 OK (detections=[], mask/prob_map 全零，无 fallback)")

    # ---------- assert 2/3/4: 单框场景，输出契约 ----------
    cascade._run_yolo = _mock_yolo_factory([(150, 200, 230, 280, 0.85)])
    res = cascade.predict(img0)
    assert res["mask"].dtype == np.uint8, f"mask dtype={res['mask'].dtype}"
    assert res["mask"].shape == (H, W)
    assert res["mask"].max() <= 1, f"mask 必须二值，max={res['mask'].max()}"
    print("[2/8] mask 契约 OK (uint8, [H,W], max<=1)")

    assert res["prob_map"].dtype == np.float32
    assert res["prob_map"].shape == (H, W)
    print("[3/8] prob_map 契约 OK (float32, [H,W])")

    for d in res["detections"]:
        x1, y1, x2, y2 = d["bbox"]
        assert all(isinstance(v, int) for v in d["bbox"]), \
            f"bbox 必须 list of int, got types {[type(v) for v in d['bbox']]}"
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, \
            f"bbox {[x1,y1,x2,y2]} 越界（image {W}x{H}）"
        assert isinstance(d["score"], float)
        assert isinstance(d["class_id"], int)
    print("[4/8] detection bbox 全部在图像边界内")

    # ---------- assert 5: 框外 prob_map == 0 ----------
    outside = _outside_all_boxes_mask((H, W), res["detections"], pad=20)
    leak = float(res["prob_map"][outside].sum())
    assert leak == 0.0, f"扩边 ROI 外的 prob_map 累加 {leak}，必须严格 0"
    print(f"[5/8] 框外 prob_map 严格 0 (leak={leak})")

    # ---------- assert 6: 多框重叠用 maximum 合并 ----------
    cascade._run_yolo = _mock_yolo_factory([
        (150, 200, 230, 280, 0.85),    # 框 A
        (200, 250, 280, 330, 0.70),    # 框 B (与 A 重叠)
    ])
    res2 = cascade.predict(img0)
    assert len(res2["detections"]) == 2, "应有 2 个 detections"
    # 验证 maximum：分别跑单框，再 maximum，应等于双框结果
    cascade._run_yolo = _mock_yolo_factory([(150, 200, 230, 280, 0.85)])
    res_a = cascade.predict(img0)
    cascade._run_yolo = _mock_yolo_factory([(200, 250, 280, 330, 0.70)])
    res_b = cascade.predict(img0)
    expected_prob = np.maximum(res_a["prob_map"], res_b["prob_map"])
    expected_mask = np.maximum(res_a["mask"], res_b["mask"])
    assert np.array_equal(res2["mask"], expected_mask), \
        "双框 mask 必须等于两个单框 mask 的 np.maximum"
    assert np.allclose(res2["prob_map"], expected_prob, atol=1e-6), \
        "双框 prob_map 必须等于两个单框 prob_map 的 np.maximum"
    # 验证不是相加：相加在重叠区会超过 1
    sum_check = res_a["prob_map"] + res_b["prob_map"]
    if (sum_check > 1.0).any():    # 重叠区相加 > 1，但 maximum 必须 <= 1
        assert res2["prob_map"].max() <= 1.0 + 1e-6, "prob_map 不应被加法污染（>1）"
    print("[6/8] 多框合并使用 np.maximum（非相加）")

    # ---------- assert 8: 6 列可视化能跑通 ----------
    cascade._run_yolo = _mock_yolo_factory([(150, 200, 230, 280, 0.85)])
    res = cascade.predict(img0)
    with tempfile.TemporaryDirectory() as td:
        vis_path = os.path.join(td, "sanity_overlay.png")
        save_cascade_overlay(
            image_np=img0, gt_mask=samples[0][1],
            detections=res["detections"], prob_map=res["prob_map"],
            pred_mask=res["mask"], dice=0.42,
            save_path=vis_path,
        )
        assert os.path.exists(vis_path), "vis 文件未生成"
        size = os.path.getsize(vis_path)
        assert size > 50_000, f"vis 文件 {size} bytes，应 > 50 KB（疑似空图）"
    print(f"[8/8] 6 列可视化 OK (size={size/1e3:.1f} KB)")

    # ---------- 所有 3 张切片都跑一遍，确认稳定 ----------
    for i, (img, gt, sid) in enumerate(samples):
        cascade._run_yolo = _mock_yolo_factory(
            [(int(W*0.3), int(H*0.4), int(W*0.5), int(H*0.6), 0.8)]
        )
        r = cascade.predict(img)
        assert r["mask"].shape == img.shape
    print(f"[stability] 3 张切片连续 predict 无报错")

    print()
    print("=" * 70)
    print("  [OK] All sanity checks passed. Ready for real weights.")
    print("=" * 70)


if __name__ == "__main__":
    main()
