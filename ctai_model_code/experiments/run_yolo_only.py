# -*- coding: utf-8 -*-
"""
Method B — 纯 YOLO11 检测，把每个 bbox 当 mask（不调用 UNet）

设计要点（来自 ablation 任务书）：
  - 只跑 YOLO，对每个 detection 把 bbox 像素直接填 1
  - 多框用 np.maximum 合并
  - **绝不调用 UNet**
  - 这一组 Dice 必然较低（矩形 vs 不规则），但 Recall 可能很高
  - 论文里"YOLO 召回好但分割粗糙"的实证

Usage:
  python -m experiments.run_yolo_only \
        --yolo_weights checkpoints/yolo11s_best.pt --split test
  python -m experiments.run_yolo_only --dry_run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
MODEL_PKG = os.path.join(PROJECT, "CTAI_model")
for p in (PROJECT, MODEL_PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_split import load_splits                                  # noqa: E402
from config import TrainConfig                                       # noqa: E402
from data.dataset import CTFullImageDataset, scan_data_directory     # noqa: E402

EXPECTED_FINGERPRINT = "84706e8b75c8f403"


def _resolve_device(d: str) -> torch.device:
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _load_yolo(weights: str | None):
    """加载 ultralytics YOLO；weights=None 时返回 None（dry_run sanity 用）。"""
    if not weights:
        print("[yolo] 未指定权重，dry_run 模式：所有切片输出全零 mask")
        return None
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"YOLO 权重不存在: {weights}")
    from ultralytics import YOLO
    return YOLO(weights)


def _yolo_predict_one(yolo, image_uint8_rgb, conf_thres, iou_thres, device):
    """单图 → bbox list。无 yolo 时返回空。"""
    if yolo is None:
        return []
    dev_str = "0" if device.type == "cuda" else "cpu"
    res = yolo.predict(image_uint8_rgb, conf=conf_thres, iou=iou_thres,
                       verbose=False, device=dev_str)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return []
    boxes = res[0].boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = boxes.conf.cpu().numpy().astype(np.float32)
    H, W = image_uint8_rgb.shape[:2]
    out = []
    for (x1, y1, x2, y2), c in zip(xyxy, conf):
        x1 = int(max(0, np.floor(x1))); y1 = int(max(0, np.floor(y1)))
        x2 = int(min(W, np.ceil(x2)));  y2 = int(min(H, np.ceil(y2)))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2, float(c)))
    return out


def _bboxes_to_mask(H, W, bboxes):
    m = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2, _ in bboxes:
        m[y1:y2, x1:x2] = np.maximum(m[y1:y2, x1:x2], 1)
    return m


def _filter_samples(data_dir, splits, split_name, limit=None):
    all_samples = scan_data_directory(data_dir)
    keep = set(splits["splits"][split_name])
    sub = [s for s in all_samples if s["person_id"] in keep]
    if limit:
        tumor = [s for s in sub if s["has_tumor"]]
        bg = [s for s in sub if not s["has_tumor"]]
        sub = (tumor[:max(1, limit // 2)] + bg[:limit])[:limit]
    return sub


def run(yolo_weights: str | None, splits_payload: dict, split_name: str = "test",
        device: str = "auto", dry_run: bool = False, limit: int = 0,
        conf_thres: float = 0.25, iou_thres: float = 0.45):
    dev = _resolve_device(device)
    yolo = _load_yolo(yolo_weights)

    data_dir = splits_payload["data_dir"]
    eff_limit = 3 if dry_run else (limit or None)
    samples = _filter_samples(data_dir, splits_payload, split_name, limit=eff_limit)
    cfg = TrainConfig()
    cfg.data_dir = data_dir
    dataset = CTFullImageDataset(data_dir, cfg, samples=samples)
    print(f"[B.yolo_only] dataset={len(dataset)} 切片  device={dev}")

    # 预热 3 张（即使 dry_run 不到 3 张也无所谓）
    if yolo is not None and len(dataset) > 0:
        warmup_n = min(3, len(dataset))
        for i in range(warmup_n):
            img_t, _, _, _ = dataset[i]
            img_np = img_t.squeeze().cpu().numpy().astype(np.float32)
            gray_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            rgb = np.stack([gray_u8] * 3, axis=-1)
            _ = _yolo_predict_one(yolo, rgb, conf_thres, iou_thres, dev)
        print(f"[B] warmup {warmup_n} 张完成")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    per_slice = []
    for idx in range(len(dataset)):
        img_t, gt_t, pid, sid = dataset[idx]
        img_np = img_t.squeeze().cpu().numpy().astype(np.float32)
        H, W = img_np.shape
        gt = (gt_t.squeeze().cpu().numpy() > 0).astype(np.uint8)
        gray_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([gray_u8] * 3, axis=-1)

        t0 = time.perf_counter()
        bboxes = _yolo_predict_one(yolo, rgb, conf_thres, iou_thres, dev)
        pred = _bboxes_to_mask(H, W, bboxes)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        inter = int(np.logical_and(pred, gt).sum())
        pf = int(pred.sum()); gf = int(gt.sum())
        has_tumor = bool(gf > 0)
        dice = (2.0 * inter) / max(pf + gf, 1)
        iou = inter / max(pf + gf - inter, 1)
        tp, fp, fn = inter, pf - inter, gf - inter
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        per_slice.append({
            "person_id": pid, "slice_id": sid, "has_tumor": has_tumor,
            "dice": dice, "iou": iou, "precision": prec, "recall": rec, "f1": f1,
            "pred_fg_pixels": pf, "gt_fg_pixels": gf,
            "intersection_pixels": inter,
            "n_detections": len(bboxes),
            "inference_ms": dt_ms,
        })
        if (idx + 1) % 100 == 0:
            avg = float(np.mean([r["inference_ms"] for r in per_slice]))
            print(f"  [{idx+1}/{len(dataset)}] avg_ms={avg:.1f}")

    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if dev.type == "cuda" else float("nan")
    summary = {
        "method": "B_yolo_only",
        "n_slices": len(per_slice),
        "elapsed_s": float(np.sum([r["inference_ms"] for r in per_slice]) / 1000.0),
        "avg_ms_per_slice": float(np.mean([r["inference_ms"] for r in per_slice]))
                            if per_slice else float("nan"),
        "peak_memory_mb": peak_mb,
        "device": str(dev),
        "conf_thres": conf_thres, "iou_thres": iou_thres,
    }
    return per_slice, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo_weights", default="")
    p.add_argument("--splits_json", default=os.path.join(PROJECT, "splits.json"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--device", default="auto")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--conf_thres", type=float, default=0.25)
    p.add_argument("--iou_thres", type=float, default=0.45)
    args = p.parse_args()

    payload = load_splits(os.path.abspath(args.splits_json))
    if payload["fingerprint"] != EXPECTED_FINGERPRINT:
        raise SystemExit(f"[FATAL] fingerprint 不一致: {payload['fingerprint']!r}")

    rows, summary = run(args.yolo_weights or None, payload, args.split,
                        args.device, args.dry_run, args.limit,
                        args.conf_thres, args.iou_thres)
    print(f"\n[B] 完成 {summary['n_slices']} 切片  "
          f"avg={summary['avg_ms_per_slice']:.1f} ms/slice")
    if rows:
        d = sum(r["dice"] for r in rows) / len(rows)
        print(f"[B] mean Dice (incl. bg) = {d:.4f}")


if __name__ == "__main__":
    main()
