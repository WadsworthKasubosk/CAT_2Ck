# -*- coding: utf-8 -*-
"""
Method A — 纯 UNet 全图滑窗（论文 baseline）

约束（来自 ablation 任务书）：
  - **必须直接 import 现有 inference.py 的 run_inference()**
  - 包装层只负责：从 splits.json 取样本 → 构建 dataset → 调 run_inference
    → 把 run_inference 的结果重打包为统一 per-slice 记录
  - 不允许任何重新实现，确保 baseline 与论文描述的"原始 UNet 方案"严格一致

run_inference 内部会做：滑窗 256×256 stride=128 + TTA (8 视图) + 后处理
（连通域筛选 + 形态学），跟原项目 evaluate.py 路径完全一致。

Usage:
  python -m experiments.run_baseline_unet \
        --unet_weights checkpoints/unet_best_ema.pth \
        --splits_json splits.json --split test
  python -m experiments.run_baseline_unet --dry_run
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)              # ctai_model_code/
MODEL_PKG = os.path.join(PROJECT, "CTAI_model")
for p in (PROJECT, MODEL_PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_split import load_splits                                       # noqa: E402
from config import TrainConfig                                           # noqa: E402
from data.dataset import CTFullImageDataset, scan_data_directory         # noqa: E402
from train import build_model                                            # noqa: E402
from inference import run_inference                                      # noqa: E402  ← 严格复用
EXPECTED_FINGERPRINT = "84706e8b75c8f403"


def _resolve_device(d: str) -> torch.device:
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _load_unet(weights, device):
    cfg = TrainConfig()
    model = build_model(cfg).to(device)
    model.eval()
    if weights and os.path.isfile(weights):
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        # 优先 EMA → model_state_dict → 裸 state_dict
        if isinstance(ckpt, dict):
            if ckpt.get("ema_state_dict"):
                state = {k: v.to(device) for k, v in ckpt["ema_state_dict"].items()}
            elif "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            else:
                state = ckpt
        else:
            state = ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[unet] {len(missing)} 缺失键（深监督头未保存属正常）")
    elif weights:
        raise FileNotFoundError(f"UNet 权重不存在: {weights}")
    else:
        print("[unet] 未指定权重，使用随机初始化（仅供 sanity / dry_run）")
    return model


def _filter_test_samples(data_dir, splits, split_name, limit=None):
    all_samples = scan_data_directory(data_dir)
    keep = set(splits["splits"][split_name])
    sub = [s for s in all_samples if s["person_id"] in keep]
    if not sub:
        raise SystemExit(f"split={split_name} 在 {data_dir} 中匹配 0 张切片")
    if limit:
        # 优先取若干含瘤切片（保证 dry_run 既有 tumor 也有 bg）
        tumor = [s for s in sub if s["has_tumor"]]
        bg = [s for s in sub if not s["has_tumor"]]
        sub = tumor[:max(1, limit // 2)] + bg[:limit]
        sub = sub[:limit]
    return sub


def run(unet_weights: str | None, splits_payload: dict, split_name: str = "test",
        device: str = "auto", dry_run: bool = False, limit: int = 0):
    """三组 ablation 共享接口：返回 (per_slice_records, summary)。"""
    dev = _resolve_device(device)
    model = _load_unet(unet_weights, dev)

    data_dir = splits_payload["data_dir"]
    eff_limit = 3 if dry_run else (limit or None)
    samples = _filter_test_samples(data_dir, splits_payload, split_name, limit=eff_limit)
    cfg = TrainConfig()
    cfg.data_dir = data_dir
    cfg.use_tta = not dry_run     # dry_run 关掉 TTA 加速
    dataset = CTFullImageDataset(data_dir, cfg, samples=samples)
    print(f"[A.baseline] dataset={len(dataset)} 切片  TTA={cfg.use_tta}  device={dev}")

    # 显存峰值统计
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 调原版 run_inference（写到临时目录，结果按 binary/*.png 二值 mask 取回）
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        results = run_inference(model, dataset, cfg, dev, output_dir=tmp)
        bin_dir = os.path.join(tmp, "binary")
        per_slice = []
        for r in results:
            pid = r["person_id"]; sid = r["slice_id"]
            bin_path = os.path.join(bin_dir, f"{pid}_{sid}.png")
            if not os.path.exists(bin_path):
                pred = np.zeros((1, 1), dtype=np.uint8)
            else:
                pred = (cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            sample = next(s for s in samples if s["person_id"] == pid and s["slice_id"] == sid)
            mask = cv2.imdecode(np.fromfile(sample["mask_path"], np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            gt = (mask > 0).astype(np.uint8) if mask is not None else np.zeros_like(pred)
            if gt.shape != pred.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            inter = int(np.logical_and(pred, gt).sum())
            pf = int(pred.sum()); gf = int(gt.sum())
            has_tumor = bool(gf > 0)
            dice = (2.0 * inter) / max(pf + gf, 1)
            iou = inter / max(pf + gf - inter, 1)
            tp = inter; fp = pf - inter; fn = gf - inter
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            per_slice.append({
                "person_id": pid, "slice_id": sid, "has_tumor": has_tumor,
                "dice": dice, "iou": iou, "precision": prec, "recall": rec, "f1": f1,
                "pred_fg_pixels": pf, "gt_fg_pixels": gf,
                "intersection_pixels": inter,
                "n_detections": 0,
                "inference_ms": float("nan"),  # 由外层用平均值填
            })
    elapsed = time.perf_counter() - t0
    avg_ms = elapsed * 1000.0 / max(len(per_slice), 1)
    for r in per_slice:
        r["inference_ms"] = avg_ms

    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if dev.type == "cuda" else float("nan")
    summary = {
        "method": "A_baseline_unet",
        "n_slices": len(per_slice),
        "elapsed_s": elapsed,
        "avg_ms_per_slice": avg_ms,
        "peak_memory_mb": peak_mb,
        "device": str(dev),
        "use_tta": cfg.use_tta,
    }
    return per_slice, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet_weights", default="")
    p.add_argument("--splits_json", default=os.path.join(PROJECT, "splits.json"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--device", default="auto")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    payload = load_splits(os.path.abspath(args.splits_json))
    if payload["fingerprint"] != EXPECTED_FINGERPRINT:
        raise SystemExit(f"[FATAL] fingerprint 不一致: {payload['fingerprint']!r}")

    rows, summary = run(args.unet_weights or None, payload, args.split,
                        args.device, args.dry_run, args.limit)
    print(f"\n[A] 完成 {summary['n_slices']} 切片  "
          f"avg={summary['avg_ms_per_slice']:.1f} ms/slice  "
          f"peak_mem={summary['peak_memory_mb']:.1f} MB")
    if rows:
        d = sum(r["dice"] for r in rows) / len(rows)
        print(f"[A] mean Dice (incl. bg) = {d:.4f}")


if __name__ == "__main__":
    main()
