# -*- coding: utf-8 -*-
"""
CLI 入口：用 cascade 对指定 split 跑全量推理 + 出指标 + 出可视化。

启动校验：
  - splits.json fingerprint == 84706e8b75c8f403
  - 若 UNet 权重含 metadata，则其 split_fingerprint 必须等于上面值（不一致硬退出）

输出目录结构（args.output_dir）：
  ├── overlay/{pid}_{phase}_{slice}.png    # 6 列大图
  ├── prob_npy/...                         # 仅当 --save_npy 时
  ├── per_slice_results.csv
  ├── per_slice_results.json
  ├── summary.md                           # 全 split 平均指标 + 分布
  └── run_config.json                      # CLI args + 权重 metadata 摘要 + git_commit
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PKG = os.path.join(HERE, "CTAI_model")
sys.path.insert(0, HERE)
sys.path.insert(0, MODEL_PKG)

from data_split import load_splits                                            # noqa: E402

# CTAI_model 内部
from config import TrainConfig                                                # noqa: E402
from data.dataset import CTFullImageDataset, scan_data_directory              # noqa: E402
from inference import _dice_score, _precision_recall                          # noqa: E402（只引用，不修改）
from inference_cascade import CascadeInference                                # noqa: E402
from vis_utils import save_cascade_overlay                                    # noqa: E402

EXPECTED_FINGERPRINT = "84706e8b75c8f403"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=HERE, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def _resolve_output_dir(arg: str | None) -> str:
    if arg:
        return os.path.abspath(arg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(os.path.join(HERE, "results", "cascade", ts))


def _filter_samples(data_dir: str, splits_payload: dict, split_name: str) -> list:
    all_samples = scan_data_directory(data_dir)
    keep = set(splits_payload["splits"][split_name])
    sub = [s for s in all_samples if s["person_id"] in keep]
    if not sub:
        raise SystemExit(
            f"[FATAL] {split_name} split 在 {data_dir} 内匹配到 0 张切片"
        )
    return sub


def _phase_short(dcm_path: str) -> str:
    """从 dcm_path 推断 a / v 后缀，用于 vis 文件命名。"""
    if "arterial phase" in dcm_path:
        return "a"
    if "venous phase" in dcm_path:
        return "v"
    return "x"


def _summary_md(out_path: str, results: list, split_name: str, n_slices: int,
                cfg_dict: dict, unet_meta: dict):
    """汇总 markdown：分含瘤/背景两组报指标，并给推理速度。"""
    tumor = [r for r in results if r["has_tumor"]]
    bg = [r for r in results if not r["has_tumor"]]

    def _avg(rs, k):
        if not rs: return float("nan")
        return float(np.mean([r[k] for r in rs]))

    def _std(rs, k):
        if not rs: return float("nan")
        return float(np.std([r[k] for r in rs]))

    # 全局 dice：含瘤切片用真实 dice；纯背景切片若预测为空 → dice=1 否则 0
    glob_dice = []
    for r in results:
        if r["has_tumor"]:
            glob_dice.append(r["dice"])
        else:
            glob_dice.append(1.0 if r["pred_fg_pixels"] == 0 else 0.0)

    n_zero_det = sum(1 for r in results if r["n_detections"] == 0)
    avg_ms = _avg(results, "inference_ms")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Cascade 推理汇总 — split = `{split_name}`\n\n")
        f.write(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- 总切片：{n_slices}（含瘤 {len(tumor)} / 背景 {len(bg)}）\n")
        f.write(f"- 0-detection 切片：{n_zero_det}（{n_zero_det/n_slices*100:.1f}%）\n")
        f.write(f"- 平均推理速度：{avg_ms:.1f} ms / 张\n\n")

        f.write("## 含瘤切片指标 (n=%d)\n\n" % len(tumor))
        f.write("| 指标 | 平均 | 标准差 |\n|---|---|---|\n")
        for k, label in [("dice", "Dice"), ("precision", "Precision"), ("recall", "Recall")]:
            f.write(f"| {label} | {_avg(tumor, k):.4f} | {_std(tumor, k):.4f} |\n")
        f.write("\n")

        f.write("## 背景切片表现 (n=%d)\n\n" % len(bg))
        if bg:
            n_clean = sum(1 for r in bg if r["pred_fg_pixels"] == 0)
            f.write(f"- 完全干净（0 假阳）切片：{n_clean} / {len(bg)} = {n_clean/len(bg)*100:.2f}%\n")
            f.write(f"- 平均假阳像素数：{_avg(bg, 'pred_fg_pixels'):.1f}\n\n")

        f.write(f"## 全局平均 Dice：**{float(np.mean(glob_dice)):.4f}**\n\n")

        f.write("## 推理配置\n\n```json\n")
        f.write(json.dumps(cfg_dict, indent=2, ensure_ascii=False))
        f.write("\n```\n\n")

        if unet_meta:
            f.write("## UNet 权重 metadata\n\n```json\n")
            f.write(json.dumps({k: unet_meta[k] for k in unet_meta if k != "patient_ids_train"
                                and k != "patient_ids_val"}, indent=2, ensure_ascii=False))
            f.write("\n```\n")
            f.write(f"\npatient_ids_train: {len(unet_meta.get('patient_ids_train', []))} 人 / "
                    f"patient_ids_val: {len(unet_meta.get('patient_ids_val', []))} 人\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo_weights", required=True,
                   help="YOLO11 权重路径（.pt）；在沙箱测试可传 'none' 跳过")
    p.add_argument("--unet_weights", required=True,
                   help="UNet 权重路径（.pth）；'none' 跳过加载（仅用随机初始化）")
    p.add_argument("--splits_json", default=os.path.join(HERE, "splits.json"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data_dir", default="",
                   help="覆盖 splits.json 中记录的数据目录")
    p.add_argument("--output_dir", default="",
                   help="结果目录；默认 results/cascade/{timestamp}")
    p.add_argument("--conf_thres", type=float, default=0.25)
    p.add_argument("--iou_thres", type=float, default=0.45)
    p.add_argument("--roi_pad", type=int, default=20)
    p.add_argument("--save_vis", type=lambda x: x.lower() in ("1", "true", "yes"),
                   default=True)
    p.add_argument("--save_npy", type=lambda x: x.lower() in ("1", "true", "yes"),
                   default=False)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0,
                   help="只跑前 N 张（>0 时生效，调试用）")
    args = p.parse_args()

    # 1) splits 校验
    splits_abs = os.path.abspath(args.splits_json)
    print(f"[splits] {splits_abs}")
    payload = load_splits(splits_abs)
    if payload["fingerprint"] != EXPECTED_FINGERPRINT:
        raise SystemExit(
            f"[FATAL] splits.json fingerprint mismatch: got {payload['fingerprint']!r}"
        )
    print(f"[splits] fingerprint OK = {payload['fingerprint']}, "
          f"split={args.split} 患者数 {len(payload['splits'][args.split])}")

    data_dir = args.data_dir or payload["data_dir"]
    print(f"[data ] {data_dir}")

    # 2) 装配 cascade
    yolo_w = None if args.yolo_weights.lower() == "none" else args.yolo_weights
    unet_w = None if args.unet_weights.lower() == "none" else args.unet_weights
    cascade = CascadeInference(
        yolo_weights=yolo_w, unet_weights=unet_w,
        device=args.device,
        conf_thres=args.conf_thres, iou_thres=args.iou_thres,
        roi_pad=args.roi_pad,
    )
    print(f"[device] {cascade.device}")
    if cascade.unet_meta:
        meta_fp = cascade.unet_meta.get("split_fingerprint")
        if meta_fp and meta_fp != EXPECTED_FINGERPRINT:
            raise SystemExit(
                f"[FATAL] UNet 权重 split_fingerprint={meta_fp!r} 与 splits.json 不一致"
            )
        print(f"[unet ] metadata: epoch={cascade.unet_meta.get('epoch')}  "
              f"best_dice={cascade.unet_meta.get('best_dice')}  "
              f"split_fp={meta_fp}")

    # 3) 数据集
    samples = _filter_samples(data_dir, payload, args.split)
    if args.limit > 0:
        samples = samples[:args.limit]
    cfg = TrainConfig()
    cfg.data_dir = data_dir
    dataset = CTFullImageDataset(data_dir, cfg, samples=samples)
    print(f"[ds   ] {len(dataset)} 张切片")

    # 4) 输出目录 + run_config.json
    out_dir = _resolve_output_dir(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    overlay_dir = os.path.join(out_dir, "overlay")
    npy_dir = os.path.join(out_dir, "prob_npy")
    if args.save_vis: os.makedirs(overlay_dir, exist_ok=True)
    if args.save_npy: os.makedirs(npy_dir, exist_ok=True)

    cfg_dict = {
        "args": vars(args),
        "splits_fingerprint": payload["fingerprint"],
        "data_dir": data_dir,
        "out_dir": out_dir,
        "git_commit": _git_commit(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(cascade.device),
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
    print(f"[out  ] {out_dir}")

    # 5) 推理循环
    results = []
    t_total = time.time()
    sample_meta = {(s["person_id"], s["slice_id"]): s for s in samples}

    for idx, payload_i in enumerate(cascade.predict_dataset(dataset)):
        pid = payload_i["person_id"]; sid = payload_i["slice_id"]
        gt = payload_i["gt_mask"]
        pred = payload_i["pred"]
        mask = pred["mask"]
        prob = pred["prob_map"]
        dets = pred["detections"]
        has_tumor = bool(gt.sum() > 0)
        dice = _dice_score(mask, gt) if (has_tumor or mask.any()) else 1.0
        prec, rec = _precision_recall(mask, gt)
        rec_dict = {
            "person_id": pid,
            "slice_id": sid,
            "has_tumor": has_tumor,
            "dice": float(dice),
            "precision": float(prec),
            "recall": float(rec),
            "n_detections": len(dets),
            "max_score": float(max((d["score"] for d in dets), default=0.0)),
            "pred_fg_pixels": int(mask.sum()),
            "gt_fg_pixels": int(gt.sum()),
            "inference_ms": float(payload_i["inference_ms"]),
        }
        results.append(rec_dict)

        # vis
        if args.save_vis:
            phase = _phase_short(sample_meta[(pid, sid)]["dcm_path"])
            vis_path = os.path.join(overlay_dir, f"{pid}_{phase}_{sid}.png")
            save_cascade_overlay(
                image_np=payload_i["image"], gt_mask=gt,
                detections=dets, prob_map=prob, pred_mask=mask,
                dice=dice, save_path=vis_path,
                title_prefix=f"{pid} / {phase} / {sid}",
            )
        # npy
        if args.save_npy:
            np.save(os.path.join(npy_dir, f"{pid}_{sid}_prob.npy"), prob)

        # 进度
        if (idx + 1) % 25 == 0 or (idx + 1) == len(dataset):
            avg_dice = float(np.mean([r["dice"] for r in results if r["has_tumor"]]) or 0)
            print(f"  [{idx+1:4d}/{len(dataset)}] tumor_avg_dice={avg_dice:.4f}  "
                  f"avg_inf_ms={float(np.mean([r['inference_ms'] for r in results])):.1f}")

    elapsed = time.time() - t_total

    # 6) per-slice csv + json
    csv_path = os.path.join(out_dir, "per_slice_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    with open(os.path.join(out_dir, "per_slice_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 7) summary.md
    _summary_md(
        os.path.join(out_dir, "summary.md"),
        results, args.split, len(dataset),
        cfg_dict, cascade.unet_meta,
    )

    # 8) finale
    print(f"\n[done] {len(results)} 张切片处理完毕，耗时 {elapsed:.1f}s")
    print(f"[done] csv     : {csv_path}")
    print(f"[done] summary : {os.path.join(out_dir, 'summary.md')}")
    if args.save_vis:
        print(f"[done] overlays: {overlay_dir}")


if __name__ == "__main__":
    main()
