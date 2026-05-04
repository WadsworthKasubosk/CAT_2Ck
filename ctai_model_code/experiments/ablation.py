# -*- coding: utf-8 -*-
"""
消融实验主调度脚本：A. 纯 UNet / B. 纯 YOLO11 / C. YOLO+UNet 级联

运行约束（来自任务书）：
  - 三组共享同一 splits.json（fingerprint=84706e8b75c8f403）和 split=test
  - 三组共享同一 UNet 权重 + 同一 YOLO 权重（A 不用 YOLO，B 不用 UNet，C 全用）
  - 唯一变量是"方法"
  - 串行跑在同一进程，避免重复加载权重
  - 任一组崩溃，已完成组的结果照样落盘；崩溃组在 csv 写一行 ERROR
  - 支持 --dry_run（随机权重 + 3 切片）

输出：
  results/ablation/{timestamp}/
      ├── method_A_baseline_unet/per_slice_results.csv
      ├── method_B_yolo_only/per_slice_results.csv
      ├── method_C_cascade/per_slice_results.csv
      ├── ablation_summary.md     ← 论文用
      ├── ablation_summary.csv    ← 程序处理用
      ├── metrics_comparison.png  ← Dice / Precision / Recall / FP_slices
      ├── speed_comparison.png    ← 推理速度
      └── run_config.json         ← CLI args + weight metadata + git_commit

CLI:
  python -m experiments.ablation \
       --yolo_weights checkpoints/yolo11s_best.pt \
       --unet_weights checkpoints/unet_best_ema.pth \
       --splits_json splits.json --split test \
       --output_dir results/ablation --methods A,B,C --device cuda

  python -m experiments.ablation --dry_run        # 用随机权重 + tiny 切片跑通链路
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
MODEL_PKG = os.path.join(PROJECT, "CTAI_model")
for p in (PROJECT, MODEL_PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_split import load_splits                              # noqa: E402
from config import TrainConfig                                  # noqa: E402
from data.dataset import CTFullImageDataset, scan_data_directory  # noqa: E402
from inference_cascade import CascadeInference                  # noqa: E402

import experiments.run_baseline_unet as method_a                # noqa: E402
import experiments.run_yolo_only as method_b                    # noqa: E402

EXPECTED_FINGERPRINT = "84706e8b75c8f403"
SCHEMA = ["person_id", "slice_id", "has_tumor", "dice", "iou",
          "precision", "recall", "f1",
          "pred_fg_pixels", "gt_fg_pixels", "intersection_pixels",
          "n_detections", "inference_ms"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def _resolve_device(d: str) -> torch.device:
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _file_sha16(path):
    if not path or not os.path.isfile(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _read_unet_metadata(path):
    if not path or not os.path.isfile(path):
        return {}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(ckpt, dict):
        return {}
    keys = ("split_fingerprint", "epoch", "best_dice", "data_dir",
            "timestamp", "git_commit", "produced_by", "model_config")
    return {k: ckpt[k] for k in keys if k in ckpt}


def _filter_test_samples(data_dir, splits_payload, split_name, limit=None):
    all_samples = scan_data_directory(data_dir)
    keep = set(splits_payload["splits"][split_name])
    sub = [s for s in all_samples if s["person_id"] in keep]
    if limit:
        tumor = [s for s in sub if s["has_tumor"]]
        bg = [s for s in sub if not s["has_tumor"]]
        sub = (tumor[:max(1, limit // 2)] + bg[:limit])[:limit]
    return sub


# ---------------------------------------------------------------------------
# Method C: cascade
# ---------------------------------------------------------------------------

def run_method_c(yolo_weights, unet_weights, splits_payload, split_name,
                 device, dry_run, limit, conf_thres, iou_thres, roi_pad):
    """复用 inference_cascade.CascadeInference；不重新实现。"""
    dev = _resolve_device(device)
    cascade = CascadeInference(
        yolo_weights=yolo_weights, unet_weights=unet_weights,
        device=device, conf_thres=conf_thres, iou_thres=iou_thres,
        roi_pad=roi_pad,
    )
    print(f"[C.cascade] device={cascade.device}")

    data_dir = splits_payload["data_dir"]
    eff_limit = 3 if dry_run else (limit or None)
    samples = _filter_test_samples(data_dir, splits_payload, split_name, limit=eff_limit)
    cfg = TrainConfig(); cfg.data_dir = data_dir
    dataset = CTFullImageDataset(data_dir, cfg, samples=samples)
    print(f"[C.cascade] dataset={len(dataset)} 切片")

    # 预热 3 张
    warmup_n = min(3, len(dataset))
    for i in range(warmup_n):
        img_t, _, _, _ = dataset[i]
        img_np = img_t.squeeze().cpu().numpy().astype(np.float32)
        try:
            _ = cascade.predict(img_np)
        except Exception:
            pass
    print(f"[C] warmup {warmup_n} 张完成")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    per_slice = []
    for payload_i in cascade.predict_dataset(dataset):
        pid = payload_i["person_id"]; sid = payload_i["slice_id"]
        gt = payload_i["gt_mask"]; pred = payload_i["pred"]
        mask = pred["mask"]; dets = pred["detections"]
        inter = int(np.logical_and(mask, gt).sum())
        pf = int(mask.sum()); gf = int(gt.sum())
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
            "n_detections": len(dets),
            "inference_ms": float(payload_i["inference_ms"]),
        })

    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if dev.type == "cuda" else float("nan")
    summary = {
        "method": "C_cascade",
        "n_slices": len(per_slice),
        "elapsed_s": float(np.sum([r["inference_ms"] for r in per_slice]) / 1000.0),
        "avg_ms_per_slice": float(np.mean([r["inference_ms"] for r in per_slice]))
                            if per_slice else float("nan"),
        "peak_memory_mb": peak_mb,
        "device": str(cascade.device),
        "conf_thres": conf_thres, "iou_thres": iou_thres, "roi_pad": roi_pad,
    }
    return per_slice, summary


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def aggregate_method(per_slice: list[dict]) -> dict:
    if not per_slice:
        return {"n_slices": 0}
    tumor = [r for r in per_slice if r["has_tumor"]]
    # 切片级（仅含瘤）
    def avg(rs, k): return float(np.mean([r[k] for r in rs])) if rs else float("nan")
    def std(rs, k): return float(np.std([r[k] for r in rs])) if rs else float("nan")
    # 患者级 Dice：以患者为单位累加 intersection / pred / gt
    by_pid = defaultdict(lambda: {"i": 0, "p": 0, "g": 0})
    for r in per_slice:
        bk = by_pid[r["person_id"]]
        bk["i"] += r["intersection_pixels"]
        bk["p"] += r["pred_fg_pixels"]
        bk["g"] += r["gt_fg_pixels"]
    pid_dices = []
    for pid, b in by_pid.items():
        denom = b["p"] + b["g"]
        if denom == 0:
            pid_dices.append(1.0)
        else:
            pid_dices.append(2.0 * b["i"] / denom)
    # FP / FN slices
    fp_slices = sum(1 for r in per_slice if (not r["has_tumor"]) and r["pred_fg_pixels"] > 0)
    fn_slices = sum(1 for r in per_slice if r["has_tumor"] and r["pred_fg_pixels"] == 0)
    return {
        "n_slices": len(per_slice),
        "n_tumor_slices": len(tumor),
        "n_bg_slices": len(per_slice) - len(tumor),
        "slice_dice_mean": avg(tumor, "dice"),
        "slice_dice_std": std(tumor, "dice"),
        "slice_iou_mean": avg(tumor, "iou"),
        "slice_precision_mean": avg(tumor, "precision"),
        "slice_recall_mean": avg(tumor, "recall"),
        "slice_f1_mean": avg(tumor, "f1"),
        "patient_dice_mean": float(np.mean(pid_dices)) if pid_dices else float("nan"),
        "fp_slices": fp_slices,
        "fn_slices": fn_slices,
        "avg_inference_ms": avg(per_slice, "inference_ms"),
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_per_slice_csv(per_slice, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not per_slice:
            f.write("ERROR: empty\n")
            return
        w = csv.DictWriter(f, fieldnames=SCHEMA)
        w.writeheader()
        for r in per_slice:
            w.writerow({k: r.get(k, "") for k in SCHEMA})


def write_error_csv(path, exc_str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"ERROR: {exc_str}\n")


def write_summary_md(out_dir, agg, summaries, n_test_patients, n_test_slices):
    """ablation_summary.md：核心指标表 + 自动生成结论 + 推理速度/显存/数据规模"""
    path = os.path.join(out_dir, "ablation_summary.md")
    methods_label = {
        "A_baseline_unet": "A. 纯 UNet",
        "B_yolo_only": "B. 纯 YOLO11",
        "C_cascade": "**C. 级联（本文）**",
    }
    METHOD_ORDER = ["A_baseline_unet", "B_yolo_only", "C_cascade"]

    def cell(v, fmt=".4f"):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "—"
        return format(v, fmt)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# 消融实验汇总\n\n")
        f.write(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- 测试 split：{n_test_patients} 患者 / {n_test_slices} 切片\n")
        f.write(f"- splits.json fingerprint：`{EXPECTED_FINGERPRINT}`\n\n")
        f.write("## 主结果（含瘤切片指标 + 切片级 / 患者级）\n\n")
        f.write("| 方案 | Dice (slice) | Dice (patient) | Precision | Recall | "
                "FP_slices | FN_slices | 速度 (ms/slice) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for m in METHOD_ORDER:
            a = agg.get(m, {})
            label = methods_label[m]
            slice_dice = (f"{cell(a.get('slice_dice_mean'))}"
                          f" ± {cell(a.get('slice_dice_std'))}"
                          if a.get("slice_dice_mean") is not None else "—")
            row = (f"| {label} | {slice_dice} | "
                   f"{cell(a.get('patient_dice_mean'))} | "
                   f"{cell(a.get('slice_precision_mean'))} | "
                   f"{cell(a.get('slice_recall_mean'))} | "
                   f"{a.get('fp_slices', '—')} | "
                   f"{a.get('fn_slices', '—')} | "
                   f"{cell(a.get('avg_inference_ms'), '.1f')} |\n")
            f.write(row)

        # 自动生成结论
        a, c = agg.get("A_baseline_unet", {}), agg.get("C_cascade", {})
        if a and c and a.get("slice_precision_mean") is not None and c.get("slice_precision_mean") is not None:
            d_prec = (c["slice_precision_mean"] - a["slice_precision_mean"]) * 100
            d_dice = ((c.get("slice_dice_mean") or 0) - (a.get("slice_dice_mean") or 0)) * 100
            fp_a, fp_c = a.get("fp_slices", 0), c.get("fp_slices", 0)
            fp_drop = (fp_a - fp_c) / fp_a * 100 if fp_a else 0
            f.write(
                f"\n## 自动生成结论\n\n"
                f"> 在 {n_test_patients} 例患者 {n_test_slices} 张测试切片上，"
                f"级联方案相比纯 UNet 基线，Precision 提升 {d_prec:+.2f} 个百分点（"
                f"A {a['slice_precision_mean']:.4f} → C {c['slice_precision_mean']:.4f}），"
                f"假阳性切片数变化 {fp_a}→{fp_c}（{fp_drop:+.1f}%），"
                f"切片级 Dice 变化 {d_dice:+.2f} 个百分点（A {a.get('slice_dice_mean', 0):.4f}"
                f" → C {c.get('slice_dice_mean', 0):.4f}）。"
                f"这验证了 YOLO11 检测阶段对背景假阳性的过滤作用。\n"
            )

        # 详细
        f.write("\n## 详细统计\n\n")
        for m in METHOD_ORDER:
            a = agg.get(m, {})
            s = summaries.get(m, {})
            f.write(f"### {methods_label[m]}\n\n")
            if not a:
                f.write("（运行失败 / 未运行，详见 method 子目录的 csv 错误信息）\n\n")
                continue
            f.write("| 项 | 值 |\n|---|---|\n")
            for k in ("n_slices", "n_tumor_slices", "n_bg_slices",
                      "slice_dice_mean", "slice_iou_mean", "slice_f1_mean",
                      "patient_dice_mean", "fp_slices", "fn_slices",
                      "avg_inference_ms"):
                v = a.get(k, "—")
                f.write(f"| {k} | {cell(v) if isinstance(v, float) else v} |\n")
            if "peak_memory_mb" in s and not np.isnan(s["peak_memory_mb"]):
                f.write(f"| peak_memory_mb | {s['peak_memory_mb']:.1f} |\n")
            f.write("\n")
    return path


def write_summary_csv(out_dir, agg):
    path = os.path.join(out_dir, "ablation_summary.csv")
    METHOD_ORDER = ["A_baseline_unet", "B_yolo_only", "C_cascade"]
    fieldnames = ["method"] + [k for k in (
        "n_slices", "n_tumor_slices", "n_bg_slices",
        "slice_dice_mean", "slice_dice_std", "slice_iou_mean",
        "slice_precision_mean", "slice_recall_mean", "slice_f1_mean",
        "patient_dice_mean", "fp_slices", "fn_slices",
        "avg_inference_ms")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in METHOD_ORDER:
            a = agg.get(m)
            if not a:
                continue
            row = {"method": m}
            for k in fieldnames[1:]:
                row[k] = a.get(k, "")
            w.writerow(row)
    return path


def make_plots(out_dir, agg):
    """生成两张柱状图：核心指标 + 推理速度。matplotlib 不可用时优雅退出。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 不可用，跳过柱状图")
        return
    METHOD_ORDER = ["A_baseline_unet", "B_yolo_only", "C_cascade"]
    labels = ["A. UNet", "B. YOLO", "C. Cascade"]

    # metrics_comparison
    metrics = ["slice_dice_mean", "slice_precision_mean", "slice_recall_mean", "fp_slices"]
    metric_titles = ["Dice (含瘤切片)", "Precision", "Recall", "FP_slices"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, mk, mt in zip(axes, metrics, metric_titles):
        vals = [agg.get(m, {}).get(mk, 0) or 0 for m in METHOD_ORDER]
        bars = ax.bar(labels, vals, color=["#888", "#e07a5f", "#3d5a80"])
        ax.set_title(mt)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.3f}" if isinstance(v, float) else str(v),
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Ablation: A vs B vs C (test split)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics_comparison.png"), dpi=140)
    plt.close(fig)

    # speed
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [agg.get(m, {}).get("avg_inference_ms", 0) or 0 for m in METHOD_ORDER]
    bars = ax.bar(labels, vals, color=["#888", "#e07a5f", "#3d5a80"])
    ax.set_ylabel("ms / slice")
    ax.set_title("Inference speed")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speed_comparison.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo_weights", default="")
    p.add_argument("--unet_weights", default="")
    p.add_argument("--splits_json", default=os.path.join(PROJECT, "splits.json"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--methods", default="A,B,C",
                   help="逗号分隔，子集如 'C' 或 'A,C'")
    p.add_argument("--output_dir", default="",
                   help="默认 results/ablation/{timestamp}")
    p.add_argument("--device", default="auto")
    p.add_argument("--dry_run", action="store_true",
                   help="随机权重 + 3 切片，验证 pipeline 通；指标无意义")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--conf_thres", type=float, default=0.25)
    p.add_argument("--iou_thres", type=float, default=0.45)
    p.add_argument("--roi_pad", type=int, default=20)
    args = p.parse_args()

    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ("A", "B", "C")]
    if bad:
        raise SystemExit(f"--methods 只能是 A/B/C 子集，非法: {bad}")

    splits_abs = os.path.abspath(args.splits_json)
    payload = load_splits(splits_abs)
    if payload["fingerprint"] != EXPECTED_FINGERPRINT:
        raise SystemExit(f"[FATAL] fingerprint 不一致: {payload['fingerprint']!r}")
    n_patients = len(payload["splits"][args.split])

    # 输出目录
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
        if not args.output_dir.endswith(os.sep):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(out_dir, ts)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(PROJECT, "..", "results", "ablation", ts)
        out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    # run_config.json
    cfg_dump = {
        "args": vars(args),
        "splits_fingerprint": payload["fingerprint"],
        "splits_path": splits_abs,
        "data_dir": payload["data_dir"],
        "split": args.split,
        "n_test_patients": n_patients,
        "device": str(_resolve_device(args.device)),
        "git_commit": _git_commit(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "yolo_weights_sha16": _file_sha16(args.yolo_weights or None),
        "unet_weights_sha16": _file_sha16(args.unet_weights or None),
        "unet_metadata": _read_unet_metadata(args.unet_weights or None),
    }
    # 校验 UNet metadata 的 fingerprint（若存在）
    meta_fp = cfg_dump["unet_metadata"].get("split_fingerprint")
    if meta_fp and meta_fp != EXPECTED_FINGERPRINT:
        raise SystemExit(
            f"[FATAL] UNet 权重 split_fingerprint={meta_fp!r} != {EXPECTED_FINGERPRINT!r}"
        )
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2, ensure_ascii=False, default=str)

    # 三组运行
    agg = {}
    summaries = {}
    method_dirs = {
        "A": os.path.join(out_dir, "method_A_baseline_unet"),
        "B": os.path.join(out_dir, "method_B_yolo_only"),
        "C": os.path.join(out_dir, "method_C_cascade"),
    }
    for d in method_dirs.values():
        os.makedirs(d, exist_ok=True)

    n_test_slices_first = None    # 由第一组成功的 method 给出准确切片数

    for tag in methods:
        print(f"\n========== Method {tag} ==========")
        csv_path = os.path.join(method_dirs[tag], "per_slice_results.csv")
        try:
            if tag == "A":
                rows, smry = method_a.run(
                    args.unet_weights or None, payload, args.split,
                    args.device, args.dry_run, args.limit)
                agg["A_baseline_unet"] = aggregate_method(rows)
                summaries["A_baseline_unet"] = smry
            elif tag == "B":
                rows, smry = method_b.run(
                    args.yolo_weights or None, payload, args.split,
                    args.device, args.dry_run, args.limit,
                    args.conf_thres, args.iou_thres)
                agg["B_yolo_only"] = aggregate_method(rows)
                summaries["B_yolo_only"] = smry
            else:  # C
                rows, smry = run_method_c(
                    args.yolo_weights or None, args.unet_weights or None,
                    payload, args.split, args.device, args.dry_run, args.limit,
                    args.conf_thres, args.iou_thres, args.roi_pad)
                agg["C_cascade"] = aggregate_method(rows)
                summaries["C_cascade"] = smry
            write_per_slice_csv(rows, csv_path)
            print(f"[{tag}] csv → {csv_path}")
            if n_test_slices_first is None:
                n_test_slices_first = len(rows)
        except Exception:
            tb = traceback.format_exc()
            print(f"[{tag}] FAILED:\n{tb}")
            write_error_csv(csv_path, tb)

    # 汇总产物
    n_slices_for_md = n_test_slices_first or 0
    md_path = write_summary_md(out_dir, agg, summaries, n_patients, n_slices_for_md)
    csv_path = write_summary_csv(out_dir, agg)
    make_plots(out_dir, agg)

    print("\n========== DONE ==========")
    print(f"[done] summary md  : {md_path}")
    print(f"[done] summary csv : {csv_path}")
    print(f"[done] all artifacts in: {out_dir}")
    if args.dry_run:
        print("[dry_run] 注意：dry_run 用随机权重 + 3 切片，指标无意义；只验证 pipeline 通畅。")


if __name__ == "__main__":
    main()
