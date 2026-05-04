# -*- coding: utf-8 -*-
"""
本地版 UNet 训练入口（cascade 方案下唯一的 UNet 训练脚本）

行为约定：
  1. 严格从 ../ctai_model_code/splits.json 读取 train/val 患者列表
  2. 启动时校验 split fingerprint == 84706e8b75c8f403，不匹配立即退出
  3. 复用 CTAI_model/train.py 的所有训练机制（EMA、深监督、cosine_warm_restarts、早停）
     但通过 monkey-patch 把内部 split_by_patient 替换为"按 splits.json 的固定划分"
  4. 训练超参全部沿用 config.py 默认值（attention_unet + 200 epoch + ...），不擅自调参
  5. 训练完成后，将 best 权重重打包为 metadata-rich 的 unet_best_ema.pth
  6. 同时输出 training_curves.png + training_log.csv 三件套

输出位置（统一在 ctai_model_code/CTAI_model/checkpoints/）：
    unet_best_ema.pth      ← 推理用产物
                              · 单一 model_state_dict（EMA 优先），weights_kind 标识来源
                              · 不含 optimizer / scheduler / scaler（剥离训练状态）
                              · metadata: best_epoch + last_epoch 分开 + split_fingerprint
                                + patient_ids_train/val + model_config + git_commit + ...
    best_model.pth         ← train.py 原始产物（含 model + ema 双副本）
    latest_checkpoint.pth  ← train.py 原始产物，含 optimizer/scheduler，恢复训练用
    training_curves.png
    training_log.csv

用法（本地）：
    cd ctai_model_code/
    python train_unet.py --epochs 200            # 默认全跑
    python train_unet.py --epochs 5 --smoke_test # 烟雾测试，5 epoch 验证流程
    python train_unet.py --resume ./CTAI_model/checkpoints/latest_checkpoint.pth  # 断点续训
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

# 把 CTAI_model 加到 sys.path（这样能直接 import 内部模块）
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "CTAI_model"))

# === tqdm 全局节流（Kaggle 长训不要把 notebook 输出灌爆）===
# 不修改 train.py 训练循环本身，只在导入前用 monkey-patch 把 tqdm 默认参数
# 调成 mininterval=10s / miniters=50。两边任何 `from tqdm import tqdm` 都会受影响。
import tqdm as _tqdm_module
_orig_tqdm = _tqdm_module.tqdm


class _ThrottledTqdm(_orig_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 3.0)   # 3 秒刷新一次，方便跟进度
        kwargs.setdefault("miniters", 20)
        kwargs.setdefault("leave", False)
        super().__init__(*args, **kwargs)


_tqdm_module.tqdm = _ThrottledTqdm

import torch  # noqa: E402

from data_split import load_splits  # noqa: E402

# 来自 CTAI_model/（必须晚于 tqdm patch，否则 train.py 已经绑了原版 tqdm 名字）
from config import TrainConfig  # noqa: E402
from data import dataset as ds_mod  # noqa: E402
import train as train_mod  # noqa: E402
# 显式覆盖 train 模块内已绑定的 tqdm 名字（它在文件顶部 from tqdm import tqdm）
train_mod.tqdm = _ThrottledTqdm

EXPECTED_FINGERPRINT = "84706e8b75c8f403"


# ---------------------------------------------------------------------------
# 1. 校验 + 注入 split
# ---------------------------------------------------------------------------

def validate_splits(splits_json_path: str):
    """加载并校验 splits.json。打印绝对路径方便排查。"""
    abs_path = os.path.abspath(splits_json_path)
    print(f"[splits] resolving splits_json: {splits_json_path}")
    print(f"[splits] absolute path        : {abs_path}")
    if not os.path.exists(abs_path):
        raise SystemExit(f"[FATAL] splits.json 不存在: {abs_path}")
    payload = load_splits(abs_path)
    fp = payload.get("fingerprint")
    if fp != EXPECTED_FINGERPRINT:
        raise SystemExit(
            f"[FATAL] splits.json fingerprint mismatch: got {fp!r}, "
            f"expected {EXPECTED_FINGERPRINT!r}. "
            f"请检查 splits.json 是否被改动，或重新生成（python data_split.py --force）。"
        )
    print(f"[splits] fingerprint OK = {fp}")
    print(f"[splits] train={len(payload['splits']['train'])} | "
          f"val={len(payload['splits']['val'])} | "
          f"test={len(payload['splits']['test'])} (test 集训练时不使用)")
    return payload


def install_split_patch(payload):
    """Monkey-patch split_by_patient 强制使用 splits.json 的固定划分。

    只切 train + val（test 留给最终评估）。
    """
    train_ids = set(payload["splits"]["train"])
    val_ids = set(payload["splits"]["val"])

    def fixed_split(samples, val_ratio=None, seed=None):
        # 校验 samples 中所有出现的患者 id 在 train ∪ val 里都有覆盖
        sample_pids = {s["person_id"] for s in samples}
        unknown = sample_pids - train_ids - val_ids - set(payload["splits"]["test"])
        if unknown:
            raise SystemExit(
                f"[FATAL] 数据目录中出现未在 splits.json 中登记的患者 ID: {sorted(unknown)[:5]}..."
            )
        train_samples = [s for s in samples if s["person_id"] in train_ids]
        val_samples = [s for s in samples if s["person_id"] in val_ids]
        # test 患者的切片直接丢弃（训练阶段绝不能见）
        print(f"[fixed_split] train={len(train_samples)} 切片 / "
              f"val={len(val_samples)} 切片 (来自 splits.json，非随机)")
        return train_samples, val_samples

    ds_mod.split_by_patient = fixed_split
    train_mod.split_by_patient = fixed_split  # train.py 直接 import 了这个名字


# ---------------------------------------------------------------------------
# 2. 训练完成后重打包 + 输出 csv
# ---------------------------------------------------------------------------

def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=HERE, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def repackage_checkpoint(src_path, dst_path, payload, config, last_epoch=None):
    """把 train.py 产出的 best_model.pth 重打包为 metadata-rich 推理产物。

    设计契约（推理产物，禁含训练状态）：
      - 只保留单一 `model_state_dict` —— 优先 EMA shadow，否则 raw model_state_dict
      - 同时保留 `weights_kind ∈ {"ema", "raw"}` 标识来源
      - **不包含** optimizer_state_dict / scheduler_state_dict / scaler_state_dict
      - 同时记录 `best_epoch`（best_dice 对应 epoch）+ `last_epoch`（最后训完 epoch）
        二者分开，便于将来排查"训练 200 epoch 但 best 锁在 epoch 1"的红旗。
    """
    if not os.path.exists(src_path):
        print(f"[WARN] 找不到源 checkpoint: {src_path}，跳过 repackage")
        return None

    raw = torch.load(src_path, map_location="cpu", weights_only=False)

    # helper: 去掉 DataParallel 的 "module." 前缀（兼容旧/新格式）
    def _strip_dp(sd):
        return {(k[7:] if k.startswith('module.') else k): v
                for k, v in sd.items()} if isinstance(sd, dict) else sd

    # 选权重：优先 EMA（推理性能最好），否则 raw
    ema_sd = raw.get("ema_state_dict")
    if isinstance(ema_sd, dict) and len(ema_sd) > 0:
        chosen_state = {k: (v.cpu() if torch.is_tensor(v) else v)
                       for k, v in _strip_dp(ema_sd).items()}
        weights_kind = "ema"
    else:
        chosen_state = _strip_dp(raw.get("model_state_dict") or {})
        weights_kind = "raw"

    enriched = {
        # 单一权重源（消除 best_model.pth 同时存 model + ema 的 2x 体积）
        "model_state_dict": chosen_state,
        "weights_kind": weights_kind,

        # epoch 语义清晰化
        "best_epoch": raw.get("epoch"),         # best_dice 触发保存时的 epoch
        "last_epoch": last_epoch,               # 训练实际跑到的最后一个 epoch
        "best_dice": raw.get("best_dice"),

        # === metadata 强制字段 ===
        "split_fingerprint": EXPECTED_FINGERPRINT,
        "patient_ids_train": payload["splits"]["train"],
        "patient_ids_val": payload["splits"]["val"],
        "data_dir": config.data_dir,
        "model_config": {
            "model": config.model,
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
            "deep_supervision": config.deep_supervision,
            "dropout_rate": config.dropout_rate,
            "ct_window_center": config.ct_window_center,
            "ct_window_width": config.ct_window_width,
        },
        "train_config": {k: v for k, v in config.__dict__.items()
                         if isinstance(v, (int, float, str, bool, type(None)))},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "produced_by": "ctai_model_code/train_unet.py",
    }
    torch.save(enriched, dst_path)
    size_mb = os.path.getsize(dst_path) / 1e6
    bd = enriched["best_dice"]
    bd_str = f"{bd:.4f}" if isinstance(bd, (int, float)) else str(bd)
    # 红旗校验：best_epoch <= 1 且 last_epoch >= 5 → 训练全程未提升
    if (enriched["best_epoch"] in (0, 1, None)) and (last_epoch or 0) >= 5:
        print(f"[repackage][WARN] best_epoch={enriched['best_epoch']} "
              f"last_epoch={last_epoch} —— 训练 {last_epoch} epoch 但 best 未离开起点，"
              f"模型可能完全没学到东西，请检查 dice 评估 / loss / lr。")
    print(f"[repackage] {dst_path}  "
          f"(best_epoch={enriched['best_epoch']}, last_epoch={last_epoch}, "
          f"best_dice={bd_str}, weights_kind={weights_kind}, size_MB={size_mb:.1f})")
    return enriched


def dump_training_log(history, csv_path):
    """history dict → 每 epoch 一行的 CSV。

    约定列：epoch, train_loss, lr, eval_dice, eval_precision, eval_recall
    train epoch 没有评估的填空
    """
    eval_map = {e: i for i, e in enumerate(history.get("eval_epoch", []))}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "lr",
                    "eval_dice", "eval_precision", "eval_recall"])
        for i, ep in enumerate(history.get("epoch", [])):
            row = [ep, history["train_loss"][i], history["lr"][i]]
            if ep in eval_map:
                j = eval_map[ep]
                row += [history["eval_dice"][j],
                        history["eval_precision"][j],
                        history["eval_recall"][j]]
            else:
                row += ["", "", ""]
            w.writerow(row)
    print(f"[csv] {csv_path}")


# ---------------------------------------------------------------------------
# 3. 入口
# ---------------------------------------------------------------------------

def build_config(args, splits_payload):
    cfg = TrainConfig()
    # 数据目录优先用命令行 --data_dir，否则用 splits.json 中记录的
    if args.data_dir:
        cfg.data_dir = args.data_dir
    else:
        cfg.data_dir = splits_payload["data_dir"]
    # 输出目录：命令行 --save_dir 优先，否则统一落 ctai_model_code/CTAI_model/checkpoints
    cfg.save_dir = args.save_dir or os.path.join(HERE, "CTAI_model", "checkpoints")
    cfg.log_dir = cfg.save_dir
    cfg.vis_dir = os.path.join(cfg.save_dir, "vis_train")
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.vis_dir, exist_ok=True)

    # 命令行可覆盖的少量参数
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.resume:
        cfg.resume = args.resume

    # 烟雾测试模式：epoch=5 + 关掉 deep eval
    if args.smoke_test:
        cfg.epochs = max(args.epochs or 5, 5)
        cfg.eval_interval = 1
        cfg.full_eval_interval = 9999
        cfg.repeats = 5  # 缩减 repeat，单 epoch 几分钟
        print("[smoke_test] epochs=%d, repeats=5, eval 每 epoch 一次" % cfg.epochs)

    cfg.__post_init__()
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_json", type=str,
                   default=os.path.join(HERE, "splits.json"),
                   help="splits.json 路径（Kaggle 上传 dataset 后用绝对路径覆盖）")
    p.add_argument("--data_dir", type=str, default="",
                   help="DICOM 数据根目录（覆盖 splits.json 里的记录值）")
    p.add_argument("--save_dir", type=str, default="",
                   help="输出目录（覆盖默认 ctai_model_code/CTAI_model/checkpoints）")
    p.add_argument("--epochs", type=int, default=None,
                   help="覆盖默认 epochs（默认 200）")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--resume", type=str, default="",
                   help="checkpoint 路径，断点续训")
    p.add_argument("--smoke_test", action="store_true",
                   help="烟雾测试：5 epoch + 缩减 repeats，验证流程能跑")
    args = p.parse_args()

    # 1. 校验 split
    payload = validate_splits(args.splits_json)

    # 2. 注入 split
    install_split_patch(payload)

    # 3. 配置
    config = build_config(args, payload)
    print(f"[config] data_dir = {config.data_dir}")
    print(f"[config] save_dir = {config.save_dir}")
    print(f"[config] model    = {config.model} (DS={config.deep_supervision})")
    print(f"[config] epochs   = {config.epochs}, batch_size={config.batch_size}")

    # 4. 训练
    train_mod.train(config)

    # 5. 取出 last_epoch（来自 latest_checkpoint history 的尾元素，最权威）+ history
    latest = os.path.join(config.save_dir, "latest_checkpoint.pth")
    last_epoch = None
    history = {}
    if os.path.exists(latest):
        latest_ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        history = latest_ckpt.get("history", {}) or {}
        if history.get("epoch"):
            last_epoch = history["epoch"][-1]
    # fallback：history 拿不到时退到 config.epochs（用户期望值）
    if last_epoch is None:
        last_epoch = config.epochs

    # 6. 重打包 best 权重（写入 best_epoch + last_epoch 双字段，剥离 optimizer 等）
    src_best = os.path.join(config.save_dir, "best_model.pth")
    dst_best = os.path.join(config.save_dir, "unet_best_ema.pth")
    repackage_checkpoint(src_best, dst_best, payload, config, last_epoch=last_epoch)

    # 7. dump training_log.csv
    if history.get("epoch"):
        csv_path = os.path.join(config.save_dir, "training_log.csv")
        dump_training_log(history, csv_path)

    print("\n[done] UNet 训练全部完成。")
    print(f"[done] 最终权重：{dst_best}")
    print(f"[done] 后续步骤：python -m CTAI_model.inference_cascade ...（待 Step 4 实现）")


if __name__ == "__main__":
    main()
