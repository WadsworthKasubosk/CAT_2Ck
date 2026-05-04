# -*- coding: utf-8 -*-
"""
全项目唯一的 train/val/test 患者划分入口。

划分规则（论文方法论章节直接引用此段）：
  - 总患者数：107（来自 直肠癌数据/，ID 范围 1001~1108，缺 1016）
  - 按 seed=42 numpy.random.RandomState shuffle 后切三段：
      train = round(107 * 0.70) = 75 人
      val   = round(107 * 0.15) = 16 人
      test  = 107 - 75 - 16    = 16 人
  - 划分粒度：**患者级**——同一患者的所有切片、所有时相（arterial / venous）
    必须落在同一个 split 中，禁止时相级或切片级划分（防止数据泄露）
  - 三个 split 患者集合互斥（make_splits 内部 assert）

持久化：
  - 输出文件：splits.json（与本脚本同目录）
  - 关键字段：splits / seed / fingerprint
  - **fingerprint = 84706e8b75c8f403**（首次生成时锁定，不再变更）

下游纪律：
  - 所有下游脚本（YOLO 数据生成、YOLO 训练、UNet 训练 & 评估、
    级联推理、消融实验）都必须 `from data_split import load_splits` 读这个 json
  - 任何脚本启动时必须校验 fingerprint，不匹配立即 raise SystemExit
  - **禁止任何脚本自行实现 train_test_split 或 split_by_patient**

用法:
  python data_split.py                          # 用默认数据目录生成 splits.json
  python data_split.py --data_dir 直肠癌数据/    # 显式指定
  python data_split.py --force                  # 覆盖已存在的 splits.json
"""
import os
import sys
import json
import argparse
import hashlib
from datetime import datetime

DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "直肠癌数据")
)
SPLITS_PATH = os.path.join(os.path.dirname(__file__), "splits.json")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def list_patients(data_dir: str):
    """枚举数据目录下的患者 ID（数字目录名），排序返回。"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    patients = []
    for name in os.listdir(data_dir):
        full = os.path.join(data_dir, name)
        if os.path.isdir(full) and name.isdigit():
            patients.append(name)
    patients.sort(key=lambda x: int(x))
    return patients


def make_splits(patients, seed=SEED):
    """按患者随机三段切分。返回 dict: {train, val, test}"""
    import numpy as np
    rng = np.random.RandomState(seed)
    shuffled = list(patients)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    # test 吃掉余数，保证三段总和 == n
    n_test = n - n_train - n_val
    train = sorted(shuffled[:n_train], key=int)
    val = sorted(shuffled[n_train:n_train + n_val], key=int)
    test = sorted(shuffled[n_train + n_val:], key=int)
    assert len(train) + len(val) + len(test) == n
    assert n_test == len(test)
    # 互斥校验
    s_t, s_v, s_e = set(train), set(val), set(test)
    assert s_t.isdisjoint(s_v) and s_t.isdisjoint(s_e) and s_v.isdisjoint(s_e)
    return {"train": train, "val": val, "test": test}


def write_splits(splits, data_dir: str, out_path: str = SPLITS_PATH):
    """持久化 splits + meta。文件中保存 patient list、seed、生成时间、数据指纹。"""
    fingerprint = hashlib.sha256(
        ("|".join(sorted(splits["train"] + splits["val"] + splits["test"]))
         + f"|seed={SEED}").encode()
    ).hexdigest()[:16]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": os.path.abspath(data_dir),
        "seed": SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "counts": {k: len(v) for k, v in splits.items()},
        "total_patients": sum(len(v) for v in splits.values()),
        "fingerprint": fingerprint,
        "splits": splits,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_splits(splits_path: str = SPLITS_PATH) -> dict:
    """下游脚本调用入口。返回完整 payload（含 splits/seed/fingerprint）。"""
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"未找到 {splits_path}。请先运行 `python data_split.py` 生成。"
        )
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_patient_set(split_name: str, splits_path: str = SPLITS_PATH) -> set:
    """便捷：返回指定 split 的患者 ID 集合（str）。"""
    payload = load_splits(splits_path)
    if split_name not in payload["splits"]:
        raise ValueError(f"未知 split: {split_name}（可选: train/val/test）")
    return set(payload["splits"][split_name])


def filter_samples_by_split(samples: list, split_name: str,
                            splits_path: str = SPLITS_PATH) -> list:
    """便捷：从 scan_data_directory 的样本列表中筛出某个 split。"""
    keep = get_patient_set(split_name, splits_path)
    return [s for s in samples if s.get("person_id") in keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out", default=SPLITS_PATH)
    p.add_argument("--force", action="store_true",
                   help="覆盖已存在的 splits.json")
    args = p.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"[ABORT] {args.out} 已存在。如需重新生成，请加 --force。")
        sys.exit(1)

    patients = list_patients(args.data_dir)
    print(f"[scan] 患者数: {len(patients)}（{patients[0]} ~ {patients[-1]}）")
    splits = make_splits(patients)
    payload = write_splits(splits, args.data_dir, args.out)

    print(f"[ok] 写入 {args.out}")
    print(f"     fingerprint = {payload['fingerprint']}")
    print(f"     train={payload['counts']['train']} | "
          f"val={payload['counts']['val']} | "
          f"test={payload['counts']['test']}")
    print(f"     val  患者: {splits['val']}")
    print(f"     test 患者: {splits['test']}")


if __name__ == "__main__":
    main()
