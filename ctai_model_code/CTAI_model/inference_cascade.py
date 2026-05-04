# -*- coding: utf-8 -*-
"""
YOLO11 + UNet 真级联推理（detect → ROI → segment）

设计契约（详见 README / sanity_check_cascade.py）：

INPUT  predict(image_np)
  - image_np : np.ndarray [H, W] float32, **已做窗宽窗位归一化到 [0, 1]**
  - 与 data/dataset.py 的 apply_ct_window(...) 输出严格对齐
  - **本类内部禁止做二次窗宽窗位 / 二次归一化**

OUTPUT  → dict
  - 'detections': list of {'bbox':[x1,y1,x2,y2] int, 'score':float, 'class_id':int}
  - 'mask'      : np.ndarray [H, W] uint8 in {0, 1}
  - 'prob_map'  : np.ndarray [H, W] float32, **框外严格为 0**

YOLO 输出 0 个框时：
  detections=[], mask 全零, prob_map 全零
  **绝不 fallback 到全图 UNet** —— 这是消融实验"YOLO 过滤无肿瘤切片→降 FP"
  论点的物理基础，破坏即破坏论点。

TODO（论文 limitation 章节）：
  当前 data/dataset.py 硬编码只读 arterial phase；venous 仅供 YOLO 训练，
  推理评估与本 cascade 都仅覆盖 arterial。后续若引入双时相评估需先扩展 dataset。
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from config import TrainConfig                                          # noqa: E402
from train import build_model                                           # noqa: E402


# ============================================================
#  CascadeInference
# ============================================================

class CascadeInference:

    # ---------------------------------------------------------
    #  ctor
    # ---------------------------------------------------------

    def __init__(
        self,
        yolo_weights: Optional[str],
        unet_weights: Optional[str],
        device: str = "auto",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        roi_pad: int = 20,
        unet_input_size: int = 256,
        unet_threshold: float = 0.5,
    ):
        self.device = self._resolve_device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.roi_pad = roi_pad
        self.unet_input_size = unet_input_size
        self.unet_threshold = unet_threshold

        self.yolo = self._load_yolo(yolo_weights)
        self.unet = self._load_unet(unet_weights)
        self.unet_meta: dict = {}      # 加载权重时填入（暴露给上层校验 split_fingerprint）

        if unet_weights:
            self._maybe_load_unet_metadata(unet_weights)

    # ---------------------------------------------------------
    #  权重加载
    # ---------------------------------------------------------

    @staticmethod
    def _resolve_device(d: str) -> torch.device:
        if d == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(d)

    def _load_yolo(self, weights: Optional[str]):
        """ultralytics YOLO 对象。weights=None 时返回 None，仅供 sanity 测试。"""
        if not weights:
            return None
        from ultralytics import YOLO        # 延迟 import，避免无 ultralytics 时 sanity 也崩
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"YOLO 权重不存在: {weights}")
        m = YOLO(weights)
        return m

    def _load_unet(self, weights: Optional[str]):
        """attention_unet 装配 + 加载权重。weights=None 时只装配（用于 sanity）。"""
        cfg = TrainConfig()                  # 用默认装配（attention_unet, DS=True, dropout=0.3）
        # 推理时禁用 deep supervision 头的训练副作用 —— 注意 forward 仍可能返回 list
        model = build_model(cfg).to(self.device)
        model.eval()

        if weights:
            if not os.path.isfile(weights):
                raise FileNotFoundError(f"UNet 权重不存在: {weights}")
            ckpt = torch.load(weights, map_location=self.device, weights_only=False)
            # 优先用 EMA shadow；其次 model_state_dict；最后裸 state_dict
            if isinstance(ckpt, dict):
                if "ema_state_dict" in ckpt and ckpt["ema_state_dict"]:
                    state = {k: v.to(self.device) for k, v in ckpt["ema_state_dict"].items()}
                elif "model_state_dict" in ckpt:
                    state = ckpt["model_state_dict"]
                else:
                    state = ckpt
            else:
                state = ckpt
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[unet] 缺失 {len(missing)} 个权重键（深监督头未保存属正常）")
            if unexpected:
                print(f"[unet] 多余 {len(unexpected)} 个权重键: {unexpected[:5]}")
        return model

    def _maybe_load_unet_metadata(self, weights: str):
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            return
        for k in ("split_fingerprint", "patient_ids_train", "patient_ids_val",
                  "epoch", "best_dice", "data_dir", "timestamp", "git_commit",
                  "produced_by", "model_config"):
            if k in ckpt:
                self.unet_meta[k] = ckpt[k]

    # ---------------------------------------------------------
    #  Stage 1：YOLO
    # ---------------------------------------------------------

    def _run_yolo(self, image_uint8_rgb: np.ndarray) -> list:
        """单张图 → list of {'bbox':[x1,y1,x2,y2] int, 'score':float, 'class_id':int}。

        sanity_check_cascade.py 会 monkey-patch 这个方法注入受控 bbox。
        """
        if self.yolo is None:
            return []
        results = self.yolo.predict(
            image_uint8_rgb,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            device=str(self.device).replace("cuda", "0") if self.device.type == "cuda" else "cpu",
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)
        cls = boxes.cls.cpu().numpy().astype(np.int64)
        out = []
        H, W = image_uint8_rgb.shape[:2]
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            x1 = int(max(0, np.floor(x1)));  y1 = int(max(0, np.floor(y1)))
            x2 = int(min(W, np.ceil(x2)));   y2 = int(min(H, np.ceil(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append({"bbox": [x1, y1, x2, y2], "score": float(c), "class_id": int(k)})
        return out

    # ---------------------------------------------------------
    #  Stage 2：UNet on ROI
    # ---------------------------------------------------------

    def _run_unet_on_roi(self, roi_f32: np.ndarray) -> tuple:
        """ROI [Hr, Wr] float32 in [0,1] → (mask uint8 [Hr,Wr], prob float32 [Hr,Wr])。

        归一化与训练对齐（dataset.py 也是直接把窗后图喂入，无二次归一），故此处也不做。
        """
        Hr, Wr = roi_f32.shape
        # resize 到 UNet 输入尺寸
        sz = self.unet_input_size
        roi_resized = cv2.resize(roi_f32, (sz, sz), interpolation=cv2.INTER_LINEAR)
        # to tensor [1,1,sz,sz]
        x = torch.from_numpy(roi_resized).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.unet(x)
            if isinstance(out, list):                 # deep supervision → 取 final
                out = out[0]
            prob = torch.sigmoid(out).squeeze().cpu().numpy().astype(np.float32)  # [sz, sz]
        # threshold at sz × sz
        mask = (prob >= self.unet_threshold).astype(np.uint8)
        # resize 回 ROI 大小：mask 用 NEAREST（不允许 LINEAR），prob 用 LINEAR
        if (Hr, Wr) != (sz, sz):
            mask = cv2.resize(mask, (Wr, Hr), interpolation=cv2.INTER_NEAREST)
            prob = cv2.resize(prob, (Wr, Hr), interpolation=cv2.INTER_LINEAR)
        return mask.astype(np.uint8), prob.astype(np.float32)

    # ---------------------------------------------------------
    #  完整推理
    # ---------------------------------------------------------

    @staticmethod
    def _validate_input(image_np: np.ndarray):
        if not isinstance(image_np, np.ndarray):
            raise TypeError(f"image_np must be ndarray, got {type(image_np)}")
        if image_np.dtype != np.float32:
            raise TypeError(f"image_np.dtype must be float32, got {image_np.dtype}")
        if image_np.ndim != 2:
            raise ValueError(f"image_np must be 2D [H,W], got ndim={image_np.ndim}")
        mn, mx = float(image_np.min()), float(image_np.max())
        if mn < -1e-6 or mx > 1.0 + 1e-6:
            raise ValueError(
                f"image_np 范围 [{mn}, {mx}] 超出 [0,1]；"
                f"很可能未做窗宽窗位/归一化。cascade 内部禁止补偿，请上游修复。"
            )

    def predict(self, image_np: np.ndarray) -> dict:
        """主入口。详见 module docstring。"""
        self._validate_input(image_np)
        H, W = image_np.shape

        # 灰度 [0,1] → uint8 RGB（YOLO 输入约定）
        gray_u8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

        # ---- Stage 1: YOLO 检测 ----
        detections = self._run_yolo(rgb)

        # ---- Stage 2: 对每个 box 在 ROI 内跑 UNet，maximum 合并 ----
        mask_full = np.zeros((H, W), dtype=np.uint8)
        prob_full = np.zeros((H, W), dtype=np.float32)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            # 扩边 + clip 到边界
            xa = max(0, x1 - self.roi_pad)
            ya = max(0, y1 - self.roi_pad)
            xb = min(W, x2 + self.roi_pad)
            yb = min(H, y2 + self.roi_pad)
            if xb <= xa or yb <= ya:
                continue
            roi = image_np[ya:yb, xa:xb]
            roi_mask, roi_prob = self._run_unet_on_roi(roi)
            mask_full[ya:yb, xa:xb] = np.maximum(mask_full[ya:yb, xa:xb], roi_mask)
            prob_full[ya:yb, xa:xb] = np.maximum(prob_full[ya:yb, xa:xb], roi_prob)

        return {"detections": detections, "mask": mask_full, "prob_map": prob_full}

    # ---------------------------------------------------------
    #  批量推理（供 run_cascade.py 用，避免重复样板）
    # ---------------------------------------------------------

    def predict_dataset(self, dataset, on_result=None):
        """对一个 CTFullImageDataset 风格的 dataset 全量推理，逐 sample yield。

        yields: dict {
            'person_id', 'slice_id',
            'image': np.float32 [H,W],
            'gt_mask': np.uint8 [H,W],
            'pred': result dict from predict(),
            'inference_ms': float,
        }
        """
        import time
        for idx in range(len(dataset)):
            image_t, gt_t, pid, sid = dataset[idx]
            image_np = image_t.squeeze().cpu().numpy().astype(np.float32)
            gt = gt_t.squeeze().cpu().numpy().astype(np.uint8)
            t0 = time.perf_counter()
            result = self.predict(image_np)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            payload = {
                "person_id": pid, "slice_id": sid,
                "image": image_np, "gt_mask": gt,
                "pred": result, "inference_ms": dt_ms,
            }
            if on_result is not None:
                on_result(idx, payload)
            yield payload
