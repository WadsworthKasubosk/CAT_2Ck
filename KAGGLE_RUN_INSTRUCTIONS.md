# Kaggle 正式训练操作 SOP

> 适用日期：2026-05-04。UNet + YOLO 并行训练，跑完即得到论文用全部权重。

---

## 1. 上传 ctai-code-and-splits.zip 新版本

项目根目录已有新生成的 `ctai-code-and-splits.zip`（~92 KB，32 文件）。

1. 打开 https://www.kaggle.com/datasets/ramyaramyarao/ctai-code-and-splits
2. 点击 **New Version** 按钮（不要新建 dataset）
3. 拖拽 `ctai-code-and-splits.zip` 上传
4. 版本备注填写：
   ```
   fix: split best_epoch/last_epoch, strip optimizer state, tqdm throttling, kaggle path patch, SMOKE_TEST=False
   ```
5. 点 **Create**，等待解压完成

> 不需要更新 `rectal-cancer-data` 和 `ctai-yolo-dataset`，这两个没变。

---

## 2. Session A：UNet 训练

### 2.1 打开 Notebook

1. Kaggle → New Notebook → File → Import Notebook → 上传 `train_unet_kaggle.ipynb`
2. 右上 **Add Data** → 搜索并挂载：
   - `ramyaramyarao/rectal-cancer-data`（最新版）
   - `ramyaramyarao/ctai-code-and-splits`（**刚上传的新版**）
   - 不需要 `ctai-yolo-dataset`

### 2.2 设置

- **Accelerator**: GPU T4 ×2
- **Internet**: On（UNet 这次不需要，但开着不碍事）
- **Persistence**: Files only（默认）

### 2.3 启动

1. 确认 Cell 3 显示 `SMOKE_TEST = False`（已改好，无需再改）
2. **Run All**（Run → Run All）
3. 关浏览器即可，Kaggle 后台续跑。**上限 12 小时，UNet 200 epoch 约 6-10 小时，安全**。

---

## 3. Session B：YOLO 训练（与 A 并行）

### 3.1 打开 Notebook

1. **另开一个浏览器标签页** → New Notebook → File → Import Notebook → 上传 `train_yolo11_kaggle.ipynb`
2. 右上 **Add Data** → 搜索并挂载：
   - `ramyaramyarao/ctai-code-and-splits`（新版）
   - `ramyaramyarao/ctai-yolo-dataset`（最新版）
   - 不需要 `rectal-cancer-data`

### 3.2 设置

- **Accelerator**: GPU T4 ×2
- **Internet**: **On**（必须装 ultralytics）
- **Persistence**: Files only

### 3.3 启动

1. 确认 Cell 4 显示 `SMOKE_TEST = False`（已改好）
2. Run All
3. 关浏览器。先跑 yolo11n×50（~30 min），再 yolo11s×100（~1-2 h），总约 2-3 小时。

---

## 4. 前 5 分钟必查（三条红线）

每个 session 启动后 **5 分钟内打开看一眼**，任一条失败立即 **Stop Session**：

### 红线 1：Cell 1 没有 DNS 报错

**UNet Cell 1** 应输出：
```
[INFO] Skipped pip install — using Kaggle pre-installed packages
torch   : 2.10.0+cu128 | CUDA: True | ...
```

**YOLO Cell 1** 应成功 `pip install ultralytics`，无 `socket.gaierror` / `Temporary failure in name resolution`。

### 红线 2：Cell 2 打印 splits OK

**UNet Cell 2**：
```
splits OK: fp=84706e8b75c8f403, train=75 | val=16 | test=16
```

**YOLO Cell 2**：
```
splits OK: fp=84706e8b75c8f403, sha16=0c83582ebd913eed
  train  images=6879  labels=6879  empty(neg)=3090
  val    images=930   labels=930   empty(neg)=665
  test   images=965   labels=965   empty(neg)=698
```

### 红线 3：Cell 4 进度条刷新间隔 ≥ 10 秒

UNet 训练启动后，`train_log_full.txt` 里 tqdm 进度条刷新间隔应 ≥10 秒（tqdm throttling 生效）。如果每秒刷几十行 → stdout 膨胀 → Stop Session，排查 `train_unet.py` 的 tqdm `mininterval` 设置。

---

## 5. 训练完成后取产物

### 5.1 下载

每个 session 的 Notebook Output 面板（右下角）→ 下载：

| Session | 产物 | 大小 |
|---|---|---|
| UNet | `unet_output.zip` | ≈126 MB |
| YOLO | `yolo_output.zip` | ≈40 MB |

### 5.2 解压到本地

```powershell
# UNet
Expand-Archive unet_output.zip -DestinationPath ctai_model_code\CTAI_model\checkpoints\ -Force
# 确认得到: unet_best_ema.pth (≈126 MB), best_model.pth, training_log.csv, training_curves.png

# YOLO
Expand-Archive yolo_output.zip -DestinationPath .\yolo_pkg_tmp\ -Force
# 将 best.pt 重命名拷贝:
cp yolo_pkg_tmp\tumor_det_n\best.pt ctai_model_code\CTAI_model\checkpoints\yolo11n_best.pt
cp yolo_pkg_tmp\tumor_det_s\best.pt ctai_model_code\CTAI_model\checkpoints\yolo11s_best.pt
Remove-Item -Recurse -Force yolo_pkg_tmp
```

### 5.3 最终 checkpoints 目录应包含

```
ctai_model_code/CTAI_model/checkpoints/
├── WEIGHTS_LOCATION.md
├── unet_best_ema.pth      (≈126 MB)
├── best_model.pth         (UNet 含 optimizer，可选)
├── training_curves.png
├── training_log.csv
├── yolo11n_best.pt        (≈5 MB)
└── yolo11s_best.pt        (≈40 MB)
```

---

## 6. 本地执行级联推理 + 消融

权重到位后在项目根目录执行：

### 6.1 级联推理（test split）

```powershell
cd 'C:\Users\da983\CAT_2Ck (3)\CAT_2Ck'

python ctai_model_code/run_cascade.py `
  --yolo_weights ctai_model_code/CTAI_model/checkpoints/yolo11s_best.pt `
  --unet_weights ctai_model_code/CTAI_model/checkpoints/unet_best_ema.pth `
  --splits_json ctai_model_code/splits.json `
  --split test `
  --data_dir '直肠癌数据' `
  --output_dir results/cascade_test `
  --device cuda
```

参数说明：
- `--conf_thres 0.25`（默认）YOLO 置信度阈值
- `--iou_thres 0.45`（默认）NMS IoU 阈值
- `--roi_pad 20`（默认）ROI 外扩像素
- `--save_vis true` 生成 overlay 大图
- `--limit N` 只跑前 N 个患者（调试用）

输出在 `results/cascade_test/`：
- `overlay/` — 6 列可视化大图
- `per_slice_results.csv` / `.json` — 逐切片指标
- `summary.md` — 全 split 汇总
- `run_config.json` — 运行参数快照

### 6.2 消融实验（test split, methods A/B/C）

```powershell
python -m ctai_model_code.experiments.ablation `
  --yolo_weights ctai_model_code/CTAI_model/checkpoints/yolo11s_best.pt `
  --unet_weights ctai_model_code/CTAI_model/checkpoints/unet_best_ema.pth `
  --splits_json ctai_model_code/splits.json `
  --split test `
  --output_dir results/ablation `
  --methods A,B,C `
  --device cuda
```

方法说明：
- **A** = YOLO + UNet cascade（完整方案）
- **B** = UNet only（无检测框引导，等同旧版）
- **C** = YOLO only（检测框直接当分割）

加上 `--dry_run` 可以用随机权重 + 3 切片快速验证管道。

### 6.3 也跑 val split（论文图表用）

```powershell
python ctai_model_code/run_cascade.py `
  --yolo_weights ... --unet_weights ... --split val `
  --output_dir results/cascade_val --device cuda
```

---

## 7. 排错速查表

| 症状 | 原因 | 解法 |
|---|---|---|
| `ConcurrencyViolation` | 多个标签页同时访问同一 session | 关掉多余标签页，刷新保留的那一个 |
| `socket.gaierror` / DNS 失败 | pip install 触发网络请求失败 | UNet：确认 Cell 1 已跳过 pip（已改好）。YOLO：确认 Internet On，重试 |
| `找不到数据目录` | 路径前缀不匹配 | 在 Cell 2 临时插入 `!find /kaggle/input -maxdepth 4 -type d` 查看实际挂载路径，更新 `KAGGLE_NS` |
| `找不到 splits.json` | ctai-code-and-splits 版本未更新 | 检查 Kaggle Data 面板确认 dataset 状态是 Attached 且版本号正确 |
| `fingerprint mismatch` | 上传了旧版或错误的 splits.json | 确认 `ctai-code-and-splits.zip` 是本轮新生成的（92 KB），重新上传 New Version |
| UNet 进度条刷屏 | tqdm mininterval 未生效 | 检查 `train_unet.py` 中 tqdm 的 `mininterval` 参数 ≥ 10 |
| `CUDA out of memory` | batch size 太大 | 当前 batch=4 应对 T4 16 GB 安全；如仍 OOM，改 `train_unet.py` 的 `--batch_size` 或降低 `--crop_size` |
| 训练超 12h | UNet 200 epoch 超过 Kaggle 上限 | 降 `--epochs`，或分两段断点续训（从 `best_model.pth` resume） |
| `unet_best_ema.pth` > 180 MB | optimizer/scheduler 未剥离 | 检查 `train_unet.py` 的 `repackage_weights()` 函数是否正确剥离了非 state_dict 字段 |
