# Kaggle 训练操作手册（UNet + YOLO 并行）

> 这一份是动手清单。准备好 Kaggle 账号后照着做即可，全程不需要本地算力。
> 两份训练 notebook 在两个独立 session 启动可以**并行**，墙钟时间 = max(UNet, YOLO)。

---

## 一、要上传的 3 个 Kaggle Dataset（一次性）

| Dataset slug | 内容 | 大小 | 说明 |
|---|---|---|---|
| `rectal-cancer-data` | 本地 `直肠癌数据/` 整目录 | ~6 GB（解压后） | 107 患者全量 DICOM，**只 UNet 用** |
| `ctai-code-and-splits` | 本地 `ctai_model_code/` 整目录（**排除** `yolo_dataset/` 和 `__pycache__/`） | ~50 KB | 含 `splits.json` + 所有 .py + 两份 .ipynb，UNet 与 YOLO 共享 |
| `ctai-yolo-dataset` | 本地 `ctai_model_code/yolo_dataset/` 整目录 | 610 MB | 8,774 张 PNG + 8,774 个 .txt，**只 YOLO 用** |

### 上传前的清理
打包 `ctai-code-and-splits` 时务必排除：
```
ctai_model_code/yolo_dataset/        # 太大，单独上传
ctai_model_code/__pycache__/         # 没用
ctai_model_code/CTAI_model/__pycache__/
ctai_model_code/CTAI_model/checkpoints/  # 本地是空的，不影响
ctai_model_code/.ipynb_checkpoints/  # 没用
```

PowerShell 一键打包示例：
```powershell
cd 'C:\Users\da983\CAT_2Ck (3)\CAT_2Ck'
# 1) 数据
Compress-Archive -Path '直肠癌数据' -DestinationPath 'rectal-cancer-data.zip'

# 2) 代码 + splits（排除 yolo_dataset / pycache）
$tmp = 'kaggle_code_tmp'
Remove-Item -Recurse -Force $tmp -ErrorAction Ignore
robocopy ctai_model_code $tmp /MIR /XD yolo_dataset __pycache__ .ipynb_checkpoints
Compress-Archive -Path $tmp -DestinationPath 'ctai-code-and-splits.zip'

# 3) yolo_dataset 单独打包
Compress-Archive -Path 'ctai_model_code\yolo_dataset' -DestinationPath 'ctai-yolo-dataset.zip'
```

上传方式：Kaggle → Datasets → New Dataset → 拖拽 zip → Kaggle 自动解压。

---

## 二、并行执行流程（建议下午开两个 session 一起跑）

### Session A：UNet 训练
1. New Notebook → **Add Data**：选 `rectal-cancer-data` + `ctai-code-and-splits`（**不需要** yolo dataset）
2. 上传 `train_unet_kaggle.ipynb` 或在 Files 区粘贴
3. 右上 **Settings**：Accelerator = `GPU T4 ×2`，Internet = ON
4. **第一遍** Run All（默认 `SMOKE_TEST = True`，5 epoch，约 15 分钟）
5. 看 Cell 5 输出：
   - `unet_best_ema.pth` 存在
   - `metadata 正常` 这行出现
   - `epoch=5, best_dice=...`（dice 值高低无所谓，关键是字段全部写入）
   - `unet_output.zip` 已生成
6. 验证通过后：
   - 编辑 Cell 3：把 `SMOKE_TEST = True` 改成 `SMOKE_TEST = False`
   - 重启 kernel（Run → Restart and Run All）
   - 完整训练 200 epoch，**约 6~10 小时**
7. 训练完成后右下角下载 `unet_output.zip`，解压到本地：
   ```
   ctai_model_code/CTAI_model/checkpoints/unet_best_ema.pth     <- 推理用
   ctai_model_code/CTAI_model/checkpoints/training_curves.png   <- 论文用
   ctai_model_code/CTAI_model/checkpoints/training_log.csv      <- 论文用
   ```

### Session B：YOLO 训练（**独立 notebook，与 A 同时运行**）
1. 另开一个 Browser Tab → New Notebook → **Add Data**：选 `ctai-code-and-splits` + `ctai-yolo-dataset`（**不需要** rectal-cancer-data）
2. 上传 `train_yolo11_kaggle.ipynb`
3. 同样选 GPU T4 ×2 + Internet ON
4. **第一遍** Run All（默认 `SMOKE_TEST = True`，yolo11n × 3 epoch，约 5 分钟）
5. Cell 5 输出应有：
   - `weights/best.pt` 存在（约 5 MB）
   - `results.png`、`results.csv`、`args.yaml`
6. 验证通过后：
   - Cell 4：`SMOKE_TEST = True` → `False`
   - Restart and Run All
   - 第一轮 yolo11n × 50 epoch（**约 30~40 分钟**）→ 第二轮 yolo11s × 100 epoch（**约 1~2 小时**）
7. 训练完成后下载 `yolo_output.zip`，解压并将 `best.pt` 重命名落到本地：
   ```
   ctai_model_code/CTAI_model/checkpoints/yolo11n_best.pt   <- from tumor_det_n/best.pt
   ctai_model_code/CTAI_model/checkpoints/yolo11s_best.pt   <- from tumor_det_s/best.pt
   ```

---

## 三、烟雾测试逐项验收清单

### UNet smoke 必须看到（Cell 5 输出区）
- [ ] `[splits] fingerprint OK = 84706e8b75c8f403`
- [ ] `[fixed_split] train=... | val=... (test 患者已剔除)`
- [ ] 5 个 epoch 全部跑完，每 epoch 有 `Loss=...` 输出
- [ ] `[repackage] .../unet_best_ema.pth (epoch=5 ...)`
- [ ] `[csv] .../training_log.csv`
- [ ] Cell 5 metadata 校验：`split_fingerprint = '84706e8b75c8f403'`、`patient_ids_train (n)= 75`、`patient_ids_val (n)= 16`
- [ ] `unet_best_ema.pth` 大小 ≈ 126 MB

### YOLO smoke 必须看到（Cell 5 输出区）
- [ ] `splits OK: fp=84706e8b75c8f403, sha16=0c83582ebd913eed`
- [ ] `train  images=6879 ... empty(neg)=3090`、`val images=930 ... empty(neg)=665`、`test images=965 ... empty(neg)=698`
- [ ] ultralytics 训练正常启动，3 epoch 全部跑完
- [ ] 产物清单包含 `weights/best.pt`、`results.png`、`results.csv`、`args.yaml`

任一项不满足 → 截图 / 复制日志 → 回到对话里发我，**不要**直接切到正式训练。

---

## 四、常见坑

| 症状 | 排查 | 解法 |
|---|---|---|
| `splits.json fingerprint mismatch` | 你改过 splits.json 但没重新生成 | 本地 `python data_split.py --force`，重新上传 ctai-code-and-splits |
| `splits.json sha16 mismatch` | splits.json 内容改了但 yolo_dataset 没重生成 | 本地 `python tools/mask_to_yolo.py --force`，重新上传 ctai-yolo-dataset |
| `找不到 splits.json / yolo_dataset/...` | dataset 没挂载或路径深 | 检查 Kaggle 右侧 Data 面板，确认 dataset 状态是 `Attached`，路径用 glob 自动定位（已加） |
| ultralytics 装不上 | Internet 没开 | Settings → Internet → On |
| GPU 不可用 | Accelerator 没选 | Settings → Accelerator → GPU T4 ×2 |
| Notebook 跑超 12 小时 | Kaggle 单 session 上限 12h | 完整 UNet 200 epoch < 10h 安全；如真碰上则降 epochs 或拆两段断点续训 |

---

## 五、训练完成后的本地落点

下载并解压两个 zip 后，本地 `ctai_model_code/CTAI_model/checkpoints/` 应有：

```
checkpoints/
├── WEIGHTS_LOCATION.md         <- 已存在的说明文档
├── unet_best_ema.pth           <- UNet 主权重（推理用，含 metadata）
├── best_model.pth              <- UNet 含 optimizer 副本（恢复训练用，可选）
├── training_curves.png         <- UNet loss/dice 曲线
├── training_log.csv            <- UNet 每 epoch 一行
├── yolo11n_best.pt             <- YOLO 第一轮（消融对照）
└── yolo11s_best.pt             <- YOLO 第二轮（论文最终）
```

到这一步，所有训练资产就绪，可以开始 Step 4：实现 `inference_cascade.py`。
