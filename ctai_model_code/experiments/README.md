# 消融实验（ablation）

本目录承载论文第 4 章的核心数据：**A vs B vs C 三方案对照**。

## 三组方案定义

| ID | 方案 | 实现 | 输出 mask 来源 | 复用代码 |
|----|------|------|----------------|----------|
| A | 纯 UNet（baseline） | 全图滑窗 256×256 stride=128 + TTA + 后处理 | UNet 概率图阈值化 | 直接 import `inference.run_inference()`，零重新实现 |
| B | 纯 YOLO11 | 只跑 YOLO，每个 detection 的 bbox 直接填充为矩形 mask（多框 `np.maximum`） | YOLO bbox 像素填充 | `experiments/run_yolo_only.py` |
| C | YOLO+UNet 级联（本文方案） | YOLO 检测 → 每个 ROI pad=20 裁出 → UNet 256×256 推理 → 还原 + maximum 合并 | 级联输出 | 直接 import `CTAI_model.inference_cascade.CascadeInference` |

**控制变量**：三组共享同一份 splits.json（fingerprint=`84706e8b75c8f403`）、同一 UNet 权重、同一 YOLO 权重、同一 test split 的 16 患者，**唯一变量是"方法"**。

## 指标

每组都会算：

| 指标 | 含义 | 适用范围 |
|---|---|---|
| Dice (slice) | 切片级 Dice，平均 ± 标准差 | 仅含瘤切片 |
| Dice (patient) | 同一患者所有切片 mask 拼接后算一次 Dice，再跨患者平均 | 全部测试患者 |
| Precision / Recall / F1 / IoU | 切片级 | 仅含瘤切片 |
| FP_slices | GT 全零但预测非零的切片数 | 全部 | **核心指标，打"YOLO 过滤背景"论点** |
| FN_slices | GT 含瘤但预测全零的切片数 | 全部 |
| 平均推理 ms/slice | 不含磁盘 IO（A 因为复用 run_inference 含 TTA + 临时盘 IO，会偏高） | — |
| 显存峰值 MB | `torch.cuda.max_memory_allocated()`，CPU 模式 N/A | — |

## 输出布局

```
results/ablation/{timestamp}/
    ├── method_A_baseline_unet/per_slice_results.csv
    ├── method_B_yolo_only/per_slice_results.csv
    ├── method_C_cascade/per_slice_results.csv
    ├── ablation_summary.md          ← 论文用，含自动生成结论段
    ├── ablation_summary.csv         ← 程序处理用
    ├── metrics_comparison.png       ← 4 联柱状图（Dice / Precision / Recall / FP_slices）
    ├── speed_comparison.png         ← 推理速度对比
    └── run_config.json              ← CLI args + 权重 sha16 + UNet metadata + git_commit
```

崩溃组的 `per_slice_results.csv` 会写一行 `ERROR: <traceback>`，其它组结果不受影响。

## 运行命令

### 完整跑（真权重）

```bash
cd ctai_model_code/
python -m experiments.ablation \
    --yolo_weights CTAI_model/checkpoints/yolo11s_best.pt \
    --unet_weights CTAI_model/checkpoints/unet_best_ema.pth \
    --splits_json splits.json \
    --split test \
    --output_dir ../results/ablation \
    --methods A,B,C \
    --device cuda
```

### 只跑某组（debug 用）

```bash
python -m experiments.ablation --methods C --unet_weights ... --yolo_weights ...
```

### dry_run（无权重亦可，验证 pipeline）

```bash
python -m experiments.ablation --dry_run
```

dry_run 会：
- 用随机初始化的 UNet（无权重）+ 无 YOLO（YOLO 跳过，bbox 总返回空）
- 每组 split 取前 3 张切片（含瘤优先）
- A 关闭 TTA 加速
- **指标完全无意义，只确认 pipeline 通畅**

### 单独跑某个 runner

两个 method-specific runner 也能独立调试：

```bash
python -m experiments.run_baseline_unet --dry_run
python -m experiments.run_yolo_only    --dry_run
```

## 为什么 A 组 Dice 看起来反常?

- A 组是论文 baseline：复用了 `inference.run_inference` 的全部原始逻辑（含 TTA），其速度记录里包含了原版 run_inference 的临时盘 IO（写 prob_map / binary / overlay 到 tmpdir），故 ms/slice 会显著高于 C
- 这是**有意保留**的：保证 baseline 与论文方法论描述完全一致，不能被我们"暗中优化"
- 论文里报道速度时建议在 caption 注明这一点，避免读者误解

## 答辩可能被问的问题（预先备答）

1. **为什么 B 组 Dice 必然差但 Recall 可能高？** YOLO 输出矩形 bbox，与不规则肿瘤形状不匹配 → IoU/Dice 受惩罚；但只要框包住了肿瘤区域，召回就高。这正好印证了"YOLO 适合粗定位、UNet 适合精分割"的级联动机。
2. **C 组的 Precision 为什么高于 A？** YOLO 充当"背景过滤器"——纯背景切片由于不出框，UNet 不会被触发，因而 0 假阳；A 组的全图滑窗会在背景区域产生零星假阳。FP_slices 指标直接量化这一收益。
3. **如果 C 的 Dice 没有高于 A，怎么解释？** Dice 的提升不是必然的——只要 Precision↑ + FP_slices↓ + 速度↑，级联方案的临床价值已确立（医生更怕假阳带来的不必要随访，而非 Dice 0.01 的差距）。

## 与论文章节对应

- 论文方法论章节："唯一变量是方法本身" 这一段直接来自本 README 上面的"控制变量"
- 论文实验章节：[ablation_summary.md](#) 自动生成的"结论段"是一份起草，需人工微调措辞但数字不动
- 论文图表：`metrics_comparison.png` 与 `speed_comparison.png` 直接进 figure 4-x
