# 基于 YOLO11 与 UNet 的直肠肿瘤 CT 图像辅助诊断系统

本仓库包含论文《基于 YOLO11 与 UNet 的直肠肿瘤 CT 图像辅助诊断系统研究》的代码、论文稿件、插图和数据样本。

## 目录结构

```
.
├── ctai_model_code/              # 模型训练与评估代码
│   ├── CTAI_model/               # YOLO11 + UNet 训练/推理核心
│   │   ├── train.py
│   │   ├── colab_train.py
│   │   ├── kaggle_train.ipynb
│   │   ├── inference.py
│   │   ├── evaluate.py
│   │   ├── config.py
│   │   ├── core/  data/  net/    # 模型结构、数据加载等
│   │   └── requirements.txt
│   ├── eval_unet.py
│   └── train_yolo11_kaggle.ipynb
│
├── 直肠癌数据_tiny/              # 数据格式样本(完整数据见 DATA_FORMAT.md)
│   └── 1002/
│       ├── arterial phase/       # 动脉期 DICOM + mask
│       └── venous phase/         # 静脉期 DICOM + mask
│
├── 论文_*.md / .docx             # 论文正文(Markdown 与 Word 版本)
├── 论文插图/  论文插图_png/      # 论文插图源文件
├── 功能介绍*.docx                # 系统功能说明文档
├── build_docx.py                 # 论文 md → docx 构建脚本
├── gen_ch5_figures.py            # 第 5 章图表生成
├── gen_fig_4_2_4_4.py            # 图 4.2/4.4 生成
├── html_to_png.py                # HTML → PNG 渲染
├── take_screenshots.py           # 系统界面截图脚本
└── 最终完善说明.md
```

## 数据格式

完整数据集 `直肠癌数据.zip` (1.4GB) 未上传到 GitHub。仓库中的 `直肠癌数据_tiny/` 提供格式样本。

详见 [DATA_FORMAT.md](DATA_FORMAT.md)。

## 未上传内容

| 路径 | 大小 | 说明 |
|------|------|------|
| `直肠癌数据.zip` | 1.4 GB | 完整 CT 数据集 |
| `直肠癌数据_small/` | 56 MB | 中等规模数据子集 |
| `实验成果/` | 362 MB | 模型权重、训练曲线、可视化结果 |
| `CTAI-master/` | 3.6 GB | 参考项目原始仓库(含 datasets/checkpoints) |

如需获取上述数据,请联系仓库所有者。
