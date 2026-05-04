# 数据格式说明

本文件描述直肠癌 CT 数据集的目录与文件格式,供其他 coding agent / 协作者参考。
仓库内 `直肠癌数据_tiny/` 是一个最小可运行样本,完全遵循以下规范。

## 顶层结构

```
直肠癌数据/
├── 1001/          # 病例 ID(4 位整数)
├── 1002/
├── 1003/
└── ...
```

每个数字目录代表一位患者 / 一个病例。

## 病例内部结构

每个病例下分两个增强期相子目录:

```
1002/
├── arterial phase/   # 动脉期
└── venous phase/     # 静脉期
```

## 切片文件命名规则

每个期相目录内,按切片编号成对存放原始 DICOM 与肿瘤分割掩膜:

```
arterial phase/
├── 10001.dcm           # 第 1 张 CT 切片 (DICOM)
├── 10001_mask.png      # 对应的二值分割 mask (PNG)
├── 10002.dcm
├── 10002_mask.png
├── 10003.dcm
├── 10003_mask.png
└── ...
```

### 文件约定

| 文件 | 格式 | 说明 |
|------|------|------|
| `<序号>.dcm` | DICOM | 原始 CT 灰度切片,含完整 DICOM tag(像素间距、窗宽窗位等) |
| `<序号>_mask.png` | PNG (8-bit) | 与 `.dcm` 同尺寸的二值掩膜:0 = 背景,255(或 1) = 肿瘤 ROI |

序号通常为 5 位整数(如 `10001`、`10002`),前两位多与病例期相相关,后三位为期相内切片序号。配对时只需按文件名前缀匹配 `<序号>` ↔ `<序号>_mask`。

## 加载示例(Python)

```python
import os
import pydicom
from PIL import Image
import numpy as np

case_dir = "直肠癌数据/1002/arterial phase"
slices = sorted(f[:-4] for f in os.listdir(case_dir) if f.endswith(".dcm"))

for sid in slices:
    dcm = pydicom.dcmread(os.path.join(case_dir, f"{sid}.dcm"))
    img = dcm.pixel_array                                      # H×W, int16
    mask = np.array(Image.open(os.path.join(case_dir, f"{sid}_mask.png")))
    mask = (mask > 0).astype(np.uint8)                         # 二值化
    # img, mask 同 shape,可直接用于 UNet 训练
```

## 与代码中路径的对应

`ctai_model_code/CTAI_model/data/` 下的数据加载器假定根目录传入后,按 `case_id / phase / slice.dcm` 三层结构遍历。如需切换数据根,修改 `config.py` 中的数据路径即可。
